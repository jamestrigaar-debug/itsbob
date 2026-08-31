"""The console: one page, six panels, rebuilt from nothing.

The previous interface grew a panel at a time and showed it. This is a rewrite
around what the thing is actually for, which is watching an assistant work and
occasionally telling it something.

Four decisions shape the layout.

**The conversation is the page, not a panel.** It gets the left half at full
height and reads like a terminal transcript, because that is the part people
reported working well. Everything else is evidence *about* the conversation and
lives on the right.

**Every panel owns its endpoint.** The old tasks panel read its data out of the
shared status payload and went dark whenever anything unrelated in that payload
was slow. Each panel here fetches only what it draws, so one broken subsystem
costs one panel and says so in place rather than blanking.

**Status is a strip, not a hunt.** Whether it is thinking, whether the daemon is
serving, whether Discord is connected, what today has cost — those are the
questions asked at a glance, so they are one row across the top and never
require opening anything.

**Nothing spins without saying why.** Every fetch is bounded, every failure is
named where it happened, and the transport indicator distinguishes connected,
reconnecting and failed. A page that hangs silently is the failure mode this
interface has actually had, twice.
"""

from __future__ import annotations

CONSOLE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>itsbob</title>
<style>
:root{
  --bg:#0f1012; --panel:#16181c; --raise:#1e2126; --line:#2a2e35;
  --text:#e8e9ea; --dim:#9aa0a8; --faint:#6b7178;
  --accent:#6ea8fe; --ok:#5bc48a; --warn:#e8b466; --bad:#e8776a;
  --D:#7a8290; --C:#5bc48a; --B:#6ea8fe; --A:#e8b466; --S:#d98ae0;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: light){
  :root{ --bg:#f6f7f9; --panel:#fff; --raise:#f0f2f5; --line:#e0e3e8;
         --text:#1a1c1f; --dim:#5c636c; --faint:#878e97; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);height:100vh;overflow:hidden;
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  display:flex;flex-direction:column}

/* status strip */
header{display:flex;align-items:center;gap:14px;padding:9px 16px;flex-wrap:wrap;
  background:var(--panel);border-bottom:1px solid var(--line);flex:none}
.brand{font-weight:600;letter-spacing:.02em;display:flex;align-items:center;gap:8px}
.lamp{width:8px;height:8px;border-radius:50%;background:var(--faint);flex:none}
.lamp.live{background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 22%,transparent)}
.lamp.busy{background:var(--warn);animation:pulse 1.1s ease-in-out infinite}
.lamp.down{background:var(--bad)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.pills{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.pill{font-size:11.5px;padding:3px 9px;border-radius:99px;border:1px solid var(--line);
  color:var(--dim);background:var(--raise);white-space:nowrap}
.pill b{color:var(--text);font-weight:600}
.pill.ok{border-color:color-mix(in srgb,var(--ok) 45%,var(--line));color:var(--ok)}
.pill.bad{border-color:color-mix(in srgb,var(--bad) 45%,var(--line));color:var(--bad)}
.pill.act{cursor:pointer}.pill.act:hover{border-color:var(--accent);color:var(--accent)}
.spacer{flex:1}
.toggle{font:inherit;font-size:12px;padding:5px 12px;border-radius:8px;cursor:pointer;
  border:1px solid var(--line);background:var(--raise);color:var(--dim);
  display:flex;align-items:center;gap:7px}
.toggle:hover{border-color:var(--accent)}
.toggle.on{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 45%,var(--line))}
.toggle .led{width:7px;height:7px;border-radius:50%;background:var(--faint)}
.toggle.on .led{background:var(--ok)}

/* two columns */
main{flex:1;display:grid;grid-template-columns:minmax(380px,1fr) minmax(420px,1fr);
  gap:1px;background:var(--line);min-height:0}
section{background:var(--bg);display:flex;flex-direction:column;min-height:0;min-width:0}
.bar{display:flex;align-items:center;gap:4px;padding:7px 12px;flex:none;
  border-bottom:1px solid var(--line);background:var(--panel)}
.bar h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;
  color:var(--faint);margin:0;font-weight:600}
.tabs{display:flex;gap:2px;margin-left:auto;flex-wrap:wrap}
.tab{font:inherit;font-size:11.5px;padding:4px 9px;border-radius:6px;cursor:pointer;
  border:1px solid transparent;background:none;color:var(--faint)}
.tab:hover{color:var(--text);background:var(--raise)}
.tab.on{color:var(--text);background:var(--raise);border-color:var(--line)}
.tab .n{margin-left:5px;font-size:10px;color:var(--accent)}
.scroll{flex:1;overflow-y:auto;padding:14px;min-height:0}
.empty{color:var(--faint);text-align:center;padding:44px 20px;line-height:1.7}

/* chat */
.msg{margin-bottom:14px;max-width:100%}
.who{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--faint);margin-bottom:4px}
.bubble{padding:9px 13px;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere}
.msg.you .bubble{background:var(--accent);color:#0b1220;margin-left:auto;max-width:82%;width:fit-content}
.msg.you .who{text-align:right}
.msg.bob .bubble{background:var(--panel);border:1px solid var(--line)}
.msg.sys .bubble{background:none;border:1px dashed var(--line);color:var(--dim);font-size:13px}
.msg.err .bubble{border-color:var(--bad);color:var(--bad)}
.trace{margin-top:6px;font-size:11.5px;color:var(--faint);font-family:var(--mono)}
form{display:flex;gap:8px;padding:11px;border-top:1px solid var(--line);
  background:var(--panel);flex:none}
textarea{flex:1;resize:none;font:inherit;padding:9px 11px;border-radius:9px;max-height:160px;
  border:1px solid var(--line);background:var(--bg);color:var(--text)}
textarea:focus{outline:none;border-color:var(--accent)}
.send{font:inherit;padding:0 18px;border-radius:9px;border:none;cursor:pointer;
  background:var(--accent);color:#0b1220;font-weight:600}
.send:disabled{opacity:.5;cursor:default}
.queued{padding:6px 12px;font-size:11.5px;color:var(--warn);border-top:1px solid var(--line);
  background:var(--panel);display:flex;gap:8px;align-items:center;flex:none}
/* `display:flex` beats the `hidden` attribute's UA `display:none`, so an
   element set hidden in JS stays on screen. Caught in a screenshot: the
   messages panel's buttons were still showing over the activity panel. */
[hidden]{display:none !important}

/* cards */
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;
  margin-bottom:9px;overflow:hidden}
.card>.top{display:flex;gap:8px;align-items:center;padding:8px 11px;
  border-bottom:1px solid var(--line);font-size:12.5px}
.card>.body{padding:9px 11px}
.tier{font-family:var(--mono);font-weight:700;font-size:11px;padding:1px 6px;
  border-radius:4px;border:1px solid currentColor}
.meta{margin-left:auto;font-size:11px;color:var(--faint);font-variant-numeric:tabular-nums}
.step{padding:6px 0;border-top:1px dashed var(--line)}
.step:first-child{border-top:none}
.call{font-family:var(--mono);font-size:12px;color:var(--accent);overflow-wrap:anywhere}
.call.bad{color:var(--bad)}
.thought{font-size:12px;color:var(--dim);margin-top:2px}
.out{font-family:var(--mono);font-size:11.5px;color:var(--faint);margin-top:4px;
  white-space:pre-wrap;overflow-wrap:anywhere;max-height:160px;overflow-y:auto}
.row{display:flex;gap:9px;align-items:flex-start;padding:8px 10px;margin-bottom:6px;
  background:var(--panel);border:1px solid var(--line);border-radius:8px;font-size:13px}
.row .grow{flex:1;min-width:0}
.sub{font-size:11.5px;color:var(--faint);margin-top:2px;overflow-wrap:anywhere}
.x{font:inherit;font-size:11px;padding:3px 8px;border-radius:6px;cursor:pointer;
  border:1px solid var(--line);background:var(--raise);color:var(--dim)}
.x:hover{border-color:var(--accent);color:var(--accent)}
.tag{font-size:10.5px;padding:1px 6px;border-radius:99px;border:1px solid var(--line);
  color:var(--faint)}

/* approvals */
.ask{border-color:var(--warn);background:color-mix(in srgb,var(--warn) 7%,var(--panel))}
.ask .top{border-color:var(--warn)}
.ask .actions{display:flex;gap:7px;margin-top:9px;flex-wrap:wrap}
.btn{font:inherit;font-size:12px;padding:6px 13px;border-radius:7px;cursor:pointer;
  border:1px solid var(--line);background:var(--raise);color:var(--text)}
.btn.yes{background:var(--ok);color:#08130d;border-color:var(--ok);font-weight:600}
.btn.no{border-color:var(--bad);color:var(--bad)}
.countdown{margin-left:auto;font-size:11px;color:var(--faint);font-variant-numeric:tabular-nums}

/* token meter */
.meter{height:7px;border-radius:99px;background:var(--raise);overflow:hidden;display:flex;margin:7px 0}
.meter span{display:block;height:100%}
.big{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stat{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:10px}
.stat>div{min-width:96px}
.stat .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--faint)}
.mini{display:flex;gap:7px;padding:8px 11px;border-top:1px solid var(--line);
  background:var(--panel);flex:none;flex-wrap:wrap}
.mini input{flex:1;min-width:90px;font:inherit;font-size:12.5px;padding:6px 9px;
  border-radius:7px;border:1px solid var(--line);background:var(--bg);color:var(--text)}
code{font-family:var(--mono);font-size:.92em;background:var(--raise);padding:1px 5px;border-radius:4px}
@media (max-width:900px){ main{grid-template-columns:1fr;grid-template-rows:1fr 1fr} }
</style></head><body>

<header>
  <div class="brand"><span class="lamp" id="lamp"></span> itsbob</div>
  <button class="toggle" id="auto" onclick="toggleAuto()"
          title="Run scheduled work continuously, and let it speak up on its own">
    <span class="led"></span><span id="auto-label">manual</span>
  </button>
  <div class="pills" id="pills"><span class="pill">starting…</span></div>
  <span class="spacer"></span>
  <button class="toggle" onclick="window.open('/messages','itsbob-messages')"
          title="Everything itsbob said without being asked, in its own window">messages<span id="unread"></span></button>
</header>

<main>
  <section>
    <div class="bar"><h2 id="chat-title">conversation</h2>
      <div class="tabs"><button class="tab" onclick="resetChat()"
        title="Start fresh. Long-term memory is kept.">new</button></div>
    </div>
    <div class="scroll" id="chat"></div>
    <div class="queued" id="queued" hidden></div>
    <form id="form">
      <textarea id="msg" rows="1" placeholder="Message itsbob…"></textarea>
      <button class="send" id="send">Send</button>
    </form>
  </section>

  <section>
    <div class="bar"><h2 id="panel-title">activity</h2>
      <div class="tabs">
        <button class="tab on" data-panel="activity">activity</button>
        <button class="tab" data-panel="memory">memory<span class="n" id="n-mem"></span></button>
        <button class="tab" data-panel="tasks">tasks<span class="n" id="n-task"></span></button>
        <button class="tab" data-panel="tokens">tokens</button>
        <button class="tab" data-panel="messages">messages<span class="n" id="n-msg"></span></button>
        <button class="tab" data-panel="system">system</button>
      </div>
    </div>
    <div class="scroll" id="panel"></div>
    <div class="mini" id="mini" hidden></div>
  </section>
</main>

<script>
"use strict";
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const TIER = {D:"--D", C:"--C", B:"--B", A:"--A", S:"--S"};

const state = {
  panel: "activity", chat: [], cards: [], live: null,
  status: null, pending: new Map(), busy: false, queue: [], errors: 0,
};

/* ---- transport ---- */
// Bounded, always. A request that never returns is what left this page on
// "connecting…" for an entire session with nothing to click.
async function api(url, opts, ms = 12000){
  const stop = new AbortController();
  const timer = setTimeout(() => stop.abort(), ms);
  let r;
  try{ r = await fetch(url, {...(opts||{}), signal: stop.signal}); }
  catch(e){
    throw new Error(e.name === "AbortError"
      ? `no answer in ${ms/1000}s — the server took the request and never replied`
      : e.message);
  }finally{ clearTimeout(timer); }
  let body = {};
  try{ body = await r.json(); }catch{ /* empty body is fine */ }
  if(!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
  return body;
}
const post = (u, d) => api(u, {method:"POST", headers:{"Content-Type":"application/json"},
                              body: JSON.stringify(d || {})});

function lamp(kind){ $("lamp").className = "lamp " + kind; }
function ago(ts){
  if(!ts) return "never";
  const d = ts*1000 - Date.now(), s = Math.abs(d)/1000;
  const t = s<90 ? `${Math.round(s)}s` : s<5400 ? `${Math.round(s/60)}m`
          : s<172800 ? `${Math.round(s/3600)}h` : `${Math.round(s/86400)}d`;
  return d > 0 ? `in ${t}` : `${t} ago`;
}
const money = n => n >= 1 ? `$${n.toFixed(2)}` : n > 0 ? `${(n*100).toFixed(2)}¢` : "$0";
const num = n => (n ?? 0).toLocaleString();

/* ---- status strip ---- */
async function refresh(){
  let s;
  try{ s = await api("/api/status"); }
  catch(e){
    state.errors++;
    $("pills").innerHTML =
      `<span class="pill bad" title="${esc(e.message)}">status unavailable</span>` +
      `<span class="pill act" onclick="refresh()">retry</span>` +
      (state.errors > 2 ? `<span class="pill">${state.errors} tries — check the terminal</span>` : "");
    lamp("down");
    return;
  }
  state.errors = 0; state.status = s;
  const p = [];

  // Serving: the daemon, not this page. They are different processes and only
  // one of them runs your schedule when the browser is closed.
  const serve = s.serving || {};
  p.push(`<span class="pill ${serve.running ? "ok" : ""}"
    title="${esc(serve.running
      ? `itsbob serve — pid ${serve.pid}, up ${Math.round((serve.uptime_s||0)/60)}m`
      : (serve.reason || "not running") + " · start it with: itsbob serve")}"
    >serving: <b>${serve.running ? "yes" : "no"}</b></span>`);

  const d = s.discord || {};
  const dWarn = d.content_warning;
  p.push(`<span class="pill ${dWarn ? "bad" : d.running ? "ok" : d.configured ? "" : "bad"}"
    title="${esc(dWarn || (d.running
      ? `listening — ${d.handled || 0} message(s) answered, ${d.mentions || 0} of them tagged`
        + (d.mention_only ? " (only answers when tagged)" : "")
      : d.configured ? "configured, not listening — turn on continuous mode"
      : (d.hint || "not configured")))}">discord: <b>${
      dWarn ? "cannot read" : d.running ? "live" : d.configured ? "idle" : "off"}</b></span>`);

  const spend = s.spend || {};
  p.push(`<span class="pill act" onclick="show('tokens')"
    title="Estimated spend today. ${num(spend.tokens)} tokens, ${
      Math.round((spend.local_share||0)*100)}% of them local and free."
    >today: <b>${money(spend.usd || 0)}</b></span>`);

  for(const [tier, info] of Object.entries(s.tiers || {})){
    if(!info.model) continue;
    p.push(`<span class="pill" style="color:var(${TIER[tier]||"--dim"})"
      title="${esc(info.label)}: ${esc(info.model)}"><b>${tier}</b></span>`);
  }
  if(s.local){
    const hit = Math.round((s.local.hit_rate ?? 0) * 100);
    p.push(`<span class="pill ${s.local.calls && !s.local.answers ? "bad" : "ok"}"
      title="${esc(`Ollama answered ${s.local.answers||0} of ${s.local.calls||0} calls`
        + (s.local.last_error ? ` — last error: ${s.local.last_error}` : ""))}"
      >local${s.local.calls ? " " + hit + "%" : ""}</span>`);
  }
  const mem = s.memory || {};
  p.push(`<span class="pill act" onclick="show('memory')"
    title="${mem.semantic_recall ? "Semantic recall is live" : "Keyword-only recall"}"
    ><b>${num(mem.records)}</b> memories</span>`);
  p.push(`<span class="pill" title="Workspace: ${esc(s.policy?.workspace||"")}">${
    esc(s.policy?.mode||"?")}${(s.policy?.auto_allow_risks||[]).includes("network")
      ? " · network open" : ""}</span>`);
  if(Object.keys(s.problems || {}).length)
    p.push(`<span class="pill bad" title="${esc(JSON.stringify(s.problems))}"
      >${Object.keys(s.problems).length} subsystem(s) failing</span>`);

  $("pills").innerHTML = p.join("");
  $("n-mem").textContent = mem.records || "";
  $("n-task").textContent = s.tasks_count || "";
  $("n-msg").textContent = s.unread || "";
  $("unread").textContent = s.unread ? ` ${s.unread}` : "";
  state.queue = s.queued || [];
  paintQueue();
  paintAuto(s.autonomous);
  if(!state.busy) lamp("live");
}

function paintAuto(a){
  const on = !!(a && a.running);
  $("auto").className = "toggle" + (on ? " on" : "");
  $("auto-label").textContent = on ? "continuous" : "manual";
  $("auto").title = on
    ? "Running its schedule, and free to speak up on its own. Click to stop."
    : "Manual: it only acts when you ask. Click to let it run its own schedule.";
}

function paintQueue(){
  const el = $("queued");
  if(!state.queue.length){ el.hidden = true; return; }
  el.hidden = false;
  el.innerHTML = `${state.queue.length} waiting: ` +
    state.queue.slice(0,3).map(q => esc((q.label || q.text || "").slice(0,42))).join(" · ") +
    ` <button class="x" onclick="clearQueue()">clear</button>`;
}

async function toggleAuto(){
  const on = !!(state.status?.autonomous?.running);
  try{ paintAuto(await post("/api/autonomous", {enabled: !on})); }
  catch(e){ alert(e.message); }
  refresh();
}
async function clearQueue(){
  try{ await post("/api/queue/clear", {}); }catch(e){ alert(e.message); }
  state.queue = []; paintQueue(); refresh();
}

/* ---- chat ---- */
function say(kind, text, extra = ""){
  state.chat.push({kind, text, extra});
  const who = {you:"you", bob:"itsbob", sys:"", err:"error"}[kind] ?? "";
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  el.innerHTML = (who ? `<div class="who">${who}</div>` : "") +
    `<div class="bubble">${esc(text)}${extra}</div>`;
  $("chat").appendChild(el);
  $("chat").scrollTop = $("chat").scrollHeight;
}
function emptyChat(){
  $("chat").innerHTML = `<p class="empty">Ask it something, or tell it something
    worth remembering.<br><kbd>Enter</kbd> sends · <kbd>Shift</kbd>+<kbd>Enter</kbd>
    for a newline<br>Keep typing while it works — messages queue and run in order.</p>`;
}

$("form").onsubmit = async ev => {
  ev.preventDefault();
  const box = $("msg"), text = box.value.trim();
  if(!text) return;
  if(!state.chat.length) $("chat").innerHTML = "";
  box.value = ""; box.style.height = "auto";
  say("you", text);
  try{
    const r = await post("/api/chat", {message: text});
    if(!r.started_now) say("sys", "queued — it is working on something else");
  }catch(e){ say("err", e.message); }
};
$("msg").addEventListener("keydown", e => {
  if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); $("form").requestSubmit(); }
});
$("msg").addEventListener("input", e => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(160, e.target.scrollHeight) + "px";
});
async function resetChat(){
  await post("/api/reset", {});
  state.chat = []; state.cards = []; emptyChat();
  if(state.panel === "activity") drawActivity();
}

/* ---- live event stream ---- */
let es;
function connect(){
  es = new EventSource("/api/stream");
  es.onopen = () => { state.errors = 0; if(!state.busy) lamp("live"); };
  es.onerror = () => lamp("down");           // EventSource retries by itself
  es.onmessage = ev => {
    let e; try{ e = JSON.parse(ev.data); }catch{ return; }
    handle(e.kind, e);
  };
}

function handle(kind, d){
  switch(kind){
    case "turn_start":
      state.busy = true; lamp("busy");
      state.live = {message: d.message, source: d.source, label: d.label, steps: [], tier: "?"};
      if(d.source !== "user" && d.message)
        say("sys", `${d.label || d.source}: started`);
      if(state.panel === "activity") drawActivity();
      break;
    case "classified":
      if(state.live){ state.live.tier = d.tier; state.live.why = d.decision?.reasoning; }
      if(state.panel === "activity") drawActivity();
      break;
    case "feasibility":
      if(state.live && !d.feasible) state.live.refused = d.reason;
      break;
    case "tool":
      if(state.live) state.live.steps.push({tool: d.name, params: d.params, thought: d.thought});
      if(state.panel === "activity") drawActivity();
      break;
    case "observation": {
      const last = state.live?.steps?.at(-1);
      if(last){ last.output = d.output; last.ok = d.ok; }
      if(state.panel === "activity") drawActivity();
      break;
    }
    case "budget_extended":
      if(state.live) state.live.extended = d.steps;
      break;
    case "memory":
      if(state.live && d.wrote)
        (state.live.wrote ||= []).push({content: d.wrote, subject: d.subject});
      break;
    case "approval_request": ask(d); break;
    case "approval_timeout":
      state.pending.delete(d.id); drawAsk();
      say("sys", `nobody answered in time, so ${d.tool} was refused`);
      break;
    case "approval_decided": state.pending.delete(d.id); drawAsk(); break;
    case "turn_end":
      state.busy = false; lamp("live");
      if(state.live){ state.live.done = true; state.live.turn = d.turn;
                      state.cards.unshift(state.live); state.live = null; }
      if(d.source === "user") say("bob", d.reply || "(no answer)");
      else if(d.reply) say("sys", `${d.label || d.source}: ${d.reply.slice(0,300)}`);
      if(state.panel === "activity") drawActivity();
      refresh();
      break;
    case "turn_error":
      state.busy = false; lamp("down");
      state.live = null; say("err", d.error || "the turn failed");
      refresh();
      break;
    case "initiative":
      say("sys", `thinking about something on its own (${d.prompt})`); break;
    case "notified":
      say("sys", `sent you a message: ${d.title}`); refresh(); break;
    case "queued": case "queue_cleared": case "autonomous": refresh(); break;
  }
}

/* ---- approvals ---- */
function ask(d){
  state.pending.set(d.id, {...d, until: Date.now() + (d.timeout || 180) * 1000});
  drawAsk(); lamp("busy");
}
function drawAsk(){
  if(state.panel === "activity") drawActivity();
}
async function decide(id, approved, remember){
  try{ await post("/api/approve", {id, approved, remember}); }
  catch(e){ say("err", e.message); }
  state.pending.delete(id); drawAsk();
}
setInterval(() => { if(state.pending.size && state.panel === "activity") drawActivity(); }, 1000);

function askCard(a){
  const left = Math.max(0, Math.round((a.until - Date.now())/1000));
  return `<div class="card ask"><div class="top">
      <b>${esc(a.tool)}</b> needs your approval
      <span class="tag">${esc(a.risk)}</span>
      <span class="countdown">${left}s — silence means no</span></div>
    <div class="body">
      ${a.reason ? `<div class="thought">${esc(a.reason)}</div>` : ""}
      <div class="out">${esc(JSON.stringify(a.params, null, 1)).slice(0, 900)}</div>
      <div class="actions">
        <button class="btn yes" onclick="decide('${a.id}',true,false)">Allow once</button>
        <button class="btn" onclick="decide('${a.id}',true,true)">Always allow ${esc(a.tool)}</button>
        <button class="btn no" onclick="decide('${a.id}',false,false)">Deny</button>
      </div></div></div>`;
}

/* ---- panels ---- */
const TITLES = {activity:"activity", memory:"memory", tasks:"scheduled tasks",
                tokens:"what it is costing", messages:"unprompted messages",
                system:"tools, APIs and scripts"};

function show(name){
  state.panel = name;
  document.querySelectorAll(".tab[data-panel]").forEach(b =>
    b.classList.toggle("on", b.dataset.panel === name));
  $("panel-title").textContent = TITLES[name];
  $("mini").hidden = true;
  ({activity:drawActivity, memory:drawMemory, tasks:drawTasks,
    tokens:drawTokens, messages:drawMessages, system:drawSystem}[name])();
}
document.querySelectorAll(".tab[data-panel]").forEach(b =>
  b.onclick = () => show(b.dataset.panel));

function stepHtml(s){
  return `<div class="step">
    <div class="call ${s.ok === false ? "bad" : ""}">${esc(s.tool)}(${
      esc(Object.entries(s.params||{}).map(([k,v]) =>
        `${k}=${String(v).slice(0,44)}`).join(", "))})</div>
    ${s.thought ? `<div class="thought">${esc(s.thought)}</div>` : ""}
    ${s.output ? `<div class="out">${esc(String(s.output).slice(0,1200))}</div>` : ""}
  </div>`;
}

function cardHtml(c, live){
  const t = c.tier || "?";
  const wrote = (c.wrote||[]).map(w =>
    `<div class="sub">remembered (${esc(w.subject||"user")}): ${esc(w.content)}</div>`).join("");
  return `<div class="card"><div class="top">
      <span class="tier" style="color:var(${TIER[t]||"--dim"})">${esc(t)}</span>
      <span>${esc((c.label || c.message || "").slice(0,60))}</span>
      <span class="meta">${live ? "working…" :
        (c.turn ? `${Math.round(c.turn.duration_ms)}ms · ${num(c.turn.tokens)} tok` : "")}</span>
    </div><div class="body">
      ${c.why ? `<div class="thought">${esc(c.why)}</div>` : ""}
      ${c.refused ? `<div class="thought">stopped before starting: ${esc(c.refused)}</div>` : ""}
      ${c.extended ? `<div class="sub">budget extended to ${c.extended} steps — it was getting somewhere</div>` : ""}
      ${(c.steps||[]).map(stepHtml).join("") ||
        `<div class="thought">${live ? "thinking…" : "answered without tools"}</div>`}
      ${wrote}
    </div></div>`;
}

function drawActivity(){
  const parts = [...state.pending.values()].map(askCard);
  if(state.live) parts.push(cardHtml(state.live, true));
  parts.push(...state.cards.slice(0, 25).map(c => cardHtml(c, false)));
  $("panel").innerHTML = parts.join("") || `<p class="empty">Every step of every turn
    lands here as it happens — the tier it chose, each tool call, and what came back.<br>
    That is the difference between "it says it did" and "it did".</p>`;
}

async function drawMemory(){
  $("mini").hidden = false;
  $("mini").innerHTML = `<input id="mq"
      placeholder="search, or type something to remember (prefix «style:» for a standing rule)…">
    <button class="x" onclick="drawMemory()">Search</button>
    <button class="x" onclick="addMemory()">Remember</button>`;
  $("mq").onkeydown = e => { if(e.key === "Enter") drawMemory(); };
  const q = $("mq")?.value?.trim() || "";
  let d;
  try{ d = await api("/api/memory?limit=40" + (q ? "&q=" + encodeURIComponent(q) : "")); }
  catch(e){ return fail(e); }
  $("panel").innerHTML = d.hits.length ? d.hits.map(h => `<div class="row">
      <div class="grow"><div>${esc(h.content)}</div>
        <div class="sub"><span class="tag">${esc(h.kind)}</span>
          <span class="tag">about ${esc(h.subject || "user")}</span>
          <span class="tag">${esc(h.horizon || "long")}-term</span>
          ${esc(h.why || "")} · ${ago(h.created_at)}</div></div>
      <button class="x" onclick="forget('${h.id}')">forget</button></div>`).join("")
    : `<p class="empty">Nothing here yet.<br>Tell it something durable — a preference,
       where something lives, a decision and why.</p>`;
}
async function addMemory(){
  const box = $("mq"); let v = box.value.trim();
  if(!v) return;
  // "style: always list every match in full" becomes a standing rule that goes
  // into the prompt on every turn, rather than a fact waiting to be recalled.
  const style = /^style\s*:/i.test(v);
  if(style) v = v.replace(/^style\s*:\s*/i, "");
  try{
    await post("/api/memory", {content: v, tags: style ? ["style"] : [],
                               kind: style ? "preference" : "fact"});
  }catch(e){ return alert(e.message); }
  box.value = ""; drawMemory(); refresh();
}
async function forget(id){
  await post("/api/memory/forget", {id}); drawMemory(); refresh();
}

async function drawTasks(){
  $("mini").hidden = false;
  $("mini").innerHTML = `<input id="tn" placeholder="name" style="max-width:100px">
    <input id="tp" placeholder="what should it do?">
    <input id="ts" placeholder="daily at 07:00" style="max-width:120px">
    <button class="x" onclick="addTask()">Add</button>`;
  let d;
  try{ d = await api("/api/tasks"); }catch(e){ return fail(e); }
  const warn = d.tasks.length && !d.runner?.autonomous && !d.runner?.serving
    ? `<div class="card"><div class="body"><div class="thought">Nothing is running
        these. Turn on continuous mode above, or run <code>itsbob serve</code>.</div>
      </div></div>` : "";
  $("panel").innerHTML = warn + (d.tasks.length ? d.tasks.map(t => `<div class="row">
      <div class="grow"><div>${esc(t.name)} <span class="tag">${esc(t.schedule)}</span>
        ${t.enabled ? "" : '<span class="tag">paused</span>'}</div>
        <div class="sub">${esc(t.prompt).slice(0,120)}</div>
        <div class="sub">next ${ago(t.next_run)} · ${t.run_count} run(s) · ${
          esc(t.last_status || "never run")}</div></div>
      <button class="x" onclick="taskAct('run','${t.id}')">run</button>
      <button class="x" onclick="taskAct('${t.enabled?"disable":"enable"}','${t.id}')">${
        t.enabled?"pause":"resume"}</button>
      <button class="x" onclick="taskAct('remove','${t.id}')">remove</button></div>`).join("")
    : `<p class="empty">No scheduled work yet.<br>Add one below — it runs under
       continuous mode, or under <code>itsbob serve</code>.</p>`);
}
async function taskAct(action, id){
  try{ await post("/api/task/" + action, {id}); }catch(e){ return alert(e.message); }
  if(action === "run") show("activity"); else drawTasks();
  refresh();
}
async function addTask(){
  const [n,p,s] = ["tn","tp","ts"].map(i => $(i).value.trim());
  if(!(n && p && s)) return alert("Name, instruction and schedule are all needed.");
  try{ await post("/api/task", {name:n, prompt:p, schedule:s}); }
  catch(e){ return alert(e.message); }
  ["tn","tp","ts"].forEach(i => $(i).value = "");
  drawTasks(); refresh();
}

async function drawTokens(){
  let d;
  try{ d = await api("/api/tokens"); }catch(e){ return fail(e); }
  const s = d.today, all = d.all_time;
  const localPct = Math.round((s.local_share || 0) * 100);
  const bar = Object.entries(s.by_purpose || {});
  const total = bar.reduce((a,[,v]) => a + v.tokens, 0) || 1;
  const colors = ["--B","--C","--A","--S","--D"];
  $("panel").innerHTML = `
    <div class="card"><div class="body">
      <div class="stat">
        <div><div class="k">today</div><div class="big">${money(s.usd)}</div></div>
        <div><div class="k">tokens</div><div class="big">${num(s.tokens)}</div></div>
        <div><div class="k">calls</div><div class="big">${num(s.calls)}</div></div>
        <div><div class="k">local &amp; free</div>
             <div class="big" style="color:var(--ok)">${localPct}%</div></div>
      </div>
      <div class="meter">${bar.map(([, v], i) =>
        `<span style="width:${100*v.tokens/total}%;background:var(${colors[i%5]})"></span>`).join("")}</div>
      <div class="sub">${bar.map(([k, v]) =>
        `${esc(k)} ${Math.round(100*v.tokens/total)}%`).join(" · ") || "nothing yet today"}</div>
      ${s.unpriced_calls ? `<div class="sub">${s.unpriced_calls} call(s) on a model with no
        published price — not counted in the total.</div>` : ""}
      <div class="sub">All time: ${money(all.usd)} over ${num(all.tokens)} tokens.
        Prices are estimates; a free allowance makes the real bill lower.</div>
    </div></div>

    <div class="card"><div class="top">where it went</div><div class="body">
      ${Object.entries(s.by_model || {}).map(([m, v]) => `<div class="step">
        <div class="call">${esc(m)}${v.local ? ' <span class="tag">free</span>' : ""}</div>
        <div class="sub">${num(v.tokens)} tokens · ${v.calls} calls · ${money(v.usd)}</div>
      </div>`).join("") || `<div class="thought">No calls today.</div>`}
    </div></div>

    <div class="card"><div class="top">recent calls</div><div class="body">
      ${d.recent.map(r => `<div class="step">
        <div class="call ${r.ok ? "" : "bad"}">${esc(r.purpose || "?")} —
          ${esc(r.model)}</div>
        <div class="sub">${num(r.tokens)} tok · ${r.latency_ms}ms · ${money(r.usd)}
          · ${ago(r.at)}</div></div>`).join("")
        || `<div class="thought">Nothing yet.</div>`}
    </div></div>`;
}

async function drawMessages(){
  let d;
  try{ d = await api("/api/messages?limit=60"); }catch(e){ return fail(e); }
  $("mini").hidden = false;
  $("mini").innerHTML = `<button class="x" onclick="readAll()">Mark all read</button>
    <button class="x" onclick="window.open('/messages','itsbob-messages')">Own window</button>`;
  $("panel").innerHTML = d.messages.length
    ? d.messages.slice().reverse().map(m => `<div class="row">
        <div class="grow"><div>${m.read ? "" : "● "}${esc(m.title)}</div>
          ${m.body ? `<div class="sub">${esc(m.body).slice(0,400)}</div>` : ""}
          <div class="sub">${esc((m.iso||"").replace("T"," "))}${
            m.task ? " · " + esc(m.task) : ""}${
            m.urgency === "high" ? ' · <span class="tag">urgent</span>' : ""}</div>
        </div></div>`).join("")
    : `<p class="empty">Nothing yet.<br>What itsbob decides to tell you unprompted —
       task results, alerts, and anything it thinks is worth saying — appears here
       and in Discord.</p>`;
}
async function readAll(){
  await post("/api/messages/read", {all:true}); drawMessages(); refresh();
}

async function drawSystem(){
  let s = state.status, scripts;
  try{
    if(!s) s = state.status = await api("/api/status");
    scripts = await api("/api/scripts");
  }catch(e){ return fail(e); }
  const services = apiRows(s);
  $("panel").innerHTML = `
    <div class="card"><div class="top">APIs and services</div><div class="body">
      <div class="thought"><b>${services.filter(a=>a.configured).length} of ${
        services.length}</b> ready. A task can only use something that is live —
        put the missing key in <code>~/.itsbob/.env</code> and restart.</div>
      ${services.map(a => `<div class="step">
        <div class="call">${esc(a.name)}
          <span class="tag" style="color:var(${a.configured?"--C":"--S"});
            border-color:currentColor">${a.configured?"live":"not set up"}</span></div>
        <div class="sub">${esc(a.description || "")}</div>
        ${a.configured ? "" : `<div class="sub">needs <code>${esc(a.key_env||"setup")}</code></div>`}
      </div>`).join("")}
    </div></div>

    <div class="card"><div class="top">scripts <span class="meta">drop a .py into ${
      esc(scripts.drop_in || "")} to add one</span></div><div class="body">
      ${scripts.scripts.map(x => `<div class="step">
        <div class="call">${esc(x.name)} <span class="tag">${x.tools.length} tool${
          x.tools.length===1?"":"s"}</span>${x.source==="broken"
            ? ' <span class="tag" style="color:var(--bad)">failed to load</span>':""}</div>
        <div class="sub">${esc(x.error || x.summary || "")}</div></div>`).join("")}
    </div></div>

    <div class="card"><div class="top">tools <span class="meta">${
      (s.tools||[]).length} available</span></div><div class="body">
      ${(s.tools||[]).map(t => `<div class="step">
        <div class="call">${esc(t.name)} <span class="tag">${esc(t.risk)}</span></div>
        <div class="sub">${esc(t.description).slice(0,160)}</div></div>`).join("")}
    </div></div>`;
}

// The catalogue only holds APIs whose key is set, so on its own it cannot show
// what you could switch on — which is the more useful half when one is missing.
function apiRows(s){
  const rows = new Map();
  for(const svc of s.services || [])
    rows.set(svc.name, {name:svc.name, configured:!!svc.configured,
                        key_env:svc.key_env, description:svc.description});
  for(const a of s.apis || [])
    rows.set(a.name, {...(rows.get(a.name)||{}), ...a, configured:!!a.configured});
  rows.set("web search", {name:"web search", configured:true, key_env:"",
    description:`No key needed — via ${s.search_backend || "duckduckgo"}.`});
  rows.set("discord", {name:"discord", configured:!!s.discord?.configured,
    key_env:"DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID",
    description:"Talk to itsbob from Discord, and let it message you there."});
  rows.set("vision", {name:"vision", configured:!!s.vision?.pillow,
    key_env:"pip install -e '.[vision]'",
    description:"Read screenshots and photos with look_at_screen."});
  return [...rows.values()].sort((a,b) =>
    (b.configured - a.configured) || a.name.localeCompare(b.name));
}

function fail(e){
  $("panel").innerHTML = `<p class="empty">Could not load this panel:<br>
    ${esc(e.message)}<br><br>
    <button class="btn" onclick="show('${state.panel}')">Try again</button></p>`;
}

/* ---- go ---- */
emptyChat();
connect();
refresh();
show("activity");
$("msg").focus();
setInterval(refresh, 15000);
document.addEventListener("keydown", e => {
  if(e.key === "/" && document.activeElement !== $("msg")){ e.preventDefault(); $("msg").focus(); }
});
</script></body></html>
"""
