"""The single-page interface, served inline.

One file, no build step, no CDN. That is a deliberate constraint rather than a
shortcut: this is a local tool for one person that must work on a laptop with
no network, and a front-end toolchain would be a second thing to install and
keep working before you could talk to your assistant.
"""

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>itsbob</title>
<style>
:root{
  --bg:#0d0f14; --panel:#141821; --raise:#1a1f2b; --line:#242b39; --line-soft:#1c2230;
  --text:#e8eaf0; --dim:#8791a5; --faint:#5a6478;
  --accent:#6ea8fe; --ok:#43c98b; --warn:#e0a458; --bad:#e5645b;
  --D:#7c8794; --C:#43c98b; --B:#6ea8fe; --A:#e0a458; --S:#e5645b;
  --radius:10px; --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f7f8fa; --panel:#fff; --raise:#f0f2f6; --line:#dfe3ea; --line-soft:#e9ecf2;
    --text:#161a22; --dim:#5c6577; --faint:#8b94a6;
    --accent:#2563eb; --ok:#0f9d58; --warn:#b45309; --bad:#dc2626;
    --D:#6b7280; --C:#0f9d58; --B:#2563eb; --A:#b45309; --S:#dc2626;
  }
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);
  font:14.5px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;overflow:hidden}
button,input,textarea,select{font:inherit;color:inherit}
button{cursor:pointer;background:none;border:none}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:var(--line);border-radius:9px}
::-webkit-scrollbar-thumb:hover{background:var(--faint)}

/* ---------- shell ---------- */
header{display:flex;align-items:center;gap:10px;padding:9px 14px;
  border-bottom:1px solid var(--line);background:var(--panel);flex-wrap:wrap;min-height:46px}
.brand{font-weight:650;letter-spacing:-.2px;display:flex;align-items:center;gap:7px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--faint);transition:background .3s}
.dot.live{background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 22%,transparent)}
.dot.busy{background:var(--accent);animation:pulse 1.1s ease-in-out infinite}
.dot.down{background:var(--bad)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto;align-items:center}
.chip{font-size:11.5px;padding:2.5px 8px;border-radius:99px;border:1px solid var(--line);
  color:var(--dim);white-space:nowrap;display:inline-flex;gap:5px;align-items:center}
.chip b{color:var(--text);font-weight:550}
.chip.ok{border-color:color-mix(in srgb,var(--ok) 40%,var(--line))}
.chip.bad{border-color:color-mix(in srgb,var(--bad) 45%,var(--line));color:var(--bad)}
.chip.act{cursor:pointer}
.chip.act:hover{border-color:var(--accent);color:var(--text)}
.auto{display:inline-flex;align-items:center;gap:7px;font-size:12px;padding:4px 11px;
  border-radius:99px;border:1px solid var(--line);color:var(--dim)}
.auto:hover{border-color:var(--faint);color:var(--text)}
.auto .led{width:7px;height:7px;border-radius:50%;background:var(--faint)}
.auto.on{border-color:var(--ok);color:var(--ok)}
.auto.on .led{background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 22%,transparent);
  animation:pulse 2.2s ease-in-out infinite}
.msg.task .bub{border-left:3px solid var(--C)}
.msg.task .who::before{content:"⟳ ";color:var(--C)}

main{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);height:calc(100vh - 46px)}
section{display:flex;flex-direction:column;min-width:0;min-height:0}
section.left{border-right:1px solid var(--line)}
.bar{display:flex;align-items:center;gap:6px;padding:7px 12px;border-bottom:1px solid var(--line);
  background:var(--panel);min-height:40px}
.bar h2{margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:var(--dim);font-weight:650}
.tabs{display:flex;gap:3px;margin-left:auto}
.tab{font-size:12px;padding:4px 10px;border-radius:7px;color:var(--dim);border:1px solid transparent}
.tab:hover{color:var(--text);background:var(--raise)}
.tab.on{color:var(--text);background:var(--raise);border-color:var(--line)}
.tab .badge{margin-left:5px;font-size:10px;color:var(--faint)}
.scroll{flex:1;overflow-y:auto;overflow-x:hidden;padding:14px;scroll-behavior:smooth}

/* ---------- chat ---------- */
.msg{margin-bottom:15px;display:flex;flex-direction:column;max-width:min(92%,760px);animation:rise .22s ease-out}
@keyframes rise{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
.msg.you{margin-left:auto;align-items:flex-end}
.who{font-size:10.5px;color:var(--faint);margin-bottom:4px;letter-spacing:.5px;text-transform:uppercase}
.bub{padding:9px 13px;border-radius:var(--radius);background:var(--panel);border:1px solid var(--line);
  white-space:pre-wrap;overflow-wrap:anywhere}
.msg.you .bub{background:color-mix(in srgb,var(--accent) 13%,var(--panel));
  border-color:color-mix(in srgb,var(--accent) 30%,var(--line))}
.msg.pending .bub{opacity:.55;border-style:dashed}
.msg.pending .who::after{content:" · waiting";color:var(--warn)}
.bub.err{border-color:var(--bad);color:var(--bad)}
.bub code{font-family:var(--mono);font-size:.9em;background:var(--raise);padding:1px 4px;border-radius:4px}
.typing{display:inline-flex;gap:4px;align-items:center;color:var(--dim);font-size:13px}
.typing i{width:5px;height:5px;border-radius:50%;background:var(--accent);animation:bounce 1.2s infinite}
.typing i:nth-child(2){animation-delay:.18s}.typing i:nth-child(3){animation-delay:.36s}
@keyframes bounce{0%,60%,100%{opacity:.25;transform:translateY(0)}30%{opacity:1;transform:translateY(-3px)}}

/* ---------- approval ---------- */
.approve{border:1px solid var(--warn);border-radius:var(--radius);background:
  color-mix(in srgb,var(--warn) 9%,var(--panel));padding:12px;margin-bottom:15px;
  animation:rise .2s ease-out;max-width:min(92%,760px)}
.approve h3{margin:0 0 4px;font-size:13px;display:flex;align-items:center;gap:7px}
.approve .cmd{font-family:var(--mono);font-size:12.5px;background:var(--bg);border:1px solid var(--line);
  border-radius:7px;padding:8px 10px;margin:8px 0;white-space:pre-wrap;overflow-wrap:anywhere;max-height:170px;overflow:auto}
.approve .why{color:var(--dim);font-size:12.5px}
.approve .acts{display:flex;gap:7px;margin-top:10px;flex-wrap:wrap;align-items:center}
.btn{padding:6px 13px;border-radius:7px;border:1px solid var(--line);font-size:13px;font-weight:550}
.btn:hover{border-color:var(--faint)}
.btn.go{background:var(--ok);border-color:var(--ok);color:#04150d}
.btn.no{background:var(--bad);border-color:var(--bad);color:#fff}
.btn.ghost{color:var(--dim)}
.countdown{margin-left:auto;font-size:11.5px;color:var(--faint);font-variant-numeric:tabular-nums}
#unread{margin-left:6px;font-size:10.5px;padding:1px 6px;border-radius:9px;
  background:var(--A);color:#fff;font-weight:600;display:none}
#unread.on{display:inline-block}

/* ---------- composer ---------- */
form{display:flex;gap:8px;padding:11px 12px;border-top:1px solid var(--line);background:var(--panel);align-items:flex-end}
textarea{flex:1;resize:none;background:var(--bg);border:1px solid var(--line);border-radius:9px;
  padding:9px 11px;min-height:42px;max-height:180px;outline:none}
textarea:focus{border-color:var(--accent)}
.send{background:var(--accent);color:#06101f;border-radius:9px;padding:0 15px;height:42px;font-weight:650}
.send:disabled{opacity:.4;cursor:default}

/* ---------- activity ---------- */
.card{border:1px solid var(--line);border-radius:var(--radius);background:var(--panel);
  margin-bottom:11px;overflow:hidden;animation:rise .2s ease-out}
.card>.top{display:flex;gap:8px;align-items:center;padding:8px 11px;border-bottom:1px solid var(--line-soft);flex-wrap:wrap}
.tier{width:19px;height:19px;border-radius:5px;display:grid;place-items:center;
  font-size:10.5px;font-weight:750;color:#08101c;flex:none}
.top .q{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
.meta{font-family:var(--mono);font-size:10.5px;color:var(--faint);white-space:nowrap}
.body{padding:9px 11px}
.step{padding:7px 0;border-top:1px dashed var(--line-soft)}
.step:first-child{border-top:none;padding-top:0}
.thought{color:var(--dim);font-size:12.5px;margin-bottom:4px}
.call{font-family:var(--mono);font-size:12px;color:var(--accent);overflow-wrap:anywhere}
.out{font-family:var(--mono);font-size:11.5px;color:var(--dim);white-space:pre-wrap;
  margin-top:4px;max-height:150px;overflow:auto;border-left:2px solid var(--line);padding-left:8px}
.out.bad{color:var(--bad);border-color:var(--bad)}
.note{font-size:12px;color:var(--dim);margin-bottom:6px}
.note b{color:var(--text);font-weight:550}
.live{border-color:var(--accent)}

/* ---------- lists ---------- */
.row{display:flex;gap:10px;padding:9px 2px;border-bottom:1px solid var(--line-soft);align-items:flex-start}
.row:last-child{border:none}
.row .grow{flex:1;min-width:0}
.row .sub{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:2px;overflow-wrap:anywhere}
.x{border:1px solid var(--line);border-radius:6px;padding:2px 8px;font-size:11px;color:var(--dim);flex:none}
.x:hover{border-color:var(--bad);color:var(--bad)}
.pill{font-size:10px;padding:1px 6px;border-radius:99px;border:1px solid var(--line);color:var(--faint)}
.empty{color:var(--faint);text-align:center;padding:44px 18px;font-size:13px;line-height:1.7}
.empty code{font-family:var(--mono);background:var(--raise);padding:1px 5px;border-radius:4px;font-size:12px}
.mini{display:flex;gap:6px;padding:10px 12px;border-top:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
.mini input{flex:1;min-width:100px;background:var(--bg);border:1px solid var(--line);
  border-radius:7px;padding:6px 9px;font-size:13px;outline:none}
.mini input:focus{border-color:var(--accent)}
.mini .btn{white-space:nowrap}
kbd{font-family:var(--mono);font-size:10.5px;border:1px solid var(--line);border-bottom-width:2px;
  border-radius:4px;padding:0 4px;color:var(--dim)}

@media(max-width:920px){
  body{overflow:auto}
  main{grid-template-columns:1fr;height:auto}
  section.left{border-right:none;border-bottom:1px solid var(--line)}
  .scroll{max-height:60vh}
}
</style></head><body>

<header>
  <div class="brand"><span class="dot" id="dot"></span> itsbob</div>
  <button class="auto" id="auto" onclick="toggleAuto()" title="Run scheduled work continuously">
    <span class="led"></span><span id="auto-label">manual</span>
  </button>
  <button class="auto" id="messages-link" onclick="window.open('/messages','itsbob-messages')"
          title="Everything itsbob said without being asked — opens in its own window">
    messages<span id="unread"></span>
  </button>
  <div class="chips" id="chips"><span class="chip">connecting…</span></div>
</header>

<main>
  <section class="left">
    <div class="bar"><h2>conversation</h2>
      <div class="tabs">
        <button class="tab" onclick="resetChat()" title="Keeps long-term memory">new</button>
      </div>
    </div>
    <div class="scroll" id="chat">
      <p class="empty">Ask it something, or tell it something worth remembering.<br>
        <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a newline · <kbd>/</kbd> to focus<br>
        Keep typing while it works — messages queue and run in order.</p>
    </div>
    <form id="form">
      <textarea id="msg" rows="1" placeholder="Message itsbob…"></textarea>
      <button class="send" id="send">Send</button>
    </form>
  </section>

  <section>
    <div class="bar"><h2 id="rt">activity</h2>
      <div class="tabs">
        <button class="tab on" data-panel="activity">activity</button>
        <button class="tab" data-panel="memory">memory <span class="badge" id="nmem"></span></button>
        <button class="tab" data-panel="tasks">tasks <span class="badge" id="ntask"></span></button>
        <button class="tab" data-panel="scripts">scripts</button>
        <button class="tab" data-panel="apis">apis <span class="badge" id="napi"></span></button>
        <button class="tab" data-panel="audit">audit</button>
      </div>
    </div>
    <div class="scroll" id="right"></div>
    <div class="mini" id="mini" hidden></div>
  </section>
</main>

<script>
"use strict";
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const TIERVAR = {D:"--D",C:"--C",B:"--B",A:"--A",S:"--S"};
const state = {panel:"activity", cards:[], live:null, status:null, pending:new Map(),
               busy:false, queue:[]};

// Every request is bounded. A server that accepts a connection and then never
// answers is indistinguishable from a slow one to `fetch`, which waits forever
// — so the page sat on "connecting…" with no error and nothing to retry. It was
// a deadlock in the status endpoint that put it there, and that is fixed, but a
// UI that can hang silently on one bad response will find another way to.
async function api(url, opts, timeoutMs = 12000){
  const stop = new AbortController();
  const timer = setTimeout(() => stop.abort(), timeoutMs);
  let r;
  try{
    r = await fetch(url, {...(opts || {}), signal: stop.signal});
  }catch(e){
    throw new Error(e.name === "AbortError"
      ? `no answer in ${timeoutMs / 1000}s — the server accepted the request but never replied`
      : e.message);
  }finally{
    clearTimeout(timer);
  }
  let body = {};
  try { body = await r.json(); } catch { /* empty body is fine */ }
  if(!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
  return body;
}
const post = (url, data) => api(url, {method:"POST", headers:{"Content-Type":"application/json"},
                                      body:JSON.stringify(data||{})});

/* ---------------- status ---------------- */
function ago(ts){
  if(!ts) return "never";
  const d = ts*1000 - Date.now(), s = Math.abs(d)/1000;
  const t = s<90 ? `${Math.round(s)}s` : s<5400 ? `${Math.round(s/60)}m`
          : s<172800 ? `${Math.round(s/3600)}h` : `${Math.round(s/86400)}d`;
  return d > 0 ? `in ${t}` : `${t} ago`;
}

async function refresh(){
  try{
    const s = await api("/api/status"); state.status = s; state.statusErrors = 0;
    const bits = [];
    for(const [tier, info] of Object.entries(s.tiers)){
      const on = !!info.model;
      bits.push(`<span class="chip ${on?"ok":"bad"}" title="${esc(info.label)}${on?": "+esc(info.model):" — nothing configured"}">
        <b>${tier}</b>${on ? esc(info.model.replace(/^gemini-/,"")) : "none"}</span>`);
    }
    if(s.local){
      // "Configured" and "answered" are different claims. Show the second.
      const hit = Math.round((s.local.hit_rate ?? 0) * 100);
      const ok = (s.local.answers ?? 0) > 0 || (s.local.calls ?? 0) === 0;
      bits.push(`<span class="chip ${ok?"ok":"bad"}" title="Ollama: ${s.local.answers ?? 0} of ${
        s.local.calls ?? 0} calls answered locally${s.local.last_error
        ? " — last error: "+esc(s.local.last_error) : ""}">local${
        s.local.calls ? " "+hit+"%" : ""}</span>`);
    }
    if(s.discord?.running)
      bits.push(`<span class="chip ok" title="Watching the Discord channel">discord</span>`);
    const m = s.memory || {};
    bits.push(`<span class="chip ${m.semantic_recall?"ok":""}" title="${m.semantic_recall
        ? "Semantic recall is live" : "Keyword-only recall"}"><b>${m.records ?? 0}</b> memories</span>`);
    bits.push(`<span class="chip" title="Workspace: ${esc(s.policy.workspace)}">${esc(s.policy.mode)}</span>`);
    if(s.queued?.length)
      bits.push(`<span class="chip act" onclick="clearQueue()"
        title="Click to drop everything waiting">${s.queued.length} queued ✕</span>`);
    if(s.autonomous?.deferrals)
      bits.push(`<span class="chip" title="${esc(s.autonomous.last_reason || "")}"
        >${s.autonomous.deferrals} deferred</span>`);
    if(s.auto_allowed?.length)
      bits.push(`<span class="chip act" onclick="alert('Allowed for this session without asking:\\n\\n'+${
        JSON.stringify(JSON.stringify(s.auto_allowed.join("\\n")))})">${s.auto_allowed.length} auto-allowed</span>`);
    $("chips").innerHTML = bits.join("");
    $("nmem").textContent = m.records ?? "";
    $("ntask").textContent = s.tasks?.length || "";
    const services = apiRows(s);
    $("napi").textContent = `${services.filter(a => a.configured).length}/${services.length}`;
    if(Object.keys(s.problems || {}).length)
      bits.push(`<span class="chip bad" title="${esc(JSON.stringify(s.problems))}"
        >${Object.keys(s.problems).length} subsystem(s) failing</span>`);
    const unread = $("unread");
    unread.textContent = s.unread || "";
    unread.className = s.unread ? "on" : "";
    state.queue = s.queued || [];
    renderQueue();
    paintAuto(s.autonomous);
    if(!state.busy) setDot("live");
  }catch(e){
    // Named plainly, with the one thing worth trying. "connecting…" forever
    // tells you nothing; this tells you where it broke.
    state.statusErrors = (state.statusErrors || 0) + 1;
    $("chips").innerHTML =
      `<span class="chip bad" title="${esc(e.message)}">status unavailable</span>` +
      `<span class="chip act" onclick="refresh()">retry</span>` +
      (state.statusErrors > 2
        ? `<span class="chip">${state.statusErrors} attempts — check the terminal running itsbob</span>`
        : "");
    setDot("down");
  }
}
function setDot(cls){ $("dot").className = "dot " + cls; }

/* ---------------- chat ---------------- */
function renderQueue(){
  document.querySelectorAll(".msg.pending").forEach(n => n.remove());
  const chat = $("chat");
  for(const item of state.queue){
    const task = item.source === "task";
    const el = document.createElement("div");
    el.className = "msg pending " + (task ? "task" : "you");
    el.innerHTML = `<div class="who">${task ? esc(item.label || "scheduled") : "you"}</div>
      <div class="bub">${esc(item.text)}</div>`;
    chat.appendChild(el);
  }
  if(state.queue.length) chat.scrollTop = chat.scrollHeight;
}

async function toggleAuto(){
  const want = !(state.status?.autonomous?.running);
  try{
    const status = await post("/api/autonomous", {enabled: want});
    paintAuto(status);
    bubble("itsbob", want
      ? "Continuous mode on — I'll run my scheduled work as it comes due. Keep talking; your messages go ahead of anything queued."
      : "Continuous mode off. Scheduled work will wait until you turn it back on, or until `itsbob serve` is running.");
  }catch(e){ alert(e.message); }
  refresh();
}
function paintAuto(status){
  const on = !!status?.running;
  $("auto").className = "auto" + (on ? " on" : "");
  $("auto-label").textContent = on
    ? (status.runs ? `continuous · ${status.runs} run${status.runs === 1 ? "" : "s"}` : "continuous")
    : "manual";
  $("auto").title = on
    ? "Running scheduled work. Click to stop."
    : "Scheduled work is not running. Click to start.";
}

function bubble(who, text, cls){
  const chat = $("chat"); chat.querySelector(".empty")?.remove();
  const el = document.createElement("div");
  el.className = "msg " + (cls || "");
  el.innerHTML = `<div class="who">${who}</div><div class="bub">${esc(text)}</div>`;
  chat.appendChild(el); chat.scrollTop = chat.scrollHeight;
  return el;
}
function thinking(){
  const el = bubble("itsbob", "");
  el.querySelector(".bub").innerHTML =
    `<span class="typing"><i></i><i></i><i></i> <span id="think-label">thinking</span></span>`;
  return el;
}
function setThinking(text){ const l = $("think-label"); if(l) l.textContent = text; }

/* ---------------- approvals ---------------- */
function approvalCard(d){
  const chat = $("chat"); chat.querySelector(".empty")?.remove();
  const el = document.createElement("div");
  el.className = "approve"; el.id = "ap-" + d.id;
  const args = Object.entries(d.params || {})
    .map(([k,v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`).join("\n");
  el.innerHTML = `
    <h3>⚠ ${esc(d.tool)} needs your approval <span class="pill">${esc(d.risk)}</span></h3>
    ${d.reason ? `<div class="why">${esc(d.reason)}</div>` : ""}
    <div class="cmd">${esc(args || "(no arguments)")}</div>
    <div class="acts">
      <button class="btn go" data-a="1">Allow once</button>
      <button class="btn no" data-a="0">Deny</button>
      <button class="btn ghost" data-a="2">Always allow ${esc(d.tool)}</button>
      <span class="countdown" id="cd-${d.id}"></span>
    </div>`;
  el.querySelectorAll("button[data-a]").forEach(b => b.onclick = () =>
    decide(d.id, b.dataset.a !== "0", b.dataset.a === "2"));
  chat.appendChild(el); chat.scrollTop = chat.scrollHeight;

  const deadline = Date.now() + (d.timeout || 180) * 1000;
  const timer = setInterval(() => {
    const left = Math.max(0, Math.round((deadline - Date.now())/1000));
    const node = $("cd-" + d.id);
    if(!node){ clearInterval(timer); return; }
    node.textContent = left ? `denied in ${left}s` : "timed out";
    if(!left) clearInterval(timer);
  }, 1000);
  state.pending.set(d.id, {el, timer});
}
async function decide(id, approved, remember){
  const entry = state.pending.get(id);
  if(entry) entry.el.querySelectorAll("button").forEach(b => b.disabled = true);
  try{ await post("/api/approve", {id, approved, remember}); }
  catch(e){ closeApproval(id, `— ${e.message}`); }
}
function closeApproval(id, verdict){
  const entry = state.pending.get(id);
  if(!entry) return;
  clearInterval(entry.timer);
  entry.el.style.opacity = ".6";
  entry.el.querySelector(".acts").innerHTML = `<span class="why">${esc(verdict)}</span>`;
  state.pending.delete(id);
}

/* ---------------- activity ---------------- */
function newCard(message){
  return {message, tier:"?", steps:[], recalled:[], wrote:[], reason:"", ms:0, tokens:0, done:false};
}
function renderCard(c, live){
  const steps = c.steps.map(s => `
    <div class="step">
      ${s.thought ? `<div class="thought">${esc(s.thought)}</div>` : ""}
      ${s.tool ? `<div class="call">${esc(s.tool)}(${esc(Object.entries(s.params||{})
          .map(([k,v]) => k+"="+JSON.stringify(v).slice(0,72)).join(", "))})</div>` : ""}
      ${s.output ? `<div class="out ${s.ok===false?"bad":""}">${esc(s.output).slice(0,1200)}</div>` : ""}
    </div>`).join("");
  return `<div class="card ${live?"live":""}">
    <div class="top">
      <span class="tier" style="background:var(${TIERVAR[c.tier]||"--B"})">${esc(c.tier)}</span>
      <span class="q">${esc(c.message)}</span>
      <span class="meta">${c.done ? `${Math.round(c.ms)}ms · ${c.tokens} tok` : "running…"}</span>
    </div>
    <div class="body">
      ${c.reason ? `<div class="note">${esc(c.reason)}</div>` : ""}
      ${c.recalled.length ? `<div class="note"><b>recalled</b> ${c.recalled
          .map(h => esc(h.content).slice(0,80)).join(" · ")}</div>` : ""}
      ${steps || `<div class="note">answering directly…</div>`}
      ${c.wrote.length ? `<div class="note"><b>remembered</b> ${c.wrote.map(esc).join(" · ")}</div>` : ""}
    </div></div>`;
}
function drawActivity(){
  const all = (state.live ? [renderCard(state.live, true)] : [])
    .concat(state.cards.map(c => renderCard(c, false)));
  $("right").innerHTML = all.length ? all.join("")
    : `<p class="empty">Every step of every turn shows up here as it happens —<br>
       the tier chosen and why, each tool call, and what came back.</p>`;
}

/* ---------------- event stream ---------------- */
let es;
function connect(){
  es = new EventSource("/api/stream");
  es.onopen = () => setDot(state.busy ? "busy" : "live");
  es.onerror = () => { setDot("down"); };   // EventSource retries by itself
  es.onmessage = ev => {
    let e; try { e = JSON.parse(ev.data); } catch { return; }
    handle(e);
  };
}
function handle(e){
  const d = e;
  switch(e.kind){
    case "turn_start": {
      state.busy = true; setDot("busy");
      // This message was waiting; promote it from greyed to a real bubble.
      const waiting = state.queue.findIndex(q => q.text === d.message);
      if(waiting !== -1) state.queue.splice(waiting, 1);
      renderQueue();
      bubble(d.source === "task" ? (d.label || "scheduled") : "you", d.message,
             d.source === "task" ? "task" : "you");
      thinking();
      state.live = newCard(d.message);
      if(state.panel === "activity") drawActivity();
      break;
    }
    case "queued":
      if(!state.queue.some(q => q.text === d.message)){
        state.queue.push({text: d.message, source: d.source || "user", label: d.label || ""});
        renderQueue();
      }
      refresh();
      break;
    case "autonomous":       refresh(); break;
    case "task_finished":    refresh(); break;
    case "deferred":
      bubble("itsbob", `Holding back "${d.task}" — ${d.reason}. Retrying in ${Math.round(d.retry_in_s/60)} min.`);
      break;
    case "notified":
      bubble("itsbob", `🔔 ${d.title}\n${d.body}`);
      break;
    case "queue_cleared":
      state.queue = []; renderQueue();
      break;
    case "classified":
      if(state.live){ state.live.tier = d.tier; state.live.reason = d.decision?.reasoning || ""; }
      setThinking(`thinking · tier ${d.tier}`);
      if(state.panel === "activity") drawActivity();
      break;
    case "memory":
      if(!state.live) break;
      if(d.recalled) state.live.recalled = d.recalled;
      if(d.wrote) state.live.wrote.push(d.wrote);
      if(state.panel === "activity") drawActivity();
      break;
    case "tool":
      if(state.live) state.live.steps.push({thought:d.thought, tool:d.name, params:d.params});
      setThinking(`running ${d.name}`);
      if(state.panel === "activity") drawActivity();
      break;
    case "observation": {
      if(!state.live) break;
      const last = [...state.live.steps].reverse().find(s => s.tool === d.tool && s.output === undefined);
      if(last){ last.output = d.output; last.ok = d.ok; }
      setThinking("thinking");
      if(state.panel === "activity") drawActivity();
      break;
    }
    case "approval_request": approvalCard(d); setThinking(`waiting for you · ${d.tool}`); break;
    case "approval_decided": closeApproval(d.id, d.approved ? "✓ allowed" : "✗ denied"); break;
    case "approval_timeout": closeApproval(d.id, "✗ timed out — denied"); break;
    case "approval_auto":    setThinking(`running ${d.tool}`); break;
    case "turn_end": {
      state.busy = false; setDot("live");
      document.querySelectorAll(".msg .typing").forEach(n => n.closest(".msg").remove());
      bubble("itsbob", d.reply || "(no answer)");
      if(state.live){
        Object.assign(state.live, {done:true, ms:d.turn.duration_ms, tokens:d.turn.tokens,
                                   tier:d.turn.tier, steps:d.turn.steps.map(s => ({
                                     thought:s.thought, tool:s.tool, params:s.params,
                                     output:s.observation, ok:s.ok}))});
        state.cards.unshift(state.live); state.live = null;
      }
      if(state.panel === "activity") drawActivity();
      refresh();
      break;
    }
    case "turn_error":
      state.busy = false; setDot("live");
      document.querySelectorAll(".msg .typing").forEach(n => n.closest(".msg").remove());
      bubble("itsbob", d.error, "").querySelector(".bub").classList.add("err");
      state.live = null;
      if(state.panel === "activity") drawActivity();
      break;
  }
}

/* ---------------- panels ---------------- */
async function drawMemory(){
  const q = $("mq")?.value || "";
  const {hits, total} = await api("/api/memory?q=" + encodeURIComponent(q));
  $("right").innerHTML = hits.length ? hits.map(h => `<div class="row">
      <div class="grow"><div>${esc(h.content)}</div>
        <div class="sub">${esc(h.kind)} · ${esc(h.why)} · ${ago(h.created_at)}${
          h.tags?.length ? " · " + h.tags.map(esc).join(", ") : ""}</div></div>
      <button class="x" onclick="forget('${h.id}')">forget</button></div>`).join("")
    : `<p class="empty">${q ? "Nothing matches that." :
        "Nothing remembered yet. Tell it something durable —<br>a preference, where something lives, a decision."}</p>`;
}
async function forget(id){ await post("/api/memory/forget", {id}); drawMemory(); refresh(); }
async function addMemory(){
  const box = $("mq"); if(!box.value.trim()) return;
  await post("/api/memory", {content: box.value.trim()});
  box.value = ""; drawMemory(); refresh();
}

async function drawTasks(){
  // Its own endpoint, not a slice of /api/status: the tasks panel used to be
  // the only one coupled to that payload, so it went dark whenever anything
  // unrelated in it was slow.
  let s;
  try{
    s = await api("/api/tasks");
  }catch(e){
    $("right").innerHTML = `<p class="empty">Could not load tasks: ${esc(e.message)}<br>
      <button class="btn" onclick="drawTasks()">Try again</button></p>`;
    return;
  }
  const note = s.tasks.length && !s.runner?.autonomous
    ? `<div class="card"><div class="body"><div class="note">
        Nothing is running these right now. Turn on continuous mode (the button
        top-left), or run <code>itsbob serve</code>.</div></div></div>`
    : "";
  $("right").innerHTML = note + (s.tasks.length ? s.tasks.map(t => `<div class="row">
      <div class="grow">
        <div>${esc(t.name)} <span class="pill">${esc(t.schedule)}</span>
          ${t.enabled ? "" : '<span class="pill">paused</span>'}</div>
        <div class="sub">${esc(t.prompt).slice(0,110)}</div>
        <div class="sub">next ${ago(t.next_run)} · ${t.run_count} run(s) · ${esc(t.last_status || "never run")}</div>
      </div>
      <button class="x" onclick="taskAct('run','${t.id}')">run</button>
      <button class="x" onclick="taskAct('${t.enabled?"disable":"enable"}','${t.id}')">${t.enabled?"pause":"resume"}</button>
      <button class="x" onclick="taskAct('remove','${t.id}')">remove</button></div>`).join("")
    : `<p class="empty">No scheduled work yet.<br>
        Add one below — it runs under continuous mode, or <code>itsbob serve</code>.</p>`);
}
async function taskAct(action, id){
  try{ await post("/api/task/" + action, {id}); }
  catch(e){ alert(e.message); return; }
  if(action === "run"){ state.panel = "activity"; syncTabs(); }
  await refresh(); render();
}
async function addTask(){
  const [n,p,s] = ["tn","tp","ts"].map(i => $(i).value.trim());
  if(!(n && p && s)) return alert("Name, instruction and schedule are all needed.");
  try{ await post("/api/task", {name:n, prompt:p, schedule:s}); }
  catch(e){ return alert(e.message); }
  // Cleared only once the server has actually accepted it — clearing on a
  // rejected schedule threw away what you typed along with the mistake.
  ["tn","tp","ts"].forEach(i => $(i).value = "");
  await refresh(); drawTasks();
}

const RISK_COLOR = {read:"--C", write:"--B", network:"--B", execute:"--A", destructive:"--S"};
async function drawScripts(){
  const {scripts} = await api("/api/scripts");
  $("right").innerHTML = scripts.map(s => `<div class="card">
      <div class="top"><span class="q"><b>${esc(s.name)}</b></span>
        <span class="meta">${s.tools.length} tool${s.tools.length === 1 ? "" : "s"}</span></div>
      <div class="body"><div class="note">${esc(s.summary)}</div>
        ${s.tools.map(t => `<div class="step">
          <div class="call">${esc(t.name)}
            <span class="pill" style="border-color:var(${RISK_COLOR[t.risk] || "--line"});
                  color:var(${RISK_COLOR[t.risk] || "--dim"})">${esc(t.risk)}</span></div>
          <div class="thought">${esc(t.description)}</div></div>`).join("")}
      </div></div>`).join("")
    || `<p class="empty">No scripts registered.</p>`;
}

// The registered catalog and the built-ins, merged. On its own the catalog
// only holds APIs whose key is already set, so it can show what works and not
// what you could switch on — which is the more useful half when something is
// missing.
function apiRows(s){
  const rows = new Map();
  for(const service of s.services || [])
    rows.set(service.name, {name:service.name, configured:!!service.configured,
                            key_env:service.key_env, description:service.description,
                            base_url:"", builtin:true});
  for(const api of s.apis || []){
    const existing = rows.get(api.name) || {};
    rows.set(api.name, {...existing, ...api, configured:!!api.configured,
                        description:api.description || existing.description});
  }
  // Capabilities that are real but live outside the API catalog.
  rows.set("web search", {name:"web search", configured:true, key_env:"",
    description:`No key needed — via ${s.search_backend || "duckduckgo"}. ` +
                (s.search_backend === "duckduckgo-html"
                  ? "Install ddgr for structured results." : "")});
  rows.set("discord", {name:"discord", configured:!!s.discord?.configured,
    key_env:"DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID",
    description:s.discord?.running
      ? "Watching the channel — it can post to you unprompted."
      : "Post to your channel unprompted, and take messages back."});
  rows.set("vision", {name:"vision", configured:!!s.vision?.pillow, key_env:"pip install -e '.[vision]'",
    description:"describe_image reads screenshots and photos. Needs pillow to resize first."});
  return [...rows.values()].sort((a, b) =>
    (b.configured - a.configured) || a.name.localeCompare(b.name));
}

async function drawApis(){
  const s = state.status || await api("/api/status");
  const rows = apiRows(s);
  const live = rows.filter(a => a.configured).length;
  // The point of this panel is answering "can I schedule a task that uses X?"
  // before writing the task, rather than at 07:00 tomorrow when it fails.
  const head = `<div class="card"><div class="body"><div class="note">
      <b>${live} of ${rows.length}</b> ready. A task can only use something that is live —
      put the missing key in <code>~/.itsbob/.env</code> and restart itsbob.
    </div></div></div>`;
  $("right").innerHTML = head + rows.map(a => `<div class="row">
      <div class="grow">
        <div>${esc(a.name)}
          <span class="pill" style="border-color:var(${a.configured ? "--C" : "--S"});
                color:var(${a.configured ? "--C" : "--S"})">${a.configured ? "live" : "not set up"}</span>
        </div>
        <div class="sub">${esc(a.description || a.base_url || "")}</div>
        ${a.configured
          ? (a.base_url ? `<div class="sub">${esc(a.base_url)}</div>` : "")
          : `<div class="sub">needs <code>${esc(a.key_env || "configuration")}</code></div>`}
      </div>
      ${a.configured && a.base_url
        ? `<button class="x" onclick="taskFromApi('${esc(a.name)}')">schedule…</button>`
        : ""}
    </div>`).join("");
}

function taskFromApi(name){
  // Drops a starting point into the task form rather than creating anything:
  // the schedule and the wording are the person's call, not ours.
  state.panel = "tasks"; syncTabs(); render();
  setTimeout(() => {
    $("tn").value = name;
    $("tp").value = `Use the ${name} API to `;
    $("ts").value = "daily at 08:00";
    $("tp").focus();
    $("tp").setSelectionRange($("tp").value.length, $("tp").value.length);
  }, 0);
}

async function drawAudit(){
  const {entries} = await api("/api/audit");
  $("right").innerHTML = entries.length ? entries.slice().reverse().map(e => `<div class="row">
      <div class="grow"><div class="call">${esc(e.tool)}</div>
        <div class="sub">${esc(e.iso)} · ${e.denied ? "DENIED" : (e.ok ? "ok" : "failed")}</div>
        ${e.error ? `<div class="out bad">${esc(e.error).slice(0,300)}</div>` : ""}</div></div>`).join("")
    : `<p class="empty">No tools have run yet.<br>Everything it does — including what it was refused — lands here.</p>`;
}

function render(){
  const mini = $("mini");
  $("rt").textContent = {activity:"activity", memory:"memory", tasks:"scheduled tasks",
                         scripts:"what it can do", apis:"configured APIs",
                         audit:"tool activity"}[state.panel];
  if(state.panel === "activity"){ mini.hidden = true; drawActivity(); }
  else if(state.panel === "memory"){
    mini.hidden = false;
    mini.innerHTML = `<input id="mq" placeholder="search or type something to remember…">
      <button class="btn" onclick="drawMemory()">Search</button>
      <button class="btn" onclick="addMemory()">Remember</button>`;
    $("mq").onkeydown = e => { if(e.key === "Enter") drawMemory(); };
    drawMemory();
  }
  else if(state.panel === "tasks"){
    mini.hidden = false;
    mini.innerHTML = `<input id="tn" placeholder="name" style="max-width:110px">
      <input id="tp" placeholder="what should it do?">
      <input id="ts" placeholder="every 30m" style="max-width:130px">
      <button class="btn" onclick="addTask()">Add</button>`;
    drawTasks();
  }
  else if(state.panel === "scripts"){ mini.hidden = true; drawScripts(); }
  else if(state.panel === "apis"){ mini.hidden = true; drawApis(); }
  else { mini.hidden = true; drawAudit(); }
}
function syncTabs(){
  document.querySelectorAll(".tab[data-panel]").forEach(b =>
    b.classList.toggle("on", b.dataset.panel === state.panel));
}
document.querySelectorAll(".tab[data-panel]").forEach(b => b.onclick = () => {
  state.panel = b.dataset.panel; syncTabs(); render();
});

/* ---------------- composer ---------------- */
const box = $("msg");
box.addEventListener("input", () => {
  box.style.height = "auto"; box.style.height = Math.min(box.scrollHeight, 180) + "px";
});
box.addEventListener("keydown", e => {
  if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); $("form").requestSubmit(); }
});
document.addEventListener("keydown", e => {
  if(e.key === "/" && document.activeElement !== box && !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)){
    e.preventDefault(); box.focus();
  }
  if(e.key === "Escape") box.blur();
});
$("form").addEventListener("submit", async e => {
  e.preventDefault();
  const text = box.value.trim(); if(!text) return;
  box.value = ""; box.style.height = "auto";
  try{
    const result = await post("/api/chat", {message: text});
    if(result.started_now){
      bubble("you", text, "you"); thinking();
    } else {
      // Queued behind work in flight: show it greyed until its turn comes.
      state.queue.push(text); renderQueue();
    }
  }catch(err){
    bubble("itsbob", err.message).querySelector(".bub").classList.add("err");
  }
});

async function clearQueue(){
  const {dropped} = await post("/api/queue/clear");
  state.queue = []; renderQueue(); refresh();
  if(dropped) bubble("itsbob", `Dropped ${dropped} queued message${dropped === 1 ? "" : "s"}.`);
}
async function resetChat(){
  await post("/api/reset");
  $("chat").innerHTML = `<p class="empty">New conversation.<br>It still remembers everything long-term.</p>`;
  state.cards = []; if(state.panel === "activity") drawActivity();
}

connect(); refresh(); render(); box.focus();
setInterval(refresh, 15000);
</script></body></html>
"""
