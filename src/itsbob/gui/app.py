"""Browser interface: chat on the left, what it is doing on the right.

The point of the right-hand panel is that an agent with tools is opaque in a
way a chatbot is not. "It said it updated the file" and "it updated the file"
are different claims, and only one of them is verifiable from a transcript. So
every step is shown as it happens: the tier chosen and why, which model
answered, each tool call with its arguments and result, what was recalled from
memory, and what was written back.

Binds to 127.0.0.1 with no authentication. It is a local interface for one
person, not a deployed service — anything that can reach the port can run
tools as you.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

__all__ = ["create_app", "run_gui"]


def create_app(home: Path | None = None, *, mode: str | None = None):
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:  # pragma: no cover - depends on install
        raise SystemExit(
            "the GUI needs Flask — install it with:  pip install -e '.[gui]'"
        ) from exc

    from ..agent import build_agent, default_home
    from ..agent.context import Conversation
    from ..daemon import TaskStore
    from ..tools import Mode

    root = Path(home) if home else default_home()
    app = Flask(__name__)
    state: dict[str, Any] = {"agent": None, "tasks": None}
    lock = threading.Lock()

    def agent():
        # Built on first use, not at import: a browser tab opening should not
        # be what discovers that a key is missing.
        with lock:
            if state["agent"] is None:
                state["agent"] = build_agent(
                    home=root,
                    mode=Mode(mode) if mode else None,
                    # No confirm handler: a web request has nobody attached to
                    # it, so confirm-gated tools are refused rather than
                    # silently approved by whoever left the tab open.
                    confirm=None,
                )
            return state["agent"]

    def tasks():
        with lock:
            if state["tasks"] is None:
                state["tasks"] = TaskStore(root / "tasks.sqlite")
            return state["tasks"]

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.get("/api/status")
    def status():
        bob = agent()
        # `is not None`, not truthiness: LongTermMemory defines __len__, so an
        # empty store is falsy and a fresh install would report no memory
        # subsystem at all rather than an empty one.
        memory = bob.memory.stats() if bob.memory is not None else {}
        return jsonify(
            {
                "home": str(root),
                "policy": bob.toolbox.policy.describe(),
                "tools": bob.toolbox.registry.names(),
                "tiers": {
                    tier: {
                        "label": info["label"],
                        "providers": [
                            {"name": row["provider"], "model": (row["models"] or [None])[0],
                             "configured": row["configured"]}
                            for row in info["providers"]
                        ],
                    }
                    for tier, info in bob.brain.describe()["tiers"].items()
                },
                "local": bob.brain.describe()["local"],
                "memory": memory,
                "apis": bob.toolbox.catalog.describe() if bob.toolbox.catalog else [],
                "tasks": [t.as_dict() for t in tasks().all()],
                "turns": len(bob.conversation),
            }
        )

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(force=True, silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "empty message"}), 400

        bob = agent()
        events: list[dict[str, Any]] = []
        turn = bob.chat(
            message,
            context=payload.get("context") or None,
            on_event=lambda e: events.append(e.as_dict()),
        )
        return jsonify({"reply": turn.final, "turn": turn.as_dict(), "events": events})

    @app.post("/api/reset")
    def reset():
        agent().conversation = Conversation()
        return jsonify({"ok": True})

    @app.get("/api/memory")
    def memory_search():
        bob = agent()
        if bob.memory is None:
            return jsonify({"hits": []})
        query = request.args.get("q", "").strip()
        limit = min(50, int(request.args.get("limit", 12)))
        hits = (
            bob.memory.search(query, limit=limit)
            if query
            else [type("H", (), {"as_dict": lambda s, r=r: _record_dict(r)})()
                  for r in bob.memory.recent(limit)]
        )
        return jsonify({"hits": [h.as_dict() for h in hits]})

    @app.post("/api/memory/forget")
    def memory_forget():
        payload = request.get_json(force=True, silent=True) or {}
        bob = agent()
        ok = bool(bob.memory is not None and bob.memory.forget(str(payload.get("id", ""))))
        return jsonify({"ok": ok})

    @app.get("/api/audit")
    def audit():
        return jsonify({"entries": agent().toolbox.audit.recent(40)})

    @app.post("/api/task")
    def task_create():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            task = tasks().create(
                str(payload.get("name", "")).strip() or "untitled",
                str(payload.get("prompt", "")).strip(),
                str(payload.get("schedule", "")).strip(),
            )
        except Exception as exc:  # noqa: BLE001 - a bad schedule is user error
            return jsonify({"error": str(exc)}), 400
        return jsonify({"task": task.as_dict()})

    @app.post("/api/task/remove")
    def task_remove():
        payload = request.get_json(force=True, silent=True) or {}
        return jsonify({"ok": tasks().remove(str(payload.get("id", "")))})

    return app


def _record_dict(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "content": record.content,
        "kind": record.kind.value,
        "created_at": record.created_at,
        "score": 0.0,
        "why": "recent",
    }


def run_gui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    home: Path | None = None,
    mode: str | None = None,
) -> None:
    app = create_app(home, mode=mode)
    url = f"http://{host}:{port}"
    print(f"itsbob gui → {url}   (ctrl-c to stop)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, use_reloader=False)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>itsbob</title><style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--text:#e6e8ee;--dim:#8b93a7;--accent:#6ea8fe;
--D:#7c8794;--C:#4cc38a;--B:#6ea8fe;--A:#e0a458;--S:#e5534b;--ok:#4cc38a;--bad:#e5534b}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;align-items:center;gap:14px;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:15px;margin:0;letter-spacing:.4px}
.pill{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--dim);white-space:nowrap}
.pill.ok{color:var(--ok);border-color:#25543f}.pill.bad{color:var(--bad);border-color:#5a2b28}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);height:calc(100vh - 45px)}
@media(max-width:900px){main{grid-template-columns:1fr;height:auto}}
section{display:flex;flex-direction:column;min-width:0;border-right:1px solid var(--line)}
.head{padding:8px 14px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
.head b{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim);font-weight:600}
.tabs{display:flex;gap:4px;margin-left:auto}
.tab{font-size:11px;padding:3px 9px;border-radius:6px;border:1px solid var(--line);background:none;color:var(--dim);cursor:pointer}
.tab.on{color:var(--text);border-color:var(--accent)}
.scroll{flex:1;overflow-y:auto;padding:14px;min-height:320px}
.msg{margin-bottom:14px;max-width:88%}
.msg.you{margin-left:auto}
.bub{padding:9px 12px;border-radius:12px;background:var(--panel);border:1px solid var(--line);white-space:pre-wrap;word-wrap:break-word}
.msg.you .bub{background:#1d2633;border-color:#2b3a4d}
.who{font-size:10px;color:var(--dim);margin-bottom:3px;text-transform:uppercase;letter-spacing:.6px}
.msg.you .who{text-align:right}
form{display:flex;gap:8px;padding:12px 14px;border-top:1px solid var(--line)}
textarea{flex:1;resize:none;background:var(--panel);border:1px solid var(--line);color:var(--text);
border-radius:8px;padding:9px 11px;font:inherit;min-height:44px;max-height:160px}
textarea:focus{outline:none;border-color:var(--accent)}
button{background:var(--accent);border:none;color:#0b1220;font-weight:600;border-radius:8px;padding:0 16px;cursor:pointer;font:inherit}
button:disabled{opacity:.45;cursor:default}
.card{border:1px solid var(--line);border-radius:9px;margin-bottom:12px;overflow:hidden;background:var(--panel)}
.card > .top{display:flex;gap:9px;align-items:center;padding:8px 11px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.tier{font-weight:700;width:20px;height:20px;border-radius:5px;display:grid;place-items:center;font-size:11px;color:#0b1220}
.body{padding:9px 11px;font-size:12.5px}
.step{padding:6px 0;border-top:1px dashed var(--line)}
.step:first-child{border-top:none}
.tool{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--accent);word-break:break-all}
.obs{color:var(--dim);white-space:pre-wrap;margin-top:3px;max-height:130px;overflow:auto;font-family:ui-monospace,Menlo,monospace;font-size:11.5px}
.obs.bad{color:#f0a29c}
.muted{color:var(--dim)}.thought{color:var(--dim);font-style:italic;margin-bottom:2px}
.row{display:flex;gap:8px;align-items:baseline;padding:5px 0;border-bottom:1px solid var(--line);font-size:12.5px}
.row:last-child{border:none}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--dim)}
.x{margin-left:auto;background:none;border:1px solid var(--line);color:var(--dim);padding:1px 7px;border-radius:5px;font-size:11px}
.empty{color:var(--dim);text-align:center;padding:40px 20px;font-size:13px}
.mini{display:flex;gap:6px;padding:10px 14px;border-top:1px solid var(--line);flex-wrap:wrap}
.mini input{flex:1;min-width:110px;background:var(--panel);border:1px solid var(--line);color:var(--text);border-radius:7px;padding:6px 9px;font:inherit;font-size:12.5px}
.spin{display:inline-block;width:9px;height:9px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:s .7s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
</style></head><body>
<header><h1>itsbob</h1><span id="strip" class="muted" style="font-size:12px">connecting…</span></header>
<main>
  <section>
    <div class="head"><b>conversation</b>
      <div class="tabs"><button class="tab" onclick="resetChat()">new conversation</button></div>
    </div>
    <div class="scroll" id="chat"><p class="empty">Ask it something, or tell it something worth remembering.</p></div>
    <form id="form">
      <textarea id="msg" placeholder="Message… (Enter to send, Shift+Enter for a newline)" rows="1"></textarea>
      <button id="send">Send</button>
    </form>
  </section>
  <section style="border-right:none">
    <div class="head"><b id="rt">what it did</b>
      <div class="tabs">
        <button class="tab on" data-panel="trace">trace</button>
        <button class="tab" data-panel="memory">memory</button>
        <button class="tab" data-panel="tasks">tasks</button>
        <button class="tab" data-panel="audit">audit</button>
      </div>
    </div>
    <div class="scroll" id="right"><p class="empty">Every step of every turn appears here.</p></div>
    <div class="mini" id="mini" hidden></div>
  </section>
</main>
<script>
const TIER={D:'--D',C:'--C',B:'--B',A:'--A',S:'--S'};
let panel='trace', traces=[];
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

async function api(url,opts){const r=await fetch(url,opts);if(!r.ok)throw new Error((await r.json().catch(()=>({}))).error||r.statusText);return r.json();}

async function status(){
  try{
    const s=await api('/api/status');
    const bits=[`<span class="pill">${esc(s.policy.mode)} mode</span>`];
    for(const [t,info] of Object.entries(s.tiers)){
      const p=info.providers.find(x=>x.configured);
      bits.push(`<span class="pill ${p?'ok':'bad'}">${t}: ${p?esc(p.model):'none'}</span>`);
    }
    if(s.local) bits.push(`<span class="pill ok">ollama</span>`);
    bits.push(`<span class="pill ${s.memory.semantic_recall?'ok':''}">${s.memory.records} memories${s.memory.semantic_recall?'':' · keyword only'}</span>`);
    bits.push(`<span class="pill">${s.tasks.length} task${s.tasks.length===1?'':'s'}</span>`);
    $('strip').innerHTML=bits.join(' ');
  }catch(e){$('strip').innerHTML=`<span class="pill bad">${esc(e.message)}</span>`;}
}

function bubble(who,text,cls){
  const chat=$('chat'); chat.querySelector('.empty')?.remove();
  const d=document.createElement('div'); d.className='msg '+(cls||'');
  d.innerHTML=`<div class="who">${who}</div><div class="bub">${esc(text)}</div>`;
  chat.appendChild(d); chat.scrollTop=chat.scrollHeight; return d;
}

function traceCard(turn,events){
  const cls=TIER[turn.tier]||'--B';
  const recalled=events.filter(e=>e.kind==='memory'&&e.data.recalled).flatMap(e=>e.data.recalled);
  const wrote=events.filter(e=>e.kind==='memory'&&e.data.wrote).map(e=>e.data.wrote);
  const cl=events.find(e=>e.kind==='classified');
  const steps=turn.steps.map(s=>`<div class="step">
      ${s.thought?`<div class="thought">${esc(s.thought)}</div>`:''}
      ${s.tool?`<div class="tool">${esc(s.tool)}(${esc(Object.entries(s.params||{}).map(([k,v])=>k+'='+JSON.stringify(v).slice(0,60)).join(', '))})</div>
        <div class="obs ${s.ok?'':'bad'}">${esc(s.observation)}</div>`:''}
      <div class="mono">tier ${esc(s.tier)} · ${esc(s.model)} · ${Math.round(s.latency_ms)}ms</div>
    </div>`).join('');
  return `<div class="card">
    <div class="top"><span class="tier" style="background:var(${cls})">${esc(turn.tier)}</span>
      <span>${esc(turn.message).slice(0,60)}</span>
      <span class="mono" style="margin-left:auto">${Math.round(turn.duration_ms)}ms · ${turn.tokens} tok</span></div>
    <div class="body">
      ${cl?`<div class="muted" style="margin-bottom:6px">${esc(cl.data.decision.reasoning)}</div>`:''}
      ${recalled.length?`<div class="muted" style="margin-bottom:6px">recalled: ${recalled.map(h=>esc(h.content).slice(0,70)).join(' · ')}</div>`:''}
      ${steps||'<div class="muted">answered directly</div>'}
      ${wrote.length?`<div class="step muted">remembered: ${wrote.map(esc).join(' · ')}</div>`:''}
    </div></div>`;
}

function render(){
  const right=$('right'), mini=$('mini');
  mini.hidden = panel==='trace'||panel==='audit';
  if(panel==='trace'){
    $('rt').textContent='what it did';
    right.innerHTML = traces.length?traces.join(''):'<p class="empty">Every step of every turn appears here.</p>';
  } else if(panel==='memory'){ $('rt').textContent='memory'; loadMemory();
    mini.innerHTML=`<input id="mq" placeholder="search memory…"><button onclick="loadMemory()">Search</button>`;
    $('mq').onkeydown=e=>{if(e.key==='Enter')loadMemory();};
  } else if(panel==='tasks'){ $('rt').textContent='scheduled tasks'; loadTasks();
    mini.innerHTML=`<input id="tn" placeholder="name"><input id="tp" placeholder="what to do…">
      <input id="ts" placeholder="every 30m"><button onclick="addTask()">Add</button>`;
  } else { $('rt').textContent='tool activity'; loadAudit(); }
}

async function loadMemory(){
  const q=$('mq')?.value||'';
  const {hits}=await api('/api/memory?q='+encodeURIComponent(q));
  $('right').innerHTML = hits.length?hits.map(h=>`<div class="row">
      <div><div>${esc(h.content)}</div><div class="mono">${esc(h.kind)} · ${esc(h.why)}</div></div>
      <button class="x" onclick="forget('${h.id}')">forget</button></div>`).join('')
    :'<p class="empty">Nothing remembered yet.</p>';
}
async function forget(id){await api('/api/memory/forget',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});loadMemory();status();}

async function loadTasks(){
  const s=await api('/api/status');
  $('right').innerHTML = s.tasks.length?s.tasks.map(t=>`<div class="row">
      <div><div>${esc(t.name)} <span class="mono">${esc(t.schedule)}</span></div>
        <div class="mono">${esc(t.prompt).slice(0,90)}</div>
        <div class="mono">${t.enabled?'enabled':'paused'} · ${t.run_count} run(s) · ${esc(t.last_status||'never run')}</div></div>
      <button class="x" onclick="rmTask('${t.id}')">remove</button></div>`).join('')
    :'<p class="empty">No scheduled tasks. Add one below — it runs when <code>itsbob serve</code> is running.</p>';
}
async function addTask(){
  try{
    await api('/api/task',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:$('tn').value,prompt:$('tp').value,schedule:$('ts').value})});
    $('tn').value=$('tp').value=$('ts').value=''; loadTasks(); status();
  }catch(e){alert(e.message);}
}
async function rmTask(id){await api('/api/task/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});loadTasks();status();}

async function loadAudit(){
  const {entries}=await api('/api/audit');
  $('right').innerHTML = entries.length?entries.slice().reverse().map(e=>`<div class="row">
      <div><div class="tool">${esc(e.tool)}</div><div class="mono">${esc(e.iso)} · ${e.denied?'DENIED':(e.ok?'ok':'failed')}</div>
      ${e.error?`<div class="obs bad">${esc(e.error)}</div>`:''}</div></div>`).join('')
    :'<p class="empty">No tools have run yet.</p>';
}

async function resetChat(){await api('/api/reset',{method:'POST'});$('chat').innerHTML='<p class="empty">New conversation. It still remembers everything long-term.</p>';}

document.querySelectorAll('.tab[data-panel]').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tab[data-panel]').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); panel=b.dataset.panel; render();
});

$('msg').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('form').requestSubmit();}});
$('form').addEventListener('submit',async e=>{
  e.preventDefault();
  const text=$('msg').value.trim(); if(!text)return;
  $('msg').value=''; $('send').disabled=true;
  bubble('you',text,'you');
  const pending=bubble('bob','…'); pending.querySelector('.bub').innerHTML='<span class="spin"></span>';
  try{
    const data=await api('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text})});
    pending.querySelector('.bub').textContent=data.reply;
    traces.unshift(traceCard(data.turn,data.events));
    if(panel==='trace')render();
    status();
  }catch(err){
    pending.querySelector('.bub').textContent='Error: '+err.message;
    pending.querySelector('.bub').style.borderColor='var(--bad)';
  }finally{$('send').disabled=false;$('msg').focus();}
});

status(); render(); $('msg').focus();
</script></body></html>
"""
