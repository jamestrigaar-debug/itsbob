"""A chat + live "thinking" monitor GUI over the complexity router.

Left: a chat box — type a game state (JSON, or plain text — plain text is
wrapped into `{"facts": {"message": "..."}}` automatically) and see itsbob's
reply. Right: a running feed of what the pipeline actually did for each
message — which tier answered, the Gatekeeper's reasoning and fingerprint,
cache hit/miss, which provider/model got called, any escalation, which
scripts ran, and latency against the 1.8s budget. That right-hand feed is
the "monitor" — it's meant to answer "why did it do that" and "which model
is it actually calling" without reading logs.

Deliberately a single Flask file with the page template inlined as a string
— there is exactly one page worth having a GUI for, so a build step or a
frontend framework would be pure overhead. Flask is an optional dependency
(the `gui` extra); everything else in this package works without it.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from typing import Any

from ..config import Settings
from ..factory import build_router
from ..llm.local import is_ollama_running
from ..router import build_complexity_router

__all__ = ["create_app", "run_gui"]

EXAMPLE_MESSAGES = [
    '{"facts": {"stamina": 15, "minute": 60}}',
    '{"facts": {"opponent_formation": "4-3-3", "minute": 78}, "events": ["opponent playing narrow tactic"]}',
    '{"facts": {"player": "star striker"}, "events": ["player unhappy about contract renegotiation"]}',
]


def _as_game_state_json(raw_text: str) -> str:
    """Accept plain text as well as JSON: wrap non-JSON input as a fact."""
    stripped = raw_text.strip()
    if not stripped:
        return "{}"
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        return json.dumps({"facts": {"message": stripped}})


def create_app():
    try:
        from flask import Flask, jsonify, render_template_string, request
    except ImportError as exc:  # pragma: no cover - exercised via cli error path
        raise ImportError(
            "Flask is required for the GUI. Install it with: pip install -e \".[gui]\""
        ) from exc

    app = Flask(__name__)
    settings = Settings.from_env()
    state_lock = threading.Lock()
    router_holder: dict[str, Any] = {"router": None}

    def get_router():
        with state_lock:
            if router_holder["router"] is None:
                router_holder["router"] = build_complexity_router(settings)
            return router_holder["router"]

    @app.get("/")
    def index():
        return render_template_string(_PAGE, examples=json.dumps(EXAMPLE_MESSAGES))

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.get("/api/status")
    def status():
        mode = os.environ.get("ITSBOB_ROUTER_MODE", "priority").strip() or "priority"

        def describe(router) -> list[dict[str, Any]]:
            return [
                {
                    "name": row["provider"],
                    "configured": row["configured"],
                    "models": row["models"],
                    "circuit_open": row["circuit_open"],
                }
                for row in router.describe()
            ]

        if mode == "google-tiered":
            router = get_router()
            tier_info = {
                "tier_b": describe(router.cloud_router),
                "tier_a": describe(router.premium_router),
            }
            providers = tier_info["tier_b"]  # for clients that only read the flat list
        else:
            tier_info = None
            providers = describe(build_router(settings))

        return jsonify(
            {
                "mode": mode,
                "local_back_brain": {
                    "reachable": is_ollama_running(),
                    "note": "ollama serve on 127.0.0.1:11434",
                },
                "cloud_providers": providers,
                "tiers": tier_info,
            }
        )

    @app.post("/api/chat")
    def chat():
        """One turn: freeform or JSON input in, a chat reply + full trace out."""
        payload = request.get_json(force=True, silent=True) or {}
        raw_text = str(payload.get("message", ""))
        goal = payload.get("goal") or "win the league"
        classify_only = bool(payload.get("classify_only"))

        state_json = _as_game_state_json(raw_text)
        try:
            parsed = json.loads(state_json)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"invalid JSON: {exc}"}), 400

        router = get_router()
        router.goal = goal

        if classify_only:
            from ..router import compress

            decision = router.gatekeeper.classify(compress(parsed))
            trace = {"classify_only": True, "decision": decision.as_dict()}
            reply = (
                f"[classify only] Tier {decision.tier.value} ({decision.tier.label}) — "
                f"{decision.reasoning}"
            )
            return jsonify({"reply": reply, "trace": trace})

        try:
            result = router.route(parsed)
        except Exception as exc:  # noqa: BLE001 - surface it in chat, don't 500 blindly
            return jsonify(
                {
                    "reply": f"Pipeline error: {type(exc).__name__}: {exc}",
                    "trace": {"error": str(exc)},
                }
            )

        trace = result.as_dict()
        trace["cache_stats"] = router.cache.stats()

        if result.needs_user:
            reply = f"⚠ {result.note} ({result.error})"
        elif result.actions:
            reply = f"{result.note}".strip() or f"Ran: {', '.join(result.actions)}"
        else:
            reply = result.note or "(no action taken)"

        return jsonify({"reply": reply, "trace": trace})

    # Kept for scripting/curl use and backward compatibility with earlier GUI versions.
    @app.post("/api/route")
    def route():
        payload = request.get_json(force=True, silent=True) or {}
        raw_state = payload.get("state")
        goal = payload.get("goal") or "win the league"
        if raw_state is None:
            return jsonify({"error": "missing 'state'"}), 400
        try:
            parsed = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"invalid JSON: {exc}"}), 400

        router = get_router()
        router.goal = goal
        try:
            result = router.route(parsed)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
        payload = result.as_dict()
        payload["cache_stats"] = router.cache.stats()
        return jsonify(payload)

    @app.post("/api/classify")
    def classify():
        payload = request.get_json(force=True, silent=True) or {}
        raw_state = payload.get("state")
        if raw_state is None:
            return jsonify({"error": "missing 'state'"}), 400
        try:
            parsed = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"invalid JSON: {exc}"}), 400

        from ..router import compress

        router = get_router()
        decision = router.gatekeeper.classify(compress(parsed))
        return jsonify(decision.as_dict())

    return app


def run_gui(*, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    app = create_app()
    url = f"http://{host}:{port}/"
    print(f"itsbob GUI running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, use_reloader=False)


_PAGE = """
<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>itsbob router</title>
<style>
  :root {
    --bg: #f7f7f5; --panel: #ffffff; --border: #e3e1db; --text: #1c1c1a;
    --muted: #6b6b63; --accent: #2f6f4f;
    --tier-d: #6b6b63; --tier-c: #2f6f4f; --tier-b: #1d6fa8; --tier-a: #a8631d; --tier-s: #a82f2f;
    --bubble-user: #eaf1ee; --bubble-bot: #f3f2ee;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; }
  header { padding: 14px 20px 10px; border-bottom: 1px solid var(--border); flex: none; }
  header h1 { margin: 0 0 4px; font-size: 18px; }
  header p { margin: 0; color: var(--muted); font-size: 12.5px; }
  #status-strip { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
  .stat { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); white-space: nowrap; }
  .stat.ok { color: var(--accent); border-color: var(--accent); }
  .stat.down { color: var(--muted); }

  main { flex: 1; display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
    gap: 0; overflow: hidden; }
  @media (max-width: 860px) { main { grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; } }

  section.chat { display: flex; flex-direction: column; min-width: 0; border-right: 1px solid var(--border); }
  #chat-log { flex: 1; overflow-y: auto; padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; }
  .bubble { max-width: 82%; padding: 9px 13px; border-radius: 12px; font-size: 13.5px; line-height: 1.45; white-space: pre-wrap; }
  .bubble.user { align-self: flex-end; background: var(--bubble-user); border-bottom-right-radius: 3px; }
  .bubble.bot { align-self: flex-start; background: var(--bubble-bot); border-bottom-left-radius: 3px; }
  .bubble.bot.tier-S { background: #fdf0f0; border: 1px solid var(--tier-s); }
  .bubble .meta { display: block; margin-top: 5px; font-size: 10.5px; color: var(--muted); }
  .badge { display: inline-flex; align-items: center; gap: 5px; padding: 2px 8px; border-radius: 999px;
    font-weight: 700; font-size: 11px; color: white; }
  .badge.D { background: var(--tier-d); } .badge.C { background: var(--tier-c); }
  .badge.B { background: var(--tier-b); } .badge.A { background: var(--tier-a); } .badge.S { background: var(--tier-s); }

  #composer { flex: none; border-top: 1px solid var(--border); padding: 10px 14px; display: flex; flex-direction: column; gap: 8px; }
  #composer .row1 { display: flex; gap: 8px; }
  #msg { flex: 1; resize: none; height: 46px; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--border);
    font-size: 13.5px; font-family: inherit; }
  #composer button { cursor: pointer; border: none; border-radius: 8px; padding: 0 16px; font-size: 13px; font-weight: 600; }
  #composer .send { background: var(--accent); color: white; }
  .row2 { display: flex; gap: 14px; align-items: center; font-size: 11.5px; color: var(--muted); flex-wrap: wrap; }
  .row2 label { display: flex; align-items: center; gap: 4px; cursor: pointer; }
  .row2 .examples span { cursor: pointer; text-decoration: underline; margin-right: 10px; }

  section.monitor { display: flex; flex-direction: column; min-width: 0; }
  section.monitor h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted);
    margin: 0; padding: 12px 16px 8px; flex: none; border-bottom: 1px solid var(--border); }
  #trace-log { flex: 1; overflow-y: auto; padding: 10px 14px; display: flex; flex-direction: column-reverse; gap: 10px; }
  .trace { border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; background: var(--panel); font-size: 12px; }
  .trace.latest { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
  .trace .head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }
  .trace dl { display: grid; grid-template-columns: 84px 1fr; gap: 3px 8px; margin: 0; }
  .trace dt { color: var(--muted); }
  .trace dd { margin: 0; word-break: break-word; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 999px; background: #eef0ea; font-size: 10.5px; margin-right: 4px; }
  .empty { color: var(--muted); font-size: 13px; padding: 16px; }
  details.raw { margin-top: 6px; }
  details.raw summary { cursor: pointer; color: var(--muted); font-size: 11px; }
  pre.raw { white-space: pre-wrap; font-size: 10.5px; background: #faf9f6; border: 1px solid var(--border);
    border-radius: 6px; padding: 8px; max-height: 160px; overflow: auto; margin: 6px 0 0; }
</style>
</head>
<body>
<header>
  <h1>itsbob — Complexity-Based Hierarchical Router</h1>
  <p>Classify First, Execute Cheapest, Fallback Gracefully. Chat on the left; watch what it actually did on the right.</p>
  <div id="status-strip"></div>
</header>
<main>
  <section class="chat">
    <div id="chat-log"></div>
    <div id="composer">
      <div class="row1">
        <textarea id="msg" placeholder='Describe a situation, or paste JSON — e.g. {"facts": {"stamina": 15, "minute": 60}}'></textarea>
        <button class="send" onclick="send()">Send</button>
      </div>
      <div class="row2">
        <label><input type="checkbox" id="classify-only"> classify only (no execution)</label>
        <span>goal:</span>
        <input id="goal" type="text" value="win the league" style="flex:0 0 160px; padding:3px 6px; border-radius:6px; border:1px solid var(--border); font-size:11.5px;">
        <span class="examples" id="examples"></span>
      </div>
    </div>
  </section>
  <section class="monitor">
    <h2>Live processing</h2>
    <div id="trace-log"><p class="empty">Nothing routed yet — send a message to see the Gatekeeper's reasoning, which model gets called, cache hits, and any escalation here.</p></div>
  </section>
</main>
<script>
const examples = {{ examples|safe }};

async function refreshStatus() {
  const el = document.getElementById('status-strip');
  try {
    const r = await fetch('/api/status'); const s = await r.json();
    const bits = [];
    bits.push(`<span class="stat">mode: ${s.mode}</span>`);
    bits.push(`<span class="stat ${s.local_back_brain.reachable ? 'ok' : 'down'}">Back Brain (Tier C): ${s.local_back_brain.reachable ? 'reachable' : 'offline → heuristic fallback'}</span>`);
    const providerPill = (p) => `<span class="stat ${p.configured ? 'ok' : 'down'}">${p.name}${p.models && p.models[0] ? ' · ' + p.models[0] : ''}: ${p.configured ? 'configured' : 'no key'}</span>`;
    if (s.tiers) {
      bits.push('<span class="stat">Tier B (Google, cheap):</span>');
      s.tiers.tier_b.forEach(p => bits.push(providerPill(p)));
      bits.push('<span class="stat">Tier A (Google, premium):</span>');
      s.tiers.tier_a.forEach(p => bits.push(providerPill(p)));
    } else {
      for (const p of s.cloud_providers) bits.push(providerPill(p));
    }
    el.innerHTML = bits.join('');
  } catch (e) { el.innerHTML = '<span class="stat down">status unavailable</span>'; }
}

function renderExamples() {
  const el = document.getElementById('examples');
  el.innerHTML = examples.map((ex, i) => `<span onclick="useExample(${i})">example ${i + 1}</span>`).join('');
}
function useExample(i) {
  document.getElementById('msg').value = examples[i];
}

function addBubble(role, text, tier) {
  const log = document.getElementById('chat-log');
  const empty = log.querySelector('.empty');
  if (empty) empty.remove();
  const b = document.createElement('div');
  b.className = 'bubble ' + role + (tier ? ' tier-' + tier : '');
  b.textContent = text;
  log.appendChild(b);
  log.scrollTop = log.scrollHeight;
  return b;
}

function tierBadge(tier, label) {
  return tier ? `<span class="badge ${tier}">${tier} · ${label || ''}</span>` : '';
}

function addTrace(data, mode) {
  const tlog = document.getElementById('trace-log');
  const empty = tlog.querySelector('.empty');
  if (empty) empty.remove();
  tlog.querySelectorAll('.trace.latest').forEach(el => el.classList.remove('latest'));

  const div = document.createElement('div');
  div.className = 'trace latest';

  if (data.error && !data.decision) {
    div.innerHTML = `<div class="head">error</div><dl><dt>message</dt><dd>${data.error}</dd></dl>`;
    tlog.prepend(div);
    return;
  }

  const d = data.decision || {};
  const tier = data.tier || (d.tier);
  const tierLabel = data.tier_label || d.tier_label;
  let head = tierBadge(tier, tierLabel);
  if (data.escalated_from) head += ` <span class="pill">escalated from ${data.escalated_from}</span>`;
  if (data.cache_hit) head += ` <span class="pill">cache hit</span>`;
  if (data.classify_only) head += ` <span class="pill">classify only</span>`;

  const rows = [];
  rows.push(`<dt>Fingerprint</dt><dd>${d.fingerprint || ''}</dd>`);
  rows.push(`<dt>Gatekeeper</dt><dd>${d.source || ''} — ${d.reasoning || ''} (${d.latency_ms ?? 0}ms)</dd>`);
  if (!data.classify_only) {
    const scripts = (data.script_results || []).map(r => `<span class="pill">${r.action}${r.ok ? '' : ' ✗'}</span>`).join(' ') || '<span class="empty" style="padding:0;">none</span>';
    rows.push(`<dt>Actions</dt><dd>${scripts}</dd>`);
    rows.push(`<dt>Model called</dt><dd>${data.provider ? data.provider + (data.model ? ' / ' + data.model : '') : '— (script/local/none)'}</dd>`);
    const budget = data.within_budget === false ? ' ⚠ over 1.8s budget' : '';
    rows.push(`<dt>Latency</dt><dd>${data.total_latency_ms ?? 0}ms${budget}</dd>`);
    if (data.cache_stats) rows.push(`<dt>Cache</dt><dd>hit rate ${(data.cache_stats.hit_rate * 100).toFixed(0)}% (${data.cache_stats.hits}/${data.cache_stats.hits + data.cache_stats.misses}), size ${data.cache_stats.size}</dd>`);
  }

  div.innerHTML = `<div class="head">${head}</div><dl>${rows.join('')}</dl>
    <details class="raw"><summary>raw JSON</summary><pre class="raw">${JSON.stringify(data, null, 2)}</pre></details>`;
  tlog.prepend(div);
}

async function send() {
  const msgEl = document.getElementById('msg');
  const text = msgEl.value.trim();
  if (!text) return;
  const goal = document.getElementById('goal').value;
  const classifyOnly = document.getElementById('classify-only').checked;

  addBubble('user', text);
  msgEl.value = '';
  const thinking = addBubble('bot', 'thinking…');

  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: text, goal, classify_only: classifyOnly })
    });
    const data = await r.json();
    thinking.remove();
    if (data.error) {
      addBubble('bot', 'error: ' + data.error);
      return;
    }
    const tier = data.trace && (data.trace.tier || (data.trace.decision && data.trace.decision.tier));
    const bubble = addBubble('bot', data.reply, tier === 'S' ? 'S' : null);
    if (data.trace && (data.trace.provider || data.trace.decision)) {
      const meta = document.createElement('span');
      meta.className = 'meta';
      const d = data.trace.decision || {};
      meta.innerHTML = tierBadge(data.trace.tier || d.tier, data.trace.tier_label || d.tier_label);
      bubble.appendChild(meta);
    }
    addTrace(data.trace, classifyOnly ? 'classify' : 'route');
  } catch (e) {
    thinking.remove();
    addBubble('bot', 'request failed: ' + e);
  }
}

document.getElementById('msg').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

renderExamples();
refreshStatus();
</script>
</body>
</html>
"""
