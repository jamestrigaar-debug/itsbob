"""A small browser GUI over the complexity router.

One page: paste (or generate an example) game state, hit Route, and see the
tier badge, the Gatekeeper's reasoning and fingerprint, whether the semantic
cache served it, which scripts ran, and the end-to-end latency against the
1.8s budget. A status strip along the top shows every provider `itsbob
doctor` would report, including the local Back Brain, refreshed on load.

Deliberately a single Flask file with the page template inlined as a string
— there is exactly one route worth having a GUI for, so a build step or a
frontend framework would be pure overhead. Flask is an optional dependency
(the `gui` extra); everything else in this package works without it.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from typing import Any

import os

from ..config import Settings
from ..factory import build_router
from ..llm.local import is_ollama_running
from ..router import build_complexity_router

__all__ = ["create_app", "run_gui"]

EXAMPLE_STATE = {
    "facts": {
        "score": "1-0",
        "minute": 78,
        "opponent_formation": "4-4-2",
        "morale": "low",
        "stamina": 34,
    },
    "events": [
        "68' Yellow card - opponent midfielder",
        "74' Corner won",
        "77' Substitution - opponent brings on fresh striker",
    ],
}


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
        return render_template_string(_PAGE, example=json.dumps(EXAMPLE_STATE, indent=2))

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
        except Exception as exc:  # noqa: BLE001 - surface it to the user, don't 500 blindly
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
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { padding: 20px 28px 10px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0 0 4px; font-size: 20px; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 24px 28px; max-width: 1200px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  section.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 0 0 12px; }
  textarea { width: 100%; min-height: 220px; font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 12.5px; padding: 10px; border-radius: 8px; border: 1px solid var(--border); resize: vertical; }
  input[type=text] { width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border); font-size: 13px; }
  label { font-size: 12px; color: var(--muted); display: block; margin: 10px 0 4px; }
  .row { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
  button { cursor: pointer; border: none; border-radius: 8px; padding: 9px 16px; font-size: 13px; font-weight: 600; }
  button.primary { background: var(--accent); color: white; }
  button.ghost { background: transparent; border: 1px solid var(--border); color: var(--text); }
  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px;
    font-weight: 700; font-size: 13px; color: white; }
  .badge.D { background: var(--tier-d); } .badge.C { background: var(--tier-c); }
  .badge.B { background: var(--tier-b); } .badge.A { background: var(--tier-a); } .badge.S { background: var(--tier-s); }
  #result { font-size: 13px; }
  #result dl { display: grid; grid-template-columns: 120px 1fr; gap: 6px 10px; margin: 12px 0; }
  #result dt { color: var(--muted); }
  #result dd { margin: 0; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef0ea; font-size: 11.5px; margin-right: 6px; }
  .empty { color: var(--muted); font-size: 13px; }
  #status-strip { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .stat { font-size: 11.5px; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border); }
  .stat.ok { color: var(--accent); border-color: var(--accent); }
  .stat.down { color: var(--muted); }
  pre.raw { white-space: pre-wrap; font-size: 11.5px; background: #faf9f6; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px; max-height: 220px; overflow: auto; }
  .alert-s { border: 2px solid var(--tier-s); background: #fdf0f0; padding: 12px; border-radius: 8px; margin-top: 10px; }
</style>
</head>
<body>
<header>
  <h1>itsbob — Complexity-Based Hierarchical Router</h1>
  <p>Classify First, Execute Cheapest, Fallback Gracefully. Paste a game state, route it, watch which tier answers.</p>
  <div id="status-strip"></div>
</header>
<main>
  <section class="panel">
    <h2>Game state</h2>
    <label for="state">Raw scraped state (JSON — flat facts, or {"facts": ..., "events": [...]})</label>
    <textarea id="state">{{ example }}</textarea>
    <label for="goal">Overarching goal (appended to cloud-tier prompts)</label>
    <input id="goal" type="text" value="win the league">
    <div class="row">
      <button class="primary" onclick="doRoute()">Route (classify + execute)</button>
      <button class="ghost" onclick="doClassify()">Classify only</button>
      <button class="ghost" onclick="loadExample()">Reset example</button>
    </div>
  </section>
  <section class="panel">
    <h2>Result</h2>
    <div id="result"><p class="empty">Nothing routed yet.</p></div>
  </section>
</main>
<script>
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

function loadExample() {
  document.getElementById('state').value = {{ example|tojson }};
}

function tierBadge(tier, label) {
  return `<span class="badge ${tier}">${tier} · ${label}</span>`;
}

function renderResult(data, mode) {
  const box = document.getElementById('result');
  if (data.error && mode !== 'route') {
    box.innerHTML = `<p class="empty">error: ${data.error}</p>`;
    return;
  }
  if (mode === 'classify') {
    box.innerHTML = `
      ${tierBadge(data.tier, data.tier_label)}
      <dl>
        <dt>Fingerprint</dt><dd>${data.fingerprint}</dd>
        <dt>Source</dt><dd>${data.source}</dd>
        <dt>Latency</dt><dd>${data.latency_ms} ms</dd>
        <dt>Reasoning</dt><dd>${data.reasoning}</dd>
      </dl>`;
    return;
  }
  const d = data.decision || {};
  const scripts = (data.script_results || []).map(r => `<span class="pill">${r.action}${r.ok ? '' : ' ✗'}</span>`).join(' ') || '<span class="empty">none</span>';
  const cache = data.cache_hit ? '<span class="pill">cache hit</span>' : '<span class="pill">cache miss</span>';
  const budget = data.within_budget ? '<span class="pill">within 1.8s budget</span>' : '<span class="pill">⚠ over 1.8s budget</span>';
  let alertBlock = '';
  if (data.needs_user) {
    alertBlock = `<div class="alert-s"><strong>Tier S — manual override required.</strong><br>${data.note}<br><small>${data.error || ''}</small></div>`;
  }
  box.innerHTML = `
    ${tierBadge(data.tier, data.tier_label)} ${data.escalated_from ? `<span class="pill">escalated from ${data.escalated_from}</span>` : ''}
    ${alertBlock}
    <dl>
      <dt>Fingerprint</dt><dd>${d.fingerprint || ''}</dd>
      <dt>Gatekeeper</dt><dd>${d.source || ''} — ${d.reasoning || ''}</dd>
      <dt>Actions</dt><dd>${scripts}</dd>
      <dt>Note</dt><dd>${data.note || ''}</dd>
      <dt>Cache</dt><dd>${cache} (${JSON.stringify(data.cache_stats || {})})</dd>
      <dt>Provider</dt><dd>${data.provider || '—'} ${data.model ? '/ ' + data.model : ''}</dd>
      <dt>Latency</dt><dd>${data.total_latency_ms} ms ${budget}</dd>
    </dl>
    <pre class="raw">${JSON.stringify(data, null, 2)}</pre>`;
}

async function doRoute() {
  const state = document.getElementById('state').value;
  const goal = document.getElementById('goal').value;
  document.getElementById('result').innerHTML = '<p class="empty">routing…</p>';
  const r = await fetch('/api/route', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ state, goal }) });
  renderResult(await r.json(), 'route');
}

async function doClassify() {
  const state = document.getElementById('state').value;
  document.getElementById('result').innerHTML = '<p class="empty">classifying…</p>';
  const r = await fetch('/api/classify', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ state }) });
  renderResult(await r.json(), 'classify');
}

refreshStatus();
</script>
</body>
</html>
"""
