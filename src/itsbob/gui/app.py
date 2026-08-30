"""The browser interface: a live view of the agent, and a place to say yes.

Chat on the left; on the right, every step as it happens — the tier chosen and
why, each tool call with its arguments and result, what was recalled from
memory and what was written back. That panel is the point: "it said it updated
the file" and "it updated the file" are different claims, and only one of them
is checkable from a transcript.

Turns run on a worker thread and stream over server-sent events, so a
ten-second turn shows its work instead of a spinner. Tools that need
confirmation raise a card in the page, which is what makes ``guarded`` mode
usable here at all — see :mod:`itsbob.gui.session`.

Binds to 127.0.0.1 with no authentication. It is a local interface for one
person, not a deployed service: anything that can reach the port can run tools
as you.
"""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path
from typing import Any

from .page import PAGE

__all__ = ["create_app", "run_gui"]


def create_app(home: Path | None = None, *, mode: str | None = None):
    try:
        from flask import Flask, Response, jsonify, request, stream_with_context
    except ImportError as exc:  # pragma: no cover - depends on install
        raise SystemExit(
            "the browser interface needs Flask — install it with:\n"
            "  pip install -e '.[gui]'   (or re-run ./install.sh)"
        ) from exc

    from ..agent import build_agent, default_home
    from ..daemon import TaskStore, parse_schedule
    from ..tools import Mode
    from .session import Session

    root = Path(home) if home else default_home()
    root.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    session = Session(
        lambda confirm: build_agent(
            home=root, mode=Mode(mode) if mode else None, confirm=confirm
        )
    )
    tasks_holder: dict[str, Any] = {}
    tasks_lock = threading.Lock()

    def tasks() -> Any:
        with tasks_lock:
            if "store" not in tasks_holder:
                tasks_holder["store"] = TaskStore(root / "tasks.sqlite")
            return tasks_holder["store"]

    def fail(message: str, status: int = 400):
        return jsonify({"error": message}), status

    # -- page --------------------------------------------------------------

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/favicon.ico")
    def favicon():
        # A 1x1 transparent GIF, so the tab shows nothing rather than a 404 in
        # the console every time the page loads.
        return Response(
            b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!"
            b"\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00"
            b"\x00\x02\x02D\x01\x00;",
            mimetype="image/gif",
        )

    # -- live stream -------------------------------------------------------

    @app.get("/api/stream")
    def stream():
        return Response(
            stream_with_context(session.listen()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",  # nginx, if anyone puts one in front
                "Connection": "keep-alive",
            },
        )

    # -- conversation ------------------------------------------------------

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(force=True, silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return fail("message is empty")
        if not session.start_turn(message, context=payload.get("context") or None):
            return fail("a turn is already running", 409)
        return jsonify({"started": True})

    @app.post("/api/approve")
    def approve():
        payload = request.get_json(force=True, silent=True) or {}
        approval_id = str(payload.get("id", ""))
        if not session.decide(
            approval_id,
            approved=bool(payload.get("approved")),
            remember=bool(payload.get("remember")),
        ):
            return fail("that request is no longer waiting — it timed out or the turn ended", 410)
        return jsonify({"ok": True})

    @app.post("/api/reset")
    def reset():
        session.reset_conversation()
        return jsonify({"ok": True})

    # -- state -------------------------------------------------------------

    @app.get("/api/status")
    def status():
        agent = session.agent
        brain = agent.brain.describe()
        return jsonify(
            {
                "home": str(root),
                "busy": session.busy,
                "policy": agent.toolbox.policy.describe(),
                "auto_allowed": sorted(session.auto_allow),
                "tools": [
                    {"name": t.name, "risk": t.risk.value, "description": t.description}
                    for t in agent.toolbox.registry.all()
                ],
                "tiers": {
                    tier: {
                        "label": info["label"],
                        "model": next(
                            (row["models"][0] for row in info["providers"]
                             if row["configured"] and row["models"]),
                            None,
                        ),
                        "provider": next(
                            (row["provider"] for row in info["providers"] if row["configured"]),
                            None,
                        ),
                    }
                    for tier, info in brain["tiers"].items()
                },
                "local": brain["local"],
                "memory": agent.memory.stats() if agent.memory is not None else {},
                "apis": agent.toolbox.catalog.describe() if agent.toolbox.catalog else [],
                "tasks": [t.as_dict() for t in tasks().all()],
                "turns": len(agent.conversation),
                "usage": brain.get("usage", {}),
            }
        )

    @app.get("/api/memory")
    def memory_search():
        agent = session.agent
        if agent.memory is None:
            return jsonify({"hits": []})
        query = request.args.get("q", "").strip()
        limit = max(1, min(50, int(request.args.get("limit", 15) or 15)))
        if query:
            hits = [h.as_dict() for h in agent.memory.search(query, limit=limit)]
        else:
            hits = [_record_dict(r) for r in agent.memory.recent(limit)]
        return jsonify({"hits": hits, "total": len(agent.memory)})

    @app.post("/api/memory")
    def memory_add():
        payload = request.get_json(force=True, silent=True) or {}
        content = str(payload.get("content", "")).strip()
        if not content:
            return fail("content is empty")
        from ..memory.base import MemoryKind, MemoryRecord

        record = session.agent.memory.add(
            MemoryRecord(
                content=content,
                kind=MemoryKind(str(payload.get("kind", "fact"))),
                importance=float(payload.get("importance", 0.6)),
                tags=tuple(payload.get("tags") or ()),
                metadata={"source": "gui"},
            )
        )
        return jsonify({"id": record.id})

    @app.post("/api/memory/forget")
    def memory_forget():
        payload = request.get_json(force=True, silent=True) or {}
        agent = session.agent
        ok = agent.memory is not None and agent.memory.forget(str(payload.get("id", "")))
        return jsonify({"ok": bool(ok)})

    @app.get("/api/audit")
    def audit():
        return jsonify({"entries": session.agent.toolbox.audit.recent(60)})

    # -- tasks -------------------------------------------------------------

    @app.post("/api/task")
    def task_create():
        payload = request.get_json(force=True, silent=True) or {}
        name = str(payload.get("name", "")).strip()
        prompt = str(payload.get("prompt", "")).strip()
        schedule = str(payload.get("schedule", "")).strip()
        if not (name and prompt and schedule):
            return fail("name, prompt and schedule are all required")
        try:
            parse_schedule(schedule)
        except Exception as exc:  # noqa: BLE001 - a bad schedule is user error
            return fail(str(exc))
        task = tasks().create(name, prompt, schedule, notify=bool(payload.get("notify", True)))
        return jsonify({"task": task.as_dict()})

    @app.post("/api/task/<action>")
    def task_action(action: str):
        payload = request.get_json(force=True, silent=True) or {}
        store = tasks()
        task = store.find(str(payload.get("id", "")))
        if task is None:
            return fail("no such task", 404)
        if action == "remove":
            return jsonify({"ok": store.remove(task.id)})
        if action in ("enable", "disable"):
            return jsonify({"ok": store.set_enabled(task.id, action == "enable")})
        if action == "run":
            if not session.start_turn(task.prompt):
                return fail("a turn is already running", 409)
            return jsonify({"started": True, "prompt": task.prompt})
        return fail(f"unknown action {action!r}", 404)

    return app


def _record_dict(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "content": record.content,
        "kind": record.kind.value,
        "tags": list(record.tags),
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
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host  # noqa: S104
    url = f"http://{shown}:{port}"
    print(f"  itsbob → {url}")
    print("  ctrl-c to stop\n")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    # threaded is the default, and load-bearing here: the SSE stream holds a
    # connection open for the life of the page, so a single-threaded server
    # would serve exactly one browser and then stop responding entirely.
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
