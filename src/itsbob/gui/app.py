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

import json
import threading
import time
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
    from ..daemon.notify import MultiSink, NoticeGate, default_sink
    from ..integrations.discord import DiscordBridge, DiscordClient, DiscordSink
    from ..tools import Mode
    from .autonomous import Autonomous
    from ..integrations.apis import builtin_status
    from ..tools.vision import pillow_available
    from ..tools.websearch import available_backend
    from ..daemon.service import daemon_status
    from ..llm.pricing import Ledger
    from .console import CONSOLE
    from .messages import MESSAGES_PAGE, MessageLog
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
    # Lazily-built subsystems, and the lock that guards building them.
    #
    # RLock, not Lock, and that is not a detail: `autonomous()` needs the task
    # store and the sink, which are lazily built the same way. With a plain
    # Lock the first request to reach it deadlocked against itself — and
    # because the thread died holding the lock, every later request that
    # touched any other subsystem (`/api/status` among them) blocked forever.
    # The page sat on "connecting…" while chat and the event stream, which
    # never take this lock, carried on working perfectly.
    #
    # Each accessor below also resolves its dependencies *before* taking the
    # lock, so the reentrancy is a safety net rather than the mechanism.
    holder: dict[str, Any] = {}
    holder_lock = threading.RLock()

    def tasks() -> Any:
        with holder_lock:
            if "store" not in holder:
                holder["store"] = TaskStore(root / "tasks.sqlite")
            return holder["store"]

    def messages() -> MessageLog:
        with holder_lock:
            if "messages" not in holder:
                holder["messages"] = MessageLog(root / "notifications.jsonl")
            return holder["messages"]

    def discord() -> DiscordBridge | None:
        """The Discord bridge, built once if a token and channel are configured."""
        with holder_lock:
            if "discord" not in holder:
                holder["discord"] = DiscordBridge.from_env(session.submit)
            return holder["discord"]

    def sink() -> Any:
        """Where proactive messages go: the usual sinks, plus Discord when set up.

        Discord is appended rather than substituted. The file log is what
        `/messages` reads, and losing it because a channel was configured would
        empty the messages window for no reason.
        """
        with holder_lock:
            if "sink" not in holder:
                base = default_sink(root, console=False)
                bridge = DiscordClient.from_env()
                if bridge is not None:
                    base = MultiSink(sinks=[*base.sinks, DiscordSink(client=bridge)])
                holder["sink"] = base
            return holder["sink"]

    def autonomous() -> Autonomous:
        if "autonomous" in holder:
            return holder["autonomous"]
        # Built before the lock is taken, not inside it.
        store, notices, brain = tasks(), sink(), session.agent.brain
        with holder_lock:
            if "autonomous" not in holder:
                holder["autonomous"] = Autonomous(
                    session, store, sink=notices, gate=NoticeGate(brain=brain)
                )
            return holder["autonomous"]

    def fail(message: str, status: int = 400):
        return jsonify({"error": message}), status

    # -- page --------------------------------------------------------------

    @app.get("/")
    def index():
        return Response(CONSOLE, mimetype="text/html")

    @app.get("/old")
    def old_index():
        """The previous interface, kept one release for anyone mid-task on it."""
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/tokens")
    def tokens():
        """What it is costing, split by model and by what the call was for.

        Token counts alone invite the wrong conclusion — a million Tier C
        tokens and a million Tier S tokens differ by about forty times in
        price — so this reports estimated money, and says which share ran
        locally for nothing.
        """
        import time as _time

        ledger = Ledger(session.agent.brain.tracker)
        midnight = _time.mktime(_time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        return jsonify(
            {
                "today": ledger.summary(since=midnight),
                "all_time": ledger.summary(),
                "recent": ledger.recent(40),
            }
        )

    @app.get("/messages")
    def messages_page():
        """The standalone messages window — everything Bob said unprompted.

        A separate page rather than a panel: proactive notices and a
        conversation are different kinds of thing, and interleaving them makes
        both harder to read. It shares the server, the log and the SSE
        machinery, so this is three routes rather than a second application.
        """
        return Response(MESSAGES_PAGE, mimetype="text/html")

    @app.get("/api/messages")
    def messages_list():
        limit = max(1, min(500, int(request.args.get("limit", 100) or 100)))
        log = messages()
        return jsonify(
            {
                "messages": log.recent(
                    limit=limit,
                    after=request.args.get("after") or None,
                    unread_only=request.args.get("unread") == "1",
                ),
                "unread": log.unread_count(),
            }
        )

    @app.get("/api/messages/stream")
    def messages_stream():
        log = messages()
        after = request.args.get("after") or log.latest_id()

        def events():
            # An immediate frame so the page can tell "connected" from "hung",
            # which an SSE stream cannot otherwise show until something happens.
            yield 'data: {"kind": "keepalive"}\n\n'
            last = time.time()
            for message in log.follow(after=after):
                if message is None:
                    # Idle. A comment frame every 20s keeps browsers and proxies
                    # from closing a connection that is working perfectly.
                    if time.time() - last > 20:
                        last = time.time()
                        yield ": keepalive\n\n"
                    continue
                last = time.time()
                yield f"data: {json.dumps(message, default=str)}\n\n"

        return Response(
            stream_with_context(events()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/api/messages/read")
    def messages_read():
        payload = request.get_json(force=True, silent=True) or {}
        log = messages()
        marked = (
            log.mark_all_read()
            if payload.get("all")
            else log.mark_read([str(i) for i in (payload.get("ids") or [])])
        )
        return jsonify({"marked": marked, "unread": log.unread_count()})

    # -- discord -----------------------------------------------------------

    @app.get("/api/discord")
    def discord_status():
        bridge = discord()
        if bridge is None:
            return jsonify(
                {
                    "configured": False,
                    "hint": "set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID in .env",
                }
            )
        return jsonify({"configured": True, **bridge.status()})

    @app.post("/api/discord")
    def discord_toggle():
        payload = request.get_json(force=True, silent=True) or {}
        bridge = discord()
        if bridge is None:
            return fail("Discord is not configured — set DISCORD_BOT_TOKEN and "
                        "DISCORD_CHANNEL_ID in .env and restart", 409)
        if bool(payload.get("enabled")):
            if not bridge.start():
                return fail(bridge.last_error or "already running", 409)
        else:
            bridge.stop()
        return jsonify({"configured": True, **bridge.status()})

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
        # Always accepted unless the queue is full: a message sent while it is
        # working is queued, not refused.
        result = session.submit(message, context=payload.get("context") or None)
        if not result["accepted"]:
            return fail(result["error"], 429)
        return jsonify(result)

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

    @app.post("/api/queue/clear")
    def queue_clear():
        return jsonify({"dropped": session.clear_queue()})

    @app.post("/api/autonomous")
    def set_autonomous():
        """Turn continuous mode on or off.

        On, itsbob runs its scheduled work by itself and you can still talk to
        it — everything goes through one queue, so nothing overlaps.
        """
        payload = request.get_json(force=True, silent=True) or {}
        runner = autonomous()
        bridge = discord()
        if bool(payload.get("enabled")):
            runner.start()
            # Turning it on means "you have the run of the place", and being
            # reachable in Discord is part of that. Making it a second switch
            # somewhere else is how you end up with a bot nobody can talk to.
            if bridge is not None and not bridge.running:
                bridge.start()
        else:
            runner.stop()
            if bridge is not None:
                bridge.stop()
        status = runner.status()
        status["discord"] = {"configured": False} if bridge is None else bridge.status()
        return jsonify(status)

    @app.get("/api/autonomous")
    def get_autonomous():
        return jsonify(autonomous().status())

    # -- state -------------------------------------------------------------

    @app.get("/api/status")
    def status():
        """Everything the header and the panels need, in one poll.

        Every section is computed behind :func:`_safe`, so a subsystem that is
        broken costs its own field and nothing else. Before, one raising call
        took the whole endpoint down — and with it the header, the tier chips,
        the tool list and the task panel, none of which had anything to do with
        whatever failed.
        """
        agent = session.agent
        problems: dict[str, str] = {}
        brain = _safe(problems, "brain", agent.brain.describe, {})

        return jsonify(
            {
                "home": str(root),
                "busy": session.busy,
                "current": session.current,
                "queued": session.queued_messages(),
                "policy": _safe(problems, "policy", lambda: agent.toolbox.policy.describe(), {}),
                "auto_allowed": sorted(session.auto_allow),
                "tools": _safe(problems, "tools", lambda: [
                    {"name": t.name, "risk": t.risk.value, "description": t.description}
                    for t in agent.toolbox.registry.all()
                ], []),
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
                    for tier, info in (brain.get("tiers") or {}).items()
                },
                "local": brain.get("local"),
                "memory": _safe(problems, "memory", lambda: (
                    agent.memory.stats() if agent.memory is not None else {}
                ), {}),
                "apis": _safe(problems, "apis", lambda: (
                    agent.toolbox.catalog.describe(agent.toolbox.env)
                    if agent.toolbox.catalog else []
                ), []),
                "tasks": _safe(problems, "tasks", lambda: [t.as_dict() for t in tasks().all()], []),
                "autonomous": holder["autonomous"].status() if "autonomous" in holder
                              else {"running": False},
                "turns": len(agent.conversation),
                "usage": brain.get("usage", {}),
                "unread": _safe(problems, "messages", lambda: messages().unread_count(), 0),
                "discord": _safe(problems, "discord", lambda: (
                    {"configured": False} if discord() is None
                    else {"configured": True, **discord().status()}
                ), {"configured": False}),
                "budget": _safe(problems, "budget", lambda: agent.guard.as_dict(), {}),
                "feasibility": _safe(problems, "feasibility", lambda: (
                    agent.feasibility.as_dict() if agent.feasibility is not None else {}
                ), {}),
                # The catalog only holds APIs whose key is present, so on its
                # own it cannot show what you could switch on. This is the
                # other half: every built-in, configured or not, with the
                # variable that would enable it.
                "services": _safe(problems, "services", lambda: builtin_status(
                    agent.toolbox.env
                ), []),
                "search_backend": _safe(problems, "search", available_backend, "unknown"),
                # The daemon, which is a different process to this one and the
                # only thing that runs your schedule with the browser closed.
                "serving": _safe(problems, "serving", lambda: daemon_status(root), {}),
                "spend": _safe(
                    problems,
                    "spend",
                    lambda: Ledger(agent.brain.tracker).summary(
                        since=__import__("time").mktime(
                            __import__("time").localtime()[:3] + (0, 0, 0, 0, 0, -1)
                        )
                    ),
                    {},
                ),
                "tasks_count": _safe(problems, "tasks_count", lambda: len(tasks()), 0),
                "vision": {"pillow": _safe(problems, "vision", pillow_available, False)},
                # Named rather than swallowed: a panel that is quietly empty
                # because something threw is worse than one that says so.
                "problems": problems,
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
                kind=MemoryKind.coerce(payload.get("kind", "fact")),
                subject=payload.get("subject") or "user",
                importance=float(payload.get("importance", 0.6)),
                tags=tuple(payload.get("tags") or ()),
                metadata={"source": "gui"},
            )
        )
        return jsonify({"id": record.id, "tags": list(record.tags)})

    @app.post("/api/memory/forget")
    def memory_forget():
        payload = request.get_json(force=True, silent=True) or {}
        agent = session.agent
        ok = agent.memory is not None and agent.memory.forget(str(payload.get("id", "")))
        return jsonify({"ok": bool(ok)})

    @app.get("/api/audit")
    def audit():
        return jsonify({"entries": session.agent.toolbox.audit.recent(60)})

    @app.get("/api/scripts")
    def scripts():
        """Every discovered script and the tools each provides."""
        from ..scripts import describe_scripts, user_scripts_dir

        allowed = set(session.agent.toolbox.registry.names())
        rows = []
        for row in describe_scripts():
            row = dict(row)
            row["tools"] = [t for t in row["tools"] if t["name"] in allowed]
            rows.append(row)
        return jsonify({"scripts": rows, "drop_in": str(user_scripts_dir())})

    # -- tasks -------------------------------------------------------------

    @app.get("/api/tasks")
    def tasks_list():
        """The task panel's own endpoint.

        It used to read the task list out of `/api/status`, which made it the
        one panel that went dark whenever anything else in that payload was
        slow or broken — the others (memory, scripts, audit) each have their
        own route and kept working. A panel should depend on the thing it
        shows and nothing else.
        """
        store = tasks()
        return jsonify(
            {
                "tasks": [t.as_dict() for t in store.all()],
                "next_due": store.next_due_at(),
                # Whether anything will actually *run* them. A schedule with no
                # runner is the most common "my task never fired", and the
                # panel is where you look when that happens.
                "runner": {
                    "autonomous": bool(
                        "autonomous" in holder and holder["autonomous"].running
                    ),
                    "serving": bool(daemon_status(root).get("running")),
                },
            }
        )

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


def _safe(problems: dict[str, str], name: str, fn: Any, fallback: Any) -> Any:
    """Run ``fn``, or record why it could not be run and return ``fallback``.

    The status endpoint is polled every fifteen seconds and feeds the whole
    interface. One subsystem raising must not blank the other nine.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - reporting is the point
        problems[name] = f"{type(exc).__name__}: {exc}"[:200]
        return fallback


def _record_dict(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "content": record.content,
        "kind": record.kind.value,
        "subject": record.subject.value,
        "horizon": record.horizon.value,
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
    print(f"  itsbob  → {url}")
    print(f"  messages → {url}/messages   (what it said without being asked)")
    print("  ctrl-c to stop\n")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    # threaded is the default, and load-bearing here: the SSE stream holds a
    # connection open for the life of the page, so a single-threaded server
    # would serve exactly one browser and then stop responding entirely.
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
