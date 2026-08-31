"""Memory as tools, so the agent decides what is worth keeping.

Recall also happens automatically before every turn (see
:mod:`itsbob.agent.context`) — these are for the cases automatic retrieval
cannot cover: writing something down on purpose, digging for something the
turn's query did not surface, and correcting or dropping a memory that turned
out to be wrong.

``forget`` is deliberately available. A memory system you can only add to
accumulates stale facts — an old address, a preference that changed — and
those are worse than no memory at all, because they are recalled with the same
confidence as the true ones.
"""

from __future__ import annotations

from typing import Any

from ..memory.base import Horizon, MemoryKind, MemoryRecord, Subject
from .base import Risk, Tool, ToolContext, ToolError, ToolResult

__all__ = ["memory_tools"]


def _store(ctx: ToolContext):
    if ctx.memory is None:
        raise ToolError("no memory store is attached")
    return ctx.memory


#: A short-horizon memory written by hand lasts this long unless promoted.
SHORT_TTL_SECONDS = 6 * 3600.0


def _remember(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    import time

    store = _store(ctx)
    # Coerced, not rejected. A wrong category is a rounding error; a rejected
    # call costs a step and a model call to discover that.
    kind = MemoryKind.coerce(params.get("kind"))
    subject = Subject.coerce(params.get("subject"))
    horizon = Horizon.coerce(params.get("horizon"))
    record = MemoryRecord(
        content=params["content"].strip(),
        kind=kind,
        subject=subject,
        horizon=horizon,
        expires_at=(
            time.time() + SHORT_TTL_SECONDS if horizon is Horizon.SHORT else None
        ),
        importance=float(params.get("importance", 0.6)),
        tags=tuple(params.get("tags") or ()),
        metadata={"source": params.get("source", "agent")},
    )
    store.add(record)
    return ToolResult(
        ok=True,
        output=(
            f"remembered [{kind.value}, about {subject.value}, {horizon.value}-term] "
            f"{record.content[:120]}"
        ),
        data={
            "id": record.id,
            "kind": kind.value,
            "subject": subject.value,
            "horizon": horizon.value,
        },
    )


def _promote(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Move a short-term memory into long-term, when it turns out to matter."""
    store = _store(ctx)
    memory_id = params["id"]
    record = store.get(memory_id)
    if record is None:
        raise ToolError(f"no memory with id {memory_id!r} — recall first to get ids")
    importance = params.get("importance")
    if not store.promote(memory_id, importance=None if importance is None else float(importance)):
        raise ToolError(f"could not promote {memory_id!r}")
    return ToolResult(
        ok=True,
        output=f"kept for good: {record.content[:120]}",
        data={"id": memory_id},
    )


def _recall(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = _store(ctx)
    hits = store.search(
        params["query"],
        limit=int(params.get("limit", 5)),
        tags=params.get("tags"),
    )
    if not hits:
        return ToolResult(ok=True, output="(nothing remembered about that)", data={"hits": []})
    lines = [
        f"- {h.record.content}  [{h.record.kind.value}, about {h.record.subject.value}, "
        f"{h.record.horizon.value}-term, {_ago(h.record.created_at)}, {h.reason}]"
        for h in hits
    ]
    return ToolResult(
        ok=True,
        output="\n".join(lines),
        data={"hits": [h.as_dict() for h in hits]},
    )


def _forget(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = _store(ctx)
    memory_id = params["id"]
    record = store.get(memory_id)
    if record is None:
        raise ToolError(f"no memory with id {memory_id!r} — recall first to get ids")
    store.forget(memory_id)
    return ToolResult(ok=True, output=f"forgot: {record.content[:120]}", data={"id": memory_id})


def _update(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    store = _store(ctx)
    fields: dict[str, Any] = {}
    if params.get("content"):
        fields["content"] = params["content"]
    if params.get("importance") is not None:
        fields["importance"] = float(params["importance"])
    if params.get("tags"):
        fields["tags"] = params["tags"]
    if not fields:
        raise ToolError("nothing to update — pass content, importance, or tags")
    if not store.update(params["id"], **fields):
        raise ToolError(f"no memory with id {params['id']!r}")
    return ToolResult(ok=True, output=f"updated {params['id']}", data=fields)


def _ago(created_at: float) -> str:
    import time

    seconds = max(0, time.time() - created_at)
    for limit, unit, size in (
        (60, "s", 1),
        (3600, "m", 60),
        (86400, "h", 3600),
        (86400 * 30, "d", 86400),
    ):
        if seconds < limit:
            return f"{int(seconds // size)}{unit} ago"
    return f"{int(seconds // (86400 * 30))}mo ago"


def memory_tools() -> list[Tool]:
    kinds = ", ".join(k.value for k in MemoryKind)
    return [
        Tool(
            name="remember",
            description=(
                "Store something worth recalling in a later conversation — a preference, "
                "a decision, a durable fact. Not for things already in this conversation. "
                "Always set `subject`: your own opinions are yours, not the user's."
            ),
            run=_remember,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "One self-contained sentence."},
                    "subject": {
                        "type": "string",
                        "description": (
                            "Who it is about: 'user' (the person), 'bob' (you — your own "
                            "taste, opinion, habit or conclusion), or 'world' (neither). "
                            "Default user. Getting this wrong makes your opinions look "
                            "like theirs, so pick deliberately."
                        ),
                    },
                    "horizon": {
                        "type": "string",
                        "description": (
                            "'long' to keep it for good, 'short' for the current thread "
                            "only (dropped after a few hours). Default long."
                        ),
                    },
                    "kind": {"type": "string", "description": f"One of: {kinds}. Default fact. Near-misses are accepted."},
                    "importance": {"type": "number", "description": "0-1. Default 0.6."},
                    "tags": {
                        "type": "array",
                        "description": (
                            "Short lowercase labels. The tag `style` is special: "
                            "anything tagged with it becomes a standing rule about "
                            "how to answer (length, format, how much detail) and is "
                            "put in front of you on every turn. Use it when the user "
                            "tells you how they want replies — 'always list every "
                            "match', 'keep it to a paragraph'."
                        ),
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="keep_memory",
            description=(
                "Promote a short-term memory to permanent, when it turns out to matter "
                "after all. Recall first to get the id."
            ),
            run=_promote,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "importance": {"type": "number", "description": "Optional new 0-1 importance."},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="recall",
            description="Search memory for anything relevant. Use when you suspect you were told before.",
            run=_recall,
            risk=Risk.READ,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "tags": {"type": "array"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="forget",
            description="Delete a memory by id, when it is wrong or out of date. Recall first to get the id.",
            run=_forget,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        ),
        Tool(
            name="update_memory",
            description="Correct an existing memory in place, keeping its id and history.",
            run=_update,
            risk=Risk.WRITE,
            mutates=True,
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "importance": {"type": "number"},
                    "tags": {"type": "array"},
                },
                "required": ["id"],
            },
        ),
    ]
