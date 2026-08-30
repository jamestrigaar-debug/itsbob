"""The messages window: everything Bob said without being asked.

The chat panel is a conversation — it only makes sense read in order, and every
line in it is a reply to the line above. Proactive output is not that. A task
result at 07:00, an alert at 14:20 and a note Bob left himself at 18:00 have
nothing to do with each other, and interleaving them into a transcript makes
both halves harder to read: the conversation acquires interruptions, and the
notices acquire a context they do not have.

So they live at ``/messages``, on their own, backed by the log the daemon
already writes. The design constraint that shaped everything here: that log is
append-only and is written by a *different process* to the one serving this
page. So:

* Read state lives in a separate small file, not in the log. Marking a message
  read must never mean rewriting a file the daemon is appending to.
* New messages are found by polling size and mtime, not by holding the file
  open. A rotation under a held handle silently stops delivering anything, and
  the failure looks exactly like "nothing has happened yet".
* Every message needs an id. Ones written before ids existed get a derived one
  — from the timestamp and title — which is stable across reads, so an old
  message can still be marked read.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from ..logfile import JsonlFile

__all__ = ["MessageLog", "MESSAGES_PAGE"]


class MessageLog:
    """Read access to ``notifications.jsonl``, plus read/unread state."""

    def __init__(self, path: str | Path, *, state_path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser()
        self.state_path = (
            Path(state_path).expanduser()
            if state_path
            else self.path.with_name("messages_read.json")
        )
        self._file = JsonlFile(self.path)
        self._lock = threading.Lock()
        self._read_ids: set[str] = self._load_state()

    # -- state -------------------------------------------------------------

    def _load_state(self) -> set[str]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        return {str(x) for x in (data.get("read") or [])}

    def _save_state(self) -> None:
        # Bounded: only the ids still present in the log matter, and an
        # unbounded set of ids for messages that rotated away years ago is a
        # slow leak with no upper limit.
        keep = list(self._read_ids)[-5000:]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"read": keep}), encoding="utf-8")
        tmp.replace(self.state_path)

    def mark_read(self, ids: list[str]) -> int:
        with self._lock:
            before = len(self._read_ids)
            self._read_ids.update(str(i) for i in ids)
            self._save_state()
            return len(self._read_ids) - before

    def mark_all_read(self) -> int:
        return self.mark_read([m["id"] for m in self.recent(limit=None)])

    # -- reading -----------------------------------------------------------

    def recent(
        self, *, limit: int | None = 100, after: str | None = None, unread_only: bool = False
    ) -> list[dict[str, Any]]:
        """Newest last, the order they are displayed in.

        ``after`` returns only what followed that id — the stream endpoint's
        cursor, and also how a reconnecting page catches up without re-rendering
        everything it already has.
        """
        rows = [_normalize(row) for row in self._file.read(None)]
        if after:
            for index, row in enumerate(rows):
                if row["id"] == after:
                    rows = rows[index + 1 :]
                    break
        for row in rows:
            row["read"] = row["id"] in self._read_ids
        if unread_only:
            rows = [row for row in rows if not row["read"]]
        return rows if limit is None else rows[-limit:]

    def unread_count(self) -> int:
        return sum(1 for row in self.recent(limit=None) if not row["read"])

    def latest_id(self) -> str | None:
        rows = self.recent(limit=1)
        return rows[-1]["id"] if rows else None

    def _fingerprint(self) -> tuple[int, float]:
        try:
            stat = self.path.stat()
        except OSError:
            return (0, 0.0)
        return (stat.st_size, stat.st_mtime)

    def follow(
        self, *, after: str | None = None, interval: float = 1.0, stop: Any = None
    ) -> Iterator[dict[str, Any] | None]:
        """Yield each new message as it is appended, and ``None`` while idle.

        Polls the file's size and mtime rather than holding it open, so a
        rotation is picked up instead of silently ending delivery. A shrinking
        file *is* a rotation, and the cursor is dropped so the new file is read
        from its start.

        The idle ``None`` is what lets the caller send a keepalive without a
        second timer: an SSE connection that says nothing for a few minutes is
        closed by browsers and proxies alike, and the page then shows an empty
        list that will never fill.
        """
        cursor = after
        seen = self._fingerprint()
        while stop is None or not stop.is_set():
            current = self._fingerprint()
            if current == seen:
                yield None
            else:
                if current[0] < seen[0]:
                    cursor = None  # rotated: start again from the new file
                seen = current
                for row in self.recent(limit=None, after=cursor):
                    cursor = row["id"]
                    yield row
            time.sleep(interval)


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    """One log line as the page expects it, whatever version wrote it."""
    row = dict(row)
    at = float(row.get("at") or 0.0)
    if not row.get("id"):
        # Derived, not random: the same old line must get the same id on every
        # read, or marking it read would never stick.
        seed = f"{at}:{row.get('title', '')}:{row.get('body', '')[:120]}"
        row["id"] = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]  # noqa: S324
    row.setdefault("title", "(no title)")
    row.setdefault("body", "")
    row.setdefault("urgency", "normal")
    row.setdefault("source", row.get("task") and "task" or "itsbob")
    row.setdefault("iso", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(at)) if at else "")
    return row


MESSAGES_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>itsbob — messages</title>
<style>
  :root {
    --bg:#f7f7f5; --panel:#fff; --ink:#1b1b19; --muted:#6b6b66;
    --line:#e3e3de; --accent:#3b6ea5; --high:#b4432f; --low:#8a8a84;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --panel:#1e1e23; --ink:#e8e8e4; --muted:#9a9a94;
            --line:#2e2e35; --accent:#7aa9d8; --high:#e08a76; --low:#76766f; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--panel);
           border-bottom:1px solid var(--line); padding:14px 20px;
           display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  h1 { font-size:16px; margin:0; font-weight:600; letter-spacing:.01em; }
  .count { font-size:13px; color:var(--muted); }
  .count b { color:var(--accent); }
  .grow { flex:1; }
  button, select { font:inherit; font-size:13px; padding:6px 12px; cursor:pointer;
           border:1px solid var(--line); border-radius:7px;
           background:var(--panel); color:var(--ink); }
  button:hover { border-color:var(--accent); }
  main { max-width:820px; margin:0 auto; padding:20px; }
  .msg { background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--line);
         border-radius:9px; padding:13px 16px; margin-bottom:11px; }
  .msg.unread { border-left-color:var(--accent); }
  .msg.high { border-left-color:var(--high); }
  .msg.low { opacity:.72; }
  .msg h2 { font-size:14.5px; margin:0 0 4px; font-weight:600; }
  .meta { font-size:12px; color:var(--muted); display:flex; gap:10px; flex-wrap:wrap; }
  .body { margin-top:8px; white-space:pre-wrap; overflow-wrap:anywhere; font-size:14px; }
  .empty { color:var(--muted); text-align:center; padding:60px 20px; }
  .dot { width:7px; height:7px; border-radius:50%; background:var(--accent);
         display:inline-block; margin-right:6px; vertical-align:middle; }
  .live { font-size:12px; color:var(--muted); }
  .live.on b { color:#3f9142; }
</style></head><body>
<header>
  <h1>itsbob — messages</h1>
  <span class="count" id="count"></span>
  <span class="grow"></span>
  <select id="filter">
    <option value="all">All</option>
    <option value="unread">Unread only</option>
    <option value="high">Urgent only</option>
  </select>
  <button id="read-all">Mark all read</button>
  <button id="open-chat">Open chat</button>
  <span class="live" id="live">connecting…</span>
</header>
<main><div id="list"><div class="empty">Nothing yet. Messages Bob sends on his own
turn up here — task results, alerts, and anything he decides is worth telling you.</div></div></main>
<script>
const list = document.getElementById("list");
const countEl = document.getElementById("count");
const filterEl = document.getElementById("filter");
const liveEl = document.getElementById("live");
let messages = [];

function render() {
  const mode = filterEl.value;
  const shown = messages.filter(m =>
    mode === "all" ? true : mode === "unread" ? !m.read : m.urgency === "high");
  const unread = messages.filter(m => !m.read).length;
  countEl.innerHTML = messages.length
    ? `${messages.length} message${messages.length === 1 ? "" : "s"}` +
      (unread ? ` · <b>${unread} unread</b>` : "")
    : "";
  document.title = unread ? `(${unread}) itsbob — messages` : "itsbob — messages";
  if (!shown.length) {
    list.innerHTML = '<div class="empty">Nothing to show with this filter.</div>';
    return;
  }
  list.innerHTML = shown.slice().reverse().map(m => `
    <article class="msg ${m.read ? "" : "unread"} ${m.urgency === "high" ? "high" : ""}
             ${m.urgency === "low" ? "low" : ""}">
      <h2>${m.read ? "" : '<span class="dot"></span>'}${esc(m.title)}</h2>
      <div class="meta"><span>${esc(m.iso || "").replace("T", " ")}</span>
        ${m.task ? `<span>task: ${esc(m.task)}</span>` : ""}
        ${m.source ? `<span>${esc(m.source)}</span>` : ""}
        ${m.urgency === "high" ? "<span>urgent</span>" : ""}</div>
      ${m.body ? `<div class="body">${esc(m.body)}</div>` : ""}
    </article>`).join("");
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
}

async function load() {
  const r = await fetch("/api/messages?limit=200");
  messages = (await r.json()).messages || [];
  render();
}

function listen() {
  const es = new EventSource("/api/messages/stream");
  es.onopen = () => { liveEl.className = "live on"; liveEl.innerHTML = "<b>live</b>"; };
  es.onerror = () => { liveEl.className = "live"; liveEl.textContent = "reconnecting…"; };
  es.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.kind === "keepalive") return;
    if (messages.some(m => m.id === msg.id)) return;
    messages.push(msg);
    render();
  };
}

document.getElementById("read-all").onclick = async () => {
  await fetch("/api/messages/read", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ all: true }),
  });
  messages.forEach(m => { m.read = true; });
  render();
};
document.getElementById("open-chat").onclick = () => window.open("/", "_blank");
filterEl.onchange = render;

load().then(listen);
</script></body></html>
"""
