"""Email, calendar, SQL, and document tools with small dependency footprints."""

from __future__ import annotations

import imaplib
import json
import os
import smtplib
import sqlite3
import uuid
import zipfile
from email import message_from_bytes
from email.header import decode_header
from email.message import EmailMessage
from html import unescape
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ..filelock import exclusive_file_lock
from .base import Risk, Tool, ToolContext, ToolError, ToolResult

MAX_OUTPUT = 12000


def _host_ok(ctx: ToolContext, url: str) -> None:
    policy = getattr(ctx, "policy", None)
    reason = policy.check_url(url) if policy is not None and hasattr(policy, "check_url") else None
    if reason:
        raise ToolError(reason)


def _email_config(ctx: ToolContext) -> tuple[str, str, str]:
    env = ctx.env
    host, user, password = (
        str(env.get(k, "")).strip()
        for k in ("ITSBOB_EMAIL_IMAP_HOST", "ITSBOB_EMAIL_USERNAME", "ITSBOB_EMAIL_PASSWORD")
    )
    if not all((host, user, password)):
        raise ToolError(
            "email is not configured; set ITSBOB_EMAIL_IMAP_HOST, ITSBOB_EMAIL_USERNAME, and ITSBOB_EMAIL_PASSWORD"
        )
    return host, user, password


def _decode(value: str) -> str:
    parts = decode_header(value or "")
    return "".join(
        part.decode(enc or "utf-8", "replace") if isinstance(part, bytes) else part
        for part, enc in parts
    )


def _email_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    host, user, password = _email_config(ctx)
    _host_ok(ctx, f"imaps://{host}")
    limit = max(1, min(50, int(params.get("limit", 10))))
    try:
        with imaplib.IMAP4_SSL(host, int(ctx.env.get("ITSBOB_EMAIL_IMAP_PORT", 993))) as mail:
            mail.login(user, password)
            mail.select("INBOX", readonly=True)
            status, data = mail.search(None, "ALL")
            if status != "OK":
                raise ToolError("mailbox search failed")
            ids = data[0].split()[-limit:][::-1]
            rows = []
            for ident in ids:
                _, fetched = mail.fetch(ident, "(BODY.PEEK[HEADER.FIELDS (DATE FROM SUBJECT)])")
                raw = b"".join(part for part in fetched if isinstance(part, tuple))
                msg = message_from_bytes(raw)
                rows.append(
                    f"{_decode(msg.get('date', ''))} — {_decode(msg.get('from', ''))} — {_decode(msg.get('subject', ''))}"
                )
    except (OSError, imaplib.IMAP4.error, ValueError) as exc:
        raise ToolError(f"email receive failed: {exc}") from exc
    return ToolResult(ok=True, output="\n".join(rows) or "No messages found.")


def _email_send(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    env = ctx.env
    host = str(env.get("ITSBOB_EMAIL_SMTP_HOST", "")).strip()
    user = str(env.get("ITSBOB_EMAIL_USERNAME", "")).strip()
    password = str(env.get("ITSBOB_EMAIL_PASSWORD", "")).strip()
    if not all((host, user, password)):
        raise ToolError(
            "email SMTP is not configured; set ITSBOB_EMAIL_SMTP_HOST, username, and password"
        )
    _host_ok(ctx, f"smtps://{host}")
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = params["to"]
    msg["Subject"] = params["subject"]
    msg.set_content(params["body"])
    try:
        with smtplib.SMTP_SSL(
            host, int(env.get("ITSBOB_EMAIL_SMTP_PORT", 465)), timeout=20
        ) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
    except (OSError, smtplib.SMTPException) as exc:
        raise ToolError(f"email send failed: {exc}") from exc
    return ToolResult(
        ok=True, output=f"Email sent to {params['to']} with subject {params['subject']!r}."
    )


def _calendar_path(ctx: ToolContext) -> Path:
    return ctx.resolve("calendar.json")


def _calendar_read(ctx: ToolContext) -> list[dict[str, Any]]:
    path = _calendar_path(ctx)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"calendar is unreadable: {exc}") from exc


def _calendar_write(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically persist the calendar so readers never see partial JSON."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise ToolError(f"calendar could not be saved: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _calendar_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _calendar_path(ctx)
    with exclusive_file_lock(path.with_name(path.name + ".lock")):
        rows = _calendar_read(ctx)
    query = str(params.get("from", ""))
    until = str(params.get("to", ""))
    rows = [
        r
        for r in rows
        if (not query or str(r.get("start", "")) >= query)
        and (not until or str(r.get("start", "")) <= until)
    ]
    return ToolResult(ok=True, output=json.dumps(rows[:100], indent=2), data={"events": rows[:100]})


def _calendar_add(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _calendar_path(ctx)
    with exclusive_file_lock(path.with_name(path.name + ".lock")):
        rows = _calendar_read(ctx)
        event = {"id": uuid.uuid4().hex[:12], **params}
        rows.append(event)
        _calendar_write(path, rows)
    return ToolResult(ok=True, output=f"Calendar event {event['id']} created.", data=event)


def _calendar_remove(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _calendar_path(ctx)
    with exclusive_file_lock(path.with_name(path.name + ".lock")):
        rows = _calendar_read(ctx)
        kept = [r for r in rows if str(r.get("id")) != str(params["id"])]
        if len(kept) == len(rows):
            raise ToolError(f"no calendar event with id {params['id']}")
        _calendar_write(path, kept)
    return ToolResult(ok=True, output=f"Calendar event {params['id']} removed.")


def _sql_query(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = ctx.resolve(params["database"], must_exist=True)
    if path.suffix.lower() not in (".db", ".sqlite", ".sqlite3"):
        raise ToolError("database must be a SQLite file (.db/.sqlite/.sqlite3)")
    query = str(params["query"]).strip()
    first = query.split(None, 1)[0].lower() if query else ""
    if first not in {"select", "with", "explain", "pragma"} or any(
        word in query.lower() for word in ("attach ", "vacuum", "load_extension")
    ):
        raise ToolError(
            "database_query is read-only and accepts SELECT, WITH, EXPLAIN, or PRAGMA statements"
        )
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            cur = db.execute(query, tuple(params.get("parameters") or ()))
            columns = [d[0] for d in cur.description or ()]
            rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchmany(200)]
    except sqlite3.Error as exc:
        raise ToolError(f"database query failed: {exc}") from exc
    output = json.dumps(rows, default=str, indent=2)[:MAX_OUTPUT]
    return ToolResult(ok=True, output=output, data={"columns": columns, "rows": rows})


def _document_text(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = ctx.resolve(params["path"], must_exist=True)
    suffix = path.suffix.lower()
    text = ""
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        except ImportError as exc:
            raise ToolError(
                "PDF parsing requires the optional documents dependency: pip install 'itsbob[documents]'"
            ) from exc
    elif suffix in (".docx", ".xlsx", ".pptx"):
        try:
            with zipfile.ZipFile(path) as archive:
                names = [
                    n
                    for n in archive.namelist()
                    if n.endswith(".xml") and ("document" in n or "sheet" in n or "slide" in n)
                ]
                chunks = []
                for name in names:
                    root = ElementTree.fromstring(archive.read(name))
                    chunks.extend(t for t in root.itertext() if t.strip())
                text = "\n".join(unescape(t) for t in chunks)
        except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise ToolError(f"office document parsing failed: {exc}") from exc
    else:
        raise ToolError("document type unsupported; use PDF, DOCX, XLSX, or PPTX")
    return ToolResult(
        ok=True,
        output=text[:MAX_OUTPUT],
        data={"path": ctx.relative(path), "truncated": len(text) > MAX_OUTPUT},
    )


def productivity_tools() -> list[Tool]:
    return [
        Tool(
            "email_list",
            "Receive a bounded list of recent email headers over IMAP.",
            _email_list,
            {"type": "object", "properties": {"limit": {"type": "integer"}}},
            Risk.NETWORK,
        ),
        Tool(
            "email_send",
            "Send an email over SMTP using configured credentials.",
            _email_send,
            {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
            Risk.NETWORK,
        ),
        Tool(
            "calendar_list",
            "List local calendar events from calendar.json.",
            _calendar_list,
            {
                "type": "object",
                "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
            },
        ),
        Tool(
            "calendar_add",
            "Create a local calendar event with title, start, and optional details.",
            _calendar_add,
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "details": {"type": "string"},
                },
                "required": ["title", "start"],
            },
            Risk.WRITE,
            True,
        ),
        Tool(
            "calendar_remove",
            "Remove a local calendar event by id.",
            _calendar_remove,
            {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            Risk.DESTRUCTIVE,
            True,
        ),
        Tool(
            "database_query",
            "Run a read-only parameterized SQLite query against a workspace database.",
            _sql_query,
            {
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "query": {"type": "string"},
                    "parameters": {"type": "array"},
                },
                "required": ["database", "query"],
            },
        ),
        Tool(
            "parse_document",
            "Extract readable text from a PDF, DOCX, XLSX, or PPTX in the workspace.",
            _document_text,
            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
    ]
