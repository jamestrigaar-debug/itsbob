"""Offline coverage for the productivity tool extensions."""

import sqlite3
import zipfile

from itsbob.tools import build_toolbox
from itsbob.agent.persona import Persona


def test_productivity_tools_are_registered(tmp_path):
    box = build_toolbox(workspace=tmp_path, mode="trusted", confirm=lambda *_: True, env={})
    assert {
        "email_list",
        "email_send",
        "calendar_list",
        "calendar_add",
        "calendar_remove",
        "database_query",
        "parse_document",
    } <= set(box.registry.names())


def test_tool_awareness_pre_prompt_keeps_descriptions_on_continuation(tmp_path):
    box = build_toolbox(workspace=tmp_path, mode="trusted", env={})
    prompt = Persona().render(
        tools=box.registry.render_for_prompt(described=False),
        tool_awareness=box.registry.render_awareness(),
        tool_names=tuple(box.registry.names()),
        continuing=True,
    )
    assert "## Tool capability guide" in prompt
    assert "email_send: Send an email" in prompt
    assert "email_send(to: string" in prompt


def test_calendar_round_trip(tmp_path):
    box = build_toolbox(workspace=tmp_path, mode="trusted", confirm=lambda *_: True, env={})
    created = box.call("calendar_add", title="Review", start="2026-09-01T10:00")
    assert created.ok
    listed = box.call("calendar_list")
    assert listed.ok and "Review" in listed.output
    assert box.call("calendar_remove", id=created.data["id"]).ok
    assert "Review" not in box.call("calendar_list").output


def test_database_query_is_read_only(tmp_path):
    db = tmp_path / "data.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("create table items (name text)")
    connection.execute("insert into items values ('one')")
    connection.commit()
    connection.close()
    box = build_toolbox(workspace=tmp_path, mode="trusted", env={})
    result = box.call("database_query", database="data.sqlite", query="select * from items")
    assert result.ok and result.data["rows"] == [{"name": "one"}]
    refused = box.call("database_query", database="data.sqlite", query="delete from items")
    assert not refused.ok and "read-only" in refused.error


def test_docx_text_is_extracted_without_office_installation(tmp_path):
    path = tmp_path / "note.docx"
    document = "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'><w:body><w:p><w:r><w:t>Hello document</w:t></w:r></w:p></w:body></w:document>"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
    box = build_toolbox(workspace=tmp_path, mode="trusted", env={})
    result = box.call("parse_document", path="note.docx")
    assert result.ok and "Hello document" in result.output
