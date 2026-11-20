"""Preserve image visibility when compaction shortens textual tool results."""

import json

from langchain_core.messages import ToolMessage

from taskharbor.agent.context_projection import (
    projected_context_messages,
    trim_tool_message,
)


def test_large_image_remains_visible_when_text_is_trimmed():
    image = {"type": "image/png", "data": "x" * 50000}
    original = ToolMessage(
        id="tool-result",
        tool_call_id="call",
        content=json.dumps({"success": True, "stdout": "a" * 30000, "display_outputs": [image]}),
        additional_kwargs={"run_id": "run"},
    )
    shortened = trim_tool_message(original, 2000, 7)
    assert shortened.id == original.id
    assert shortened.tool_call_id == original.tool_call_id
    assert shortened.additional_kwargs["run_id"] == "run"
    assert shortened.additional_kwargs["context_trimmed"]
    assert json.loads(shortened.content)["display_outputs"] == [image]
    assert json.loads(original.content)["stdout"] == "a" * 30000
    projected = projected_context_messages([shortened], True)[0]
    assert isinstance(projected.content, list)
    assert projected.content[1]["image_url"]["url"] == "data:image/png;base64," + image["data"]
    assert "read_history seq=7" in projected.content[0]["text"]
    assert len(projected.content[0]["text"]) < 2500


def test_media_encoding_alone_does_not_consume_text_budget():
    original = ToolMessage(
        id="tool",
        tool_call_id="call",
        content=json.dumps(
            {"stdout": "done", "display_outputs": [{"type": "image/png", "data": "x" * 50000}]}
        ),
    )
    assert trim_tool_message(original, 2000, 1) is original
    without_vision = projected_context_messages([original], False)[0]
    assert isinstance(without_vision.content, str)
    assert len(without_vision.content) < 200


def test_direct_multimodal_blocks_survive_text_trimming():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}}
    original = ToolMessage(
        id="tool",
        tool_call_id="call",
        content=[{"type": "text", "text": "a" * 10000}, image],
    )
    shortened = trim_tool_message(original, 1000, 2)
    assert shortened.content[1] == image
    assert "seq=2" in shortened.content[0]["text"]
    assert len(shortened.content[0]["text"]) <= 1000


def test_json_without_media_still_has_a_valid_preview_envelope():
    original = ToolMessage(tool_call_id="call", content=json.dumps({"stdout": "a" * 10000}))
    shortened = trim_tool_message(original, 1000, 9)
    payload = json.loads(shortened.content)
    assert payload["archive_reference"] == {"tool": "read_history", "seq": 9}
    assert "seq=9" in payload["preview"]
