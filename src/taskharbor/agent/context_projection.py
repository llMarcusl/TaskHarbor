"""Trim textual tool evidence without breaking rich-output projection."""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from .message_projection import prepare_messages_for_llm


def _preview(text, limit, seq):
    reference = f"\n[Full original: read_history seq={seq}]"
    width = max(1, (limit - len(reference) - 3) // 2)
    return text[:width] + "\n…\n" + text[-width:] + reference


def _trim_text(text, limit, seq):
    if len(text) <= limit:
        return text
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _preview(text, limit, seq)
    if not isinstance(parsed, dict):
        return _preview(text, limit, seq)

    # Keep the original media envelope: VisionMiddleware consumes these fields
    # after compaction, removing bulky encodings and attaching image_url blocks.
    media = parsed.get("display_outputs")
    evidence = {key: value for key, value in parsed.items() if key != "display_outputs"}
    serialized = json.dumps(evidence, ensure_ascii=False)
    if media and len(serialized) <= limit:
        return text
    preview = {
        "preview": _preview(serialized, limit, seq),
        "archive_reference": {"tool": "read_history", "seq": seq},
    }
    if "success" in parsed:
        preview["success"] = parsed["success"]
    if media:
        preview["display_outputs"] = media
    return json.dumps(preview, ensure_ascii=False)


def trim_tool_message(message, limit, seq):
    """Return a bounded text preview while preserving media, IDs and call pairing.

    Raw evidence must already have been archived. Large binary encodings remain
    in the work message for existing vision projection, and are excluded from
    textual context measurement by ``projected_context_messages``.
    """
    if not isinstance(message, ToolMessage):
        return message
    if isinstance(message.content, str):
        content = _trim_text(message.content, limit, seq)
    elif isinstance(message.content, list):
        content = [
            {**block, "text": _trim_text(block["text"], limit, seq)}
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            else block
            for block in message.content
        ]
    else:
        return message
    if content == message.content:
        return message
    return message.model_copy(
        update={
            "content": content,
            "additional_kwargs": {**message.additional_kwargs, "context_trimmed": True},
        }
    )


def projected_context_messages(messages, vision):
    """Measure the same rich-output view that the model actually receives."""
    return prepare_messages_for_llm(messages, vision)
