"""Project persisted rich tool outputs into lightweight model inputs."""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage


def strip_images_from_messages(messages: list) -> list:
    """Return a copy of *messages* with all image_url blocks removed from ToolMessages."""
    cleaned: list = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
            text_blocks = [b for b in msg.content if b.get("type") != "image_url"]
            if len(text_blocks) == 1 and text_blocks[0].get("type") == "text":
                new_content = text_blocks[0]["text"]
            else:
                new_content = text_blocks or ""
            cleaned.append(
                ToolMessage(
                    content=new_content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )
        else:
            cleaned.append(msg)
    return cleaned


def prepare_messages_for_llm(messages: list, vision: bool) -> list:
    """Return a lightweight copy of *messages* suitable for the LLM.

    Heavy ``display_outputs`` payloads (base64 images, HTML, audio …) in
    ToolMessages are replaced with brief summaries so they don't bloat the
    context window.  When *vision* is enabled, image outputs are re-attached
    as efficient ``image_url`` content blocks so the model can still "see"
    them without polluting the text context.

    The full data remains in the persisted state so the frontend can render
    rich outputs as usual.
    """
    prepared: list = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            prepared.append(msg)
            continue

        if isinstance(msg.content, list):
            text_parts = [b for b in msg.content if b.get("type") == "text"]
            raw_text = text_parts[0]["text"] if text_parts else ""
        elif isinstance(msg.content, str):
            raw_text = msg.content
        else:
            prepared.append(msg)
            continue

        try:
            parsed = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            prepared.append(msg)
            continue

        if not isinstance(parsed, dict) or not parsed.get("display_outputs"):
            if isinstance(msg.content, list):
                prepared.append(
                    ToolMessage(content=raw_text, tool_call_id=msg.tool_call_id, name=msg.name)
                )
            else:
                prepared.append(msg)
            continue

        display_outputs = parsed.pop("display_outputs")
        summaries: list[str] = []
        image_blocks: list[dict] = []
        for out in display_outputs:
            mime = out.get("type", "unknown")
            if mime.startswith("image/") and vision and out.get("data"):
                image_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{out['data']}"},
                    }
                )
                summaries.append(f"[{mime} image displayed to user]")
            else:
                summaries.append(f"[{mime} output displayed to user]")

        if summaries:
            parsed["display_outputs_summary"] = summaries

        new_text = json.dumps(parsed, ensure_ascii=False)

        if image_blocks:
            new_content: str | list = [{"type": "text", "text": new_text}] + image_blocks
        else:
            new_content = new_text

        prepared.append(
            ToolMessage(content=new_content, tool_call_id=msg.tool_call_id, name=msg.name)
        )

    return prepared
