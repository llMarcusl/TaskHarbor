"""Current-thread raw archive access with bounded text pagination."""

import json

from langchain_core.tools import tool
from langgraph.config import get_config

from .memory.extractor import evidence_text
from .memory.service import get_memory_service


def create_history_tool(service=None):
    @tool
    def read_history(seq: int, offset: int = 0, max_chars: int = 8000) -> str:
        """Read original archived text at seq in this thread; use offset for long results."""
        if seq < 1 or offset < 0:
            return "Invalid seq or offset."
        current = service or get_memory_service()
        current.start()
        thread = get_config().get("configurable", {}).get("thread_id")
        owner = current.repository.owner_for_thread(thread)
        if not owner:
            return "Thread unavailable."
        rows = current.repository.history_page(owner, thread, seq - 1, 1, seq)
        if not rows:
            return "No archived message at this seq."
        text = evidence_text(rows[0]["payload"])
        width = max(1, min(8000, max_chars))
        return json.dumps(
            {
                "seq": seq,
                "text": text[offset : offset + width],
                "next_offset": offset + width if offset + width < len(text) else None,
            },
            ensure_ascii=False,
        )

    return read_history
