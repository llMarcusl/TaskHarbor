"""Two extraction triggers; neither mutates the graph's messages or checkpoint."""

from __future__ import annotations

import asyncio
import logging

from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config

from .service import get_memory_service

logger = logging.getLogger(__name__)


class _MemoryMiddleware(AgentMiddleware):
    kind: str

    def __init__(self, service=None):
        self.service = service

    def _observe(self, state, *, schedule, thread_id):
        try:
            service = self.service or get_memory_service()
            service.observe(self.kind, state["messages"], thread_id, schedule=schedule)
        except Exception as exc:
            logger.warning("Memory archive/queue unavailable (%s)", type(exc).__name__)

    def before_model(self, state, runtime):
        self._observe(state, schedule=self.kind == "session", thread_id=self._thread_id())

    async def abefore_model(self, state, runtime):
        await asyncio.to_thread(
            self._observe, state, schedule=self.kind == "session", thread_id=self._thread_id()
        )

    @staticmethod
    def _thread_id():
        return get_config().get("configurable", {}).get("thread_id")


class SessionMemoryMiddleware(_MemoryMiddleware):
    """Observe tool-complete boundaries before/after model calls; threshold in worker."""

    kind = "session"

    def after_model(self, state, runtime):
        self._observe(state, schedule=True, thread_id=self._thread_id())

    async def aafter_model(self, state, runtime):
        await asyncio.to_thread(self._observe, state, schedule=True, thread_id=self._thread_id())


class LongTermMemoryMiddleware(_MemoryMiddleware):
    """Debounce raw user/final-assistant evidence after a complete agent turn."""

    kind = "long_term"

    def after_agent(self, state, runtime):
        self._observe(state, schedule=True, thread_id=self._thread_id())

    async def aafter_agent(self, state, runtime):
        await asyncio.to_thread(self._observe, state, schedule=True, thread_id=self._thread_id())
