"""Unified clients for infrastructure and external APIs."""

from taskharbor.clients.aegra import (
    DEFAULT_ASSISTANT_ID,
    DEFAULT_BASE_URL,
    STREAM_TIMEOUT,
    AegraClient,
    SSEEvent,
)
from taskharbor.clients.infra import (
    get_checkpointer,
    get_chat_model,
    get_db_conninfo,
    get_db_pool,
)

__all__ = [
    # Aegra API
    "AegraClient",
    "SSEEvent",
    "DEFAULT_BASE_URL",
    "DEFAULT_ASSISTANT_ID",
    "STREAM_TIMEOUT",
    # Infrastructure
    "get_db_conninfo",
    "get_db_pool",
    "get_checkpointer",
    "get_chat_model",
]
