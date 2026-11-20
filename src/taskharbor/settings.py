"""Application settings via Pydantic Settings.

Loads from environment variables and ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """TaskHarbor configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM (provider-agnostic) ──
    CHAT_MODEL: str = "gpt-4o"
    CHAT_MODEL_PROVIDER: str = "openai"
    CHAT_MODEL_API_KEY: str
    CHAT_MODEL_BASE_URL: str | None = None
    CHAT_MODEL_SUPPORTS_VISION: bool | None = None

    # Independent extraction model. No implicit fallback to the primary model/key.
    SESSION_MEMORY_ENABLED: bool = False
    LONG_TERM_MEMORY_ENABLED: bool = False
    MEMORY_INJECTION_ENABLED: bool = True
    SESSION_MEMORY_INJECTION_MAX_CHARS: int = Field(default=6000, ge=0, le=64000)
    LONG_TERM_MEMORY_INJECTION_MAX_CHARS: int = Field(default=6000, ge=0, le=100000)
    MEMORY_INJECTION_MAX_CHARS: int = Field(default=13000, ge=0, le=164000)
    MEMORY_MODEL: str | None = None
    MEMORY_MODEL_PROVIDER: str | None = None
    MEMORY_MODEL_API_KEY: str | None = None
    MEMORY_MODEL_BASE_URL: str | None = None
    CONTEXT_COMPACTION_ENABLED: bool = False
    # Explicit model-name -> total context tokens; unknown models fail closed when enabled.
    CONTEXT_MODEL_WINDOWS: dict[str, int] = Field(default_factory=dict)
    CONTEXT_OUTPUT_RESERVE: int = Field(default=8192, ge=1)
    CONTEXT_SAFETY_TOKENS: int = Field(default=2048, ge=0)
    CONTEXT_TRIGGER_RATIO: float = Field(default=0.8, gt=0, lt=1)
    CONTEXT_TARGET_RATIO: float = Field(default=0.5, gt=0, lt=1)
    CONTEXT_KEEP_MESSAGES: int = Field(default=6, ge=1)
    CONTEXT_TOOL_MAX_CHARS: int = Field(default=12000, ge=1000)
    CONTEXT_OLD_TOOL_MAX_CHARS: int = Field(default=2000, ge=500)
    CONTEXT_IMAGE_TOKENS: int = Field(default=4096, ge=1)
    MEMORY_TIMEOUT_SECONDS: int = Field(default=30, ge=1, le=300)
    MEMORY_OUTPUT_TOKENS: int = Field(default=4096, ge=256, le=16384)
    MEMORY_INPUT_CHARS: int = Field(default=48000, ge=16000, le=500000)
    MEMORY_WORKERS: int = Field(default=2, ge=1, le=16)
    MEMORY_POLL_SECONDS: float = Field(default=1, ge=0.05, le=60)
    MEMORY_MAX_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    MEMORY_CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0, le=1)
    MEMORY_MAX_FACTS: int = Field(default=100, ge=1, le=1000)
    MEMORY_STANDALONE_OWNER: str | None = None
    SESSION_MEMORY_INITIAL_TOKENS: int = Field(default=10000, ge=1)
    SESSION_MEMORY_STEP_TOKENS: int = Field(default=5000, ge=1)
    SESSION_MEMORY_MAX_CHARS: int = Field(default=16000, ge=1000, le=64000)
    LONG_TERM_MEMORY_MAX_CHARS: int = Field(default=24000, ge=5000, le=100000)
    LONG_TERM_MEMORY_DEBOUNCE_SECONDS: float = Field(default=30, ge=0, le=300)
    LONG_TERM_MEMORY_MAX_DELAY_SECONDS: float = Field(default=300, ge=1, le=3600)

    @model_validator(mode="after")
    def _memory_model_required(self):
        if (
            self.SESSION_MEMORY_ENABLED
            or self.LONG_TERM_MEMORY_ENABLED
            or self.CONTEXT_COMPACTION_ENABLED
        ):
            if not self.MEMORY_MODEL or not self.MEMORY_MODEL_PROVIDER:
                raise ValueError(
                    "Memory extraction requires MEMORY_MODEL and MEMORY_MODEL_PROVIDER"
                )
        if self.CONTEXT_COMPACTION_ENABLED:
            if not self.CONTEXT_MODEL_WINDOWS or any(
                n <= self.CONTEXT_OUTPUT_RESERVE + self.CONTEXT_SAFETY_TOKENS
                for n in self.CONTEXT_MODEL_WINDOWS.values()
            ):
                raise ValueError("Compaction requires explicit usable CONTEXT_MODEL_WINDOWS")
            if self.CONTEXT_TARGET_RATIO >= self.CONTEXT_TRIGGER_RATIO:
                raise ValueError("Context target must be below trigger")
        return self

    # ── Container ──
    CONTAINER_MEMORY_LIMIT: str = "2048m"
    CONTAINER_CPU_QUOTA: int = 200_000
    CONTAINER_PIDS_LIMIT: int = 512
    CONTAINER_TMPFS_SIZE: str = "200m"
    CONTAINER_ORPHAN_MIN_AGE_SECONDS: int = 300
    CONTAINER_EXECUTION_TIMEOUT_SECONDS: int = 30
    CONTAINER_MAX_SESSIONS: int = 5
    CONTAINER_MAX_SESSIONS_PER_THREAD: int = 3
    CONTAINER_EXECUTE_AS_ROOT: bool = False
    CONTAINER_NETWORK_ENABLED: bool = True

    # ── Session Lifecycle / GC ──
    SESSION_IDLE_TTL_SECONDS: int = Field(default=600, ge=0)
    SESSION_MAX_LIFETIME_SECONDS: int = 7200
    SESSION_GC_INTERVAL_SECONDS: int = 60
    SESSION_MAX_ACTIVE_THREADS: int = 10
    SANDBOX_LEASE_TTL_SECONDS: int = Field(default=120, ge=10)
    SANDBOX_QUEUE_TIMEOUT_SECONDS: float = Field(default=30, ge=0, le=300)
    SANDBOX_MAX_WAITERS: int = Field(default=16, ge=0)
    SANDBOX_STANDALONE_OWNER: str | None = None
    CONTAINER_TERMINAL_TIMEOUT_SECONDS: int = Field(default=300, ge=1, le=3600)
    CONTAINER_INSTALL_TIMEOUT_SECONDS: int = Field(default=600, ge=1, le=3600)

    # Tool decisions are Agent-local. Rules are trusted server configuration.
    TOOL_SECURITY_RULES: list[dict[str, Any]] = Field(default_factory=list)
    TOOL_APPROVAL_TTL_SECONDS: int = Field(default=1800, ge=10, le=86400)
    TOOL_MAX_CONCURRENCY: int = Field(default=4, ge=1, le=32)
    AGENT_MAX_STEPS: int = Field(default=1000, ge=25, le=10000)
    SUBAGENTS_ENABLED: bool = True
    SUBAGENT_MAX_CONCURRENCY: int = Field(default=3, ge=1, le=8)
    SUBAGENT_MAX_TASKS_PER_RUN: int = Field(default=12, ge=1, le=50)
    SUBAGENT_TIMEOUT_SECONDS: int = Field(default=900, ge=1, le=3600)
    SUBAGENT_MAX_STEPS: int = Field(default=80, ge=4, le=300)
    SUBAGENT_PROTOCOL_URL: str | None = None
    HOST_COMMANDS_ENABLED: bool = False
    HOST_COMMAND_TIMEOUT_SECONDS: int = Field(default=60, ge=1, le=300)
    HOST_COMMAND_MAX_TIMEOUT_SECONDS: int = Field(default=300, ge=1, le=300)

    # ── Output Limits (characters) ──
    MAX_STDOUT_CHARS: int = 50000
    MAX_STDERR_CHARS: int = 120000
    MAX_RESULT_CHARS: int = 30000
    MAX_TRACEBACK_CHARS: int = 8000

    # ── Encryption ──
    ENCRYPTION_KEY: str = ""

    # ── Storage (uploads + exports) ──
    STORAGE_DIR: Path = Path("./storage")
    IMPORT_ALLOWED_DIRS: list[Path] = []

    # ── API (for download URLs in export_files) ──
    API_BASE_URL: str = "http://127.0.0.1:8000"

    # ── Agent ──
    MAX_ITERATIONS: int = 25

    @field_validator("IMPORT_ALLOWED_DIRS", mode="before")
    @classmethod
    def _parse_import_dirs(cls, v: Any) -> list[Path]:
        if isinstance(v, str):
            return [Path(d.strip()) for d in v.split(",") if d.strip()]
        return v

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        root = Path(__file__).resolve().parent.parent.parent
        self.STORAGE_DIR = (
            self.STORAGE_DIR.resolve()
            if self.STORAGE_DIR.is_absolute()
            else (root / self.STORAGE_DIR).resolve()
        )
        self.IMPORT_ALLOWED_DIRS = [
            p.resolve() if p.is_absolute() else (root / p).resolve()
            for p in self.IMPORT_ALLOWED_DIRS
        ]
        return self

    # ── Checkpointer (PostgreSQL, shared with Aegra) ──
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
