"""Per-agent model selection using the existing configurable settings."""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from taskharbor.agent.configuration import Configuration
from taskharbor.settings import Settings


class ModelRegistry:
    """Keep the default model and cache models created for runtime overrides."""

    def __init__(self, default: BaseChatModel, settings: Settings) -> None:
        self.default = default
        self.default_key = self._key(
            settings.CHAT_MODEL,
            settings.CHAT_MODEL_PROVIDER,
            settings.CHAT_MODEL_API_KEY,
            settings.CHAT_MODEL_BASE_URL,
        )
        self.overrides: dict[str, BaseChatModel] = {}

    @staticmethod
    def _key(model: str, provider: str, api_key: str | None, base_url: str | None) -> str:
        return f"{model}|{provider}|{hash(api_key or '')}|{base_url or ''}"

    def resolve(self, cfg: Configuration) -> BaseChatModel:
        key = self._key(
            cfg.chat_model, cfg.chat_model_provider, cfg.chat_model_api_key, cfg.chat_model_base_url
        )
        if key == self.default_key:
            return self.default
        if key not in self.overrides:
            kwargs: dict[str, Any] = {
                "model": cfg.chat_model,
                "model_provider": cfg.chat_model_provider,
            }
            if cfg.chat_model_base_url:
                kwargs["base_url"] = cfg.chat_model_base_url
            if cfg.chat_model_api_key:
                kwargs["api_key"] = cfg.chat_model_api_key
            # create_agent binds the tools after middleware selects the model.
            self.overrides[key] = init_chat_model(**kwargs)
        return self.overrides[key]
