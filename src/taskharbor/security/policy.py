"""Small deterministic deny/ask/allow policy. No model-written conditions."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from taskharbor.security.registry import ToolSpec


class ToolRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    effect: Literal["allow", "deny", "ask"]
    tool: str | None = None
    group: str | None = None
    executor: Literal["sandbox", "host", "connector"] | None = None
    cwd: str | None = None

    @model_validator(mode="after")
    def meaningful_match(self):
        if not any((self.tool, self.group, self.executor, self.cwd)):
            raise ValueError("A tool rule needs a tool, group, executor, or cwd")
        if self.cwd and not Path(self.cwd).is_absolute():
            raise ValueError("Rule cwd must be an absolute directory")
        return self

    def matches(self, spec: ToolSpec, arguments: dict) -> bool:
        if self.tool is not None and self.tool != spec.name:
            return False
        if self.group is not None and self.group != spec.group:
            return False
        if self.executor is not None and self.executor != spec.executor:
            return False
        if self.cwd:
            candidate = arguments.get("cwd")
            if not isinstance(candidate, str) or not Path(candidate).is_absolute():
                return False
            try:
                Path(candidate).resolve().relative_to(Path(self.cwd).resolve())
            except (ValueError, OSError):
                return False
        return True


class PolicyEngine:
    def __init__(self, rules=()):
        self.rules = tuple(ToolRule.model_validate(rule) for rule in rules)

    def decide(self, spec: ToolSpec, arguments: dict) -> str:
        matches = [rule for rule in self.rules if rule.matches(spec, arguments)]
        if any(rule.effect == "deny" for rule in matches):
            return "deny"
        if any(rule.effect == "ask" for rule in matches):
            return "ask"
        if any(rule.effect == "allow" for rule in matches):
            return "allow"
        return "unmatched"
