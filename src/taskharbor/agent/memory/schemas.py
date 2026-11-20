"""Bounded model outputs and deterministic memory merge rules."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Text = Annotated[str, Field(min_length=1, max_length=600)]
Items = Annotated[list[Text], Field(max_length=12)]
Sources = Annotated[list[str], Field(min_length=1, max_length=20)]
SectionName = Literal[
    "workContext",
    "personalContext",
    "topOfMind",
    "cognitiveStyle",
    "recentMonths",
    "earlierContext",
    "longTermBackground",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionMemory(StrictModel):
    goals: Items = Field(default_factory=list)
    constraints: Items = Field(default_factory=list)
    completed: Items = Field(default_factory=list)
    in_progress: Items = Field(default_factory=list)
    next_steps: Items = Field(default_factory=list)
    decisions: Items = Field(default_factory=list)
    artifacts: Items = Field(default_factory=list)
    runtime_facts: Items = Field(default_factory=list)
    errors: Items = Field(default_factory=list)


class SectionUpdate(StrictModel):
    name: SectionName
    content: Text
    source_ids: Sources
    confidence: float = Field(ge=0, le=1)
    scope: Literal["user", "project", "session"]
    durability: Literal["durable", "temporary"]
    authority: Literal["descriptive", "instruction"]


class FactChange(StrictModel):
    action: Literal["add", "update", "retract", "reinforce"]
    target_id: str | None = None
    content: Text
    category: Literal[
        "preference", "knowledge", "context", "behavior", "goal", "correction", "cognitive"
    ]
    confidence: float = Field(ge=0, le=1)
    scope: Literal["user", "project", "session"]
    durability: Literal["durable", "temporary"]
    authority: Literal["descriptive", "instruction"]
    source_ids: Sources
    explicit_correction: bool = False


class LongTermDelta(StrictModel):
    sections: list[SectionUpdate] = Field(default_factory=list, max_length=7)
    facts: list[FactChange] = Field(default_factory=list, max_length=30)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def empty_long_term() -> dict:
    return {"user_context": {}, "history": {}, "facts": []}


def merge_long_term(
    old, delta, *, user_sources, thread_id, threshold, max_facts, source_times=None
):
    """Only current raw human evidence may authorize durable descriptive changes.

    Revision checks are performed by the repository. Exact repeated facts merge;
    semantic corrections require an explicit target and correction evidence.
    """
    result = copy.deepcopy(old or empty_long_term())
    source_times = source_times or {}

    def observed_at(ids):
        return max((source_times.get(i, "") for i in ids), default="")

    def add_evidence(target, change):
        proof = {"source_thread_id": thread_id, "source_ids": change.source_ids}
        proofs = target.get(
            "sources",
            [{"source_thread_id": target["source_thread_id"], "source_ids": target["source_ids"]}],
        )
        if proof not in proofs:
            proofs.append(proof)
        target["sources"] = proofs[-8:]
        target["confidence"] = max(target["confidence"], change.confidence)
        target["observed_at"] = max(target.get("observed_at", ""), observed_at(change.source_ids))

    for section in delta.sections:
        if (
            section.confidence < threshold
            or not set(section.source_ids) <= user_sources
            or section.scope != "user"
            or section.durability != "durable"
            or section.authority != "descriptive"
        ):
            continue
        group = (
            "history"
            if section.name in ("recentMonths", "earlierContext", "longTermBackground")
            else "user_context"
        )
        timestamp = observed_at(section.source_ids)
        previous = result[group].get(section.name, {})
        if previous.get("observed_at", "") > timestamp:
            continue
        result[group][section.name] = {
            "content": section.content,
            "source_ids": section.source_ids,
            "source_thread_id": thread_id,
            "confidence": section.confidence,
            "observed_at": timestamp,
        }
    facts = {fact["id"]: fact for fact in result["facts"]}
    for change in delta.facts:
        if (
            change.confidence < threshold
            or change.scope != "user"
            or change.durability != "durable"
            or change.authority != "descriptive"
            or (change.category == "correction" and change.confidence < 0.95)
            or not set(change.source_ids) <= user_sources
        ):
            continue
        target = facts.get(change.target_id)
        timestamp = observed_at(change.source_ids)
        if change.action in ("update", "retract"):
            if target is None or not change.explicit_correction or change.confidence < 0.95:
                continue
            if target.get("observed_at", "") > timestamp:
                continue
            del facts[target["id"]]
            if change.action == "retract":
                continue
        if change.action == "reinforce":
            if target is None or change.confidence < 0.9:
                continue
            # Repeated processing never increments a score or counter.
            add_evidence(target, change)
            continue
        key = digest(" ".join(change.content.casefold().split()))
        candidate = {
            "id": key,
            "content": change.content,
            "category": change.category,
            "confidence": change.confidence,
            "source_ids": change.source_ids,
            "source_thread_id": thread_id,
            "observed_at": timestamp,
            "sources": [{"source_thread_id": thread_id, "source_ids": change.source_ids}],
        }
        if key not in facts:
            facts[key] = candidate
        else:
            add_evidence(facts[key], change)
    result["facts"] = sorted(
        facts.values(),
        key=lambda f: (f["category"] == "correction", f["confidence"], f["id"]),
        reverse=True,
    )[:max_facts]
    return result
