"""Recover model input after interruption; retain the original checkpoint evidence."""

from __future__ import annotations

from dataclasses import replace

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config

INTERRUPTED_TOOL_RESULT = (
    "This tool call was interrupted and no result was received. "
    "The underlying operation may still be running; do not assume it failed or retry "
    "a side-effecting operation without checking its state."
)


def repair_tool_messages(messages: list) -> list:
    """Pair results immediately after their AI message in a model-only copy.

    Match within each assistant turn so reused call IDs cannot consume another
    turn's results. Keep actual results, fill only missing ones, and omit orphan
    or duplicate results from the request. Never mutate messages or their content.
    """
    patched = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            if not isinstance(message, ToolMessage):
                patched.append(message)
            index += 1
            continue

        end = index + 1
        while end < len(messages) and not isinstance(messages[end], AIMessage):
            end += 1
        results = {}
        for candidate in messages[index + 1 : end]:
            if isinstance(candidate, ToolMessage):
                results.setdefault(candidate.tool_call_id, candidate)

        patched.append(message)
        for call in message.tool_calls:
            call_id = call["id"]
            patched.append(
                results[call_id]
                if call_id in results
                else ToolMessage(
                    content=INTERRUPTED_TOOL_RESULT,
                    tool_call_id=call_id,
                    name=call["name"],
                    status="error",
                )
            )
        patched.extend(
            candidate
            for candidate in messages[index + 1 : end]
            if not isinstance(candidate, ToolMessage)
        )
        index = end
    return patched


class ToolHistoryMiddleware(AgentMiddleware):
    """Repair only the input passed to the model, including old broken threads."""

    def wrap_model_call(self, request, handler):
        return handler(request.override(messages=repair_tool_messages(request.messages)))

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(messages=repair_tool_messages(request.messages)))


class RunIdentityMiddleware(AgentMiddleware):
    """Persist the owning run on new AI tool calls for reliable UI status lookup."""

    @staticmethod
    def _stamp(response):
        run_id = get_config().get("configurable", {}).get("run_id")
        if not run_id:
            return response
        return replace(
            response,
            result=[
                message.model_copy(
                    update={
                        "additional_kwargs": {
                            **message.additional_kwargs,
                            "sandbox_run_id": str(run_id),
                        }
                    }
                )
                if isinstance(message, AIMessage) and message.tool_calls
                else message
                for message in response.result
            ],
        )

    def wrap_model_call(self, request, handler):
        return self._stamp(handler(request))

    async def awrap_model_call(self, request, handler):
        return self._stamp(await handler(request))
