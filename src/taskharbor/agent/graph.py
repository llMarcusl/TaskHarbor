"""Aegra / LangGraph Platform entry point."""

from taskharbor.agent.factory import build_agent
from taskharbor.sandbox import get_manager

__all__ = ["build_agent", "graph"]

graph = build_agent(manager=get_manager(), checkpointer=None)
