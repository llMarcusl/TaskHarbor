"""Independent Aegra worker graph with service-injected persistence."""

from taskharbor.agent.subagents.factory import build_worker

graph = build_worker()
