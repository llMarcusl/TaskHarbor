import { useEffect, useMemo, useState } from "react";
import type { Message, ToolRunStatus } from "../types";
import * as api from "../api/aegra";
import { unresolvedRunIds } from "../lib/tool-status";

export interface LiveRun {
  threadId: string;
  runId: string;
  status: ToolRunStatus;
}

/** Query only runs with missing results. Poll active runs after a page reload. */
export function useToolRunStates(
  threadId: string | null,
  messages: Message[],
  liveRun: LiveRun | null,
  streaming: boolean,
  reload: () => Promise<void>,
): Record<string, ToolRunStatus> {
  const [snapshot, setSnapshot] = useState<{
    threadId: string | null;
    runs: Record<string, ToolRunStatus>;
  }>({ threadId: null, runs: {} });
  const idsKey = JSON.stringify(unresolvedRunIds(messages));
  // Once the local stream is gone, an active status must be checked again.
  // A cancellation response can still report running while the worker settles.
  const liveIsAuthoritative = streaming || !["pending", "running"].includes(liveRun?.status ?? "");
  const liveId = liveRun?.threadId === threadId && liveIsAuthoritative ? liveRun.runId : undefined;

  useEffect(() => {
    if (!threadId) return;
    const ids = (JSON.parse(idsKey) as string[]).filter((id) => id !== liveId);
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    const poll = async () => {
      const settled = await Promise.allSettled(
        ids.map((id) => api.getRun(threadId, id, AbortSignal.any([
          controller.signal, AbortSignal.timeout(10_000),
        ]))),
      );
      if (controller.signal.aborted) return;
      const runs: Record<string, ToolRunStatus> = {};
      settled.forEach((result, index) => {
        runs[ids[index]] = result.status === "fulfilled" ? result.value.status : "unknown";
      });
      setSnapshot({ threadId, runs });
      // The checkpoint may have gained a real result while this page was offline.
      if (!streaming && ids.length > 0) await reload();
      if (controller.signal.aborted) return;
      const active = Object.values(runs).some((status) => ["pending", "running"].includes(status));
      const retry = settled.some((result) => result.status === "rejected") && ++attempts < 3;
      if (active || retry) timer = setTimeout(() => void poll(), 2000);
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [threadId, idsKey, liveId, streaming, reload]);

  return useMemo(() => ({
    ...(liveRun?.threadId === threadId ? { [liveRun.runId]: liveRun.status } : {}),
    ...(snapshot.threadId === threadId ? snapshot.runs : {}),
    ...(liveRun?.threadId === threadId && liveIsAuthoritative
      ? { [liveRun.runId]: liveRun.status } : {}),
  }), [snapshot, threadId, liveRun, liveIsAuthoritative]);
}
