import { useState, useCallback, useRef, useEffect } from "react";
import type { AsyncTask, Message, Settings, UploadedFileMeta, ToolApproval, SSEEvent, ThreadState } from "../types";
import * as api from "../api/aegra";
import { formatFileSize } from "../lib/utils";
import { useToolRunStates, type LiveRun } from "./useToolRunStates";
import { approvalBlocksInput, approvalNeedsResume } from "../lib/approvals";
import { taskProgressMessages } from "../lib/subagent-tasks";

function modelConfiguration(settings: Settings): Record<string, string> {
  const configurable: Record<string, string> = {
    chat_model: settings.chatModel,
    chat_model_provider: settings.chatModelProvider,
    chat_model_base_url: settings.chatModelBaseUrl ?? "",
    chat_model_supports_vision: String(settings.supportsVision),
  };
  if (settings.chatModelApiKey) configurable.chat_model_api_key = settings.chatModelApiKey;
  return configurable;
}

function buildUserContent(text: string, fileMetas: UploadedFileMeta[]): string {
  if (fileMetas.length === 0) return text;
  const fileBlock = fileMetas
    .map((f) => `- \`${f.name}\` (${formatFileSize(f.size)}) saved at \`${f.path}\``)
    .join("\n");
  return [text, `\n\n**Uploaded files:**\n${fileBlock}`].filter(Boolean).join("\n");
}

interface ActiveRequest {
  controller: AbortController;
  threadId: string | null;
  runId?: string;
  stopRequested: boolean;
  cancellation?: Promise<void>;
}

export function useChat(
  threadId: string | null,
  settings: Settings,
  createThread: () => Promise<{ thread_id: string }>,
) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [serverState, setServerState] = useState<{ threadId: string | null; busy: boolean }>({
    threadId: null, busy: false,
  });
  const serverBusy = serverState.threadId === threadId && serverState.busy;
  const [runFailure, setRunFailure] = useState<{ threadId: string; message: string } | null>(null);
  const runError = runFailure?.threadId === threadId ? runFailure.message : null;
  const [taskSnapshot, setTaskSnapshot] = useState<{
    threadId: string | null; tasks: Record<string, AsyncTask>; waiting: boolean;
  }>({ threadId: null, tasks: {}, waiting: false });
  const asyncTasks = taskSnapshot.threadId === threadId ? taskSnapshot.tasks : {};
  const awaitingSubtasks = taskSnapshot.threadId === threadId && taskSnapshot.waiting;
  const [liveRun, setLiveRun] = useState<LiveRun | null>(null);
  const activeRef = useRef<ActiveRequest | null>(null);
  const viewThreadRef = useRef(threadId);
  viewThreadRef.current = threadId;
  const loadVersionRef = useRef(0);
  const [approvalSnapshot, setApprovalSnapshot] = useState<{
    threadId: string | null; records: ToolApproval[]; loaded: boolean;
  }>({ threadId: null, records: [], loaded: false });
  const [approvalError, setApprovalError] = useState<string | null>(null);
  const [busyApprovalIds, setBusyApprovalIds] = useState<ReadonlySet<string>>(new Set());
  const busyApprovalsRef = useRef(new Set<string>());
  const queuedResumeRef = useRef(new Map<string, ToolApproval>());
  const childResumesRef = useRef(new Set<string>());
  const deferredAnswerIdsRef = useRef(new Set<string>());
  const resumeApprovalRef = useRef<((approval: ToolApproval) => Promise<void>) | null>(null);
  const approvalGenerationRef = useRef(0);
  const approvalRequestRef = useRef(0);
  const approvals = approvalSnapshot.threadId === threadId ? approvalSnapshot.records : [];
  const approvalsLoaded = !threadId || (approvalSnapshot.threadId === threadId && approvalSnapshot.loaded);
  const approvalBlocked = approvals.some(approvalBlocksInput);

  const continueQueuedApproval = useCallback((records: ToolApproval[], targetThreadId: string) => {
    if (viewThreadRef.current !== targetThreadId) return;
    for (const queued of queuedResumeRef.current.values()) {
      const record = records.find((item) => item.id === queued.id);
      // A response begun before the saved decision can still contain pending.
      if (!record || record.status === "pending") continue;
      if (!approvalNeedsResume(record)) {
        queuedResumeRef.current.delete(queued.id);
        continue;
      }
      const executionThread = record.execution_thread_id ?? record.thread_id;
      if (record.execution_active || childResumesRef.current.has(executionThread) ||
          (executionThread === targetThreadId && activeRef.current)) continue;
      queuedResumeRef.current.delete(queued.id);
      void resumeApprovalRef.current?.(record);
    }
  }, []);

  const applyGraphState = useCallback((targetThreadId: string, state: ThreadState["values"]) => {
    if (viewThreadRef.current !== targetThreadId) return;
    setTaskSnapshot({
      threadId: targetThreadId,
      tasks: state.async_tasks ?? {},
      waiting: Boolean(state.awaiting_subtasks?.length),
    });
  }, []);

  useEffect(() => () => { activeRef.current?.controller.abort(); }, []);

  const loadMessages = useCallback(async () => {
    const version = ++loadVersionRef.current;
    if (activeRef.current?.threadId === threadId && activeRef.current) return;
    if (!threadId) {
      setMessages([]);
      return;
    }
    try {
      const history = await api.getConversationMessages(threadId, AbortSignal.timeout(30_000));
      if (viewThreadRef.current === threadId && loadVersionRef.current === version &&
          activeRef.current?.threadId !== threadId) {
        setMessages(history);
      }
    } catch (err) {
      console.error("Failed to load messages:", err);
    }
  }, [threadId]);

  const refreshApprovals = useCallback(async (targetThreadId = threadId, signal?: AbortSignal) => {
    if (!targetThreadId) return [] as ToolApproval[];
    const generation = approvalGenerationRef.current;
    const requestId = ++approvalRequestRef.current;
    const records = await api.listApprovals(targetThreadId, signal ?? AbortSignal.timeout(10_000));
    if (viewThreadRef.current === targetThreadId && generation === approvalGenerationRef.current &&
        requestId === approvalRequestRef.current) {
      setApprovalSnapshot({ threadId: targetThreadId, records, loaded: true });
      setApprovalError((previous) => previous?.startsWith("Could not check tool approvals:") ? null : previous);
      continueQueuedApproval(records, targetThreadId);
    }
    return records;
  }, [threadId, continueQueuedApproval]);

  useEffect(() => {
    ++approvalGenerationRef.current;
    setApprovalError(null);
    busyApprovalsRef.current = new Set();
    setBusyApprovalIds(new Set());
    queuedResumeRef.current.clear();
    deferredAnswerIdsRef.current.clear();
    if (!threadId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let previous = "";
    const poll = async () => {
      try {
        const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(10_000)]);
        const [approvalsResult, stateResult, threadResult] = await Promise.allSettled([
          refreshApprovals(threadId, signal), api.getThreadState(threadId, signal), api.getThread(threadId, signal),
        ]);
        if (controller.signal.aborted) return;
        if (threadResult.status === "fulfilled" && viewThreadRef.current === threadId) {
          const busy = threadResult.value.status === "busy";
          setServerState({ threadId, busy });
          if (activeRef.current?.threadId !== threadId) {
            if (busy || threadResult.value.status === "error") {
              try {
                const run = await api.getLatestRun(threadId, signal);
                if (run && viewThreadRef.current === threadId && !controller.signal.aborted &&
                    activeRef.current?.threadId !== threadId) {
                  setLiveRun({ threadId, runId: run.run_id, status: run.status });
                  setRunFailure(run.status === "error" ? {
                    threadId,
                    message: run.error_message || "The main run failed before completing its response.",
                  } : null);
                }
              } catch { /* Keep the busy state until the next successful status check. */ }
            } else {
              setLiveRun((previous) => previous?.threadId === threadId ? null : previous);
              setRunFailure(null);
            }
          }
        }
        if (controller.signal.aborted) return;
        if (stateResult.status === "fulfilled" && viewThreadRef.current === threadId) {
          const values = stateResult.value.values;
          const tasks = await api.refreshAsyncTasks(values.async_tasks ?? {}, signal);
          if (controller.signal.aborted) return;
          applyGraphState(threadId, { ...values, async_tasks: tasks });
          if (activeRef.current?.threadId !== threadId) {
            const incoming = taskProgressMessages(values.messages ?? [], values.async_tasks ?? {}, deferredAnswerIdsRef.current);
            setMessages((previous) => api.mergeMessages(
              taskProgressMessages(previous, {}, deferredAnswerIdsRef.current), incoming,
            ));
          }
        }
        if (approvalsResult.status === "rejected") throw approvalsResult.reason;
        const records = approvalsResult.value;
        const signature = JSON.stringify(records.map((record) => [record.id, record.status, record.result]));
        if (previous && previous !== signature) void loadMessages();
        previous = signature;
      } catch (err) {
        if (!controller.signal.aborted && viewThreadRef.current === threadId) {
          setApprovalError(`Could not check tool approvals: ${err instanceof Error ? err.message : String(err)}`);
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(() => void poll(), 2500);
      }
    };
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [threadId, refreshApprovals, loadMessages, applyGraphState]);

  const runStates = useToolRunStates(threadId, messages, liveRun, streaming, loadMessages);
  const remoteRunId = Object.keys(runStates).find((id) =>
    ["pending", "running", "stopping"].includes(runStates[id]),
  );

  const consumeStream = useCallback(async (
    active: ActiveRequest, events: AsyncGenerator<SSEEvent>, safeError = false,
  ) => {
    for await (const event of events) {
      if (event.event === "metadata" && typeof event.data === "object" && event.data) {
        const runId = (event.data as Record<string, unknown>).run_id;
        if (typeof runId === "string") {
          active.runId = runId;
          if (!active.stopRequested) setLiveRun({ threadId: active.threadId!, runId, status: "running" });
        }
      }
      if (event.event === "error") {
        if (safeError) throw new Error("The server could not continue the saved tool call.");
        const data = event.data;
        const obj = typeof data === "object" && data ? data as Record<string, unknown> : null;
        throw new Error(typeof data === "string" ? data : String(obj?.message ?? obj?.error ?? "An unexpected error occurred."));
      }
      if (event.event === "values" && typeof event.data === "object" && event.data &&
          viewThreadRef.current === active.threadId) {
        const incoming = ((event.data as Record<string, unknown>).messages ?? []) as Message[];
        const values = event.data as ThreadState["values"];
        const visible = taskProgressMessages(incoming, values.async_tasks ?? {}, deferredAnswerIdsRef.current);
        setMessages((prev) => api.mergeMessages(taskProgressMessages(prev, {}, deferredAnswerIdsRef.current), visible));
        if (active.threadId) applyGraphState(active.threadId, event.data as ThreadState["values"]);
      }
    }
  }, [applyGraphState]);

  const finishRequest = useCallback(async (active: ActiveRequest, reloadHistory = true) => {
    await active.cancellation;
    let records: ToolApproval[] = [];
    // Keep the request occupied until approval state is refreshed: an SSE close
    // may mean "waiting for approval", rather than a completed turn.
    if (active.threadId) {
      try { records = await refreshApprovals(active.threadId); } catch { /* The poll retries. */ }
    }
    if (!active.stopRequested) setLiveRun(null);
    if (activeRef.current === active) activeRef.current = null;
    setStreaming(false);
    if (!active.threadId) return;
    if (!active.stopRequested) continueQueuedApproval(records, active.threadId);
    if (!reloadHistory) return;
    const version = ++loadVersionRef.current;
    try {
      const history = await api.getConversationMessages(active.threadId, AbortSignal.timeout(30_000));
      if (viewThreadRef.current === active.threadId && loadVersionRef.current === version &&
          !activeRef.current) setMessages(history);
    } catch { /* Keep the last streamed snapshot on a failed refresh. */ }
  }, [refreshApprovals, continueQueuedApproval]);

  const stopStreaming = useCallback(() => {
    queuedResumeRef.current.clear();
    const active = activeRef.current;
    if (!active) {
      // After reload there is no local SSE controller, but the persisted owner
      // still lets the user stop a running (or stale pre-restart) Aegra run.
      if (!threadId || !remoteRunId || runStates[remoteRunId] === "stopping") return;
      setLiveRun({ threadId, runId: remoteRunId, status: "stopping" });
      void api.cancelRun(threadId, remoteRunId).then((run) => {
        if (viewThreadRef.current === threadId) {
          setLiveRun({ threadId, runId: remoteRunId, status: run.status });
        }
      }).catch((err) => {
        console.error("Could not confirm run cancellation:", err);
        if (viewThreadRef.current === threadId) setLiveRun(null);
      }).finally(() => void loadMessages());
      return;
    }
    if (active.stopRequested) return;
    active.stopRequested = true;
    if (!active.threadId || !active.runId) {
      active.controller.abort();
      return;
    }
    const stoppedThreadId = active.threadId;
    const stoppedRunId = active.runId;
    setLiveRun({ threadId: stoppedThreadId, runId: stoppedRunId, status: "stopping" });
    active.cancellation = api.cancelRun(stoppedThreadId, stoppedRunId)
      .then((run) => {
        if (activeRef.current === active) {
          setLiveRun({ threadId: stoppedThreadId, runId: stoppedRunId, status: run.status });
        }
      })
      .catch((err) => {
        console.error("Could not confirm run cancellation:", err);
        if (activeRef.current === active) setLiveRun(null);
      })
      .finally(() => active.controller.abort());
  }, [threadId, remoteRunId, runStates, loadMessages]);

  const sendMessage = useCallback(async (content: string, files?: File[]) => {
    if ((!content.trim() && !files?.length) || activeRef.current || remoteRunId || serverBusy ||
        approvalBlocked || awaitingSubtasks || !approvalsLoaded) return;
    const active: ActiveRequest = {
      controller: new AbortController(), threadId, stopRequested: false,
    };
    activeRef.current = active;
    ++loadVersionRef.current;
    setLiveRun(null);
    setStreaming(true);
    let completed = false;
    try {
      if (!active.threadId) {
        const thread = await createThread();
        active.threadId = thread.thread_id;
        if (!active.stopRequested) setMessages([]);
      }
      if (active.stopRequested) return;
      const targetThreadId = active.threadId;
      let fileMetas: UploadedFileMeta[] = [];
      if (files?.length) {
        try {
          fileMetas = await api.uploadFiles(targetThreadId, files);
        } catch (err) {
          console.error("File upload error:", err);
        }
      }
      if (active.stopRequested) return;
      const fullContent = buildUserContent(content, fileMetas);
      if (!fullContent.trim()) return;
      const messageId = crypto.randomUUID();
      if (viewThreadRef.current === targetThreadId || viewThreadRef.current === null) {
        setMessages((prev) => [...prev, { id: messageId, type: "human", content: fullContent }]);
      }
      const onRunCreated = (runId: string) => {
        active.runId = runId;
        if (!active.stopRequested) {
          setLiveRun({ threadId: targetThreadId, runId, status: "running" });
        }
      };
      await consumeStream(active, api.streamRun(
        targetThreadId, [{ id: messageId, role: "human", content: fullContent }], modelConfiguration(settings),
        active.controller.signal, onRunCreated,
      ));
      completed = true;
    } catch (err) {
      if (!active.stopRequested && !active.controller.signal.aborted) {
        console.error("Stream error:", err);
        let paused = false;
        try { paused = (await refreshApprovals(active.threadId)).some(approvalBlocksInput); } catch { /* Show original error. */ }
        if (!paused && viewThreadRef.current === active.threadId) {
          setMessages((prev) => [...prev, {
            type: "ai", content: `Error: ${err instanceof Error ? err.message : String(err)}`,
          }]);
        }
      }
    } finally {
      await finishRequest(active, completed || active.stopRequested);
    }
  }, [threadId, settings, createThread, remoteRunId, serverBusy, approvalBlocked, awaitingSubtasks, approvalsLoaded,
    consumeStream, finishRequest, refreshApprovals]);

  const resumeApproval = useCallback(async (approval: ToolApproval) => {
    if (!threadId || approval.thread_id !== threadId ||
        !approvalNeedsResume(approval)) return;
    queuedResumeRef.current.set(approval.id, approval);
    const executionThread = approval.execution_thread_id ?? threadId;
    if (executionThread !== threadId) {
      if (approval.execution_active || childResumesRef.current.has(executionThread)) return;
      childResumesRef.current.add(executionThread);
      queuedResumeRef.current.delete(approval.id);
      try {
        await api.resumeChildApproval(approval, modelConfiguration(settings));
      } catch (err) {
        if (viewThreadRef.current === threadId) {
          if (err instanceof Error && err.message === "API error 409") {
            queuedResumeRef.current.set(approval.id, approval);
          } else {
            setApprovalError(`Continuation could not be confirmed. Check the status before retrying: ${err instanceof Error ? err.message : String(err)}`);
          }
        }
      } finally {
        childResumesRef.current.delete(executionThread);
        try { await refreshApprovals(threadId); } catch { /* The poll retries. */ }
      }
      return;
    }
    // Each execution thread owns its saved interrupts and active run.
    if (activeRef.current || approval.execution_active) return;
    queuedResumeRef.current.delete(approval.id);
    const active: ActiveRequest = {
      controller: new AbortController(), threadId, stopRequested: false,
    };
    activeRef.current = active;
    setStreaming(true);
    setApprovalError(null);
    ++loadVersionRef.current;
    const onCreated = (runId: string) => {
      active.runId = runId;
      setLiveRun({ threadId, runId, status: "running" });
    };
    try {
      await consumeStream(active, api.resumeRun(threadId, approval.id, modelConfiguration(settings),
        active.controller.signal, onCreated, approval), true);
    } catch (err) {
      if (!active.stopRequested && !active.controller.signal.aborted && viewThreadRef.current === threadId) {
        if (err instanceof Error && err.message === "Stream error 409") {
          // The server can still be finishing the previous run; the approval poll retries its wake.
          queuedResumeRef.current.set(approval.id, approval);
        } else {
          queuedResumeRef.current.delete(approval.id);
          setApprovalError(`Continuation could not be confirmed. Check the status before retrying: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
    } finally {
      await finishRequest(active);
    }
  }, [threadId, settings, consumeStream, finishRequest, refreshApprovals]);
  resumeApprovalRef.current = resumeApproval;

  const decideApproval = useCallback(async (approval: ToolApproval, decision: "allow" | "deny") => {
    if (!threadId || approval.thread_id !== threadId ||
        busyApprovalsRef.current.has(approval.id) || busyApprovalsRef.current.has("cancel") ||
        approval.status !== "pending") return;
    ++approvalGenerationRef.current;
    busyApprovalsRef.current.add(approval.id);
    setBusyApprovalIds(new Set(busyApprovalsRef.current));
    setApprovalError(null);
    try {
      const record = await api.decideApproval(threadId, approval.id, decision);
      ++approvalGenerationRef.current;
      ++approvalRequestRef.current;
      if (viewThreadRef.current === threadId) {
        setApprovalSnapshot((prev) => ({ threadId, loaded: true, records: [
          ...prev.records.filter((item) => item.id !== record.id), record,
        ] }));
        if (approvalNeedsResume(record)) void resumeApproval(record);
      }
    } catch (err) {
      if (viewThreadRef.current === threadId) {
        setApprovalError(`The decision could not be confirmed. Refresh its status before trying again: ${err instanceof Error ? err.message : String(err)}`);
      }
      try { await refreshApprovals(threadId); } catch { /* The poll retries. */ }
    } finally {
      if (viewThreadRef.current === threadId) {
        busyApprovalsRef.current.delete(approval.id);
        setBusyApprovalIds(new Set(busyApprovalsRef.current));
      }
    }
  }, [threadId, resumeApproval, refreshApprovals]);

  const cancelApprovalRound = useCallback(async () => {
    if (!threadId || busyApprovalsRef.current.has("cancel")) return;
    ++approvalGenerationRef.current;
    queuedResumeRef.current.clear();
    busyApprovalsRef.current.add("cancel");
    setBusyApprovalIds(new Set(busyApprovalsRef.current));
    setApprovalError(null);
    try {
      await api.cancelApprovals(threadId);
      const records = await refreshApprovals(threadId);
      const cancelled = records.filter((record) => record.status === "cancelled" && approvalNeedsResume(record));
      for (const record of cancelled) void resumeApproval(record);
      if (!cancelled.length && !activeRef.current) await loadMessages();
    } catch (err) {
      if (viewThreadRef.current === threadId) {
        setApprovalError(`Cancellation could not be confirmed: ${err instanceof Error ? err.message : String(err)}`);
      }
    } finally {
      if (viewThreadRef.current === threadId) {
        busyApprovalsRef.current.delete("cancel");
        setBusyApprovalIds(new Set(busyApprovalsRef.current));
      }
    }
  }, [threadId, refreshApprovals, resumeApproval, loadMessages]);

  return {
    messages, asyncTasks, awaitingSubtasks, runError, streaming: streaming || Boolean(remoteRunId) || serverBusy, runStates,
    sendMessage, stopStreaming, loadMessages, setMessages,
    approvals, approvalBlocked, approvalsLoaded, approvalError, busyApprovalIds,
    decideApproval, resumeApproval, cancelApprovalRound,
  };
}
