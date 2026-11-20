import type {
  Thread,
  ThreadState,
  SSEEvent,
  Settings,
  UploadedFileMeta,
  Run,
  Message,
  ToolApproval,
  SecurityConfig,
  AsyncTask,
} from "../types";

const BASE_URL = "/api";
const ASSISTANT_ID = "taskharbor";

async function request<T>(path: string, options?: RequestInit, safeError = false): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    if (safeError) throw new Error(`API error ${res.status}`);
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    await request("/health");
    return true;
  } catch {
    return false;
  }
}

export async function getSettings(): Promise<Partial<Settings>> {
  return request<Partial<Settings>>("/settings");
}

export async function saveSettings(
  settings: Settings,
): Promise<Partial<Settings>> {
  return request<Partial<Settings>>("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export function getSecurityConfig(): Promise<SecurityConfig> {
  return request<SecurityConfig>("/security/config", undefined, true);
}

export async function listApprovals(threadId: string, signal?: AbortSignal): Promise<ToolApproval[]> {
  const response = await request<{ approvals: ToolApproval[] }>(`/threads/${encodeURIComponent(threadId)}/approvals`, { signal }, true);
  return response.approvals;
}

export function decideApproval(threadId: string, approvalId: string, decision: "allow" | "deny"): Promise<ToolApproval> {
  return request<ToolApproval>(`/threads/${encodeURIComponent(threadId)}/approvals/${encodeURIComponent(approvalId)}/decision`, {
    method: "POST", body: JSON.stringify({ decision }), signal: AbortSignal.timeout(15_000),
  }, true);
}

export function cancelApprovals(threadId: string): Promise<{ status: "cancelled" }> {
  return request<{ status: "cancelled" }>(`/threads/${encodeURIComponent(threadId)}/approvals/cancel`, {
    method: "POST", signal: AbortSignal.timeout(15_000),
  }, true);
}

// ── Sandbox sessions ──────────────────────────────────────────────────────

export interface ActiveSession {
  session_id: string;
  container_id: string;
  runtime: string;
  status: string;
  thread_id: string | null;
  created_at: string;
  last_activity: string;
}

export async function listSessions(): Promise<ActiveSession[]> {
  return request<ActiveSession[]>("/sessions");
}

export async function killSession(sessionId: string): Promise<void> {
  await request(`/sessions/${sessionId}`, { method: "DELETE" });
}

// ── Threads ───────────────────────────────────────────────────────────────

export async function createThread(
  metadata?: Record<string, unknown>,
): Promise<Thread> {
  return request<Thread>("/threads", {
    method: "POST",
    body: JSON.stringify({ metadata: metadata ?? {} }),
  });
}

export async function listThreads(limit = 50): Promise<Thread[]> {
  const visible: Thread[] = [];
  const pageSize = Math.min(limit, 100);
  let offset = 0;
  while (visible.length < limit) {
    const threads = await request<Thread[]>("/threads/search", {
      method: "POST",
      body: JSON.stringify({ limit: pageSize, offset }),
    });
    visible.push(...threads.filter((thread) => thread.metadata?.sandbox_worker !== true));
    if (threads.length < pageSize) break;
    offset += threads.length;
  }
  return visible.slice(0, limit);
}

export async function deleteThread(threadId: string): Promise<void> {
  await fetch(`${BASE_URL}/threads/${threadId}`, { method: "DELETE" });
}

export function getThread(threadId: string, signal?: AbortSignal): Promise<Thread> {
  return request<Thread>(`/threads/${encodeURIComponent(threadId)}`, { signal }, true);
}

export async function getLatestRun(threadId: string, signal?: AbortSignal): Promise<Run | undefined> {
  const runs = await request<Run[]>(`/threads/${encodeURIComponent(threadId)}/runs?limit=1`, { signal }, true);
  return runs[0];
}

export async function getThreadState(
  threadId: string,
  signal?: AbortSignal,
): Promise<ThreadState> {
  return request<ThreadState>(`/threads/${threadId}/state`, { signal });
}

export function visibleMessages(messages: Message[]): Message[] {
  return messages.filter((m) => !m.additional_kwargs?.memory_internal ||
    m.additional_kwargs?.context_notice);
}

export function mergeMessages(previous: Message[], incoming: Message[]): Message[] {
  const merged = [...visibleMessages(previous)];
  for (const message of visibleMessages(incoming)) {
    // Trimmed work-state output must never replace the complete archived/UI result.
    const index = message.id ? merged.findIndex((m) => m.id === message.id) : -1;
    if (index >= 0) {
      if (!message.additional_kwargs?.context_trimmed) merged[index] = message;
    } else merged.push(message);
  }
  return merged;
}

export async function getConversationMessages(threadId: string, signal?: AbortSignal): Promise<Message[]> {
  const state = await getThreadState(threadId, signal);
  let history: Message[] = [];
  let after = 0;
  while (true) {
    const page = await request<{ enabled: boolean; messages: Message[]; next_seq: number | null }>(
      `/threads/${threadId}/history?after_seq=${after}`, { signal },
    );
    history = mergeMessages(history, page.messages);
    if (!page.enabled || page.next_seq === null) break;
    if (page.next_seq <= after) throw new Error("Invalid history cursor");
    after = page.next_seq;
  }
  return mergeMessages(history, state.values?.messages ?? []);
}

export function getRun(threadId: string, runId: string, signal?: AbortSignal): Promise<Run> {
  return request<Run>(`/threads/${threadId}/runs/${runId}`, { signal });
}

/** Parent checkpoints can be paused while their independent children advance. */
export async function refreshAsyncTasks(tasks: Record<string, AsyncTask>, signal?: AbortSignal): Promise<Record<string, AsyncTask>> {
  const updated = { ...tasks };
  await Promise.all(Object.values(tasks).map(async (task) => {
    if (["success", "error", "cancelled", "timeout", "interrupted"].includes(task.status)) return;
    try {
      const path = `/threads/${encodeURIComponent(task.thread_id)}`;
      const [thread, runs] = await Promise.all([
        request<{ status: string }>(path, { signal }, true),
        request<Run[]>(`${path}/runs?limit=1`, { signal }, true),
      ]);
      const run = runs[0];
      if (!run) return;
      const status = thread.status === "interrupted" ? "waiting" :
        run.status === "interrupted" ? "cancelled" : run.status;
      updated[task.task_id] = { ...task, run_id: run.run_id, status, last_checked_at: new Date().toISOString() };
    } catch {
      // A failed refresh retains the last confirmed checkpoint status.
    }
  }));
  return updated;
}

export function cancelRun(threadId: string, runId: string): Promise<Run> {
  return request<Run>(`/threads/${threadId}/runs/${runId}/cancel?action=cancel&wait=1`, {
    method: "POST",
    signal: AbortSignal.timeout(10_000),
  });
}

export async function uploadFiles(
  threadId: string,
  files: File[],
): Promise<UploadedFileMeta[]> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  const res = await fetch(`${BASE_URL}/threads/${threadId}/files/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload error ${res.status}: ${text}`);
  }
  return res.json();
}

export function getDownloadUrl(
  threadId: string,
  sessionId: string,
  path: string,
  artifactId?: string,
): string {
  const url = `${BASE_URL}/threads/${encodeURIComponent(threadId)}/files/download?session_id=${encodeURIComponent(sessionId)}&path=${encodeURIComponent(path)}`;
  return artifactId ? `${url}&artifact_id=${encodeURIComponent(artifactId)}` : url;
}

export async function* streamRun(
  threadId: string,
  messages: { id?: string; role: string; content: string }[],
  configurable?: Record<string, string>,
  signal?: AbortSignal,
  onRunCreated?: (runId: string) => void,
): AsyncGenerator<SSEEvent> {
  const payload: Record<string, unknown> = {
    assistant_id: ASSISTANT_ID,
    input: { messages },
    stream_mode: ["values"],
  };
  if (configurable) {
    payload.config = { configurable };
  }

  yield* streamPayload(threadId, payload, signal, onRunCreated);
}

/** Resume the saved call by ID. Permission and frozen arguments stay on the server. */
export async function* resumeRun(
  threadId: string,
  approvalId: string,
  configurable?: Record<string, string>,
  signal?: AbortSignal,
  onRunCreated?: (runId: string) => void,
  target?: Pick<ToolApproval, "interrupt_id" | "assistant_id">,
): AsyncGenerator<SSEEvent> {
  const payload: Record<string, unknown> = {
    assistant_id: target?.assistant_id ?? ASSISTANT_ID,
    command: { resume: approvalResumeValue(approvalId, target?.interrupt_id) },
    stream_mode: ["values"],
  };
  if (configurable) payload.config = { configurable };
  yield* streamPayload(threadId, payload, signal, onRunCreated, true);
}

export function approvalResumeValue(approvalId: string, interruptId?: string | null): Record<string, unknown> {
  const decision = { approval_id: approvalId };
  return interruptId ? { [interruptId]: decision } : decision;
}

/** Child graphs resume independently of the parent's open response stream. */
export function resumeChildApproval(approval: ToolApproval, configurable?: Record<string, string>): Promise<Run> {
  return request<Run>(`/threads/${encodeURIComponent(approval.execution_thread_id ?? approval.thread_id)}/runs`, {
    method: "POST",
    body: JSON.stringify({
      assistant_id: approval.assistant_id ?? "taskharbor-worker",
      command: { resume: approvalResumeValue(approval.id, approval.interrupt_id) },
      ...(configurable ? { config: { configurable } } : {}),
    }),
    signal: AbortSignal.timeout(15_000),
  }, true);
}

async function* streamPayload(
  threadId: string,
  payload: Record<string, unknown>,
  signal?: AbortSignal,
  onRunCreated?: (runId: string) => void,
  safeError = false,
): AsyncGenerator<SSEEvent> {

  const res = await fetch(`${BASE_URL}/threads/${threadId}/runs/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) {
    if (safeError) throw new Error(`Stream error ${res.status}`);
    const text = await res.text();
    throw new Error(`Stream error ${res.status}: ${text}`);
  }

  // Available before the first SSE event, even when a tool starts immediately.
  const location = res.headers.get("Content-Location");
  const runId = location?.match(/\/runs\/([^/]+)\/?$/)?.[1];
  if (runId) onRunCreated?.(runId);

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";
  const dataLines: string[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      } else if (line === "") {
        if (dataLines.length > 0) {
          const raw = dataLines.join("\n");
          let parsed: unknown;
          try {
            parsed = JSON.parse(raw);
          } catch {
            parsed = raw;
          }
          yield { event: eventType, data: parsed };
        }
        eventType = "";
        dataLines.length = 0;
      }
    }
  }

  if (dataLines.length > 0) {
    const raw = dataLines.join("\n");
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = raw;
    }
    yield { event: eventType, data: parsed };
  }
}
