import type { AsyncTask, Message, ToolApproval } from "../types";

export const TASK_STATUS_LABELS = {
  queued: "Queued",
  running: "Running",
  waiting_approval: "Awaiting approval",
  completed: "Completed",
  failed: "Failed",
  cancelling: "Stopping",
  cancelled: "Cancelled",
  timed_out: "Timed out",
  pending: "Queued",
  waiting: "Awaiting approval",
  success: "Completed",
  error: "Failed",
  timeout: "Timed out",
  interrupted: "Interrupted",
} as const;

export interface TaskSnapshot {
  task_id: string;
  status: keyof typeof TASK_STATUS_LABELS;
  summary?: string;
  error?: string;
  agent_name?: string;
}

/** Hold a model's final-answer draft while its required worker outcomes are pending. */
export function taskProgressMessages(
  messages: Message[], asyncTasks: Record<string, AsyncTask>, deferredIds: Set<string>,
): Message[] {
  const pending = Object.values(asyncTasks).some((task) =>
    ["pending", "running", "waiting", "queued"].includes(task.status),
  );
  const messageKey = (message: Message) => message.id ?? JSON.stringify([message.type, message.content]);
  if (pending) {
    const latestAssistant = [...messages].reverse().find((message) => ["ai", "AIMessage", "AIMessageChunk"].includes(message.type));
    if (latestAssistant && !latestAssistant.tool_calls?.length) deferredIds.add(messageKey(latestAssistant));
  }
  return messages.filter((message) => !deferredIds.has(messageKey(message)));
}

function readSnapshot(value: unknown): TaskSnapshot | null {
  if (!value || typeof value !== "object") return null;
  const data = value as Record<string, unknown>;
  const taskId = data.task_id ?? data.thread_id;
  if (typeof taskId !== "string" || typeof data.status !== "string" ||
      !Object.prototype.hasOwnProperty.call(TASK_STATUS_LABELS, data.status)) return null;
  return {
    task_id: taskId,
    status: data.status as TaskSnapshot["status"],
    summary: typeof data.summary === "string" ? data.summary : typeof data.result === "string" ? data.result : undefined,
    error: typeof data.error === "string" ? data.error : undefined,
    agent_name: typeof data.agent_name === "string" ? data.agent_name : undefined,
  };
}

/** Graph task state takes precedence over historical tool messages. */
export function collectTaskSnapshots(
  messages: Message[], approvals: ToolApproval[] = [], asyncTasks: Record<string, AsyncTask> = {},
): TaskSnapshot[] {
  const snapshots = new Map<string, TaskSnapshot>();
  for (const message of messages) {
    if (message.type !== "tool" || !["task", "wait_tasks", "cancel_task", "start_async_task", "check_async_task",
      "update_async_task", "cancel_async_task", "list_async_tasks"].includes(message.name ?? "")) continue;
    const text = typeof message.content === "string" ? message.content :
      message.content.filter((block) => block.type === "text").map((block) => block.text ?? "").join("");
    let data: unknown;
    try { data = JSON.parse(text); } catch { continue; }
    const entries = data && typeof data === "object" && "tasks" in data ?
      (data as { tasks: unknown }).tasks : data;
    for (const entry of Array.isArray(entries) ? entries : [entries]) {
      const snapshot = readSnapshot(entry);
      if (snapshot) snapshots.set(snapshot.task_id, snapshot);
    }
  }
  for (const task of Object.values(asyncTasks)) {
    const snapshot = readSnapshot(task);
    if (snapshot) snapshots.set(snapshot.task_id, { ...snapshots.get(snapshot.task_id), ...snapshot,
      summary: snapshot.summary ?? snapshots.get(snapshot.task_id)?.summary });
  }
  for (const approval of approvals) {
    if (!approval.task_id || approval.task_id === "root") continue;
    const snapshot = snapshots.get(approval.task_id);
    if (!snapshot || ["completed", "failed", "cancelled", "timed_out", "success", "error", "timeout", "interrupted"].includes(snapshot.status)) continue;
    if (approval.status === "pending") snapshots.set(snapshot.task_id, { ...snapshot, status: "waiting_approval" });
  }
  return [...snapshots.values()];
}
