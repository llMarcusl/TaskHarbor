import type { Message, ToolApproval, ToolCall } from "../types";

export function approvalNeedsResume(approval: ToolApproval): boolean {
  // Child cancellation is finalized by the service, which retires its run/thread.
  // A checkpoint can still contain the old interrupt while the thread is idle;
  // resuming it would race cleanup and Aegra correctly rejects that request.
  const child = Boolean(approval.task_id && approval.task_id !== "root") ||
    Boolean(approval.execution_thread_id && approval.execution_thread_id !== approval.thread_id);
  if (child && approval.status === "cancelled") return false;
  return approval.transport !== "live" && approval.result == null &&
    ["approved", "denied", "expired", "cancelled", "superseded"].includes(approval.status);
}

export function approvalBlocksInput(approval: ToolApproval): boolean {
  return ["pending", "approved", "executing"].includes(approval.status) ||
    (approval.status === "unknown" && approval.result == null) || approvalNeedsResume(approval);
}

export function approvalsForPanel(approvals: ToolApproval[]): ToolApproval[] {
  const ordered = [...approvals].sort((a, b) =>
    (a.batch_index ?? 0) - (b.batch_index ?? 0) ||
    (a.created_at ?? "").localeCompare(b.created_at ?? ""),
  );
  // An unanswered child must not hide another task's approval.
  return ordered.filter(approvalBlocksInput);
}

export function approvalForTool(
  approvals: ToolApproval[], message: Message, call: ToolCall,
): ToolApproval | undefined {
  const runId = message.additional_kwargs?.sandbox_run_id;
  const taskId = message.additional_kwargs?.sandbox_task_id;
  return [...approvals].reverse().find((approval) =>
    approval.tool_call_id === call.id && approval.tool_name === call.name &&
    (!approval.task_id || approval.task_id === "root" || approval.task_id === taskId) &&
    (typeof runId !== "string" || approval.run_id === runId),
  );
}

export const APPROVAL_LABELS: Record<ToolApproval["status"], string> = {
  pending: "Awaiting approval",
  approved: "Approved · waiting to run",
  denied: "Denied",
  expired: "Approval expired",
  cancelled: "Cancelled",
  executing: "Executing approved call",
  succeeded: "Completed",
  failed: "Failed",
  unknown: "Result unknown · do not retry",
  superseded: "Environment or call changed",
};
