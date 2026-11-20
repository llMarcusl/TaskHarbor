import { AlertTriangle, Loader2, ShieldCheck, Square } from "lucide-react";
import type { ToolApproval } from "../types";
import { APPROVAL_LABELS, approvalBlocksInput, approvalNeedsResume, approvalsForPanel } from "../lib/approvals";

interface ToolApprovalPanelProps {
  approvals: ToolApproval[];
  busyIds: ReadonlySet<string>;
  streaming: boolean;
  error: string | null;
  onDecision: (approval: ToolApproval, decision: "allow" | "deny") => void;
  onResume: (approval: ToolApproval) => void;
  onCancel: () => void;
}

function displayValue(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "";
}

function expiryLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function ToolApprovalPanel({
  approvals, busyIds, error, onDecision, onResume, onCancel,
}: ToolApprovalPanelProps) {
  const visible = approvalsForPanel(approvals);
  if (visible.length === 0 && !error) return null;
  const cancelling = busyIds.has("cancel");

  return (
    <div className="space-y-3" aria-label="Tool confirmations" aria-live="polite">
      {error && (
        <p role="alert" className="rounded-lg border border-red-900/50 bg-red-950/30 px-3 py-2 text-xs text-red-300">
          {error}
        </p>
      )}
      {visible.map((approval) => {
        const args = approval.arguments;
        const execution = approval.execution_context ?? {};
        const location = approval.executor === "host" ? "This computer" :
          approval.executor === "sandbox" ? "Sandbox" : "Remote service";
        const shell = execution.powershell_path ?? execution.shell;
        const pending = approval.status === "pending";
        const busy = cancelling || busyIds.has(approval.id);
        return (
          <div key={approval.id} className="min-w-0 rounded-xl border border-amber-700/60 bg-amber-950/20 p-4 text-sm">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <ShieldCheck size={17} className="shrink-0 text-amber-400" />
              <span className="font-medium text-zinc-100">{approval.tool_name}</span>
              <span className="rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300">
                {approval.task_id && approval.task_id !== "root" ? `${approval.agent_name ?? "Task"} · ${approval.task_id}` : "Main agent"}
              </span>
              <span className="rounded border border-zinc-700 px-2 py-0.5 text-xs text-zinc-300">{location}</span>
              <span className="text-xs text-amber-300">{APPROVAL_LABELS[approval.status]}</span>
            </div>
            <p className="mb-3 whitespace-pre-wrap break-words text-xs text-zinc-300">{approval.reason}</p>
            {typeof args.command === "string" ? (
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-zinc-950 p-3 font-mono text-xs text-zinc-100">{args.command}</pre>
            ) : (
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-zinc-950 p-3 font-mono text-xs text-zinc-100">{JSON.stringify(args, null, 2)}</pre>
            )}
            <dl className="mt-3 grid min-w-0 grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
              {args.cwd != null && <><dt className="text-zinc-500">Directory</dt><dd className="whitespace-pre-wrap break-all text-zinc-200">{displayValue(args.cwd)}</dd></>}
              {shell != null && <><dt className="text-zinc-500">Shell</dt><dd className="break-all text-zinc-200">{displayValue(shell)}</dd></>}
              {args.timeout != null && <><dt className="text-zinc-500">Timeout</dt><dd className="text-zinc-200">{displayValue(args.timeout)} seconds</dd></>}
              <dt className="text-zinc-500">Expires</dt><dd className="text-zinc-300">{expiryLabel(approval.expires_at)}</dd>
            </dl>
            {approval.executor === "host" && pending && (
              <p className="mt-3 text-xs text-amber-200/80">Runs with this computer's account permissions. The working directory does not limit file access.</p>
            )}
            {approval.status === "unknown" && (
              <p className="mt-3 flex items-start gap-2 text-xs text-amber-200"><AlertTriangle size={14} className="shrink-0" />
                Execution started, but its result could not be confirmed. Check actual effects before attempting the operation again. Cancelling does not undo changes.
              </p>
            )}
            {approval.status === "superseded" && (
              <p className="mt-3 text-xs text-zinc-300">The original call will not execute. Workspace files are retained; the agent must prepare a new environment if needed.</p>
            )}
            <div className="mt-4 flex flex-wrap items-center gap-2">
              {pending && <>
                <button type="button" disabled={busy} onClick={() => onDecision(approval, "allow")}
                  className="rounded-lg bg-indigo-600 px-3 py-2 text-xs font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50">Allow once</button>
                <button type="button" disabled={busy} onClick={() => onDecision(approval, "deny")}
                  className="rounded-lg border border-zinc-600 px-3 py-2 text-xs text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50">Deny once</button>
              </>}
              {approvalNeedsResume(approval) && (
                <button type="button" disabled={busy || approval.execution_active} onClick={() => onResume(approval)}
                  className="rounded-lg bg-indigo-600 px-3 py-2 text-xs font-medium text-white hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50">
                  {approval.status === "approved" ? "Continue approved call" : "Continue without executing"}
                </button>
              )}
              {approvalBlocksInput(approval) && (
                <button type="button" disabled={busy} onClick={onCancel}
                  className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"><Square size={11} /> Cancel round</button>
              )}
              {(busyIds.has(approval.id) || cancelling || approval.status === "executing") &&
                <Loader2 size={14} className="animate-spin text-zinc-400" />}
            </div>
          </div>
        );
      })}
    </div>
  );
}
