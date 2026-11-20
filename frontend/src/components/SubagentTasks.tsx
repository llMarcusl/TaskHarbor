import type { AsyncTask, Message, ToolApproval } from "../types";
import { collectTaskSnapshots, TASK_STATUS_LABELS } from "../lib/subagent-tasks";

export function SubagentTasks({ messages, approvals, asyncTasks }: {
  messages: Message[]; approvals: ToolApproval[]; asyncTasks: Record<string, AsyncTask>;
}) {
  const tasks = collectTaskSnapshots(messages, approvals, asyncTasks);
  if (!tasks.length) return null;
  return (
    <section aria-label="Subagent tasks" className="space-y-2 rounded-lg border border-zinc-800 p-3 text-xs">
      <p className="font-medium text-zinc-400">Subtasks</p>
      {tasks.map((task) => (
        <div key={task.task_id} className="border-t border-zinc-800 pt-2 first:border-0">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <code className="break-all text-zinc-300">{task.agent_name ? `${task.agent_name} · ` : ""}{task.task_id}</code>
            <span className={["waiting_approval", "waiting"].includes(task.status) ? "text-amber-300" :
              ["failed", "timed_out", "error", "timeout", "interrupted"].includes(task.status) ? "text-red-300" : "text-zinc-400"}>
              {TASK_STATUS_LABELS[task.status]}
            </span>
          </div>
          {(task.summary || task.error) && <p className="mt-1 whitespace-pre-wrap break-words text-zinc-400">{task.error || task.summary}</p>}
        </div>
      ))}
    </section>
  );
}
