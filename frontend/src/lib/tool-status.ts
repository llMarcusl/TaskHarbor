import type { Message, ToolRunStatus } from "../types";

export function isAI(message: Message): boolean {
  return ["ai", "AIMessage", "AIMessageChunk"].includes(message.type);
}

export function isTool(message: Message): boolean {
  return message.type === "tool" || message.type === "ToolMessage";
}

export function messageRunId(message: Message): string | undefined {
  const id = message.additional_kwargs?.sandbox_run_id;
  return typeof id === "string" && id.length > 0 ? id : undefined;
}

/** Scope results to the issuing assistant message, including providers reusing IDs. */
export function indexToolResults(messages: Message[]): Map<number, Map<string, Message>> {
  const index = new Map<number, Map<string, Message>>();
  let owner = -1;
  let calls = new Set<string>();
  messages.forEach((message, position) => {
    if (isAI(message)) {
      owner = position;
      calls = new Set(message.tool_calls?.map((call) => call.id));
      index.set(owner, new Map());
    } else if (isTool(message) && message.tool_call_id && calls.has(message.tool_call_id)) {
      const results = index.get(owner)!;
      if (!results.has(message.tool_call_id)) results.set(message.tool_call_id, message);
    }
  });
  return index;
}

export function unresolvedRunIds(messages: Message[]): string[] {
  const results = indexToolResults(messages);
  const ids = new Set<string>();
  messages.forEach((message, position) => {
    const runId = messageRunId(message);
    if (runId && message.tool_calls?.some((call) => !results.get(position)?.has(call.id))) {
      ids.add(runId);
    }
  });
  return [...ids].sort();
}

export type MissingToolStatus = "running" | "stopping" | "interrupted" | "unknown";

export function missingToolStatus(
  message: Message,
  runs: Record<string, ToolRunStatus>,
): MissingToolStatus {
  const id = messageRunId(message);
  const status = id ? runs[id] : undefined;
  if (status === "pending" || status === "running") return "running";
  if (status === "stopping") return "stopping";
  if (!status || status === "unknown") return "unknown";
  return "interrupted";
}
