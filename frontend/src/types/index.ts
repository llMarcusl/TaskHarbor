export interface ContentBlock {
  type: string;
  text?: string;
  thinking?: string;
  reasoning?: string;
  image_url?: { url: string };
}

export interface DisplayOutput {
  type: string;  // MIME type (image/png, text/html, audio/wav, etc.)
  data: string;  // raw string (HTML/SVG) or base64 (binary types)
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface Message {
  id?: string;
  type: string;
  content: string | ContentBlock[];
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  additional_kwargs?: Record<string, unknown>;
  status?: "success" | "error";
}

export type RunStatus = "pending" | "running" | "success" | "error" | "interrupted" | "timeout";
export type ToolRunStatus = RunStatus | "stopping" | "unknown";
export interface Run {
  run_id: string;
  status: RunStatus;
  error_message?: string | null;
}

export interface Thread {
  thread_id: string;
  metadata: Record<string, unknown>;
  status?: "idle" | "busy" | "interrupted" | "error";
  created_at?: string;
  updated_at?: string;
}

export interface ThreadState {
  values: {
    messages?: Message[];
    async_tasks?: Record<string, AsyncTask>;
    awaiting_subtasks?: string[] | null;
  };
}

export interface AsyncTask {
  task_id: string;
  agent_name: string;
  thread_id: string;
  run_id: string;
  status: string;
  created_at: string;
  last_checked_at: string;
  last_updated_at: string;
}

export interface SSEEvent {
  event: string;
  data: unknown;
}

export interface Settings {
  chatModel: string;
  chatModelProvider: string;
  chatModelApiKey: string;
  chatModelApiKeyHint: string;
  chatModelBaseUrl: string;
  supportsVision: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  chatModel: "gpt-4o",
  chatModelProvider: "openai",
  chatModelApiKey: "",
  chatModelApiKeyHint: "",
  chatModelBaseUrl: "",
  supportsVision: true,
};

export interface UploadedFileMeta {
  name: string;
  path: string;
  size: number;
}

export type ApprovalStatus = "pending" | "approved" | "denied" | "expired" |
  "cancelled" | "executing" | "succeeded" | "failed" | "unknown" | "superseded";

export interface ToolApproval {
  id: string;
  thread_id: string;
  execution_thread_id?: string;
  interrupt_id?: string | null;
  assistant_id?: string;
  run_id: string | null;
  tool_call_id: string;
  batch_index?: number;
  execution_active?: boolean;
  transport?: "live" | "checkpoint";
  task_id?: string | null;
  agent_name?: string | null;
  tool_name: string;
  group: string;
  executor: string;
  arguments: Record<string, unknown>;
  execution_context?: Record<string, unknown>;
  reason: string;
  status: ApprovalStatus;
  created_at: string;
  expires_at: string;
  result?: unknown;
}

export interface SecurityConfig {
  host_commands_enabled: boolean;
  host_shell: string;
  host_supported: boolean;
  default_host_timeout: number;
  approval_ttl_seconds: number;
}

export const RUNTIME_LANGUAGE: Record<string, string> = {
  python: "python",
  node: "javascript",
  r: "r",
};
