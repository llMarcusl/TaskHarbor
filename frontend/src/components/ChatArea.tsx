import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { Send, Loader2, Paperclip, Square, X, Upload } from "lucide-react";
import type { AsyncTask, Message, ToolRunStatus, ToolApproval } from "../types";
import { indexToolResults, missingToolStatus } from "../lib/tool-status";
import { MessageBubble } from "./MessageBubble";
import { extractSessions } from "../lib/utils";
import { ToolApprovalPanel } from "./ToolApprovalPanel";
import { SubagentTasks } from "./SubagentTasks";

interface ChatAreaProps {
  messages: Message[];
  streaming: boolean;
  runStates: Record<string, ToolRunStatus>;
  runError: string | null;
  onSendMessage: (content: string, files?: File[]) => void;
  onStopStreaming: () => void;
  threadId: string | null;
  approvals: ToolApproval[];
  asyncTasks: Record<string, AsyncTask>;
  awaitingSubtasks: boolean;
  approvalBlocked: boolean;
  approvalsLoaded: boolean;
  approvalError: string | null;
  busyApprovalIds: ReadonlySet<string>;
  onApprovalDecision: (approval: ToolApproval, decision: "allow" | "deny") => void;
  onApprovalResume: (approval: ToolApproval) => void;
  onApprovalCancel: () => void;
}

export function ChatArea({
  messages,
  streaming,
  runStates,
  runError,
  onSendMessage,
  onStopStreaming,
  threadId,
  approvals,
  asyncTasks,
  awaitingSubtasks,
  approvalBlocked,
  approvalsLoaded,
  approvalError,
  busyApprovalIds,
  onApprovalDecision,
  onApprovalResume,
  onApprovalCancel,
}: ChatAreaProps) {
  const [input, setInput] = useState("");
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const dragCounterRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const followBottomRef = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const approvalStateKey = approvals.map((approval) => `${approval.id}:${approval.status}`).join("|");

  useEffect(() => {
    followBottomRef.current = true;
  }, [threadId]);

  useEffect(() => {
    const container = scrollRef.current;
    if (container && followBottomRef.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages, approvalStateKey, asyncTasks, awaitingSubtasks, threadId, runError]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [threadId]);

  const toolResults = useMemo(() => indexToolResults(messages), [messages]);

  const sessionRuntimes = useMemo(() => {
    const sessions = extractSessions(messages);
    const map = new Map<string, string>();
    for (const [sid, info] of sessions) {
      if (info.runtime) map.set(sid, info.runtime);
    }
    return map;
  }, [messages]);

  const handleSubmit = () => {
    const text = input.trim();
    if ((!text && pendingFiles.length === 0) || streaming || approvalBlocked || awaitingSubtasks || !approvalsLoaded) return;
    followBottomRef.current = true;
    setInput("");
    const files = pendingFiles.length > 0 ? [...pendingFiles] : undefined;
    setPendingFiles([]);
    onSendMessage(text, files);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setPendingFiles((prev) => [...prev, ...newFiles]);
    }
    e.target.value = "";
  };

  const removeFile = (index: number) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== index));
  };

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current += 1;
    if (e.dataTransfer.types.includes("Files")) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current = 0;
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setPendingFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)]);
    }
  }, []);

  return (
    <div
      className="relative flex min-h-0 min-w-0 flex-1 flex-col w-full"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {isDragging && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-zinc-900/80 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-indigo-500 bg-zinc-800/90 px-12 py-10">
            <Upload size={40} className="text-indigo-400" />
            <p className="text-lg font-medium text-zinc-200">Solte os arquivos aqui</p>
            <p className="text-sm text-zinc-400">Os arquivos serão anexados à mensagem</p>
          </div>
        </div>
      )}
      <div
        ref={scrollRef}
        onScroll={(event) => {
          const container = event.currentTarget;
          followBottomRef.current = container.scrollHeight - container.scrollTop - container.clientHeight < 80;
        }}
        className="flex min-h-0 min-w-0 flex-1 flex-col w-full overflow-y-auto px-4 py-6"
      >
        <div className="flex w-full min-w-0 flex-1 flex-col space-y-3">
          {messages.length === 0 && (
            <div className="flex h-full items-center justify-center pt-20 text-zinc-500">
              <p>Send a message to start the conversation.</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble
              key={i}
              message={msg}
              messageIndex={i}
              allMessages={messages}
              streaming={streaming}
              toolResults={toolResults.get(i)}
              missingStatus={missingToolStatus(msg, runStates)}
              threadId={threadId}
              sessionRuntimes={sessionRuntimes}
              approvals={approvals}
            />
          ))}
          {streaming && !approvalBlocked && (
            <div className="flex justify-start">
              <div className="flex items-center gap-2 rounded-2xl rounded-bl-md bg-zinc-800 px-4 py-2.5 text-sm text-zinc-400">
                <Loader2 size={14} className="animate-spin" />
                Thinking...
              </div>
            </div>
          )}
          <SubagentTasks messages={messages} approvals={approvals} asyncTasks={asyncTasks} />
          {runError && <div role="alert" className="rounded-xl border border-red-900/60 bg-red-950/30 p-3 text-sm text-red-200">
            <p>Main task stopped: {runError}</p>
            <p className="mt-1 text-xs">Completed subtask files are retained. Send a follow-up to continue.</p>
          </div>}
          {awaitingSubtasks && <div className="flex items-center gap-3 text-xs text-zinc-400">
            <p role="status">Waiting for subtask results…</p>
            <button type="button" disabled={busyApprovalIds.has("cancel")} onClick={onApprovalCancel}
              className="text-zinc-300 hover:text-white disabled:opacity-50">Cancel round</button>
          </div>}
          <ToolApprovalPanel
            approvals={approvals}
            busyIds={busyApprovalIds}
            streaming={streaming}
            error={approvalError}
            onDecision={onApprovalDecision}
            onResume={onApprovalResume}
            onCancel={onApprovalCancel}
          />
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="flex w-full min-w-0 shrink-0 flex-wrap border-t border-zinc-800 bg-zinc-900 p-4">
        <div className="flex w-full min-w-0 flex-col">
          {approvalBlocked && <p className="mb-2 text-xs text-amber-300">Resolve the tool confirmation or cancel this round before sending another message.</p>}
          {!approvalsLoaded && <p className="mb-2 text-xs text-zinc-500">Checking saved tool confirmations…</p>}
          {pendingFiles.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {pendingFiles.map((f, i) => (
                <div
                  key={i}
                  className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-800 px-2 py-1 text-xs text-zinc-300"
                >
                  <Paperclip size={12} className="text-zinc-500" />
                  <span className="max-w-[150px] truncate">{f.name}</span>
                  <button
                    onClick={() => removeFile(i)}
                    className="text-zinc-500 hover:text-zinc-300"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="flex w-full min-w-0 items-end gap-3">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={handleFileSelect}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={approvalBlocked || awaitingSubtasks || !approvalsLoaded}
              className="flex h-10 shrink-0 items-center justify-center rounded-xl border border-zinc-700 bg-zinc-800 px-3 text-zinc-400 transition-colors hover:bg-zinc-700 hover:text-zinc-200"
              title="Upload files"
            >
              <Paperclip size={18} />
            </button>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message..."
              rows={1}
              className="min-w-0 flex-1 resize-none overflow-y-auto rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
              style={{ minHeight: "2.5rem" }}
            />
            <button
              onClick={streaming ? onStopStreaming : handleSubmit}
              disabled={!streaming && (approvalBlocked || awaitingSubtasks || !approvalsLoaded || (!input.trim() && pendingFiles.length === 0))}
              className="flex h-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 px-4 text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
              title={streaming ? "Stop response" : "Send message"}
              aria-label={streaming ? "Stop response" : "Send message"}
            >
              {streaming ? <Square size={18} fill="currentColor" /> : <Send size={18} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
