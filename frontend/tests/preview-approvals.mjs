// Manual UI fixture: npm run build && node tests/preview-approvals.mjs
// In-memory API only. Never executes host commands or contacts Docker/database.
// First resume intentionally fails, to exercise "decision saved, retry continuation".
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";

const dist = resolve(dirname(fileURLToPath(import.meta.url)), "../dist");
const threadId = "approval-fixture";
const args = { command: "Get-Content -LiteralPath '.\\报告.txt'", cwd: "E:\\示例项目", timeout: 60 };
const approval = {
  id: "approval-1", thread_id: threadId, run_id: "initial-run", tool_call_id: "host-call",
  tool_name: "execute_host_command", group: "terminal", executor: "host", arguments: args,
  execution_context: { shell: "PowerShell", powershell_path: "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" },
  reason: "Host commands require confirmation before running on this computer.",
  status: "pending", created_at: new Date().toISOString(),
  expires_at: new Date(Date.now() + 30 * 60_000).toISOString(), result: null,
};
const messages = [
  { id: "human-1", type: "human", content: "Read 报告.txt on my computer" },
  { id: "ai-1", type: "ai", content: "", additional_kwargs: { sandbox_run_id: "initial-run" },
    tool_calls: [{ id: "host-call", name: "execute_host_command", args }] },
];
let resumeAttempts = 0;
let executions = 0;
const json = (res, data, status = 200) => {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
};
createServer(async (req, res) => {
  try {
    const path = new URL(req.url, "http://localhost").pathname;
    if (!path.startsWith("/api/")) {
      const file = resolve(dist, path === "/" ? "index.html" : `.${path}`);
      if (!file.startsWith(dist + "/") && !file.startsWith(dist + "\\")) return json(res, {}, 404);
      res.writeHead(200, { "Content-Type": { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" }[extname(file)] ?? "application/octet-stream" });
      return res.end(await readFile(file));
    }
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
    if (path === "/api/settings") return json(res, {});
    if (path === "/api/health") return json(res, { status: "ok" });
    if (path === "/api/sessions") return json(res, []);
    if (path === "/api/security/config") return json(res, {
      host_commands_enabled: true, host_supported: true, host_shell: "PowerShell",
      default_host_timeout: 60, approval_ttl_seconds: 1800,
    });
    if (path === "/api/threads/search") return json(res, [{ thread_id: threadId, metadata: { title: "Approval fixture" } }]);
    if (path.endsWith("/state")) return json(res, { values: { messages } });
    if (path.includes("/history")) return json(res, { enabled: false, messages: [], next_seq: null });
    if (path.endsWith("/approvals")) return json(res, { approvals: [approval] });
    if (path.endsWith("/approvals/approval-1/decision")) {
      if (approval.status !== "pending") return json(res, { error: "Already decided" }, 409);
      approval.status = body.decision === "allow" ? "approved" : "denied";
      return json(res, approval);
    }
    if (path.endsWith("/approvals/cancel")) {
      approval.status = "cancelled";
      return json(res, { status: "cancelled" });
    }
    if (path.endsWith("/runs/stream")) {
      if (body.command?.resume?.approval_id !== approval.id || body.input) return json(res, { error: "Invalid resume" }, 400);
      if (++resumeAttempts === 1) return json(res, { error: "Simulated connection failure before execution" }, 503);
      if (approval.status === "approved") {
        approval.status = "executing";
        ++executions;
        await new Promise((done) => setTimeout(done, 700));
        approval.status = "succeeded";
        approval.result = { stdout: "中文内容", exit_code: 0 };
      } else approval.result ??= { success: false, error: "Call was not permitted" };
      if (!messages.some((message) => message.tool_call_id === "host-call")) {
        messages.push({ id: "tool-1", type: "tool", content: JSON.stringify(approval.result), tool_call_id: "host-call" });
        messages.push({ id: "ai-2", type: "ai", content: `Fixture complete. Actual host executions: 0. Simulated executions: ${executions}.` });
      }
      res.writeHead(200, { "Content-Type": "text/event-stream", "Content-Location": `/threads/${threadId}/runs/resumed-run` });
      return res.end(`event: values\ndata: ${JSON.stringify({ messages })}\n\n`);
    }
    if (path.includes("/runs/")) return json(res, { run_id: path.split("/").at(-1), status: "interrupted" });
    json(res, {}, 404);
  } catch (err) {
    console.error(err);
    if (!res.headersSent) json(res, { error: String(err) }, 500);
    else res.end();
  }
}).listen(5188, "127.0.0.1", () => console.log("Approval fixture: http://127.0.0.1:5188"));
