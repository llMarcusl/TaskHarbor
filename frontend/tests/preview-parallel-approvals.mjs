// Manual UI fixture. Build first, then node tests/preview-parallel-approvals.mjs.
// In-memory responses only: never runs commands, Docker, a model, or a database.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const dist = resolve(dirname(fileURLToPath(import.meta.url)), "../dist");
const threadId = "parallel-approval-fixture";
const origin = "fixture-origin";
const events = [];
let active = false;
let stream = null;
const approvals = ["A", "B"].map((label, index) => ({
  id: label, thread_id: threadId, run_id: origin, tool_call_id: `call-${label}`,
  batch_index: index, tool_name: "execute_host_command", group: "terminal", executor: "host",
  arguments: { command: `Write-Output '${label} fixture only'`, cwd: "E:\\SandboxAgent", timeout: 30 },
  execution_context: { shell: "PowerShell" }, reason: "Preview only: no commands execute.",
  status: "pending", created_at: new Date().toISOString(),
  expires_at: new Date(Date.now() + 30 * 60_000).toISOString(), result: null,
}));
const messages = [
  { id: "human", type: "human", content: "Test approving B while A is still running" },
  { id: "assistant", type: "ai", content: "", additional_kwargs: { sandbox_run_id: origin },
    tool_calls: approvals.map((record) => ({ id: record.tool_call_id, name: record.tool_name, args: record.arguments })) },
];
const json = (res, value, status = 200) => {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(value));
};
const publicApprovals = () => approvals.map((record) => ({ ...record, execution_active: active }));
const writeValues = () => stream?.write(`event: values\ndata: ${JSON.stringify({ messages })}\n\n`);

function finish() {
  for (const record of approvals) {
    if (record.status !== "denied" && record.status !== "cancelled") {
      record.status = "succeeded";
      record.result = { type: "tool", content: JSON.stringify({ success: true, stdout: record.id }),
        tool_call_id: record.tool_call_id, name: record.tool_name };
    }
    messages.push({ id: `result-${record.id}`, type: "tool", tool_call_id: record.tool_call_id,
      name: record.tool_name, content: JSON.stringify(record.result) });
  }
  messages.push({ id: "summary", type: "ai", content: "Both fixture calls completed. B was approved while A was running. Actual commands executed: 0." });
  active = false;
  events.push("batch complete");
  writeValues();
  stream?.end();
  stream = null;
}

createServer(async (req, res) => {
  try {
    const path = new URL(req.url, "http://localhost").pathname;
    if (path === "/fixture/status") return json(res, { active, events, approvals: publicApprovals() });
    if (!path.startsWith("/api/")) {
      const file = resolve(dist, path === "/" ? "index.html" : `.${path}`);
      if (!file.startsWith(dist + sep)) return json(res, {}, 404);
      const contents = await readFile(file);
      res.writeHead(200, { "Content-Type": { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" }[extname(file)] ?? "application/octet-stream" });
      return res.end(contents);
    }
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
    if (path === "/api/settings") return json(res, {});
    if (path === "/api/security/config") return json(res, { host_commands_enabled: true, host_supported: true, host_shell: "PowerShell", default_host_timeout: 60, approval_ttl_seconds: 1800 });
    if (path === "/api/threads/search") return json(res, [{ thread_id: threadId, metadata: { title: "Parallel approval fixture" } }]);
    if (path.endsWith("/state")) return json(res, { values: { messages } });
    if (path.includes("/history")) return json(res, { enabled: false, messages: [], next_seq: null });
    if (path.endsWith("/approvals")) return json(res, { approvals: publicApprovals() });
    const decision = path.match(/\/approvals\/([AB])\/decision$/);
    if (decision) {
      const record = approvals.find((item) => item.id === decision[1]);
      if (record.status !== "pending") return json(res, {}, 409);
      events.push(`decide ${record.id} while active=${active}`);
      record.status = body.decision === "allow" ? "approved" : "denied";
      json(res, { ...record, execution_active: active });
      if (active && record.id === "B") {
        if (record.status === "approved") record.status = "executing";
        events.push("B started before A completed");
        setTimeout(finish, 1000);
      }
      return;
    }
    if (path.endsWith("/approvals/cancel")) {
      for (const record of approvals) { record.status = "cancelled"; record.result = { cancelled: true }; }
      json(res, { status: "cancelled" });
      if (active) finish();
      return;
    }
    if (path.endsWith("/runs/stream")) {
      if (active) return json(res, {}, 409);
      if (!body.command?.resume) return json(res, {}, 400);
      active = true;
      stream = res;
      approvals[0].status = "executing";
      events.push("A started; waiting for B decision");
      res.writeHead(200, { "Content-Type": "text/event-stream", "Content-Location": `/threads/${threadId}/runs/fixture-resume` });
      writeValues();
      return;
    }
    if (path.includes("/runs/")) return json(res, { run_id: path.split("/").at(-1), status: active && !path.endsWith(origin) ? "running" : "interrupted" });
    json(res, {}, 404);
  } catch (error) {
    if (res.headersSent) res.destroy();
    else json(res, { error: error.message }, error.code === "ENOENT" ? 404 : 500);
  }
}).listen(Number(process.argv[2] ?? 18923), "127.0.0.1", () => {
  process.stdout.write(`Parallel approval fixture: http://127.0.0.1:${process.argv[2] ?? 18923}\n`);
});
