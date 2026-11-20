// Manual browser fixture: npm run build, then node tests/preview-interruption.mjs.
// Uses an in-memory mock API; never contacts the application database or Docker.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";

const dist = resolve(dirname(fileURLToPath(import.meta.url)), "../dist");
const threadId = "interruption-fixture";
const runs = { old: { run_id: "old", status: "interrupted" } };
const pending = new Map();
let sequence = 0;
const messages = [
  { type: "human", content: "Previous interrupted conversation", id: "h0" },
  { type: "ai", content: "", id: "a0", additional_kwargs: { sandbox_run_id: "old" },
    tool_calls: [{ id: "old-tool", name: "create_session", args: { language: "python" } }] },
  { type: "human", content: "Older message without run metadata", id: "h-legacy" },
  { type: "ai", content: "", id: "a-legacy",
    tool_calls: [{ id: "legacy", name: "execute_code", args: { code: "print(1)" } }] },
];
const json = (res, data, status = 200) => {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
};
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, "http://localhost").pathname;
    if (!path.startsWith("/api/")) {
      const file = resolve(dist, path === "/" ? "index.html" : `.${path}`);
      if (!file.startsWith(dist + "/") && !file.startsWith(dist + "\\")) return json(res, {}, 404);
      const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
      res.writeHead(200, { "Content-Type": types[extname(file)] ?? "application/octet-stream" });
      res.end(await readFile(file));
      return;
    }
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
    if (path === "/api/settings") return json(res, {});
    if (path === "/api/health") return json(res, { status: "ok" });
    if (path === "/api/sessions") return json(res, []);
    if (path.endsWith("/approvals")) return json(res, { approvals: [] });
    if (path === "/api/security/config") return json(res, {
      host_commands_enabled: false, host_supported: true, host_shell: "PowerShell",
      default_host_timeout: 60, approval_ttl_seconds: 1800,
    });
    if (path === "/api/threads/search") return json(res, [{ thread_id: threadId, metadata: { title: "Interruption test" } }]);
    if (path === "/api/threads" && req.method === "POST") return json(res, { thread_id: threadId, metadata: {} });
    if (path.endsWith("/state")) return json(res, { values: { messages } });
    if (path.endsWith("/runs/stream")) {
      const id = `run-${++sequence}`;
      runs[id] = { run_id: id, status: "running" };
      messages.push({ type: "human", content: body.input.messages[0].content, id: `h${sequence}` });
      messages.push({ type: "ai", content: "", id: `a${sequence}`,
        additional_kwargs: { sandbox_run_id: id },
        tool_calls: [{ id: `call-${sequence}`, name: "create_session", args: { language: "python", dependencies: {} } }] });
      res.writeHead(200, { "Content-Type": "text/event-stream", "Content-Location": `/threads/${threadId}/runs/${id}` });
      res.write(`event: metadata\ndata: ${JSON.stringify({ run_id: id })}\n\n`);
      res.write(`event: values\ndata: ${JSON.stringify({ messages })}\n\n`);
      pending.set(id, res);
      res.on("close", () => pending.delete(id));
      return;
    }
    const match = path.match(/\/runs\/([^/]+)(\/cancel)?$/);
    if (match) {
      const run = runs[match[1]];
      if (!run) return json(res, {}, 404);
      if (match[2]) {
        run.status = "interrupted";
        pending.get(run.run_id)?.end();
        console.log(`Cancelled ${run.run_id}`);
      }
      return json(res, run);
    }
    json(res, {}, 404);
  } catch (err) {
    console.error(err);
    if (!res.headersSent) json(res, {}, 500);
    else res.end();
  }
});
server.listen(5187, "127.0.0.1", () => console.log("Mock interruption UI: http://127.0.0.1:5187"));
