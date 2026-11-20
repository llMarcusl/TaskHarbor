import assert from "node:assert/strict";
import test from "node:test";
import { collectTaskSnapshots, taskProgressMessages } from "../src/lib/subagent-tasks.ts";
import { refreshAsyncTasks } from "../src/api/aegra.ts";

const result = (name, data) => ({ type: "tool", name, content: JSON.stringify(data) });

test("task submission is a queued snapshot, not a completed task", () => {
  const tasks = collectTaskSnapshots([result("task", { task_id: "A", status: "queued" })]);
  assert.equal(tasks[0].status, "queued");
});

test("wait and cancellation snapshots update the matching task without dropping siblings", () => {
  const tasks = collectTaskSnapshots([
    result("task", { task_id: "A", status: "queued" }),
    result("task", { task_id: "B", status: "queued" }),
    result("wait_tasks", { tasks: [
      { task_id: "A", status: "completed", summary: "Checked data" },
      { task_id: "B", status: "running" },
    ] }),
    result("cancel_task", { task_id: "B", status: "cancelled" }),
  ]);
  assert.deepEqual(tasks.map(({ task_id, status }) => [task_id, status]), [["A", "completed"], ["B", "cancelled"]]);
  assert.equal(tasks[0].summary, "Checked data");
});

test("pending live approval identifies its child but cannot overwrite a terminal result", () => {
  const messages = [result("wait_tasks", { tasks: [
    { task_id: "A", status: "running" }, { task_id: "B", status: "completed" },
  ] })];
  const approvals = ["A", "B"].map((task_id) => ({ task_id, transport: "live", status: "pending" }));
  assert.deepEqual(collectTaskSnapshots(messages, approvals).map((task) => task.status), ["waiting_approval", "completed"]);
});

test("ordinary tool output and malformed task data do not create task cards", () => {
  assert.deepEqual(collectTaskSnapshots([
    result("execute_code", { task_id: "A", status: "running" }),
    result("task", { task_id: "B", status: "invented" }),
    { type: "tool", name: "task", content: "not JSON" },
  ]), []);
});

test("checkpoint async_tasks stay visible after tool history compaction", () => {
  const tasks = collectTaskSnapshots([], [], {
    A: { task_id: "A", thread_id: "A", agent_name: "taskharbor-worker", run_id: "run-A", status: "waiting" },
    B: { task_id: "B", thread_id: "B", agent_name: "taskharbor-worker", run_id: "run-B", status: "success" },
  });
  assert.deepEqual(tasks.map(({ task_id, status }) => [task_id, status]), [["A", "waiting"], ["B", "success"]]);
  assert.equal(tasks[0].agent_name, "taskharbor-worker");
});

test("graph status wins over old tool output while retaining the result summary", () => {
  const tasks = collectTaskSnapshots([
    result("check_async_task", { thread_id: "A", status: "success", result: "Checked records" }),
  ], [], { A: { task_id: "A", status: "success", agent_name: "taskharbor-worker" } });
  assert.equal(tasks[0].status, "success");
  assert.equal(tasks[0].summary, "Checked records");
});

test("checkpoint child approvals identify waiting work without changing terminal tasks", () => {
  const state = { A: { task_id: "A", status: "running" }, B: { task_id: "B", status: "success" } };
  const approvals = ["A", "B"].map((task_id) => ({ task_id, transport: "checkpoint", status: "pending" }));
  assert.deepEqual(collectTaskSnapshots([], approvals, state).map((task) => task.status), ["waiting_approval", "success"]);
});

test("child task refresh follows the newest run and distinguishes approvals from cancellation", async (t) => {
  t.mock.method(globalThis, "fetch", async (url) => {
    if (url.includes("/runs?")) return Response.json([{ run_id: "resumed-run", status: "interrupted" }]);
    return Response.json({ status: url.endsWith("/A") ? "interrupted" : "idle" });
  });
  const tasks = await refreshAsyncTasks({
    A: { task_id: "A", thread_id: "A", run_id: "old-run", status: "running" },
    B: { task_id: "B", thread_id: "B", run_id: "old-run", status: "running" },
  });
  assert.equal(tasks.A.run_id, "resumed-run");
  assert.equal(tasks.A.status, "waiting");
  assert.equal(tasks.B.status, "cancelled");
});

test("failed child status refresh retains the confirmed task state", async (t) => {
  const confirmed = { task_id: "A", thread_id: "A", run_id: "run-A", status: "running" };
  t.mock.method(globalThis, "fetch", async () => new Response("unavailable", { status: 503 }));
  assert.deepEqual(await refreshAsyncTasks({ A: confirmed }), { A: confirmed });
});

test("a final-answer draft stays hidden through worker notifications until the collected answer arrives", () => {
  const hidden = new Set();
  const history = { id: "history", type: "ai", content: "Previous round answer" };
  const start = { id: "start", type: "ai", content: "", tool_calls: [{ id: "s", name: "start_async_task", args: {} }] };
  const draft = { id: "draft", type: "ai", content: "Premature answer" };
  const pending = { A: { task_id: "A", status: "running" } };
  assert.deepEqual(taskProgressMessages([history, start], pending, hidden), [history, start]);
  assert.deepEqual(taskProgressMessages([history, start, draft], pending, hidden), [history, start]);
  const notification = { id: "notification", type: "human", content: "Task finished", additional_kwargs: { lc_source: "async_subagent" } };
  const final = { id: "final", type: "ai", content: "Collected answer" };
  assert.deepEqual(taskProgressMessages([history, start, draft, notification, final],
    { A: { task_id: "A", status: "success" } }, hidden), [history, start, notification, final]);
});
