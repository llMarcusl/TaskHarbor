import assert from "node:assert/strict";
import test from "node:test";
import { approvalBlocksInput, approvalNeedsResume, approvalForTool, approvalsForPanel } from "../src/lib/approvals.ts";
import { listApprovals, decideApproval, cancelApprovals, resumeRun, resumeChildApproval, listThreads, getThread, getLatestRun } from "../src/api/aegra.ts";

const approval = (status, result) => ({
  id: "approval-1", thread_id: "thread-1", run_id: "run-1", tool_call_id: "call-1",
  tool_name: "execute_host_command", group: "terminal", executor: "host",
  arguments: { command: "Get-Content README.md", cwd: "E:\\Project", timeout: 60 },
  reason: "Host command needs confirmation", status, result,
});

test("pending, execution and unresolved decisions block new messages after refresh", () => {
  for (const status of ["pending", "approved", "executing", "unknown", "denied", "expired", "cancelled", "superseded"]) {
    assert.equal(approvalBlocksInput(approval(status)), true, status);
  }
  for (const status of ["denied", "expired", "cancelled", "superseded", "unknown", "succeeded", "failed"]) {
    assert.equal(approvalBlocksInput(approval(status, { message: "Settled" })), false, status);
  }
});

test("only decided unconsumed calls offer resume; unknown execution never offers retry", () => {
  for (const status of ["approved", "denied", "expired", "cancelled", "superseded"]) {
    assert.equal(approvalNeedsResume(approval(status)), true, status);
    assert.equal(approvalNeedsResume(approval(status, { success: false })), false, status);
  }
  for (const status of ["pending", "executing", "unknown", "succeeded", "failed"]) {
    assert.equal(approvalNeedsResume(approval(status)), false, status);
  }
});

test("root and child live approvals are consumed by their running invocation without graph resume", () => {
  for (const taskId of [null, "task-A"]) {
    for (const status of ["pending", "approved", "denied", "expired", "cancelled", "superseded", "executing"]) {
      const record = { ...approval(status), transport: "live", task_id: taskId };
      assert.equal(approvalNeedsResume(record), false, `${taskId ?? "root"}: ${status}`);
    }
    assert.equal(approvalBlocksInput({ ...approval("pending"), transport: "live", task_id: taskId }), true);
    assert.equal(approvalBlocksInput({ ...approval("denied"), transport: "live", task_id: taskId }), false);
  }
});

test("cancelled child approvals never resume even before cleanup saves their result", () => {
  for (const identity of [
    { task_id: "child-A" },
    { execution_thread_id: "child-A" },
  ]) {
    const record = { ...approval("cancelled"), transport: "checkpoint", ...identity };
    assert.equal(approvalNeedsResume(record), false);
    assert.equal(approvalBlocksInput(record), false);
    assert.deepEqual(approvalsForPanel([record]), []);
  }
  const root = { ...approval("cancelled"), transport: "checkpoint", task_id: null,
    execution_thread_id: "thread-1" };
  assert.equal(approvalNeedsResume(root), true);
  assert.equal(approvalBlocksInput(root), true);
  for (const status of ["approved", "denied"]) {
    assert.equal(approvalNeedsResume({ ...approval(status), task_id: "child-A" }), true);
  }
});

test("a child approval with a reused tool ID does not attach to the main transcript", () => {
  const child = { ...approval("pending"), task_id: "task-A", transport: "live" };
  const call = { id: "call-1", name: "execute_host_command", args: {} };
  assert.equal(approvalForTool([child], { additional_kwargs: { sandbox_run_id: "run-1" } }, call), undefined);
  assert.equal(approvalForTool([child], {
    additional_kwargs: { sandbox_run_id: "run-1", sandbox_task_id: "task-A" },
  }, call), child);
});

test("approval badge cannot leak to a tool call from a different run", () => {
  const record = approval("pending");
  const call = { id: "call-1", name: "execute_host_command", args: {} };
  assert.equal(approvalForTool([record], { additional_kwargs: { sandbox_run_id: "run-1" } }, call), record);
  assert.equal(approvalForTool([record], { additional_kwargs: { sandbox_run_id: "other-run" } }, call), undefined);
  assert.equal(approvalForTool([record], {}, { ...call, name: "execute_terminal" }), undefined);
});

test("cancelled execution warnings stay in history without stacking over the next approval", () => {
  const cancelledExecution = {
    ...approval("unknown", { success: false, error: "Execution cancelled; actual effects are unknown" }),
    id: "old-approval", run_id: "old-run", tool_call_id: "old-call",
  };
  const completed = { ...approval("succeeded", { success: true }), id: "completed-approval" };
  const pending = { ...approval("pending"), id: "new-approval", run_id: "new-run" };
  const unresolvedExecution = { ...approval("unknown"), id: "unresolved-approval" };
  const records = [cancelledExecution, completed, pending, unresolvedExecution];

  assert.deepEqual(approvalsForPanel(records), [pending, unresolvedExecution]);
  assert.deepEqual(approvalsForPanel([cancelledExecution, completed, unresolvedExecution]), [unresolvedExecution]);
  assert.equal(approvalBlocksInput(unresolvedExecution), true);
  assert.equal(approvalNeedsResume(unresolvedExecution), false);
  assert.equal(approvalForTool(
    records,
    { additional_kwargs: { sandbox_run_id: "old-run" } },
    { id: "old-call", name: "execute_host_command", args: {} },
  ), cancelledExecution);
});

test("all approvals remain independently visible while earlier tools run", () => {
  const a = { ...approval("pending"), id: "A", batch_index: 0 };
  const b = { ...approval("pending"), id: "B", batch_index: 1 };
  const c = { ...approval("pending"), id: "C", batch_index: 2 };
  // API history is newest-first; approval order comes from the model's batch.
  assert.deepEqual(approvalsForPanel([c, b, a]), [a, b, c]);
  a.status = "approved";
  assert.deepEqual(approvalsForPanel([c, b, a]), [a, b, c]);
  a.status = "executing";
  b.status = "approved";
  assert.deepEqual(approvalsForPanel([c, b, a]), [a, b, c]);
  c.status = "denied";
  c.result = { success: false };
  a.status = "succeeded";
  a.result = { success: true };
  b.status = "executing";
  assert.deepEqual(approvalsForPanel([c, b, a]), [b]);
});

test("decision sends only an allow/deny decision, never changed tool parameters", async (t) => {
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/api/threads/thread-1/approvals/approval-1/decision");
    assert.equal(options.method, "POST");
    assert.deepEqual(JSON.parse(options.body), { decision: "allow" });
    return Response.json(approval("approved"));
  });
  assert.equal((await decideApproval("thread-1", "approval-1", "allow")).status, "approved");
});

test("resume uses the saved approval ID and creates no new human input or approved flag", async (t) => {
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/api/threads/thread-1/runs/stream");
    assert.deepEqual(JSON.parse(options.body), {
      assistant_id: "taskharbor", command: { resume: { approval_id: "approval-1" } },
      stream_mode: ["values"], config: { configurable: { chat_model: "test" } },
    });
    return new Response('event: values\ndata: {"messages":[]}\n\n', {
      headers: { "Content-Location": "/threads/thread-1/runs/resumed-run" },
    });
  });
  const seen = [];
  for await (const event of resumeRun("thread-1", "approval-1", { chat_model: "test" }, undefined, (id) => seen.push(id))) {
    assert.equal(event.event, "values");
    assert.deepEqual(seen, ["resumed-run"]);
  }
});

test("refresh and cancellation target only the current thread; API failures stay failures", async (t) => {
  let call = 0;
  t.mock.method(globalThis, "fetch", async (url, options) => {
    if (++call === 1) {
      assert.equal(url, "/api/threads/thread-1/approvals");
      return Response.json({ approvals: [approval("pending")] });
    }
    assert.equal(url, "/api/threads/thread-1/approvals/cancel");
    assert.equal(options.method, "POST");
    return new Response("not confirmed", { status: 409 });
  });
  assert.equal((await listApprovals("thread-1"))[0].status, "pending");
  await assert.rejects(cancelApprovals("thread-1"), /409/);
});

test("approval and resume errors do not display raw server response bodies", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("secret-test-token must not appear", { status: 500 }));
  await assert.rejects(decideApproval("thread-1", "approval-1", "allow"), (error) => {
    assert.match(error.message, /500/);
    assert.doesNotMatch(error.message, /secret-test-token/);
    return true;
  });
  await assert.rejects(async () => {
    for await (const _event of resumeRun("thread-1", "approval-1")) { /* no values on error */ }
  }, (error) => {
    assert.match(error.message, /500/);
    assert.doesNotMatch(error.message, /secret-test-token/);
    return true;
  });
});

test("child approval resumes its execution thread and interrupt without opening a parent stream", async (t) => {
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/api/threads/child-1/runs");
    assert.deepEqual(JSON.parse(options.body), {
      assistant_id: "taskharbor-worker", command: { resume: { "interrupt-1": { approval_id: "approval-1" } } },
      config: { configurable: { chat_model: "selected-model", chat_model_provider: "openai" } },
    });
    return Response.json({ run_id: "child-resume", status: "pending" });
  });
  const resumed = await resumeChildApproval({ ...approval("approved"), execution_thread_id: "child-1",
    interrupt_id: "interrupt-1", assistant_id: "taskharbor-worker" }, { chat_model: "selected-model", chat_model_provider: "openai" });
  assert.equal(resumed.run_id, "child-resume");
});

test("parent resume addresses the specific saved interrupt", async (t) => {
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/api/threads/thread-1/runs/stream");
    assert.deepEqual(JSON.parse(options.body).command, { resume: { "parent-interrupt": { approval_id: "approval-1" } } });
    return new Response("event: values\ndata: {}\n\n");
  });
  for await (const _event of resumeRun("thread-1", "approval-1", undefined, undefined, undefined,
    { interrupt_id: "parent-interrupt", assistant_id: "taskharbor" })) { /* consume */ }
});

test("internal worker threads stay out of the conversation list", async (t) => {
  t.mock.method(globalThis, "fetch", async () => Response.json([
    { thread_id: "parent", metadata: { source: "frontend" } },
    { thread_id: "child", metadata: { sandbox_worker: true, parent_thread_id: "parent" } },
    { thread_id: "legacy", metadata: {} },
  ]));
  assert.deepEqual((await listThreads()).map((thread) => thread.thread_id), ["parent", "legacy"]);
});

test("worker threads do not consume the conversation page limit", async (t) => {
  t.mock.method(globalThis, "fetch", async (_url, options) => {
    const { limit, offset } = JSON.parse(options.body);
    assert.equal(limit, 2);
    return Response.json(offset === 0 ? [
      { thread_id: "child", metadata: { sandbox_worker: true } },
      { thread_id: "parent", metadata: {} },
    ] : [{ thread_id: "older-parent", metadata: {} }]);
  });
  assert.deepEqual((await listThreads(2)).map((thread) => thread.thread_id), ["parent", "older-parent"]);
});

test("a notification-created parent run can be located without an open browser stream", async (t) => {
  t.mock.method(globalThis, "fetch", async (url) => {
    if (url === "/api/threads/parent") return Response.json({ thread_id: "parent", status: "busy", metadata: {} });
    assert.equal(url, "/api/threads/parent/runs?limit=1");
    return Response.json([{ run_id: "notification-resume", status: "running" }]);
  });
  assert.equal((await getThread("parent")).status, "busy");
  assert.equal((await getLatestRun("parent")).run_id, "notification-resume");
});
