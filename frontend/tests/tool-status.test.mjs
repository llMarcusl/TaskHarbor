import assert from "node:assert/strict";
import test from "node:test";
import { indexToolResults, missingToolStatus, unresolvedRunIds } from "../src/lib/tool-status.ts";

const ai = (run, id = "call") => ({
  type: "ai", content: "", tool_calls: [{ id, name: "create_session", args: {} }],
  additional_kwargs: run ? { sandbox_run_id: run } : {},
});

test("old missing result never becomes running when a new run starts", () => {
  const states = { old: "interrupted", next: "running" };
  assert.equal(missingToolStatus(ai("old"), states), "interrupted");
  assert.equal(missingToolStatus(ai("next"), states), "running");
  assert.equal(missingToolStatus(ai(), states), "unknown");
});

test("reload, terminal states, pending cancellation and query failures", () => {
  for (const status of ["success", "error", "interrupted", "timeout"]) {
    assert.equal(missingToolStatus(ai("run"), { run: status }), "interrupted");
  }
  assert.equal(missingToolStatus(ai("run"), {}), "unknown");
  assert.equal(missingToolStatus(ai("run"), { run: "unknown" }), "unknown");
  assert.equal(missingToolStatus(ai("run"), { run: "stopping" }), "stopping");
  assert.equal(missingToolStatus(ai("run"), { run: "pending" }), "running");
});

test("partial and reused call IDs match only the issuing assistant turn", () => {
  const first = ai("old");
  const second = ai("new");
  second.tool_calls.push({ id: "pending", name: "execute_code", args: {} });
  const real = { type: "tool", content: "real result", tool_call_id: "call" };
  const messages = [first, { type: "human", content: "continue" }, second, real];
  const results = indexToolResults(messages);
  assert.equal(results.get(0).get("call"), undefined);
  assert.equal(results.get(2).get("call"), real);
  assert.equal(results.get(2).get("pending"), undefined);
  assert.deepEqual(unresolvedRunIds(messages), ["new", "old"]);
});
