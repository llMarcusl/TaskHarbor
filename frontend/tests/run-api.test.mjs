import assert from "node:assert/strict";
import test from "node:test";
import { cancelRun, getRun, streamRun } from "../src/api/aegra.ts";

test("stream exposes its run ID before yielding values, so immediate stop can cancel it", async (t) => {
  const seen = [];
  t.mock.method(globalThis, "fetch", async (_url, options) => {
    assert.equal(JSON.parse(options.body).stream_mode[0], "values");
    return new Response('event: values\ndata: {"messages":[]}\n\n', {
      headers: { "Content-Location": "/threads/t/runs/r1" },
    });
  });
  for await (const event of streamRun("t", [], {}, undefined, (id) => seen.push(id))) {
    assert.deepEqual(seen, ["r1"]);
    assert.equal(event.event, "values");
  }
});

test("cancel targets the owning run and waits for run settlement", async (t) => {
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/api/threads/t/runs/r1/cancel?action=cancel&wait=1");
    assert.equal(options.method, "POST");
    assert.ok(options.signal);
    return Response.json({ run_id: "r1", status: "interrupted" });
  });
  assert.equal((await cancelRun("t", "r1")).status, "interrupted");
});

test("unavailable run status is an error, never fabricated running or interrupted", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("missing", { status: 404 }));
  await assert.rejects(getRun("t", "missing"), /404/);
});
