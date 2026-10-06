import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { type Server } from "node:http";
import { startServer } from "../src/server.js";

let server: Server;
let base: string;

before(async () => {
  server = await new Promise<Server>((resolve) => {
    const s = startServer(0);
    s.once("listening", () => resolve(s));
  });
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("无端口");
  base = `http://127.0.0.1:${addr.port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

async function post(body: unknown): Promise<{ status: number; json: any }> {
  const res = await fetch(`${base}/compute`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  return { status: res.status, json };
}

test("POST /compute 返回输出与梯度", async () => {
  const { status, json } = await post({
    nodes: [
      {
        id: "x",
        op: "const",
        value: [
          [1, 2],
          [3, 4],
        ],
      },
      { id: "y", op: "const", value: [[1], [1]] },
      { id: "z", op: "add", inputs: ["x", "y"] },
      { id: "s", op: "sum", inputs: ["z"] },
    ],
    outputs: ["s"],
    gradInputs: ["x", "y"],
  });
  assert.equal(status, 200);
  assert.equal(json.outputs[0], 14);
  assert.deepEqual(json.grads.x, [
    [1, 1],
    [1, 1],
  ]);
  assert.deepEqual(json.grads.y, [[2], [2]]);
});

test("POST /compute 非法形状返回 422 且无部分结果", async () => {
  const { status, json } = await post({
    nodes: [
      { id: "a", op: "const", value: [1, 2] },
      { id: "b", op: "const", value: [1, 2, 3] },
      { id: "c", op: "add", inputs: ["a", "b"] },
    ],
    outputs: ["c"],
    gradInputs: ["a"],
  });
  assert.equal(status, 422);
  assert.equal(json.error.code, "BROADCAST_INCOMPATIBLE");
  assert.equal(json.outputs, undefined);
  assert.equal(json.grads, undefined);
});

test("POST /compute 拒绝非法常量 (400)；JSON 中 NaN 会被序列化为 null", async () => {
  // JSON 规范无 NaN：JSON.stringify(NaN) === 'null'，服务端按坏常量拒绝
  const res1 = await fetch(`${base}/compute`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      nodes: [{ id: "a", op: "const", value: NaN }],
      outputs: ["a"],
    }),
  });
  const body1 = await res1.json();
  assert.equal(res1.status, 400);
  assert.equal(body1.error.code, "BAD_CONSTANT");

  // 运行时溢出（前向产生 Infinity）走 422
  const { status, json } = await post({
    nodes: [
      { id: "x", op: "const", value: 1e308 },
      { id: "y", op: "const", value: 1e308 },
      { id: "z", op: "mul", inputs: ["x", "y"] },
    ],
    outputs: ["z"],
  });
  assert.equal(status, 422);
  assert.equal(json.error.code, "NON_FINITE_VALUE");
});

test("GET /health", async () => {
  const res = await fetch(`${base}/health`);
  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { ok: true });
});

test("非法 JSON 返回 400", async () => {
  const res = await fetch(`${base}/compute`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{not json",
  });
  assert.equal(res.status, 400);
});
