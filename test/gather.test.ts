import { test } from "node:test";
import assert from "node:assert/strict";
import { compute } from "../src/engine.js";
import { GraphError } from "../src/errors.js";
import type { NodeSpec, TensorValue } from "../src/graph.js";

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------

function expectGraphError(code: string, fn: () => unknown): void {
  let thrown: unknown;
  try {
    fn();
  } catch (e) {
    thrown = e;
  }
  assert.ok(
    thrown instanceof GraphError,
    `期望 GraphError(${code})，实际 ${String(thrown)}`,
  );
  assert.equal((thrown as GraphError).code, code);
}

function flatten(v: TensorValue): number[] {
  if (typeof v === "number") return [v];
  if (v.every((x: unknown) => typeof x === "number")) return v as number[];
  const out: number[] = [];
  for (const row of v as number[][]) out.push(...row);
  return out;
}

function shapeOf(v: TensorValue): number[] {
  if (typeof v === "number") return [];
  if (v.every((x: unknown) => typeof x === "number")) return [v.length];
  return [v.length, (v[0] as number[]).length];
}

function unflatten(flat: number[], shape: number[]): TensorValue {
  if (shape.length === 0) return flat[0] as number;
  if (shape.length === 1) return flat;
  const [m, n] = shape as [number, number];
  const out: number[][] = [];
  for (let i = 0; i < m; i++) out.push(flat.slice(i * n, (i + 1) * n));
  return out;
}

function closeEnough(actual: number, expected: number, tol: number): boolean {
  if (Number.isNaN(actual) || !Number.isFinite(actual)) return false;
  return Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected));
}

/** 中心有限差分（f 为输出全部元素之和，与引擎单位种子梯度一致） */
function checkGradsWithFD(
  nodes: NodeSpec[],
  outputId: string,
  inputIds: string[],
  h = 1e-5,
  tol = 1e-6,
): void {
  const res = compute({ nodes, outputs: [outputId], gradInputs: inputIds });
  for (const id of inputIds) {
    const original = nodes.find((n) => n.id === id)!.value as TensorValue;
    const analytic = flatten(res.grads[id] as TensorValue);
    const flat = flatten(original);
    for (let i = 0; i < flat.length; i++) {
      const evalAt = (delta: number) => {
        const perturbed = flat.map((x, j) => (j === i ? x + delta : x));
        const value = unflatten(perturbed, shapeOf(original));
        const nextNodes = nodes.map((n) =>
          n.id === id ? { ...n, value } : n,
        );
        const out = compute({ nodes: nextNodes, outputs: [outputId] })
          .outputs[0] as TensorValue;
        return flatten(out).reduce((a, b) => a + b, 0);
      };
      const numeric = (evalAt(h) - evalAt(-h)) / (2 * h);
      assert.equal(
        analytic.length,
        flat.length,
        `梯度元素数不一致: ${id} (${analytic.length} vs ${flat.length})`,
      );
      assert.ok(
        closeEnough(analytic[i] as number, numeric, tol),
        `梯度不匹配 ${id}[${i}]: 解析=${analytic[i]} 差分=${numeric}`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// 前向
// ---------------------------------------------------------------------------

test("gather 前向：一维默认轴，按 indices 顺序采样且允许重复", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: [10, 20, 30, 40] },
      { id: "g", op: "gather", inputs: ["x"], indices: [2, 0, 2, 3] },
    ],
    outputs: ["g", "g"],
  });
  assert.deepEqual(res.outputs, [
    [30, 10, 30, 40],
    [30, 10, 30, 40],
  ]);
});

test("gather 前向：二维 axis=0 按行采样，输出形状 [L,n]", () => {
  const res = compute({
    nodes: [
      {
        id: "M",
        op: "const",
        value: [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
      },
      { id: "g", op: "gather", inputs: ["M"], axis: 0, indices: [2, 0, 2] },
    ],
    outputs: ["g"],
  });
  assert.deepEqual(res.outputs[0], [
    [7, 8, 9],
    [1, 2, 3],
    [7, 8, 9],
  ]);
});

test("gather 前向：二维 axis=1 按列采样，输出形状 [m,L]", () => {
  const res = compute({
    nodes: [
      {
        id: "M",
        op: "const",
        value: [
          [1, 2, 3],
          [4, 5, 6],
        ],
      },
      { id: "g", op: "gather", inputs: ["M"], axis: 1, indices: [2, 2, 0] },
    ],
    outputs: ["g"],
  });
  assert.deepEqual(res.outputs[0], [
    [3, 3, 1],
    [6, 6, 4],
  ]);
});

test("gather 前向：负轴按秩归一（-1 即 axis=1，-2 即 axis=0）", () => {
  const nodes: NodeSpec[] = [
    {
      id: "M",
      op: "const",
      value: [
        [1, 2, 3],
        [4, 5, 6],
      ],
    },
    { id: "g1", op: "gather", inputs: ["M"], axis: -1, indices: [0, 2] },
    { id: "g2", op: "gather", inputs: ["M"], axis: -2, indices: [1, 0] },
  ];
  const res = compute({ nodes, outputs: ["g1", "g2"] });
  assert.deepEqual(res.outputs[0], [
    [1, 3],
    [4, 6],
  ]);
  assert.deepEqual(res.outputs[1], [
    [4, 5, 6],
    [1, 2, 3],
  ]);
});

test("gather 前向：可串联（gather 的输出再被 gather）", () => {
  const res = compute({
    nodes: [
      {
        id: "M",
        op: "const",
        value: [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
      },
      { id: "g", op: "gather", inputs: ["M"], axis: 0, indices: [2, 0, 1] },
      { id: "h", op: "gather", inputs: ["g"], axis: 1, indices: [1, 1, 0] },
    ],
    outputs: ["h"],
  });
  assert.deepEqual(res.outputs[0], [
    [8, 8, 7],
    [2, 2, 1],
    [5, 5, 4],
  ]);
});

test("gather 前向：采样结果进入矩阵乘与广播加法（形状/坐标正确）", () => {
  const res = compute({
    nodes: [
      {
        id: "X",
        op: "const",
        value: [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
      },
      {
        id: "W",
        op: "const",
        value: [
          [2, 1, -1],
          [1, 1, 2],
        ],
      },
      { id: "p", op: "gather", inputs: ["X"], axis: 0, indices: [2, 0] },
      { id: "C", op: "matmul", inputs: ["p", "W"] },
      { id: "b", op: "const", value: [10, 20, 30] },
      { id: "Y", op: "add", inputs: ["C", "b"] },
    ],
    outputs: ["p", "C", "Y"],
  });
  assert.deepEqual(res.outputs[0], [
    [5, 6],
    [1, 2],
  ]);
  assert.deepEqual(res.outputs[1], [
    [16, 11, 7],
    [4, 3, 3],
  ]);
  assert.deepEqual(res.outputs[2], [
    [26, 31, 37],
    [14, 23, 33],
  ]);
});

// ---------------------------------------------------------------------------
// 反向：重复索引与共享分支必须累加
// ---------------------------------------------------------------------------

test("gather 反向：一维重复索引的贡献全部加回原位置", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: [10, 20, 30, 40] },
      { id: "g", op: "gather", inputs: ["x"], indices: [2, 0, 2, 3] },
      { id: "s", op: "sum", inputs: ["g"] },
    ],
    outputs: ["s"],
    gradInputs: ["x"],
  });
  // 位置 2 被采两次 => 2；位置 1 未采 => 0
  assert.deepEqual(res.grads.x, [1, 0, 2, 1]);
});

test("gather 反向：二维 axis=0 重复行逐元素累加", () => {
  const res = compute({
    nodes: [
      {
        id: "M",
        op: "const",
        value: [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
      },
      { id: "g", op: "gather", inputs: ["M"], axis: 0, indices: [2, 0, 2] },
      { id: "s", op: "sum", inputs: ["g"] },
    ],
    outputs: ["s"],
    gradInputs: ["M"],
  });
  assert.deepEqual(res.grads.M, [
    [1, 1, 1],
    [0, 0, 0],
    [2, 2, 2],
  ]);
});

test("gather 反向：二维 axis=1（及负轴）重复列逐元素累加", () => {
  const build = (axis: number) =>
    compute({
      nodes: [
        {
          id: "M",
          op: "const",
          value: [
            [1, 2, 3],
            [4, 5, 6],
          ],
        },
        { id: "g", op: "gather", inputs: ["M"], axis, indices: [2, 2, 0] },
        { id: "s", op: "sum", inputs: ["g"] },
      ],
      outputs: ["s"],
      gradInputs: ["M"],
    });
  const expected = [
    [1, 0, 2],
    [1, 0, 2],
  ];
  assert.deepEqual(build(1).grads.M, expected);
  assert.deepEqual(build(-1).grads.M, expected);
});

test("gather 反向：共享 gather 节点的两条下游路径都累加（精确值）", () => {
  const nodes: NodeSpec[] = [
    { id: "x", op: "const", value: [1, 2, 3] },
    { id: "g", op: "gather", inputs: ["x"], indices: [1, 1, 2] },
    { id: "a", op: "mul", inputs: ["g", "g"] },
    { id: "f", op: "sum", inputs: ["a"] },
  ];
  const res = compute({ nodes, outputs: ["f"], gradInputs: ["x"] });
  assert.equal(res.outputs[0], 17);
  assert.deepEqual(res.grads.x, [0, 8, 6]);
  checkGradsWithFD(nodes, "f", ["x"]);
});

test("gather 反向：同一输入被两个 gather 节点共享，贡献跨节点累加", () => {
  // g1 取 [x2, x0]，g2 取 [x2, x2]；f = sum(g1) + sum(g2*g2)
  // dx2 直接 1 次 + 作为平方 4*x2=12 => 13；dx0 = 1；dx1 = 0
  const nodes: NodeSpec[] = [
    { id: "x", op: "const", value: [1, 2, 3] },
    { id: "g1", op: "gather", inputs: ["x"], indices: [2, 0] },
    { id: "g2", op: "gather", inputs: ["x"], indices: [2, 2] },
    { id: "q", op: "mul", inputs: ["g2", "g2"] },
    { id: "s1", op: "sum", inputs: ["g1"] },
    { id: "s2", op: "sum", inputs: ["q"] },
    { id: "f", op: "add", inputs: ["s1", "s2"] },
  ];
  const res = compute({ nodes, outputs: ["f"], gradInputs: ["x"] });
  assert.equal(res.outputs[0], 4 + 18);
  assert.deepEqual(res.grads.x, [1, 0, 13]);
  checkGradsWithFD(nodes, "f", ["x"]);
});

test("gather 反向：axis=0 采样行进入 matmul + 广播偏置 + ReLU（有限差分）", () => {
  const nodes: NodeSpec[] = [
    {
      id: "X",
      op: "const",
      value: [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.1],
      ],
    },
    {
      id: "W",
      op: "const",
      value: [
        [0.2, 0.3, -0.1],
        [0.1, -0.2, 0.4],
      ],
    },
    { id: "p", op: "gather", inputs: ["X"], axis: 0, indices: [2, 0, 2] },
    { id: "H", op: "matmul", inputs: ["p", "W"] },
    { id: "b", op: "const", value: [1, 2, 3] },
    { id: "Hb", op: "add", inputs: ["H", "b"] },
    { id: "R", op: "relu", inputs: ["Hb"] },
    { id: "f", op: "sum", inputs: ["R"] },
  ];
  checkGradsWithFD(nodes, "f", ["X", "W", "b"], 1e-5, 1e-5);
});

test("gather 反向：axis=1 采样列后做矩阵乘（有限差分）", () => {
  const nodes: NodeSpec[] = [
    {
      id: "M",
      op: "const",
      value: [
        [0.3, 0.5, 0.2],
        [0.4, 0.1, 0.6],
      ],
    }, // 2x3
    { id: "G", op: "gather", inputs: ["M"], axis: 1, indices: [2, 0, 2, 1] }, // 2x4
    {
      id: "B",
      op: "const",
      value: [
        [0.2, 0.3],
        [0.4, -0.2],
        [0.1, 0.5],
        [-0.3, 0.2],
      ],
    }, // 4x2
    { id: "C", op: "matmul", inputs: ["G", "B"] },
    { id: "f", op: "sum", inputs: ["C"] },
  ];
  checkGradsWithFD(nodes, "f", ["M", "B"], 1e-5, 1e-5);
});

test("gather 反向：一维采样结果参与广播逐元素运算（有限差分）", () => {
  const nodes: NodeSpec[] = [
    { id: "x", op: "const", value: [0.3, -0.5, 0.8, 0.2] },
    { id: "g", op: "gather", inputs: ["x"], indices: [3, 0, 3, 2] },
    {
      id: "W",
      op: "const",
      value: [
        [0.2, 0.4, -0.1, 0.3],
        [0.5, -0.2, 0.3, 0.1],
      ],
    },
    { id: "Y", op: "mul", inputs: ["W", "g"] }, // [2,4] 广播 [4]
    { id: "f", op: "sum", inputs: ["Y"] },
  ];
  checkGradsWithFD(nodes, "f", ["x", "W"], 1e-5, 1e-5);
});

test("gather 反向：串联 gather 的有限差分核对", () => {
  const nodes: NodeSpec[] = [
    {
      id: "M",
      op: "const",
      value: [
        [0.2, 0.5, 0.3],
        [0.4, 0.1, 0.6],
        [0.7, 0.2, 0.1],
      ],
    },
    { id: "g", op: "gather", inputs: ["M"], axis: 0, indices: [2, 0, 1] },
    { id: "h", op: "gather", inputs: ["g"], axis: 1, indices: [1, 1, 0, 2] },
    { id: "f", op: "sum", inputs: ["h"] },
  ];
  checkGradsWithFD(nodes, "f", ["M"], 1e-5, 1e-5);
});

test("gather 反向：与输出不连通的常量返回全 0", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: [1, 2, 3] },
      { id: "z", op: "const", value: [9, 9] },
      { id: "g", op: "gather", inputs: ["x"], indices: [0, 2] },
      { id: "s", op: "sum", inputs: ["g"] },
    ],
    outputs: ["s"],
    gradInputs: ["x", "z"],
  });
  assert.deepEqual(res.grads.x, [1, 0, 1]);
  assert.deepEqual(res.grads.z, [0, 0]);
});

// ---------------------------------------------------------------------------
// 校验失败：执行前拒绝，不返回部分结果
// ---------------------------------------------------------------------------

test("gather 拒绝：索引越界（含负轴归一后的轴）", () => {
  const M = [
    [1, 2],
    [3, 4],
  ];
  const req = (extra: Record<string, unknown>): NodeSpec[] => [
    { id: "M", op: "const", value: M },
    { id: "g", op: "gather", inputs: ["M"], ...extra } as NodeSpec,
  ];
  expectGraphError("INVALID_REQUEST", () =>
    compute({ nodes: req({ indices: [0, 2] }), outputs: ["g"] }),
  );
  expectGraphError("INVALID_REQUEST", () =>
    compute({ nodes: req({ axis: 1, indices: [0, 2] }), outputs: ["g"] }),
  );
  expectGraphError("INVALID_REQUEST", () =>
    compute({ nodes: req({ axis: -1, indices: [2] }), outputs: ["g"] }),
  );
  // 一维越界
  expectGraphError("INVALID_REQUEST", () =>
    compute({
      nodes: [
        { id: "v", op: "const", value: [1, 2, 3] },
        { id: "g", op: "gather", inputs: ["v"], indices: [3] },
      ],
      outputs: ["g"],
    }),
  );
});

test("gather 拒绝：非法索引类型（负数 / 小数 / NaN / 字符串 / null）", () => {
  const base = (indices: unknown): Parameters<typeof compute>[0] => ({
    nodes: [
      { id: "v", op: "const", value: [1, 2, 3] },
      { id: "g", op: "gather", inputs: ["v"], indices } as NodeSpec,
    ],
    outputs: ["g"],
  });
  expectGraphError("INVALID_REQUEST", () => compute(base([0, -1])));
  expectGraphError("INVALID_REQUEST", () => compute(base([0, 1.5])));
  expectGraphError("INVALID_REQUEST", () => compute(base([NaN])));
  expectGraphError("INVALID_REQUEST", () => compute(base(["0"])));
  expectGraphError("INVALID_REQUEST", () => compute(base([null])));
  expectGraphError("INVALID_REQUEST", () => compute(base(0)));
  expectGraphError("INVALID_REQUEST", () => compute(base(undefined)));
  expectGraphError("INVALID_REQUEST", () => compute(base([])));
});

test("gather 拒绝：indices 超过 128 项（128 项恰好合法）", () => {
  const big = Array.from({ length: 129 }, (_, i) => i % 4);
  expectGraphError("INVALID_REQUEST", () =>
    compute({
      nodes: [
        { id: "v", op: "const", value: [1, 2, 3, 4] },
        { id: "g", op: "gather", inputs: ["v"], indices: big },
      ],
      outputs: ["g"],
    }),
  );
  const ok = compute({
    nodes: [
      { id: "v", op: "const", value: [1, 2, 3, 4] },
      {
        id: "g",
        op: "gather",
        inputs: ["v"],
        indices: Array.from({ length: 128 }, (_, i) => i % 4),
      },
    ],
    outputs: ["g"],
  });
  assert.equal((ok.outputs[0] as number[]).length, 128);
});

test("gather 拒绝：非法 axis（越界 / 小数 / 字符串），含负轴", () => {
  const req = (axis: unknown): Parameters<typeof compute>[0] => ({
    nodes: [
      {
        id: "M",
        op: "const",
        value: [
          [1, 2],
          [3, 4],
        ],
      },
      { id: "g", op: "gather", inputs: ["M"], axis, indices: [0] } as NodeSpec,
    ],
    outputs: ["g"],
  });
  expectGraphError("INVALID_REQUEST", () => compute(req(2)));
  expectGraphError("INVALID_REQUEST", () => compute(req(-3)));
  expectGraphError("INVALID_REQUEST", () => compute(req(1.5)));
  expectGraphError("INVALID_REQUEST", () => compute(req("0")));
});

test("gather 拒绝：标量输入（秩必须为 1 或 2）", () => {
  expectGraphError("INVALID_REQUEST", () =>
    compute({
      nodes: [
        { id: "x", op: "const", value: 5 },
        { id: "g", op: "gather", inputs: ["x"], indices: [0] },
      ],
      outputs: ["g"],
    }),
  );
});

test("gather 形状错误继续向下游传播：输出与 matmul 不兼容返回 MATRIX_SHAPE", () => {
  expectGraphError("MATRIX_SHAPE", () =>
    compute({
      nodes: [
        {
          id: "M",
          op: "const",
          value: [
            [1, 2, 3],
            [4, 5, 6],
          ],
        }, // 2x3
        { id: "g", op: "gather", inputs: ["M"], axis: 0, indices: [0, 1] }, // 2x3
        {
          id: "B",
          op: "const",
          value: [
            [1, 2],
            [3, 4],
          ],
        }, // 2x2：内维 3 != 2
        { id: "C", op: "matmul", inputs: ["g", "B"] },
      ],
      outputs: ["C"],
    }),
  );
});

test("gather 参与元素预算：超限在执行前失败（无部分结果）", () => {
  expectGraphError("ELEMENT_LIMIT", () =>
    compute({
      nodes: [
        {
          id: "M",
          op: "const",
          value: [
            [1, 2, 3],
            [4, 5, 6],
          ],
        }, // 6
        { id: "g", op: "gather", inputs: ["M"], axis: 0, indices: [0, 1] }, // 6
      ],
      outputs: ["g"],
      maxElements: 11, // 6 + 6 = 12 > 11
    }),
  );
  // 放宽到 12 即通过
  const ok = compute({
    nodes: [
      {
        id: "M",
        op: "const",
        value: [
          [1, 2, 3],
          [4, 5, 6],
        ],
      },
      { id: "g", op: "gather", inputs: ["M"], axis: 0, indices: [0, 1] },
    ],
    outputs: ["g"],
    maxElements: 12,
  });
  assert.equal((ok.outputs[0] as number[][]).length, 2);
});

test("gather 非法请求不返回 outputs / grads", () => {
  let thrown: unknown;
  try {
    compute({
      nodes: [
        { id: "v", op: "const", value: [1, 2] },
        { id: "g", op: "gather", inputs: ["v"], indices: [5] },
      ],
      outputs: ["g"],
      gradInputs: ["v"],
    });
  } catch (e) {
    thrown = e;
  }
  assert.ok(thrown instanceof GraphError);
  assert.equal((thrown as GraphError).code, "INVALID_REQUEST");
});

// ---------------------------------------------------------------------------
// 证据一致：接口输出与输入更新后的重新求值
// ---------------------------------------------------------------------------

test("gather：修改输入常量后重新求值得到更新后的结果（无陈旧缓存）", () => {
  const build = (vals: number[]): Parameters<typeof compute>[0] => ({
    nodes: [
      { id: "x", op: "const", value: vals },
      { id: "g", op: "gather", inputs: ["x"], indices: [2, 0, 2] },
      { id: "s", op: "sum", inputs: ["g"] },
    ],
    outputs: ["g", "s"],
    gradInputs: ["x"],
  });
  // 两个输出 g 与 s 均以全 1 为种子梯度，故 x 的梯度为两份贡献之和：[2,0,4]
  const r1 = compute(build([10, 20, 30]));
  assert.deepEqual(r1.outputs, [[30, 10, 30], 70]);
  assert.deepEqual(r1.grads.x, [2, 0, 4]);

  const r2 = compute(build([11, 21, 31]));
  assert.deepEqual(r2.outputs, [[31, 11, 31], 73]);
  assert.deepEqual(r2.grads.x, [2, 0, 4]);

  // 连续两次求值彼此独立、结果一致
  assert.deepEqual(compute(build([11, 21, 31])).outputs, r2.outputs);
});

test("gather：共享分支的前向证据与重新计算一致（菱形 DAG）", () => {
  const nodes: NodeSpec[] = [
    {
      id: "M",
      op: "const",
      value: [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
    },
    { id: "g", op: "gather", inputs: ["M"], axis: 0, indices: [2, 0] }, // 2x2
    {
      id: "B",
      op: "const",
      value: [
        [2, 1],
        [1, 3],
      ],
    },
    { id: "c1", op: "matmul", inputs: ["g", "B"] },
    { id: "c2", op: "matmul", inputs: ["B", "g"] }, // 2x2
    { id: "s1", op: "sum", inputs: ["c1"] },
    { id: "s2", op: "sum", inputs: ["c2"] },
    { id: "f", op: "add", inputs: ["s1", "s2"] },
  ];
  const r1 = compute({ nodes, outputs: ["g", "c1", "c2", "f"], gradInputs: ["M", "B"] });
  const r2 = compute({ nodes, outputs: ["g", "c1", "c2", "f"], gradInputs: ["M", "B"] });
  assert.deepEqual(r1.outputs, r2.outputs);
  assert.deepEqual(r1.grads, r2.grads);
  // 手算：g=[[5,6],[1,2]]，B=[[2,1],[1,3]]
  // c1 = [[16,23],[4,7]] sum=50；c2 = B@g = [[11,14],[8,12]] sum=45；f=95
  assert.deepEqual(r1.outputs[1], [
    [16, 23],
    [4, 7],
  ]);
  assert.deepEqual(r1.outputs[2], [
    [11, 14],
    [8, 12],
  ]);
  assert.equal(r1.outputs[3], 95);
  checkGradsWithFD(nodes, "f", ["M", "B"], 1e-5, 1e-5);
});
