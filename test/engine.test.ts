import { test } from "node:test";
import assert from "node:assert/strict";
import { compute } from "../src/engine.js";
import { GraphError } from "../src/errors.js";
import { MAX_NODES } from "../src/graph.js";
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

/** 嵌套张量扁平化为一维数组（行主序），用于与有限差分逐一比较 */
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

function closeEnough(actual: number, expected: number, tol: number): boolean {
  if (Number.isNaN(actual) || !Number.isFinite(actual)) return false;
  return Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected));
}

/**
 * 中心有限差分梯度：
 * grad_i = (f(x + h e_i) - f(x - h e_i)) / (2h)
 * f 为“输出全部元素之和”，与引擎的单位种子梯度定义一致。
 * h 取 1e-5；测试构造均避开 ReLU 零点等不可导点。
 */
function finiteDiff(
  nodes: NodeSpec[],
  outputId: string,
  inputId: string,
  original: TensorValue,
  h = 1e-5,
): number[] {
  const flat = flatten(original);
  const grad: number[] = new Array(flat.length);
  for (let i = 0; i < flat.length; i++) {
    const makeReq = (delta: number) => {
      const perturbed = flat.map((x, j) => (j === i ? x + delta : x));
      const value = unflatten(perturbed, shapeOf(original));
      const nextNodes = nodes.map((n) =>
        n.id === inputId ? { ...n, value } : n,
      );
      return compute({
        nodes: nextNodes,
        outputs: [outputId],
      }).outputs[0] as TensorValue;
    };
    const plus = scalarSum(makeReq(h));
    const minus = scalarSum(makeReq(-h));
    grad[i] = (plus - minus) / (2 * h);
  }
  return grad;
}

function scalarSum(v: TensorValue): number {
  return flatten(v).reduce((a, b) => a + b, 0);
}

function unflatten(flat: number[], shape: number[]): TensorValue {
  if (shape.length === 0) return flat[0];
  if (shape.length === 1) return flat;
  const [m, n] = shape;
  const out: number[][] = [];
  for (let i = 0; i < m; i++) out.push(flat.slice(i * n, (i + 1) * n));
  return out;
}

/** 用有限差分核对所有请求输入的梯度 */
function checkGradsWithFD(
  nodes: NodeSpec[],
  outputId: string,
  inputIds: string[],
  fdTol = 1e-6,
): void {
  const res = compute({ nodes, outputs: [outputId], gradInputs: inputIds });
  for (const id of inputIds) {
    const original = nodes.find((n) => n.id === id)!.value as TensorValue;
    const analytic = flatten(res.grads[id] as TensorValue);
    const numeric = finiteDiff(nodes, outputId, id, original);
    assert.equal(
      analytic.length,
      numeric.length,
      `梯度元素数不一致: ${id} (${analytic.length} vs ${numeric.length})`,
    );
    for (let i = 0; i < analytic.length; i++) {
      assert.ok(
        closeEnough(analytic[i], numeric[i], fdTol),
        `梯度不匹配 ${id}[${i}]: 解析=${analytic[i]} 差分=${numeric[i]}`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// 前向计算
// ---------------------------------------------------------------------------

test("前向：常量与标量算术", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: 3 },
      { id: "y", op: "const", value: 4 },
      { id: "z", op: "add", inputs: ["x", "y"] },
      { id: "w", op: "mul", inputs: ["z", "x"] }, // (3+4)*3 = 21
    ],
    outputs: ["w"],
  });
  assert.deepEqual(res.outputs, [21]);
  assert.deepEqual(res.grads, {});
});

test("前向：二维乘一维广播，逐元素坐标必须正确（行主序步距）", () => {
  // M[i,j] * v[j]：每一行乘同一个 v
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
      { id: "v", op: "const", value: [10, 20, 30] },
      { id: "Y", op: "mul", inputs: ["M", "v"] },
    ],
    outputs: ["Y"],
  });
  assert.deepEqual(res.outputs[0], [
    [10, 40, 90],
    [40, 100, 180],
  ]);
});

test("前向：二维加一维，再被二维逐元素乘（复合广播坐标）", () => {
  const res = compute({
    nodes: [
      {
        id: "A",
        op: "const",
        value: [
          [1, 2],
          [3, 4],
        ],
      },
      { id: "b", op: "const", value: [100, 200] },
      { id: "C", op: "add", inputs: ["A", "b"] },
      {
        id: "D",
        op: "const",
        value: [
          [2, 3],
          [4, 5],
        ],
      },
      { id: "E", op: "mul", inputs: ["C", "D"] },
    ],
    outputs: ["E"],
  });
  assert.deepEqual(res.outputs[0], [
    [202, 606],
    [412, 1020],
  ]);
});

test("前向：一维逐元素与广播", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: [1, 2, 3] },
      { id: "s", op: "const", value: 10 },
      { id: "b", op: "const", value: [1, 1, 1] },
      { id: "y", op: "mul", inputs: ["x", "s"] },
      { id: "z", op: "add", inputs: ["y", "b"] },
    ],
    outputs: ["z"],
  });
  assert.deepEqual(res.outputs, [[11, 21, 31]]);
});

test("前向：二维矩阵乘 + 全量求和 + ReLU", () => {
  const res = compute({
    nodes: [
      {
        id: "A",
        op: "const",
        value: [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
      }, // 3x2
      {
        id: "B",
        op: "const",
        value: [
          [1, 0, -1],
          [0, 1, 1],
        ],
      }, // 2x3
      { id: "C", op: "matmul", inputs: ["A", "B"] }, // 3x3
      { id: "R", op: "relu", inputs: ["C"] },
      { id: "s", op: "sum", inputs: ["R"] },
    ],
    outputs: ["C", "R", "s"],
  });
  assert.deepEqual(res.outputs[0], [
    [1, 2, 1],
    [3, 4, 1],
    [5, 6, 1],
  ]);
  assert.deepEqual(res.outputs[1], [
    [1, 2, 1],
    [3, 4, 1],
    [5, 6, 1],
  ]);
  assert.equal(res.outputs[2], 24);
});

test("前向：ReLU 在零点输出 0、负值归零", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: [-2, 0, 3] },
      { id: "r", op: "relu", inputs: ["x"] },
    ],
    outputs: ["r"],
  });
  assert.deepEqual(res.outputs, [[0, 0, 3]]);
});

test("前向：多输出含重复 id，按出现顺序返回", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: 5 },
      { id: "y", op: "const", value: 7 },
      { id: "z", op: "add", inputs: ["x", "y"] },
    ],
    outputs: ["z", "x", "z"],
  });
  assert.deepEqual(res.outputs, [12, 5, 12]);
});

// ---------------------------------------------------------------------------
// 梯度：基础算子精确值
// ---------------------------------------------------------------------------

test("梯度：标量 add/mul 链式法则", () => {
  // f = (x+y) * x, x=3 y=4
  // df/dx = y + 2x = 10, df/dy = x = 3
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: 3 },
      { id: "y", op: "const", value: 4 },
      { id: "a", op: "add", inputs: ["x", "y"] },
      { id: "f", op: "mul", inputs: ["a", "x"] },
    ],
    outputs: ["f"],
    gradInputs: ["x", "y"],
  });
  assert.equal(res.grads.x, 10);
  assert.equal(res.grads.y, 3);
});

test("梯度：sum 广播回每个元素", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: [1, 2, 3] },
      { id: "s", op: "sum", inputs: ["x"] },
    ],
    outputs: ["s"],
    gradInputs: ["x"],
  });
  assert.deepEqual(res.grads.x, [1, 1, 1]);
});

test("梯度：ReLU 严格大于零才传导，零点导数为零", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: [-1.5, 0, 2.5] },
      { id: "r", op: "relu", inputs: ["x"] },
      { id: "s", op: "sum", inputs: ["r"] },
    ],
    outputs: ["s"],
    gradInputs: ["x"],
  });
  assert.deepEqual(res.grads.x, [0, 0, 1]);
});

test("梯度：与输出不连通的常量输入返回全零梯度", () => {
  const res = compute({
    nodes: [
      { id: "x", op: "const", value: 3 },
      { id: "z", op: "const", value: 9 },
      { id: "y", op: "add", inputs: ["x", "x"] },
    ],
    outputs: ["y"],
    gradInputs: ["x", "z"],
  });
  assert.equal(res.grads.x, 2);
  assert.equal(res.grads.z, 0);
});

// ---------------------------------------------------------------------------
// 梯度：共享子图（多路径累加）—— 有限差分
// ---------------------------------------------------------------------------

test("梯度：共享子图（菱形）所有路径累加", () => {
  //      x
  //    /   \
  //  x*x    x+x
  //    \   /
  //     add -> sum
  // f = x^2 + 2x, f'(3) = 8
  const nodes: NodeSpec[] = [
    { id: "x", op: "const", value: 3 },
    { id: "p", op: "mul", inputs: ["x", "x"] },
    { id: "d", op: "add", inputs: ["x", "x"] },
    { id: "f", op: "add", inputs: ["p", "d"] },
  ];
  const res = compute({ nodes, outputs: ["f"], gradInputs: ["x"] });
  assert.equal(res.grads.x, 8);
  checkGradsWithFD(nodes, "f", ["x"]);
});

test("梯度：共享二维子节点在两个矩阵乘中被使用", () => {
  // C1 = A @ B, C2 = D @ A，输出 sum(C1) + sum(C2)
  const nodes: NodeSpec[] = [
    {
      id: "A",
      op: "const",
      value: [
        [1, 2],
        [3, -1],
      ],
    },
    {
      id: "B",
      op: "const",
      value: [
        [2, 0],
        [-1, 1],
      ],
    },
    {
      id: "D",
      op: "const",
      value: [
        [1, 1],
        [0, 2],
      ],
    },
    { id: "C1", op: "matmul", inputs: ["A", "B"] },
    { id: "C2", op: "matmul", inputs: ["D", "A"] },
    { id: "s1", op: "sum", inputs: ["C1"] },
    { id: "s2", op: "sum", inputs: ["C2"] },
    { id: "f", op: "add", inputs: ["s1", "s2"] },
  ];
  checkGradsWithFD(nodes, "f", ["A", "B", "D"], 1e-6);
});

// ---------------------------------------------------------------------------
// 梯度：矩阵乘 —— 有限差分
// ---------------------------------------------------------------------------

test("梯度：matmul 两侧梯度的有限差分核对", () => {
  const nodes: NodeSpec[] = [
    {
      id: "A",
      op: "const",
      value: [
        [1, 2, 3],
        [4, 5, 6],
      ],
    }, // 2x3
    {
      id: "B",
      op: "const",
      value: [
        [1, -1],
        [2, 0],
        [-1, 1],
      ],
    }, // 3x2
    { id: "C", op: "matmul", inputs: ["A", "B"] },
    { id: "f", op: "sum", inputs: ["C"] },
  ];
  checkGradsWithFD(nodes, "f", ["A", "B"]);

  // 解析值核对：sum(A@B) 对 A 的梯度 = 1^T(按行复制) * B 列和
  // B = [[1,-1],[2,0],[-1,1]]，各 k 行元素之和为 [0,2,0]
  const res = compute({ nodes, outputs: ["f"], gradInputs: ["A", "B"] });
  assert.deepEqual(res.grads.A, [
    [0, 2, 0],
    [0, 2, 0],
  ]);
  // dB = A^T @ 1 = 各列元素之和 [5,7,9] 按列复制
  assert.deepEqual(res.grads.B, [
    [5, 5],
    [7, 7],
    [9, 9],
  ]);
});

test("梯度：matmul 与逐元素、ReLU 混合", () => {
  // 选择全正的初值，避免有限分差异到 ReLU 零点
  const nodes: NodeSpec[] = [
    {
      id: "X",
      op: "const",
      value: [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.1],
      ],
    }, // 3x2
    {
      id: "W",
      op: "const",
      value: [
        [0.2, 0.3, -0.1],
        [0.1, -0.2, 0.4],
      ],
    }, // 2x3
    { id: "b", op: "const", value: [1, 2, 3] },
    { id: "H", op: "matmul", inputs: ["X", "W"] }, // 3x3
    { id: "Hb", op: "add", inputs: ["H", "b"] },
    { id: "R", op: "relu", inputs: ["Hb"] },
    { id: "f", op: "sum", inputs: ["R"] },
  ];
  checkGradsWithFD(nodes, "f", ["X", "W", "b"], 1e-5);
});

// ---------------------------------------------------------------------------
// 梯度：两轴广播 —— 有限差分
// ---------------------------------------------------------------------------

test("梯度：二维 [m,n] 与一维 [n] 广播（沿轴 0 降维）", () => {
  const nodes: NodeSpec[] = [
    {
      id: "A",
      op: "const",
      value: [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
    }, // 3x2
    { id: "b", op: "const", value: [10, 20] }, // 沿行广播
    { id: "Y", op: "add", inputs: ["A", "b"] },
    { id: "f", op: "sum", inputs: ["Y"] },
  ];
  const res = compute({ nodes, outputs: ["f"], gradInputs: ["A", "b"] });
  assert.deepEqual(res.grads.A, [
    [1, 1],
    [1, 1],
    [1, 1],
  ]);
  assert.deepEqual(res.grads.b, [3, 3]); // 扩展维度求和
  checkGradsWithFD(nodes, "f", ["A", "b"]);
});

test("梯度：二维 [m,n] 与标量 mul（两轴都求和降维）", () => {
  const nodes: NodeSpec[] = [
    {
      id: "A",
      op: "const",
      value: [
        [1, 2],
        [3, -1],
      ],
    },
    { id: "s", op: "const", value: 2.5 },
    { id: "Y", op: "mul", inputs: ["A", "s"] },
    { id: "f", op: "sum", inputs: ["Y"] },
  ];
  const res = compute({ nodes, outputs: ["f"], gradInputs: ["A", "s"] });
  assert.deepEqual(res.grads.A, [
    [2.5, 2.5],
    [2.5, 2.5],
  ]);
  assert.equal(res.grads.s, 5); // 1+2+3-1
  checkGradsWithFD(nodes, "f", ["A", "s"]);
});

test("梯度：一维 [m] 与二维 [m,n] 通过尾部广播（新增轴）", () => {
  // [m] 对齐到 [m,n] 时按尾部维度解释为 [1,m]？不——尾部对齐：[m] -> [1? ]
  // 实际上 [m] 与 [m,n] 尾部对齐为 [m] 对 [m,n] 的后 1 维 => [1? ] 不匹配
  // 合法组合应是 [n] 与 [m,n]。这里用 [n] 与 [m,n]，另加一个 [m,1] x [1,n]。
  const nodes: NodeSpec[] = [
    { id: "M", op: "const", value: [[1], [2], [3]] }, // [3,1]
    { id: "V", op: "const", value: [4, 5] }, // [2] 尾部对齐 -> [1,2]
    { id: "Y", op: "mul", inputs: ["M", "V"] }, // [3,2]
    { id: "f", op: "sum", inputs: ["Y"] },
  ];
  const res = compute({ nodes, outputs: ["f"], gradInputs: ["M", "V"] });
  // d/dM: V 沿轴1求和 => [4+5] 每行
  assert.deepEqual(res.grads.M, [[9], [9], [9]]);
  // d/dV: M 沿轴0求和 => [1+2+3] 每列
  assert.deepEqual(res.grads.V, [6, 6]);
  checkGradsWithFD(nodes, "f", ["M", "V"]);
});

test("梯度：广播 mul 两侧都被扩展（[m,1] x [1,n]）", () => {
  const nodes: NodeSpec[] = [
    { id: "U", op: "const", value: [[1], [-2], [3]] }, // [3,1]
    { id: "W", op: "const", value: [[2, -3]] }, // [1,2]
    { id: "Y", op: "mul", inputs: ["U", "W"] }, // [3,2]
    { id: "R", op: "relu", inputs: ["Y"] },
    { id: "f", op: "sum", inputs: ["R"] },
  ];
  checkGradsWithFD(nodes, "f", ["U", "W"], 1e-5);
});

// ---------------------------------------------------------------------------
// 校验失败：非法维度、NaN/无穷、环、预算、节点上限
// ---------------------------------------------------------------------------

test("拒绝 NaN / 无穷常量", () => {
  expectGraphError("NON_FINITE_VALUE", () =>
    compute({ nodes: [{ id: "x", op: "const", value: NaN }], outputs: ["x"] }),
  );
  expectGraphError("NON_FINITE_VALUE", () =>
    compute({
      nodes: [{ id: "x", op: "const", value: [1, Infinity] }],
      outputs: ["x"],
    }),
  );
  expectGraphError("NON_FINITE_VALUE", () =>
    compute({
      nodes: [{ id: "x", op: "const", value: [[1], [NaN]] }],
      outputs: ["x"],
    }),
  );
});

test("拒绝三维常量与锯齿数组", () => {
  expectGraphError("UNSUPPORTED_RANK", () =>
    compute({
      nodes: [{ id: "x", op: "const", value: [[[1]]] as unknown as number }],
      outputs: ["x"],
    }),
  );
  expectGraphError("BAD_CONSTANT", () =>
    compute({
      nodes: [{ id: "x", op: "const", value: [[1, 2], [3]] }],
      outputs: ["x"],
    }),
  );
});

test("拒绝广播不兼容", () => {
  expectGraphError("BROADCAST_INCOMPATIBLE", () =>
    compute({
      nodes: [
        { id: "a", op: "const", value: [1, 2, 3] },
        { id: "b", op: "const", value: [1, 2] },
        { id: "c", op: "add", inputs: ["a", "b"] },
      ],
      outputs: ["c"],
    }),
  );
  expectGraphError("BROADCAST_INCOMPATIBLE", () =>
    compute({
      nodes: [
        {
          id: "A",
          op: "const",
          value: [
            [1, 2],
            [3, 4],
          ],
        },
        { id: "B", op: "const", value: [[1, 2, 3]] },
        { id: "C", op: "mul", inputs: ["A", "B"] },
      ],
      outputs: ["C"],
    }),
  );
});

test("拒绝非法矩阵乘（一维输入 / 内维不匹配）", () => {
  expectGraphError("MATRIX_SHAPE", () =>
    compute({
      nodes: [
        { id: "a", op: "const", value: [1, 2, 3] },
        { id: "b", op: "const", value: [[1], [2], [3]] },
        { id: "c", op: "matmul", inputs: ["a", "b"] },
      ],
      outputs: ["c"],
    }),
  );
  expectGraphError("MATRIX_SHAPE", () =>
    compute({
      nodes: [
        { id: "A", op: "const", value: [[1, 2, 3]] }, // 1x3
        { id: "B", op: "const", value: [[1, 2]] }, // 1x2
        { id: "C", op: "matmul", inputs: ["A", "B"] },
      ],
      outputs: ["C"],
    }),
  );
});

test("拒绝有环图", () => {
  expectGraphError("CYCLE", () =>
    compute({
      nodes: [
        { id: "a", op: "add", inputs: ["b", "c"] },
        { id: "b", op: "const", value: 1 },
        { id: "c", op: "add", inputs: ["a", "b"] },
      ],
      outputs: ["a"],
    }),
  );
});

test("拒绝未知节点 / 重复 id / 非法请求", () => {
  expectGraphError("UNKNOWN_NODE", () =>
    compute({
      nodes: [{ id: "a", op: "add", inputs: ["x", "y"] }],
      outputs: ["a"],
    }),
  );
  expectGraphError("DUPLICATE_ID", () =>
    compute({
      nodes: [
        { id: "a", op: "const", value: 1 },
        { id: "a", op: "const", value: 2 },
      ],
      outputs: ["a"],
    }),
  );
  expectGraphError("INVALID_GRAD_INPUT", () =>
    compute({
      nodes: [
        { id: "x", op: "const", value: 1 },
        { id: "y", op: "relu", inputs: ["x"] },
      ],
      outputs: ["y"],
      gradInputs: ["y"],
    }),
  );
});

test("节点数超过 100 被拒绝", () => {
  const nodes: NodeSpec[] = [];
  for (let i = 0; i <= MAX_NODES; i++) {
    nodes.push({ id: `n${i}`, op: "const", value: i });
  }
  expectGraphError("TOO_MANY_NODES", () => compute({ nodes, outputs: ["n0"] }));
  // 恰好 100 个合法
  const okNodes = nodes.slice(0, MAX_NODES);
  assert.equal(compute({ nodes: okNodes, outputs: ["n99"] }).outputs[0], 99);
});

test("元素总数预算在分配前生效（含常量与中间结果）", () => {
  expectGraphError("ELEMENT_LIMIT", () =>
    compute({
      nodes: [
        { id: "a", op: "const", value: [1, 2, 3] },
        { id: "b", op: "const", value: [4, 5, 6] },
        { id: "c", op: "add", inputs: ["a", "b"] },
      ],
      outputs: ["c"],
      maxElements: 5, // 3+3+3 = 9 > 5
    }),
  );
  // 放宽预算即通过
  assert.deepEqual(
    compute({
      nodes: [
        { id: "a", op: "const", value: [1, 2, 3] },
        { id: "b", op: "const", value: [4, 5, 6] },
        { id: "c", op: "add", inputs: ["a", "b"] },
      ],
      outputs: ["c"],
      maxElements: 9,
    }).outputs,
    [[5, 7, 9]],
  );
});

test("运行时溢出产生的 Infinity 被拒绝", () => {
  expectGraphError("NON_FINITE_VALUE", () =>
    compute({
      nodes: [
        { id: "x", op: "const", value: 1e308 },
        { id: "y", op: "const", value: 1e308 },
        { id: "z", op: "mul", inputs: ["x", "y"] },
      ],
      outputs: ["z"],
    }),
  );
});

test("失败时不返回部分梯度（反向阶段产生非有限值整体失败）", () => {
  // f = relu(x) * y；x 取正值保证前向有限。
  // y 取极大值时 f 溢出 —— 仍属前向；改为让“梯度”溢出：
  // f = sum(x_large * y_shared)：x=1e308 标量，y=[1,1,...]，前向每个乘积 1e308（有限），
  // 长度 3 求和即溢出，用长度 1 保证前向有限，而对 x 的梯度 = sum(y) 可控。
  // 采用：f = x*y1 + x*y2，x=1e308, y=2 => 前向 2e308 溢出，不合适。
  // 正解：前向有限但 dx 溢出：f = sum(X @ B)，X 为 1x2 全 1e200，B 为 2x200 全 1e200
  // 每个输出 = 2e400 溢出 —— 也不行。
  // 用 ReLU 门控使前向为 0 而梯度路径巨大：f = relu(-1) * huge 在前向为 0（有限），
  // 但 d/dhuge 经过 relu(-1)=0 也是 0。改为对门控输入：
  // f = relu(g) * h，g=-1（负），h=1e308：前向 0；d/dh = relu(g)=0；
  // d/dg = 0（g<0）。无法制造。
  //
  // 最终方案：f = relu(g) * h，g=1，h=1e308 => 前向 = 1e308 有限；
  // d/dg = h = 1e308 有限；d/dh = 1。需要梯度 inf：令输出对 g 的梯度
  // 经过“sum 广播”：g 为标量，relu(g) 标量，乘以 H=[1e308, 1e308]（有限），
  // 前向 = [1e308, 1e308] 有限，再 sum => 2e308 = Infinity，前向失败。
  // 去掉 sum，输出直接是向量（梯度种子为全 1 向量）：d/dg = 2e308 = Infinity，
  // 而前向 [1e308,1e308] 有限。
  expectGraphError("NON_FINITE_VALUE", () =>
    compute({
      nodes: [
        { id: "g", op: "const", value: 1 },
        { id: "r", op: "relu", inputs: ["g"] },
        { id: "H", op: "const", value: [1e308, 1e308] },
        { id: "f", op: "mul", inputs: ["r", "H"] },
      ],
      outputs: ["f"],
      gradInputs: ["g", "H"],
    }),
  );
});

// ---------------------------------------------------------------------------
// 随机图 fuzz：有限差分
// ---------------------------------------------------------------------------

// 简单确定性 PRNG（mulberry32）
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface PoolEntry {
  id: string;
  shape: number[];
  /** 对应常量节点 id（该条目本身即可作为被求导叶子） */
  isConst: boolean;
}

/**
 * 构造随机合法图：
 * - 形状仅取 标量 / [2] / [3] / [2,3] / [3,2]
 * - 随机选择可兼容的算子
 * - 常量初值取 [-1, 1] 且偏置为正，配合“正区间”常量保证 ReLU 输入远离零点
 */
function buildRandomGraph(seed: number): {
  nodes: NodeSpec[];
  outputId: string;
  leafIds: string[];
} {
  const rng = makeRng(seed);
  const shapes: number[][] = [[], [2], [3], [2, 3], [3, 2]];
  const nodes: NodeSpec[] = [];
  const pool: PoolEntry[] = [];
  const leafIds: string[] = [];

  const randConst = (): number => 0.2 + rng() * 0.8; // [0.2, 1.0]，全正避开 ReLU 零点
  let counter = 0;
  const freshId = (prefix: string) => `${prefix}${counter++}`;

  const addConst = (shape: number[]): PoolEntry => {
    const id = freshId("c");
    const total = shape.reduce((a, b) => a * b, 1);
    const flat = Array.from({ length: total }, randConst);
    let value: TensorValue;
    if (shape.length === 0) value = flat[0];
    else if (shape.length === 1) value = flat;
    else {
      const rows: number[][] = [];
      for (let i = 0; i < shape[0]; i++) {
        rows.push(flat.slice(i * shape[1], (i + 1) * shape[1]));
      }
      value = rows;
    }
    nodes.push({ id, op: "const", value });
    const entry = { id, shape, isConst: true };
    pool.push(entry);
    leafIds.push(id);
    return entry;
  };

  // 初始 4 个常量叶子
  for (let i = 0; i < 4; i++)
    addConst(shapes[Math.floor(rng() * shapes.length)]);

  const canBroadcast = (a: number[], b: number[]): boolean => {
    try {
      const rank = Math.max(a.length, b.length);
      for (let k = 0; k < rank; k++) {
        const da = k < rank - a.length ? 1 : a[k - (rank - a.length)];
        const db = k < rank - b.length ? 1 : b[k - (rank - b.length)];
        if (da !== db && da !== 1 && db !== 1) return false;
      }
      return true;
    } catch {
      return false;
    }
  };

  const ops: Array<"add" | "mul" | "matmul" | "sum" | "relu"> = [
    "add",
    "add",
    "mul",
    "mul",
    "matmul",
    "matmul",
    "sum",
    "relu",
  ];

  // 追加 8~14 个内部节点
  const internal = 8 + Math.floor(rng() * 7);
  for (let s = 0; s < internal && nodes.length < 60; s++) {
    const op = ops[Math.floor(rng() * ops.length)];
    if (op === "matmul") {
      const lefts = pool.filter((p) => p.shape.length === 2);
      if (lefts.length === 0) {
        addConst(shapes[3 + Math.floor(rng() * 2)]);
        s--;
        continue;
      }
      const a = lefts[Math.floor(rng() * lefts.length)];
      const rights = pool.filter(
        (p) => p.shape.length === 2 && p.shape[0] === a.shape[1],
      );
      if (rights.length === 0) {
        addConst([a.shape[1], 1 + Math.floor(rng() * 3)]);
        s--;
        continue;
      }
      const b = rights[Math.floor(rng() * rights.length)];
      const id = freshId("n");
      nodes.push({ id, op, inputs: [a.id, b.id] });
      pool.push({ id, shape: [a.shape[0], b.shape[1]], isConst: false });
    } else if (op === "sum" || op === "relu") {
      const a = pool[Math.floor(rng() * pool.length)];
      const id = freshId("n");
      nodes.push({ id, op, inputs: [a.id] });
      pool.push({ id, shape: op === "sum" ? [] : a.shape, isConst: false });
    } else {
      // 尝试找可广播对
      let a: PoolEntry | null = null;
      let b: PoolEntry | null = null;
      for (let tries = 0; tries < 20; tries++) {
        const ca = pool[Math.floor(rng() * pool.length)];
        const cb = pool[Math.floor(rng() * pool.length)];
        if (canBroadcast(ca.shape, cb.shape)) {
          a = ca;
          b = cb;
          break;
        }
      }
      if (!a || !b) {
        addConst([]);
        s--;
        continue;
      }
      const rank = Math.max(a.shape.length, b.shape.length);
      const outShape: number[] = [];
      for (let k = 0; k < rank; k++) {
        const da =
          k < rank - a.shape.length ? 1 : a.shape[k - (rank - a.shape.length)];
        const db =
          k < rank - b.shape.length ? 1 : b.shape[k - (rank - b.shape.length)];
        outShape.push(Math.max(da, db));
      }
      const id = freshId("n");
      nodes.push({ id, op, inputs: [a.id, b.id] });
      pool.push({ id, shape: outShape, isConst: false });
    }
  }

  const outputId = pool[pool.length - 1].id;
  return { nodes, outputId, leafIds };
}

test("fuzz：200 个随机图的梯度与有限差分一致", () => {
  for (let seed = 1; seed <= 200; seed++) {
    const { nodes, outputId, leafIds } = buildRandomGraph(seed);
    // 只对最多 4 个叶子做差分（每个叶子元素数小，成本可控）
    const chosen = leafIds.slice(0, 4);
    assert.doesNotThrow(
      () => checkGradsWithFD(nodes, outputId, chosen, 1e-4),
      `随机图 seed=${seed} 梯度校验失败`,
    );
  }
});

test("fuzz：随机图前向两次计算结果一致（无共享状态）", () => {
  const { nodes, outputId } = buildRandomGraph(42);
  const r1 = compute({ nodes, outputs: [outputId] });
  const r2 = compute({ nodes, outputs: [outputId] });
  assert.deepEqual(r1.outputs, r2.outputs);
});
