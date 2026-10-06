import {
  gatherShape,
  normalizeAxis,
  validateGatherFields,
} from "./gather.js";
import { GraphError } from "./errors.js";
import {
  Tensor,
  parseTensor,
  numElements,
  broadcastShape,
  type TensorValue,
} from "./tensor.js";

export type { TensorValue } from "./tensor.js";

export type Op = "const" | "add" | "mul" | "matmul" | "sum" | "relu" | "gather";

export interface NodeSpec {
  id: string;
  op: Op;
  /** const：嵌套数值数组；其余算子：输入节点 id */
  value?: TensorValue;
  axis?: number;
  indices?: number[];
  inputs?: string[];
}

export interface ComputeRequest {
  /** 至多 100 个节点 */
  nodes: NodeSpec[];
  /** 要求计算值的节点 id（可重复，按出现顺序返回） */
  outputs: string[];
  /** 要求梯度的常量输入节点 id */
  gradInputs?: string[];
  /** 前向张量元素总数上限（默认 10_000_000，在分配前校验） */
  maxElements?: number;
}

export interface PreparedNode {
  spec: NodeSpec;
  shape: number[];
  inputNodes: PreparedNode[];
  constant: Tensor | null;
  /** 拓扑序中的位置 */
  topoIndex: number;
  /** gather：结构校验后的 indices（结构校验阶段填充） */
  gatherIndices?: number[];
  /** gather：按输入秩归一后的 axis（形状推断阶段填充） */
  gatherAxis?: 0 | 1;
}

export interface PreparedGraph {
  nodes: PreparedNode[];
  byId: Map<string, PreparedNode>;
  topo: PreparedNode[];
  outputNodes: PreparedNode[];
  gradInputNodes: PreparedNode[];
  maxElements: number;
}

export const MAX_NODES = 100;
export const DEFAULT_MAX_ELEMENTS = 10_000_000;

/**
 * 图准备：请求校验 -> 常量解析 -> 拓扑排序（环检测）-> 形状推断 -> 元素预算。
 * 本函数在任何数据缓冲区分配之前完成全部可预期校验。
 */
export function prepareGraph(req: unknown): PreparedGraph {
  if (typeof req !== "object" || req === null || Array.isArray(req)) {
    throw new GraphError("INVALID_REQUEST", "请求体必须是对象");
  }
  const r = req as ComputeRequest;

  if (!Array.isArray(r.nodes)) {
    throw new GraphError("INVALID_REQUEST", "nodes 必须是数组");
  }
  if (r.nodes.length === 0) {
    throw new GraphError("INVALID_REQUEST", "至少需要一个节点");
  }
  if (r.nodes.length > MAX_NODES) {
    throw new GraphError(
      "TOO_MANY_NODES",
      `节点数 ${r.nodes.length} 超过上限 ${MAX_NODES}`,
    );
  }
  if (!Array.isArray(r.outputs) || r.outputs.length === 0) {
    throw new GraphError("INVALID_REQUEST", "outputs 必须是非空字符串数组");
  }

  let maxElements = DEFAULT_MAX_ELEMENTS;
  if (r.maxElements !== undefined) {
    maxElements = r.maxElements;
    if (
      typeof maxElements !== "number" ||
      !Number.isInteger(maxElements) ||
      maxElements <= 0
    ) {
      throw new GraphError("INVALID_REQUEST", "maxElements 必须是正整数");
    }
  }

  // ---- 结构校验 & 常量解析（先于任何形状工作） ----
  const byId = new Map<string, PreparedNode>();
  for (const spec of r.nodes) {
    if (typeof spec !== "object" || spec === null) {
      throw new GraphError("INVALID_REQUEST", "节点必须是对象");
    }
    if (typeof spec.id !== "string" || spec.id.length === 0) {
      throw new GraphError("INVALID_REQUEST", "节点 id 必须是非空字符串");
    }
    if (byId.has(spec.id)) {
      throw new GraphError("DUPLICATE_ID", `重复的节点 id: ${spec.id}`);
    }
    if (typeof spec.op !== "string") {
      throw new GraphError("INVALID_REQUEST", `节点 ${spec.id} 缺少 op`);
    }

    let constant: Tensor | null = null;
    let gatherIndices: number[] | undefined;
    switch (spec.op) {
      case "const":
        if (spec.inputs !== undefined) {
          throw new GraphError(
            "INVALID_REQUEST",
            `常量节点 ${spec.id} 不接受 inputs`,
          );
        }
        if (spec.value === undefined) {
          throw new GraphError(
            "BAD_CONSTANT",
            `常量节点 ${spec.id} 缺少 value`,
          );
        }
        constant = parseTensor(spec.value);
        break;
      case "add":
      case "mul":
        if (!Array.isArray(spec.inputs) || spec.inputs.length !== 2) {
          throw new GraphError(
            "INVALID_REQUEST",
            `节点 ${spec.id} (${spec.op}) 需要恰好 2 个输入`,
          );
        }
        break;
      case "matmul":
        if (!Array.isArray(spec.inputs) || spec.inputs.length !== 2) {
          throw new GraphError(
            "INVALID_REQUEST",
            `节点 ${spec.id} (matmul) 需要恰好 2 个输入`,
          );
        }
        break;
      case "gather":
        if (!Array.isArray(spec.inputs) || spec.inputs.length !== 1) {
          throw new GraphError(
            "INVALID_REQUEST",
            `节点 ${spec.id} (gather) 需要恰好 1 个输入`,
          );
        }
        // axis / indices 结构校验（越界检查在形状推断、输入形状已知后进行）
        gatherIndices = validateGatherFields(
          spec.id,
          spec.axis,
          spec.indices,
        ).indices;
        break;
      case "sum":
      case "relu":
        if (!Array.isArray(spec.inputs) || spec.inputs.length !== 1) {
          throw new GraphError(
            "INVALID_REQUEST",
            `节点 ${spec.id} (${spec.op}) 需要恰好 1 个输入`,
          );
        }
        break;
      default:
        throw new GraphError(
          "INVALID_REQUEST",
          `节点 ${spec.id} 的 op 非法: ${String(spec.op)}`,
        );
    }

    byId.set(spec.id, {
      spec,
      shape: [],
      inputNodes: [],
      constant,
      topoIndex: -1,
      ...(gatherIndices !== undefined ? { gatherIndices } : {}),
    });
  }

  // ---- 输出与梯度输入校验 ----
  const outputNodes: PreparedNode[] = [];
  for (const id of r.outputs) {
    if (typeof id !== "string") {
      throw new GraphError("INVALID_REQUEST", "outputs 元素必须是字符串 id");
    }
    const node = byId.get(id);
    if (!node) throw new GraphError("UNKNOWN_NODE", `输出节点不存在: ${id}`);
    outputNodes.push(node);
  }

  const gradInputNodes: PreparedNode[] = [];
  if (r.gradInputs !== undefined) {
    if (!Array.isArray(r.gradInputs)) {
      throw new GraphError("INVALID_REQUEST", "gradInputs 必须是字符串数组");
    }
    for (const id of r.gradInputs) {
      if (typeof id !== "string") {
        throw new GraphError(
          "INVALID_GRAD_INPUT",
          "gradInputs 元素必须是字符串 id",
        );
      }
      const node = byId.get(id);
      if (!node)
        throw new GraphError("UNKNOWN_NODE", `梯度输入节点不存在: ${id}`);
      if (node.spec.op !== "const") {
        throw new GraphError(
          "INVALID_GRAD_INPUT",
          `只能对常量输入求梯度: ${id}`,
        );
      }
      if (gradInputNodes.includes(node)) {
        throw new GraphError(
          "INVALID_GRAD_INPUT",
          `gradInputs 中存在重复: ${id}`,
        );
      }
      gradInputNodes.push(node);
    }
  }

  // ---- 解析输入引用 ----
  for (const node of byId.values()) {
    if (node.spec.op === "const") continue;
    for (const inputId of node.spec.inputs as string[]) {
      const dep = byId.get(inputId);
      if (!dep) {
        throw new GraphError(
          "UNKNOWN_NODE",
          `节点 ${node.spec.id} 引用了不存在的节点 ${inputId}`,
        );
      }
      node.inputNodes.push(dep);
    }
  }

  // ---- 拓扑排序（DFS，带环检测；顺序在同优先级下按节点声明顺序，保证确定性） ----
  const topo: PreparedNode[] = [];
  const state = new Map<PreparedNode, 0 | 1 | 2>(); // 0=未访问 1=在栈上 2=完成
  const visit = (node: PreparedNode, stack: string[]): void => {
    const s = state.get(node) ?? 0;
    if (s === 1) {
      throw new GraphError(
        "CYCLE",
        `计算图存在环: ${[...stack, node.spec.id].join(" -> ")}`,
      );
    }
    if (s === 2) return;
    state.set(node, 1);
    for (const dep of node.inputNodes) {
      visit(dep, [...stack, node.spec.id]);
    }
    state.set(node, 2);
    node.topoIndex = topo.length;
    topo.push(node);
  };
  for (const node of byId.values()) visit(node, []);

  // ---- 形状推断 ----
  for (const node of topo) {
    switch (node.spec.op) {
      case "const":
        node.shape = (node.constant as Tensor).shape;
        break;
      case "add":
      case "mul": {
        const a = node.inputNodes[0] as PreparedNode;
        const b = node.inputNodes[1] as PreparedNode;
        node.shape = inferBinaryShape(
          node.spec.op,
          a.shape,
          b.shape,
          node.spec.id,
        );
        break;
      }
      case "matmul": {
        const a = node.inputNodes[0] as PreparedNode;
        const b = node.inputNodes[1] as PreparedNode;
        node.shape = inferMatmulShape(a.shape, b.shape, node.spec.id);
        break;
      }
      case "gather": {
        const axisRaw = node.spec.axis ?? 0;
        const inputShape = node.inputNodes[0].shape;
        // gatherShape 内部完成秩 / 越界 / 负轴归一校验；
        // 归一后的 axis 存到 PreparedNode，供前向与反向直接使用（此时已全部验证通过）。
        node.shape = gatherShape(
          inputShape,
          axisRaw,
          node.gatherIndices as number[],
        );
        node.gatherAxis = normalizeAxis(axisRaw, inputShape.length);
        break;
      }
      case "sum":
        node.shape = [];
        break;
      case "relu":
        node.shape = (node.inputNodes[0] as PreparedNode).shape;
        break;
    }
  }

  // ---- 元素总数预算（分配前）：统计所有节点（含常量） ----
  let total = 0;
  for (const node of topo) {
    total += numElements(node.shape);
    if (total > maxElements) {
      throw new GraphError(
        "ELEMENT_LIMIT",
        `前向张量元素总数超过上限：已达 ${total}，上限 ${maxElements}（节点 ${node.spec.id}）`,
      );
    }
  }

  return {
    nodes: Array.from(byId.values()),
    byId,
    topo,
    outputNodes,
    gradInputNodes,
    maxElements,
  };
}

function inferBinaryShape(
  op: "add" | "mul",
  a: number[],
  b: number[],
  id: string,
): number[] {
  if (a.length > 2 || b.length > 2) {
    throw new GraphError("UNSUPPORTED_RANK", `节点 ${id} 的输入维度超过二维`);
  }
  try {
    return broadcastShape(a, b);
  } catch (e) {
    if (e instanceof GraphError) {
      throw new GraphError(
        "BROADCAST_INCOMPATIBLE",
        `节点 ${id} (${op}) 的输入无法广播: ${e.message}`,
      );
    }
    throw e;
  }
}

function inferMatmulShape(a: number[], b: number[], id: string): number[] {
  if (a.length !== 2 || b.length !== 2) {
    throw new GraphError(
      "MATRIX_SHAPE",
      `节点 ${id} (matmul) 的两个输入都必须是二维矩阵，实际秩为 ${a.length} 和 ${b.length}`,
    );
  }
  if (a[1] !== b[0]) {
    throw new GraphError(
      "MATRIX_SHAPE",
      `节点 ${id} (matmul) 内维不匹配: [${a[0]},${a[1]}] x [${b[0]},${b[1]}]`,
    );
  }
  return [a[0], b[1]];
}
