import { gatherValue, gatherGradient } from "./gather.js";
import { GraphError } from "./errors.js";
import {
  prepareGraph,
  type ComputeRequest,
  type PreparedGraph,
  type PreparedNode,
} from "./graph.js";
import {
  Tensor,
  alignedStrides,
  assertFinite,
  numElements,
  toValue,
  zeros,
  type TensorValue,
} from "./tensor.js";
import {
  forwardAdd,
  forwardMatmul,
  forwardMul,
  forwardRelu,
  forwardSum,
} from "./ops.js";

export interface ComputeResult {
  outputs: TensorValue[];
  /** 与请求中 gradInputs 同序 */
  grads: Record<string, TensorValue>;
}

/** 图计算入口：成功返回完整输出与全部梯度；失败抛出 GraphError，不返回部分结果 */
export function compute(req: ComputeRequest): ComputeResult {
  // 1) 完整校验 + 形状推断 + 元素预算（在任何缓冲区分配之前）
  const graph = prepareGraph(req);

  // 2) 前向：只算输出可达到的节点（拓扑序）；共享节点只计算一次
  const values = runForward(graph);

  // 3) 反向：共享节点的梯度累加所有路径，广播梯度沿扩展维度求和
  const grads = runBackward(graph, values);

  // 4) 序列化
  return {
    outputs: graph.outputNodes.map((n) => toValue(values.get(n) as Tensor)),
    grads: Object.fromEntries(
      graph.gradInputNodes.map((n) => [
        n.spec.id,
        toValue(grads.get(n) as Tensor),
      ]),
    ),
  };
}

// ---------------------------------------------------------------------------
// 前向
// ---------------------------------------------------------------------------

function runForward(graph: PreparedGraph): Map<PreparedNode, Tensor> {
  const values = new Map<PreparedNode, Tensor>();
  const needed = collectReachable(graph, graph.outputNodes);

  for (const node of graph.topo) {
    if (!needed.has(node)) continue;
    const t = evalNode(node, values);
    assertFinite(t, node.spec.id);
    values.set(node, t);
  }
  return values;
}

function evalNode(
  node: PreparedNode,
  values: Map<PreparedNode, Tensor>,
): Tensor {
  const in0 = node.inputNodes[0] as PreparedNode;
  const in1 = node.inputNodes[1] as PreparedNode;
  switch (node.spec.op) {
    case "const":
      return node.constant as Tensor;
    case "add":
      return forwardAdd(
        values.get(in0) as Tensor,
        values.get(in1) as Tensor,
        node.shape,
      );
    case "mul":
      return forwardMul(
        values.get(in0) as Tensor,
        values.get(in1) as Tensor,
        node.shape,
      );
    case "matmul":
      return forwardMatmul(
        values.get(in0) as Tensor,
        values.get(in1) as Tensor,
      );
    case "gather":
      return gatherValue(
        values.get(in0) as Tensor,
        node.spec.axis ?? 0,
        node.spec.indices!,
      );
    case "sum":
      return forwardSum(values.get(in0) as Tensor);
    case "relu":
      return forwardRelu(values.get(in0) as Tensor);
  }
}

function collectReachable(
  graph: PreparedGraph,
  roots: PreparedNode[],
): Set<PreparedNode> {
  const seen = new Set<PreparedNode>();
  const stack = [...roots];
  while (stack.length) {
    const n = stack.pop() as PreparedNode;
    if (seen.has(n)) continue;
    seen.add(n);
    for (const dep of n.inputNodes) stack.push(dep);
  }
  return seen;
}

// ---------------------------------------------------------------------------
// 反向模式自动微分
// ---------------------------------------------------------------------------

function runBackward(
  graph: PreparedGraph,
  values: Map<PreparedNode, Tensor>,
): Map<PreparedNode, Tensor> {
  const grads = new Map<PreparedNode, Tensor>();

  // 多个输出时，每个输出均以全 1 为种子梯度（对标量/向量/矩阵所有元素求和）；
  // 重复输出 id 的种子也会叠加。
  for (const out of graph.outputNodes) {
    addGrad(grads, out, onesLike(values.get(out) as Tensor));
  }

  // 只需要从“输出可达且能到达所请求梯度输入”的节点反传。
  const needed = collectBackwardNeeded(graph);

  // 逆拓扑序：所有下游路径的梯度在处理该节点前已累加完毕。
  for (let idx = graph.topo.length - 1; idx >= 0; idx--) {
    const node = graph.topo[idx];
    if (!needed.has(node)) continue;
    const g = grads.get(node);
    if (g === undefined) continue;
    backwardNode(node, g, values, grads);
  }

  // 梯度同样不允许 NaN / 无穷
  for (const node of graph.gradInputNodes) {
    const g = grads.get(node);
    if (g !== undefined) assertFinite(g, node.spec.id);
  }

  // 与输出不连通（或贡献恒为 0）的梯度输入：梯度为 0
  for (const node of graph.gradInputNodes) {
    if (!grads.has(node)) {
      grads.set(node, zeros(node.shape));
    }
  }
  return grads;
}

function collectBackwardNeeded(graph: PreparedGraph): Set<PreparedNode> {
  const reachableFromOutputs = collectReachable(graph, graph.outputNodes);
  if (graph.gradInputNodes.length === 0) return reachableFromOutputs;

  // 反向图上从所请求输入出发能到达的节点（即这些输入的祖先链上的节点本身）
  const canReachInput = new Set<PreparedNode>();
  const children = new Map<PreparedNode, PreparedNode[]>();
  for (const n of graph.topo) {
    for (const dep of n.inputNodes) {
      const list = children.get(dep);
      if (list) list.push(n);
      else children.set(dep, [n]);
    }
  }
  const stack = [...graph.gradInputNodes];
  while (stack.length) {
    const n = stack.pop() as PreparedNode;
    if (canReachInput.has(n)) continue;
    canReachInput.add(n);
    for (const parent of children.get(n) ?? []) stack.push(parent);
  }

  const result = new Set<PreparedNode>();
  for (const n of reachableFromOutputs) {
    if (canReachInput.has(n)) result.add(n);
  }
  // 输出种子节点也需要（可能其本身不是任何 gradInput 的祖先……但它是根）
  for (const o of graph.outputNodes) result.add(o);
  return result;
}

function onesLike(t: Tensor): Tensor {
  const g = zeros(t.shape);
  g.data.fill(1);
  return g;
}

/** 共享节点的梯度：首次出现时分配零缓冲，之后逐元素累加所有路径的贡献 */
function addGrad(
  grads: Map<PreparedNode, Tensor>,
  node: PreparedNode,
  contribution: Tensor,
): void {
  const existing = grads.get(node);
  if (existing === undefined) {
    grads.set(node, contribution);
    return;
  }
  const a = existing.data;
  const b = contribution.data;
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

/**
 * 把“与输出同形状、已含广播结果”的贡献降维到操作数形状：
 * - 操作数缺失（补 1）的轴：整条轴求和
 * - 操作数自身为 1 而被扩展的轴：整条轴求和
 * 即广播的逆运算。
 */
function reduceBroadcast(full: Tensor, operandShape: number[]): Tensor {
  const outShape = full.shape;
  const rank = outShape.length;

  // 快速路径：形状完全相同，无需降维。
  // 注意必须拷贝：上游梯度缓冲可能被多个后继共享（例如 d=x+x 有两条边），
  // 直接返回别名会让后续 addGrad 的累加污染其他路径。
  let identical = operandShape.length === rank;
  if (identical) {
    for (let k = 0; k < rank; k++) {
      if (operandShape[k] !== outShape[k]) {
        identical = false;
        break;
      }
    }
  }
  if (identical) {
    return { shape: operandShape, data: Float64Array.from(full.data) };
  }

  const out = zeros(operandShape);
  const total = numElements(outShape);
  // 输出坐标 -> 操作数扁平索引；补位维度与被扩展的 size-1 轴步距为 0，
  // 多个输出元素因此累加到同一操作数位置，即“沿扩展维度求和”。
  const operandStrides = alignedStrides(operandShape, outShape);

  for (let flat = 0; flat < total; flat++) {
    let rem = flat;
    let ti = 0;
    for (let k = rank - 1; k >= 0; k--) {
      const coord = rem % outShape[k];
      rem = Math.floor(rem / outShape[k]);
      ti += coord * operandStrides[k];
    }
    out.data[ti] += full.data[flat];
  }
  return out;
}

function backwardNode(
  node: PreparedNode,
  g: Tensor,
  values: Map<PreparedNode, Tensor>,
  grads: Map<PreparedNode, Tensor>,
): void {
  // 输入数量已在 prepareGraph 阶段按 op 校验
  const aNode = node.inputNodes[0] as PreparedNode;
  const bNode = node.inputNodes[1] as PreparedNode;
  const a = values.get(aNode) as Tensor | undefined;
  const b = bNode ? values.get(bNode) : undefined;

  switch (node.spec.op) {
    case "const":
      // 常量（非被求导输入）不继续反传
      return;

    case "gather":
      addGrad(
        grads,
        aNode,
        gatherGradient(a!, node.spec.axis ?? 0, node.spec.indices!, g),
      );
      return;
    case "add":
      // d(a+b)/da = 1：梯度沿广播扩展的轴求和
      addGrad(grads, aNode, reduceBroadcast(g, (a as Tensor).shape));
      addGrad(grads, bNode, reduceBroadcast(g, (b as Tensor).shape));
      return;

    case "mul": {
      // d(a*b)/da = b，d(a*b)/db = a；逐元素乘后再做广播逆降维
      const gaFull = mulBroadcast(g, b as Tensor, node.shape);
      const gbFull = mulBroadcast(g, a as Tensor, node.shape);
      addGrad(grads, aNode, reduceBroadcast(gaFull, (a as Tensor).shape));
      addGrad(grads, bNode, reduceBroadcast(gbFull, (b as Tensor).shape));
      return;
    }

    case "matmul": {
      // C = A @ B，A:[m,k] B:[k,n] C:[m,n]
      // dA = g @ B^T  -> [m,k]
      // dB = A^T @ g  -> [k,n]
      const A = a as Tensor;
      const B = b as Tensor;
      const [m, k] = A.shape;
      const n = B.shape[1];
      const dA = zeros([m, k]);
      const dB = zeros([k, n]);
      for (let i = 0; i < m; i++) {
        for (let p = 0; p < k; p++) {
          let s = 0;
          for (let j = 0; j < n; j++)
            s += g.data[i * n + j] * B.data[p * n + j];
          dA.data[i * k + p] = s;
        }
      }
      for (let p = 0; p < k; p++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let i = 0; i < m; i++)
            s += A.data[i * k + p] * g.data[i * n + j];
          dB.data[p * n + j] = s;
        }
      }
      addGrad(grads, aNode, dA);
      addGrad(grads, bNode, dB);
      return;
    }

    case "sum":
      // 全量求和：上游标量梯度广播回输入的每一个元素
      addGrad(grads, aNode, scalarExpand(g, (a as Tensor).shape));
      return;

    case "relu": {
      // 严格 > 0 处导数为 1，零点规定为 0
      const x = a as Tensor;
      const gx = zeros(x.shape);
      for (let i = 0; i < x.data.length; i++) {
        gx.data[i] = x.data[i] > 0 ? g.data[i] : 0;
      }
      addGrad(grads, aNode, gx);
      return;
    }
  }
}

/** g（输出形状）按输出坐标取操作数对应元素相乘，结果保持输出形状（用于 mul 反向） */
function mulBroadcast(g: Tensor, operand: Tensor, outShape: number[]): Tensor {
  const out = zeros(outShape);
  const strides = alignedStrides(operand.shape, outShape);
  const total = numElements(outShape);
  const rank = outShape.length;
  for (let flat = 0; flat < total; flat++) {
    let rem = flat;
    let idx = 0;
    for (let k = rank - 1; k >= 0; k--) {
      const coord = rem % outShape[k];
      rem = Math.floor(rem / outShape[k]);
      idx += coord * strides[k];
    }
    out.data[flat] = g.data[flat] * operand.data[idx];
  }
  return out;
}

/** 标量上游梯度扩展为指定形状的全量拷贝 */
function scalarExpand(g: Tensor, shape: number[]): Tensor {
  if (g.shape.length !== 0 || g.data.length !== 1) {
    throw new GraphError("SHAPE_MISMATCH", "sum 的上游梯度必须是标量");
  }
  const out = zeros(shape);
  out.data.fill(g.data[0]);
  return out;
}
