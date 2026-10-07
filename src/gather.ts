import { GraphError } from "./errors.js";
import { Tensor, zeros } from "./tensor.js";

/** indices 静态参数上限（见 README「索引采样」） */
export const MAX_GATHER_INDICES = 128;

export interface GatherParams {
  /** 按输入秩归一化后的轴（0 起） */
  axis: number;
  /** 校验通过的整数索引（保持请求顺序，允许重复） */
  indices: number[];
}

/**
 * 校验并归一化 gather 的静态参数（axis / indices）。
 * 所有失败都在图准备（形状推断）阶段、任何前向缓冲区分配之前抛出 GraphError：
 * - 输入必须是一维或二维
 * - axis 缺省为 0；必须为整数，负轴按秩归一化后不得越界
 * - indices 必须是非空整数数组（至多 128 项），
 *   每项必须是所选轴上合法的非负整数
 */
export function prepareGather(
  nodeId: string,
  inputShape: readonly number[],
  axis: unknown,
  indices: unknown,
): GatherParams {
  const rank = inputShape.length;
  if (rank !== 1 && rank !== 2) {
    throw new GraphError(
      "INVALID_REQUEST",
      `节点 ${nodeId} (gather) 的输入必须是一维或二维张量，实际秩为 ${rank}`,
    );
  }

  let ax: number;
  if (axis === undefined) {
    ax = 0;
  } else {
    if (typeof axis !== "number" || !Number.isInteger(axis)) {
      throw new GraphError(
        "INVALID_REQUEST",
        `节点 ${nodeId} (gather) 的 axis 必须是整数，实际为 ${String(axis)}`,
      );
    }
    ax = axis;
  }
  // 负轴按秩归一
  if (ax < 0) ax += rank;
  if (ax < 0 || ax >= rank) {
    throw new GraphError(
      "INVALID_REQUEST",
      `节点 ${nodeId} (gather) 的 axis 越界（输入秩为 ${rank}）`,
    );
  }

  if (!Array.isArray(indices)) {
    throw new GraphError(
      "INVALID_REQUEST",
      `节点 ${nodeId} (gather) 的 indices 必须是非空整数数组`,
    );
  }
  if (indices.length === 0) {
    throw new GraphError(
      "INVALID_REQUEST",
      `节点 ${nodeId} (gather) 的 indices 不能为空`,
    );
  }
  if (indices.length > MAX_GATHER_INDICES) {
    throw new GraphError(
      "INVALID_REQUEST",
      `节点 ${nodeId} (gather) 的 indices 至多 ${MAX_GATHER_INDICES} 项，实际 ${indices.length} 项`,
    );
  }

  const dim = inputShape[ax];
  const resolved: number[] = new Array(indices.length);
  for (let j = 0; j < indices.length; j++) {
    const idx = indices[j];
    if (typeof idx !== "number" || !Number.isInteger(idx) || idx < 0) {
      throw new GraphError(
        "INVALID_REQUEST",
        `节点 ${nodeId} (gather) 的 indices[${j}] 必须是非负整数，实际为 ${String(idx)}`,
      );
    }
    if (idx >= (dim as number)) {
      throw new GraphError(
        "INVALID_REQUEST",
        `节点 ${nodeId} (gather) 的 indices[${j}]=${idx} 超出轴 ${ax} 的长度 ${dim}`,
      );
    }
    resolved[j] = idx;
  }

  return { axis: ax, indices: resolved };
}

/** 输出形状：复制输入形状，把被采样轴的长度替换为索引项数 */
export function gatherShape(
  inputShape: readonly number[],
  params: GatherParams,
): number[] {
  const out = inputShape.slice();
  out[params.axis] = params.indices.length;
  return out;
}

/**
 * 前向：沿 axis 按 indices 顺序采样，输出在该轴上的长度为 indices.length。
 * 重复索引合法：同一元素可按顺序多次出现在输出中。
 * 返回持有独立数据副本的张量，不与输入或其他分支共享缓冲。
 */
export function gatherValue(input: Tensor, params: GatherParams): Tensor {
  const { axis, indices } = params;
  const shape = input.shape;
  const out = zeros(gatherShape(shape, params));
  const data = input.data;
  const L = indices.length;

  if (shape.length === 1) {
    // 一维：axis 归一化后必为 0
    for (let j = 0; j < L; j++) out.data[j] = data[indices[j]];
    return out;
  }

  const m = shape[0];
  const n = shape[1];
  if (axis === 0) {
    // 按行采样：out[j, k] = in[indices[j], k]
    for (let j = 0; j < L; j++) {
      const src = indices[j] * n;
      const dst = j * n;
      for (let k = 0; k < n; k++) {
        out.data[dst + k] = data[src + k];
      }
    }
  } else {
    // 按列采样：out[i, j] = in[i, indices[j]]
    for (let i = 0; i < m; i++) {
      const srcRow = i * n;
      const dstRow = i * L;
      for (let j = 0; j < L; j++) {
        out.data[dstRow + j] = data[srcRow + indices[j]];
      }
    }
  }
  return out;
}

/**
 * 反向：把每个输出位置的梯度加回它采样自的输入位置。
 * 重复索引对应的多个输出位置逐元素累加到同一输入位置；
 * 返回独立的零初始化缓冲，共享分支之间的跨路径累加由引擎 addGrad 完成。
 */
export function gatherGradient(
  inputShape: readonly number[],
  params: GatherParams,
  grad: Tensor,
): Tensor {
  const { axis, indices } = params;
  const out = zeros(inputShape.slice());
  const dst = out.data;
  const src = grad.data;
  const L = indices.length;

  if (inputShape.length === 1) {
    for (let j = 0; j < L; j++) {
      dst[indices[j]] += src[j];
    }
    return out;
  }

  const m = inputShape[0];
  const n = inputShape[1];
  if (axis === 0) {
    // grad 形状 [L, n]：第 j 行整行加回输入的第 indices[j] 行（重复行累加）
    for (let j = 0; j < L; j++) {
      const dstRow = indices[j] * n;
      const srcRow = j * n;
      for (let k = 0; k < n; k++) {
        dst[dstRow + k] += src[srcRow + k];
      }
    }
  } else {
    // grad 形状 [m, L]：第 j 列整列加回输入的第 indices[j] 列（重复列累加）
    for (let i = 0; i < m; i++) {
      const dstRow = i * n;
      const srcRow = i * L;
      for (let j = 0; j < L; j++) {
        dst[dstRow + indices[j]] += src[srcRow + j];
      }
    }
  }
  return out;
}
