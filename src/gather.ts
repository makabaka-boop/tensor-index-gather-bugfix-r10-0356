import { GraphError } from "./errors.js";
import { Tensor, zeros } from "./tensor.js";

/** indices 项数上限（README：一维非空 indices，至多 128 项） */
export const MAX_INDICES = 128;

/**
 * gather 参数的结构层校验（在任何缓冲区分配之前完成）：
 * - axis 缺省为 0；若给出必须是整数（负轴稍后按秩归一）
 * - indices 必须是非空数组，长度 <= 128，元素全为非负整数
 * 索引是否落在所选轴长度之内属于形状层校验（见 gatherShape）。
 */
export function validateGatherFields(
  id: string,
  axis: unknown,
  indices: unknown,
): { axis: number | undefined; indices: number[] } {
  let validatedAxis: number | undefined;
  if (axis !== undefined) {
    if (typeof axis !== "number" || !Number.isInteger(axis)) {
      throw new GraphError(
        "INVALID_REQUEST",
        `节点 ${id} (gather) 的 axis 必须是整数，实际为 ${String(axis)}`,
      );
    }
    validatedAxis = axis;
  }

  if (!Array.isArray(indices) || indices.length === 0) {
    throw new GraphError(
      "INVALID_REQUEST",
      `节点 ${id} (gather) 的 indices 必须是非空数组`,
    );
  }
  if (indices.length > MAX_INDICES) {
    throw new GraphError(
      "INVALID_REQUEST",
      `节点 ${id} (gather) 的 indices 至多 ${MAX_INDICES} 项，实际为 ${indices.length} 项`,
    );
  }
  const validatedIndices = new Array<number>(indices.length);
  for (let j = 0; j < indices.length; j++) {
    const ix = indices[j];
    if (typeof ix !== "number" || !Number.isInteger(ix) || ix < 0) {
      throw new GraphError(
        "INVALID_REQUEST",
        `节点 ${id} (gather) 的 indices[${j}] 必须是非负整数，实际为 ${String(ix)}`,
      );
    }
    validatedIndices[j] = ix;
  }
  return { axis: validatedAxis, indices: validatedIndices };
}

/** 负轴按秩归一：-1 -> rank-1；归一后仍越界则失败 */
export function normalizeAxis(axis: number, rank: number): 0 | 1 {
  const a = axis < 0 ? axis + rank : axis;
  if (a < 0 || a >= rank) {
    throw new GraphError(
      "INVALID_REQUEST",
      `axis=${axis} 超出秩 ${rank} 的合法范围 [${-rank}, ${rank - 1}]`,
    );
  }
  return a as 0 | 1;
}

/**
 * gather 形状推断（同时完成形状层校验）：
 * - 输入必须是一维或二维
 * - axis（负轴归一后）必须在秩范围内
 * - 每个索引必须是所选轴上合法的非负整数
 * 输出形状 = 输入形状但被选轴长度替换为 indices 长度（按 indices 顺序，可重复）。
 */
export function gatherShape(
  shape: readonly number[],
  axis: number,
  indices: readonly number[],
): number[] {
  const rank = shape.length;
  if (rank !== 1 && rank !== 2) {
    throw new GraphError(
      "INVALID_REQUEST",
      `gather 输入必须是一维或二维张量，实际秩为 ${rank}`,
    );
  }
  const a = normalizeAxis(axis, rank);
  const dim = shape[a] as number;
  for (let j = 0; j < indices.length; j++) {
    const ix = indices[j] as number;
    if (ix >= dim) {
      throw new GraphError(
        "INVALID_REQUEST",
        `indices[${j}]=${ix} 超出所选轴 ${a} 的长度 ${dim}`,
      );
    }
  }
  const out = shape.slice();
  out[a] = indices.length;
  return out;
}

/**
 * 前向 gather：沿 axis 按 indices 顺序切片采样，重复索引重复参与。
 * 返回独立分配的缓冲，不与输入共享内存（共享子图 / 修改输入后重算都安全）。
 */
export function gatherValue(
  input: Tensor,
  axisRaw: number,
  indices: readonly number[],
): Tensor {
  const rank = input.shape.length;
  const axis = normalizeAxis(axisRaw, rank);
  const k = indices.length;
  const src = input.data;

  if (rank === 1) {
    const data = new Float64Array(k);
    for (let j = 0; j < k; j++) data[j] = src[indices[j] as number];
    return { shape: [k], data };
  }

  const m = input.shape[0] as number;
  const n = input.shape[1] as number;

  if (axis === 0) {
    // 选取整行，输出 [k, n]；按 indices 顺序复制，重复行重复复制
    const data = new Float64Array(k * n);
    for (let j = 0; j < k; j++) {
      const row = indices[j] as number;
      const srcStart = row * n;
      const dstStart = j * n;
      for (let q = 0; q < n; q++) data[dstStart + q] = src[srcStart + q];
    }
    return { shape: [k, n], data };
  }

  // axis === 1：每行内按 indices 取列，输出 [m, k]
  const data = new Float64Array(m * k);
  for (let i = 0; i < m; i++) {
    const rowStart = i * n;
    const dstStart = i * k;
    for (let j = 0; j < k; j++) {
      data[dstStart + j] = src[rowStart + (indices[j] as number)];
    }
  }
  return { shape: [m, k], data };
}

/**
 * 反向 gather：把输出位置 j 的梯度按 indices[j] 散射回原轴位置。
 * 重复索引 / 共享分支的多次贡献全部累加（+=，不是赋值）；
 * 二维轴 0 时整行（含所有列）逐元素累加。
 */
export function gatherGradient(
  input: Tensor,
  axisRaw: number,
  indices: readonly number[],
  grad: Tensor,
): Tensor {
  const rank = input.shape.length;
  const axis = normalizeAxis(axisRaw, rank);
  const k = indices.length;
  const out = zeros(input.shape);
  const dst = out.data;
  const g = grad.data;

  if (rank === 1) {
    for (let j = 0; j < k; j++) dst[indices[j] as number] += g[j];
    return out;
  }

  const m = input.shape[0] as number;
  const n = input.shape[1] as number;

  if (axis === 0) {
    for (let j = 0; j < k; j++) {
      const row = indices[j] as number;
      const srcStart = j * n;
      const dstStart = row * n;
      for (let q = 0; q < n; q++) dst[dstStart + q] += g[srcStart + q];
    }
    return out;
  }

  // axis === 1
  for (let i = 0; i < m; i++) {
    const rowStart = i * n;
    const gStart = i * k;
    for (let j = 0; j < k; j++) {
      dst[rowStart + (indices[j] as number)] += g[gStart + j];
    }
  }
  return out;
}
