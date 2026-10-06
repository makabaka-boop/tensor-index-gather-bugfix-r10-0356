import { GraphError } from "./errors.js";

/**
 * 内部张量：形状（长度 0/1/2，分别表示标量、一维、二维）+ 行主序扁平数据。
 * 所有算子都在该表示上工作；仅在入站解析与出站序列化时使用嵌套数组。
 */
export interface Tensor {
  shape: number[];
  data: Float64Array;
}

/** 线上协议中的 JSON 张量：number / number[] / number[][] */
export type TensorValue = number | number[] | number[][];

export function numElements(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/**
 * 把嵌套数组解析为内部张量。严格校验：
 * - 只接受标量 / 一维 / 二维（更深拒绝）
 * - 行必须等长（拒绝锯齿数组）
 * - 元素必须是有限数值（拒绝 NaN / Infinity）
 */
export function parseTensor(value: unknown): Tensor {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new GraphError("NON_FINITE_VALUE", "常量包含 NaN 或无穷值");
    }
    return { shape: [], data: Float64Array.of(value) };
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new GraphError("BAD_CONSTANT", "不允许空数组常量");
    }
    if (value.every((x) => typeof x === "number")) {
      const data = new Float64Array(value.length);
      for (let i = 0; i < value.length; i++) {
        const x = value[i] as number;
        if (!Number.isFinite(x)) {
          throw new GraphError(
            "NON_FINITE_VALUE",
            `常量元素 [${i}] 为 NaN 或无穷值`,
          );
        }
        data[i] = x;
      }
      return { shape: [value.length], data };
    }
    if (value.every((row) => Array.isArray(row))) {
      const rows = value as unknown[][];
      const m = rows.length;
      const n = (rows[0] as unknown[]).length;
      if (n === 0) {
        throw new GraphError("BAD_CONSTANT", "不允许长度为 0 的维度");
      }
      for (let i = 1; i < m; i++) {
        if ((rows[i] as unknown[]).length !== n) {
          throw new GraphError(
            "BAD_CONSTANT",
            `二维常量的第 ${i} 行长度不一致（锯齿数组）`,
          );
        }
      }
      const data = new Float64Array(m * n);
      for (let i = 0; i < m; i++) {
        const row = rows[i] as unknown[];
        for (let j = 0; j < n; j++) {
          const x = row[j];
          if (typeof x !== "number") {
            throw new GraphError(
              "UNSUPPORTED_RANK",
              "仅支持标量、一维、二维张量",
            );
          }
          if (!Number.isFinite(x)) {
            throw new GraphError(
              "NON_FINITE_VALUE",
              `常量元素 [${i}][${j}] 为 NaN 或无穷值`,
            );
          }
          data[i * n + j] = x;
        }
      }
      return { shape: [m, n], data };
    }
    throw new GraphError(
      "UNSUPPORTED_RANK",
      "常量数组元素必须全部为数值，且维度不超过二维",
    );
  }

  throw new GraphError("BAD_CONSTANT", "常量必须是数值或数值数组");
}

/** 内部张量 -> 嵌套数组（标量 -> number，一维 -> number[]，二维 -> number[][]） */
export function toValue(t: Tensor): TensorValue {
  const s = t.shape;
  if (s.length === 0) return t.data[0];
  if (s.length === 1) return Array.from(t.data);
  const m = s[0] as number;
  const n = s[1] as number;
  const out: number[][] = new Array(m);
  for (let i = 0; i < m; i++) {
    out[i] = Array.from(t.data.subarray(i * n, (i + 1) * n));
  }
  return out;
}

export function zeros(shape: number[]): Tensor {
  return { shape, data: new Float64Array(numElements(shape)) };
}

/**
 * 按尾部维度广播求输出形状。维度数取较大者，缺失维度视为 1；
 * 对应维度必须相等或其中之一为 1。
 */
export function broadcastShape(a: number[], b: number[]): number[] {
  const rank = Math.max(a.length, b.length);
  const out: number[] = new Array(rank);
  for (let k = 0; k < rank; k++) {
    const da = k < rank - a.length ? 1 : a[k - (rank - a.length)];
    const db = k < rank - b.length ? 1 : b[k - (rank - b.length)];
    if (da !== db && da !== 1 && db !== 1) {
      throw new GraphError(
        "BROADCAST_INCOMPATIBLE",
        `维度 ${k} 上 ${da} 与 ${db} 无法按尾部维度广播`,
      );
    }
    out[k] = Math.max(da, db);
  }
  return out;
}

/**
 * 操作数相对对齐后输出形状的步距。
 * 操作数缺失（补 1）的维度步距为 0；自身维度为 1 而被扩展的维度步距也是 0
 * （该坐标恒取 0）。
 */
export function alignedStrides(shape: number[], outShape: number[]): number[] {
  const rank = outShape.length;
  const offset = rank - shape.length;
  const strides = new Array<number>(rank).fill(0);
  // 行主序：第 k 维步距 = 其右侧各维长度之积。
  // 从最后一维（步距 1）向左累乘；尺寸为 1 而被扩展的维恒取同一元素（步距 0）。
  let stride = 1;
  for (let k = rank - 1; k >= 0; k--) {
    if (k >= offset) {
      const d = shape[k - offset];
      if (d !== 1) strides[k] = stride;
      stride *= d;
    }
    // k < offset：补位维度，步距保持 0；stride 乘补位的 1，无变化
  }
  return strides;
}

/** 校验所有数据均为有限值（前向结果检查，拒绝运行时溢出产生的 Infinity/NaN） */
export function assertFinite(t: Tensor, nodeId: string): void {
  for (let i = 0; i < t.data.length; i++) {
    if (!Number.isFinite(t.data[i])) {
      throw new GraphError(
        "NON_FINITE_VALUE",
        `节点 ${nodeId} 的计算结果含 NaN 或无穷值（位置 ${i}）`,
      );
    }
  }
}
