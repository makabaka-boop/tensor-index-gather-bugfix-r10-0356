import { Tensor, alignedStrides, numElements, zeros } from "./tensor.js";

/** 逐元素二元运算（按尾部维度广播）。out = f(a, b) */
function elementwiseBinary(
  a: Tensor,
  b: Tensor,
  outShape: number[],
  f: (x: number, y: number) => number,
): Tensor {
  const out = zeros(outShape);
  const sa = alignedStrides(a.shape, outShape);
  const sb = alignedStrides(b.shape, outShape);
  const total = numElements(outShape);
  const rank = outShape.length;

  for (let flat = 0; flat < total; flat++) {
    let rem = flat;
    let ia = 0;
    let ib = 0;
    for (let k = rank - 1; k >= 0; k--) {
      const coord = rem % outShape[k];
      rem = Math.floor(rem / outShape[k]);
      ia += coord * sa[k];
      ib += coord * sb[k];
    }
    out.data[flat] = f(a.data[ia], b.data[ib]);
  }
  return out;
}

export function forwardAdd(a: Tensor, b: Tensor, outShape: number[]): Tensor {
  return elementwiseBinary(a, b, outShape, (x, y) => x + y);
}

export function forwardMul(a: Tensor, b: Tensor, outShape: number[]): Tensor {
  return elementwiseBinary(a, b, outShape, (x, y) => x * y);
}

export function forwardMatmul(a: Tensor, b: Tensor): Tensor {
  const m = a.shape[0] as number;
  const k = a.shape[1] as number;
  const n = b.shape[1] as number;
  const out = zeros([m, n]);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aik = a.data[i * k + p];
      for (let j = 0; j < n; j++) {
        out.data[i * n + j] += aik * b.data[p * n + j];
      }
    }
  }
  return out;
}

/** 全量求和：输出标量 */
export function forwardSum(a: Tensor): Tensor {
  let acc = 0;
  for (let i = 0; i < a.data.length; i++) acc += a.data[i];
  return { shape: [], data: Float64Array.of(acc) };
}

export function forwardRelu(a: Tensor): Tensor {
  const out = zeros(a.shape);
  for (let i = 0; i < a.data.length; i++) {
    out.data[i] = a.data[i] > 0 ? a.data[i] : 0;
  }
  return out;
}
