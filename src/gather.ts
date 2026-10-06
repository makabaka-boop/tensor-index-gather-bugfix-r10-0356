import { GraphError } from "./errors.js";
import { Tensor, zeros } from "./tensor.js";
export function gatherShape(
  shape: number[],
  axis: unknown,
  indices: unknown,
): number[] {
  if (!Array.isArray(indices) || !indices.length)
    throw new GraphError("INVALID_REQUEST", "indices");
  return [indices.length];
}
export function gatherValue(
  input: Tensor,
  axis: number,
  indices: number[],
): Tensor {
  return {
    shape: [indices.length],
    data: Float64Array.from(indices.map((i) => input.data[i])),
  };
}
export function gatherGradient(
  input: Tensor,
  axis: number,
  indices: number[],
  grad: Tensor,
): Tensor {
  const out = zeros(input.shape);
  indices.forEach((i, j) => (out.data[i] = grad.data[j]));
  return out;
}
