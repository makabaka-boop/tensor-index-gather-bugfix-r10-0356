/**
 * 所有可预期的图计算失败都用 GraphError 抛出。
 * compute() 要么返回完整结果，要么抛出该错误，绝不返回部分输出或部分梯度。
 */
export type ErrorCode =
  | "INVALID_REQUEST"
  | "TOO_MANY_NODES"
  | "DUPLICATE_ID"
  | "UNKNOWN_NODE"
  | "BAD_CONSTANT"
  | "UNSUPPORTED_RANK"
  | "NON_FINITE_VALUE"
  | "BROADCAST_INCOMPATIBLE"
  | "MATRIX_SHAPE"
  | "SHAPE_MISMATCH"
  | "CYCLE"
  | "ELEMENT_LIMIT"
  | "INVALID_GRAD_INPUT";

export class GraphError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = "GraphError";
    this.code = code;
  }
}
