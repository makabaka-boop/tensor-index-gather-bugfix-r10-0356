import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
  type Server,
} from "node:http";
import { GraphError } from "./errors.js";
import { compute } from "./engine.js";
import type { ComputeRequest } from "./graph.js";

export const DEFAULT_PORT = 8080;
const MAX_BODY_BYTES = 4 * 1024 * 1024;

const STATUS_BY_CODE: Record<GraphError["code"], number> = {
  INVALID_REQUEST: 400,
  TOO_MANY_NODES: 400,
  DUPLICATE_ID: 400,
  UNKNOWN_NODE: 400,
  BAD_CONSTANT: 400,
  UNSUPPORTED_RANK: 400,
  NON_FINITE_VALUE: 422,
  BROADCAST_INCOMPATIBLE: 422,
  MATRIX_SHAPE: 422,
  SHAPE_MISMATCH: 422,
  CYCLE: 422,
  ELEMENT_LIMIT: 422,
  INVALID_GRAD_INPUT: 400,
};

export function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

async function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(
          new GraphError(
            "INVALID_REQUEST",
            `请求体超过 ${MAX_BODY_BYTES} 字节上限`,
          ),
        );
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

export async function handler(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, { ok: true });
    return;
  }
  if (req.method !== "POST" || req.url !== "/compute") {
    sendJson(res, 404, {
      error: { code: "NOT_FOUND", message: "使用 POST /compute" },
    });
    return;
  }

  let parsed: unknown;
  try {
    const raw = await readBody(req);
    parsed = JSON.parse(raw);
  } catch (e) {
    if (e instanceof GraphError) {
      sendJson(res, 400, { error: { code: e.code, message: e.message } });
      return;
    }
    sendJson(res, 400, {
      error: { code: "INVALID_REQUEST", message: "请求体不是合法 JSON" },
    });
    return;
  }

  try {
    const result = compute(parsed as ComputeRequest);
    sendJson(res, 200, result);
  } catch (e) {
    if (e instanceof GraphError) {
      // 失败只返回错误，不返回部分输出或部分梯度
      sendJson(res, STATUS_BY_CODE[e.code], {
        error: { code: e.code, message: e.message },
      });
      return;
    }
    sendJson(res, 500, { error: { code: "INTERNAL", message: "内部错误" } });
  }
}

export function startServer(port: number = DEFAULT_PORT): Server {
  const server = createServer(handler);
  server.listen(port, () => {
    console.log(`tensor graph service listening on :${port} (POST /compute)`);
  });
  return server;
}

// 仅在直接运行该模块时启动 HTTP 服务（被测试导入时不启动）
if (
  process.argv[1] &&
  import.meta.url === new URL(`file://${process.argv[1]}`).href
) {
  startServer(Number(process.env.PORT ?? DEFAULT_PORT));
}
