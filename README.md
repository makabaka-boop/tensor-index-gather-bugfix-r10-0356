# tensor-graph-service

接收至多 **100 个节点**的张量计算图并完成**前向求值 + 反向模式自动微分**的 TypeScript / Node 服务。

- 张量秩：标量、一维、二维（`Float64` 行主序存储）
- 算子：`const`（常量）、`add` / `mul`（逐元素，按**尾部维度**广播）、`matmul`（二维矩阵乘）、`sum`（全量求和 → 标量）、`relu`、`gather`（沿轴按索引采样，支持重复索引与负轴）
- 图可共享子节点（DAG），不得有环
- 自动推断全部形状；分配前限制前向张量元素总数（`maxElements`，默认 1000 万）
- 拒绝 NaN / 无穷值、非法维度、锯齿数组、维度 >2 的常量
- 梯度：共享节点累加所有路径；广播产生的梯度沿扩展维度求和；**ReLU 在零点导数为 0**
- 失败只返回错误码与信息，**绝不返回部分输出或部分梯度**

## 运行

```bash
npm install
npm run build          # tsc -> dist/
npm start              # 启动 HTTP 服务，默认 :8080（可用 PORT 覆盖）
npm test               # 编译并运行全部测试（node:test）
```

## HTTP API

### `POST /compute`

请求体：

```json
{
  "nodes": [
    { "id": "X", "op": "const", "value": [[1, 2], [3, 4]] },
    { "id": "b", "op": "const", "value": [10, 20] },
    { "id": "L", "op": "add", "inputs": ["X", "b"] },
    { "id": "R", "op": "relu", "inputs": ["L"] },
    { "id": "f", "op": "sum", "inputs": ["R"] }
  ],
  "outputs": ["f", "R"],
  "gradInputs": ["X", "b"],
  "maxElements": 10000000
}
```

- `nodes[].op`：
  - `const`：字段 `value` 为 number / number[] / number[][]
  - `add`、`mul`、`matmul`：`inputs` 长度为 2
  - `sum`、`relu`：`inputs` 长度为 1
  - `gather`：`inputs` 长度为 1，字段 `axis`（整数，缺省 0，支持负轴按秩归一）与 `indices`（非空整数数组，长度 1–128，元素为所选轴上合法的非负整数，可重复）。输出形状与输入同秩，被选轴长度替换为 `indices.length`
- `outputs`：要求值的节点 id（可重复，按顺序返回）。每个输出都以**全 1 种子梯度**参与反传，即对“所有输出元素之和”求导。
- `gradInputs`：可选，必须是 `const` 节点 id；返回与其同形状的梯度。不连通的输入返回全 0。
- `maxElements`：可选正整数；所有节点（含常量）形状元素数之和超过即拒绝（在任何缓冲区分配前）。

响应：`{ "outputs": [...], "grads": { "id": ... } }`，`grads` 的键与 `gradInputs` 对应。

### `GET /health` → `200 {"ok":true}`

### 错误码（响应体 `{"error":{"code","message"}}`）

| code | 含义 | HTTP |
| --- | --- | --- |
| `INVALID_REQUEST` / `DUPLICATE_ID` / `UNKNOWN_NODE` / `BAD_CONSTANT` / `INVALID_GRAD_INPUT` / `TOO_MANY_NODES` | 请求结构问题（含 gather 的空/超长 `indices`、非整数或越界索引、非法 `axis`、非一/二维输入） | 400 |
| `UNSUPPORTED_RANK` | 秩超过二维（常量） | 400 |
| `NON_FINITE_VALUE` | 常量或计算结果含 NaN / Infinity | 422 |
| `BROADCAST_INCOMPATIBLE` | 尾部维度无法广播 | 422 |
| `MATRIX_SHAPE` | matmul 输入非二维或内维不匹配 | 422 |
| `CYCLE` | 图中存在环 | 422 |
| `ELEMENT_LIMIT` | 前向张量元素总数超限 | 422 |

> JSON 规范不支持 NaN：`JSON.stringify(NaN)` 会变成 `null`，因此经 HTTP 传入的 `NaN` 常量按 `BAD_CONSTANT`(400) 拒绝；计算过程中溢出产生的 Infinity 按 `NON_FINITE_VALUE`(422) 拒绝。

## 代码结构

```
src/
  errors.ts   GraphError + 错误码
  tensor.ts   内部张量、JSON 解析/序列化、广播形状与步距、有限性校验
  graph.ts    请求校验、拓扑排序（环检测）、形状推断、元素预算
  ops.ts      前向算子
  gather.ts   gather 参数校验 / 形状推断 / 前向采样 / 反向散射
  engine.ts   前向执行 + 反向模式 AD（共享累加 / 广播降维 / ReLU 零点）
  server.ts   HTTP 入口
test/
  engine.test.ts  精确断言 + 中心有限差分（h=1e-5，避开不可导点）+ 200 随机图 fuzz
  server.test.ts  HTTP 端到端
```

## 自动微分要点

- 逆拓扑序反传；每个节点的梯度缓冲首次出现时分配，之后各条路径逐元素**累加**。
- 广播逆运算：把输出坐标按行主序映射回操作数扁平索引，补位维度与尺寸为 1 的扩展轴步距为 0，多个输出元素因而累加到同一输入位置（沿扩展轴求和）。
- 贡献缓冲一律持有独立副本，避免与其他路径共享可变缓冲造成污染。
- `matmul`：`C=A@B` ⇒ `dA = g@Bᵀ`，`dB = Aᵀ@g`。
- `sum`：标量上游梯度扩展回输入形状。
- `gather`：前向按 `indices` 顺序沿轴切片（重复索引重复取值，输出持有独立副本）；反向把第 j 个输出位置的梯度散射回 `indices[j]`，重复索引与共享分支的贡献全部**累加**（非覆盖）。



## 索引采样
计算图增加 gather：恰一个输入、一维非空 indices（1–128 项）及 axis（默认 0）。输入须为一维或二维；负轴按秩归一，索引必须是所选轴上合法的非负整数。输出与输入同秩，仅把该轴长度替换为 indices 长度，按 indices 顺序采样，重复索引合法（同一元素可多次参与前向）。反传按 indices 把贡献散射回原位置并**累加**：重复索引、共享分支、多个输出种子的贡献全部计入；索引不是可微输入。

非法 indices（空、超 128 项、非整数、负数或越界）、非法 axis（非整数或归一后越界）或非一/二维输入均在任何缓冲区分配前以 `INVALID_REQUEST`(400) 失败，不返回部分结果。该算子参与既有形状预算（输出元素数计入总数）、共享 DAG、广播、矩阵乘和统一 HTTP 失败处理。
