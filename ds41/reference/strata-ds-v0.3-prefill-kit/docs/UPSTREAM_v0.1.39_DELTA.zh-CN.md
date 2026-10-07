# Strata-DS v0.2.1 设计更新：吸收 Strata v0.1.39

核查日期：2026-10-04。交付类型：**增量设计、开发任务、实验契约及离线验证工具，不是已实现的 DeepSeek 推理引擎**。

## 0. 给负责人和本地 Codex 的结论

需要更新，但不推翻上一版。**DeepSeek V4.1 Flash、coolbho3k EXL3 3 bpw 主线、校准混合 Q2 对照、Engram 放 SSD、11 个硬件配方全部保留。**第一个完整模型基线仍为 EXL3 / 256 GiB RAM / 24 GiB 独显；先模型正确性，再逐档减少主存。

此次实质更新是：

1. 把固定主存/显存预留提升为按加载、预填充、解码、会话切换、图捕获分别计算的 **阶段内存账本**；先保住必须常驻的权重，再分配专家缓存。
2. **专家文件 I/O 策略在驻留分配完成后决定**，根据实际会读到的专家范围，而不是整个模型包大小；Engram 的 I/O 与专家策略分开。
3. **预填充块长、专家传输环、热点缓存、会话槽位共享同一预算**；以字节而不是固定专家数量分配。
4. 把 **Responses API 和实际 Codex 工具循环**提前纳入验收；先用脚本化假后端测试协议，随后接入真实模型，不等 GPU 优化全部完成才补服务接口。
5. 多会话增加“配置槽位数 S”和“实际请求数 C”两个独立变量，默认仍 S=1；建立 solo↔batch、长短请求抢占、取消、回放与惩罚项一致性测试。
6. 严格区别同字节存储正确性、固定数学路径一致性、不同算子/批量的浮点差异、量化质量与业务性能。
7. 加入 Linux/cgroup/CPU 亲和性观测；旧 GPU、SYCL、更多 GPU 和 Qwen 专用图优化不挤占首轮 EXL3 工作。

这些是下一步实现的要求。本包新代码只做离线规划与契约校验；没有下载完整模型、编译 Strata、执行 CUDA、创建云实例或修改远端仓库。

## 1. 证据固定与审查范围

| 项目 | 固定值 |
|---|---|
| 上游仓库 | `Niko1221/Strata` |
| 新标签 | `v0.1.39` |
| 新标签实际 commit | `6f32ec070f23ced9f50e704d854d775da52591ab` |
| 前一轮读取的基线 commit | `99f3dbd0b21d1401b3769e0c0d963913607f380b` |
| release 发布时间 | 2026-10-04 12:32:47 UTC，即新加坡/北京时间 20:32:47 |
| release ID | `403010090` |
| 本次对照 | release、固定标签 BATCHING、Responses、内存与专家来源、预填充源文件；预填充文件与旧基线做定点比较 |
| 本次未做 | 全仓库每行审计、上游构建、全部 PR diff 和测试执行、完整模型/GPU 验证 |

以上身份来自上游 API。[U1,U2] 不使用继续变化的 `main` 作为实验版本。标签也不是密码学意义的不可移动锁：本地 checkout 必须再次核对 SHA。下载二进制还需校验其 SHA-256；Windows Qwen 引擎不是 Linux EXL3 引擎。

注意两个易误导之处：其一，标签中的部分注释写 `0.1.39b`，而 release 对比写 `0.1.38`，以实际 SHA 和开关行为为准。其二，PR #583 的页面状态是 closed / merged=false，但其设计已在标签代码中重新实现/吸收；不能按 PR 状态认定功能不存在，也不能盲目 cherry-pick 原 PR 分支覆盖发布版。[U5,U6,U11]

原 EXL3、Engram、Q2 锁定信息没有因 Strata 更新而更改。本轮没有重新验证这些量化资产；原锁文件中的 null 仍然阻止真实执行。

## 2. 变更—移植—验收矩阵

| 上游内容 | 对 Strata-DS 的决定 | 优先级与原任务 | 不能照搬之处 |
|---|---|---|---|
| #577 文件层按实际工作集、驻留完成后重算 | 必须吸收；拆分专家 buffered/direct 与 Engram 行 I/O | P0 / SD-004、007 | 不能以全 shard 大小判断；不能把刚分配的 RAM 再扣一次 |
| #620 先加载 head/logits；#633 容器内存预检 | 必须吸收；所有强制权重与阶段峰值先预算 | P0 / SD-001、007 | head 只是一个例子；DS attention/router/shared/HC 等都需真实清单 |
| #583 byte-budget ring、chunk 联合选择 | 必须重新实现资源契约 | P0 设计、P1 加速 / SD-010 | 不抄 384 slots、Qwen blob 字节、chunk 网格和全专家流式阈值 |
| #451 Responses API / Codex 工具循环 | 提前做独立协议适配层 | P0 协议、P1 模型 E2E / SD-014 | 不用 Qwen 模板、effort 映射；不把 Base64 当加密 |
| #465/#559 真正多请求解码、#656 长短提示让行 | 吸收调度思路、重建 DS 独立状态 | P1 / SD-011、012 | 不抄 GDN/QSA/PLE 状态；不承诺默认精确一致 |
| batch/solo 的缓存和采样差异 | 增加严格能力与采样契约 | P0 设计、P1 实现 | 惩罚项不能静默失效；上游 exact 开关不是 DS 开关 |
| #646 减少 launch/host roundtrip | 后置，依赖正确 eager 后端和真实 profile | P2 / SD-002、004、017 | Qwen MTP 验证和 EXL3 kernel 不同；全层 GPU 驻留条件不自动成立 |
| #650 THP、#626 >64 CPU affinity | 先观测，受控 A/B | P1 probe / P2 调优 | 不能改 Vast 宿主机 sysctl；不认为大页必然生效 |
| #606 finite / 单 token 重复终止 | finite 校验必须，重复守卫单独标记 | P1 / SD-006、014 | 正常生成结束、启发式截断、数值错误不能混为成功 |
| #605 HDD 上 PLE 改为 RAM | 不采用 DS 全表 RAM fallback | P0 部署预检 | DS Engram 数百 GB，检测不适配应拒绝/明确降级而非悄悄全加载 |
| CUDA12 旧卡、SYCL、HIP 扩展 | 暂缓；能力矩阵留扩展位 | P2 | 上游 Qwen 编译成功不证明 EXL3 可在同卡运行 |
| #578/#598/#663 多 GPU 分工 | 暂缓，保持单卡主线 | P2 | 不导入多 GPU 工作量或社区速度到 16/24 GiB 单卡结果 |

来源：[U1,U3–U9]。本矩阵是工程决策，不是全部能力已移植。

## 3. P0：阶段内存账本和文件层决策

### 3.1 旧预算的地位改变，不改旧硬件配方

v0.2 的“主存保留 32 GiB / GPU 非权重保留 6 GiB”仍可用于粗筛机器，但**不能作为打开 N 个槽位后的准入证明**。新代码必须为同一配方生成：

```
MemoryPlan(identity, device, phase, configured_slots, context, chunk, precision):
  mandatory_weights_bytes
  persistent_state_bytes
  expert_cache_bytes
  ring_bytes
  temporary_tensor_bytes
  copy_overlap_bytes
  graph_pool_bytes
  external_runtime_bytes
  safety_margin_bytes
  available_snapshot_id
```

“configured_slots”是实际已分配的槽位，不是当前活跃请求数。如果实现懒分配，必须记录 allocated/active 的真实数量；不得看到仅 C=1 就漏掉为 S=5 预留的状态。

每个阶段报告 `peak = sum(simultaneously_live_terms)`，全流程报告各阶段最大值。不要把互斥阶段的 workspace 全相加，也不能把能同时存在的加载副本、DMA 环、旧 graph 池漏掉。层/形状切换选择不同 kernel 时，workspace 未必随 chunk 单调变化；没有证明就枚举候选形状，不盲用二分。

顺序：header/能力预检 → 分配必须的非专家张量和输出 head/logits → 分配已批准槽位及有界工作区 → 以剩余空间分配专家缓存 → 报告 effective 资源 → warmup/capture 复核峰值。缺任一强制大小时返回 UNKNOWN/BLOCKED，不先填 0。[U1,U8]

### 3.2 专家工作集不是模型文件大小

上游 #577 的错误是用全部文件大小、过早判断绕过页缓存，导致 RAM 本来能缓存的重填充反复读取 SSD；修复后只考虑主存副本以外实际会读的专家，并在主存副本建立后重算。[U1,U7,U8]

我们的设计把输入语义写死：

```
available_after_residency = 已完成本轮驻留分配之后的可用主存观测
future_reserve = 尚未分配的 session/staging/workspace 等 + 余量
file_expert_working_set = 该阶段可能从文件取的专家区间的去重字节总和
room = max(0, available_after_residency - future_reserve)
```

`room >= file_expert_working_set` 只是 buffered 候选，不是能永久缓存的保证。否则把 direct 作为候选，前提是对齐、文件系统、容器权限和读取实现均合格；未实现时保持 bounded buffered + 内存约束，不因追求 direct 要求 privileged。

关键细节：

- 不再次减去已经反映在 available_after_residency 中的主存专家缓存。
- 不把 SSD-only Engram 全表加进“专家文件工作集”，但 Engram 热行、页缓存、实际 I/O 与两队列互扰必须进入总账本/性能观测。
- **GPU 热专家也可能需要文件回填**：若 prefill 借用其槽位，且 RAM 没有副本，则归还后的补载必须算进该阶段读集，不能机械地减掉“所有 GPU 专家”。
- 使用固定 tensor ranges 计算 union，不能重复计 gate/up/down 共享片段，也不能按平均 bpw 猜文件 offset。
- 实际 `memory.current`、干净文件页可回收量、祖先 cgroup 限额、swap 与系统可用量共同观测。可回收页是估计，不当成已预留现金一样无条件花掉。[U8,U12]
- 每个 trial 锁定策略；运行中切换 buffered/direct 必须排空 I/O，并发布新的策略代际和原因，不静默变化。

至少提供 `auto_advice`、`buffered`、`direct` 三种**项目内部设计值**；它们不是当前引擎已实现的启动参数。新离线函数只输出试验候选，`can_launch` 始终 false。

### 3.3 必须覆盖的回归

全模型 shard 含巨型 Engram 而专家冷集很小；驻留构建前后 available 变化；GPU 热槽位借出/归还；脏页不可回收；cgroup 实际额度小于宿主机；元数据 unknown；另一个进程导致可用量下降。测试至少检查：不误全量 clone、不双扣 RAM、不虚报 direct、内存压力不能改变模型输出。

## 4. P0/P1：以字节统一 prefill、ring、cache、slots

Strata 标签中新增按 pack 最大 expert blob 换算 ring 的逻辑，但预算默认值仍从 Qwen Q2_0 的测量导出。其预填充代码还有 N=2560、K=10、NE=512 等架构常量，因此不能直接接 EXL3 权重。[U5,U6]

DeepSeek 实现的 slot stride 必须来自**完整可执行 expert bundle**：gate/up/down code、尺度、旋转/置换、校正数据、padding，以及所选 kernel 的布局约束。单独的 dequant scratch 进入另一个阶段项；不能反量化整个专家池。

```
ring_slots = floor(ring_budget_bytes / exact_aligned_slot_stride)
ring_allocated_bytes = ring_slots * exact_aligned_slot_stride
```

最小在途数量由执行算法要求，而不是写死 16。如果预算只能放 0 或少于算法所需，报告不合格或选择已验证的低内存算法，**不能通过 clamp(min_slots) 超支分配**。

联合选择应评估 `(chunk_tokens, ring_bytes, cache_loan_bytes, allocated_slots)`；满足 mandatory/state/graph/scratch/安全余量之后，按真实测量优化 TTFT 与持续 decode 延迟。不是“最大 chunk 一定最快”，也不是“最深 ring 一定最快”。大幅降低热点缓存可能让 prefill 后的长串 decode 得不偿失。

建议首轮候选 ring 256/512/1024 MiB，只用于实验搜索范围，不作为最低或最优值；每个候选先精确验算。chunk 从后端支持列表中选取，首轮保持数学路径固定。短 append 64/256 token 不应为了大吞吐路径多读一整层的冷专家；先观测路由去重，再决定 stream-all。Qwen 的 stream-all 阈值不是 DeepSeek 阈值。

### 借用显存的生命周期

`ACTIVE_DECODE` → 请求可借额度 → 等待最后消费 event → 租出 slot → prefill DMA/compute → 等待最后消费 event → 按 backing 恢复需要的 expert → 发布新 residency generation → 解码继续。全过程每个指针只能代表经过验证的那个 expert。

当 decode 正在运行时默认 `prefill_borrow=false`，先得到可复现实验基线。后续借用优化独立开关：借用前冻结对应 slot，保护 lease；取消 prompt 不等于 DMA 已结束。记录 `loan_bytes`、`refill_bytes`、等待与隐藏时间。无此证据不宣布 overlap 收益。

必须 A/B：旧固定策略 / byte ring / byte ring+chunk 联合；不同量化各自验证。上游 release 已报告一些配置不提速甚至下降，所以不能规定“打开新开关即通过性能门槛”。[U1]

## 5. P1：并发不再只有“concurrency=5”

### 5.1 三个数必须分开

- `requested_slots`：配置希望分配的上限。
- `effective_slots`：引擎真正分配/报告的上限。
- `offered_concurrency`：客户端同时或按到达计划提交的请求数。

上游会在显存不够时缩减槽位并告知服务器。[U3] 我们面向严格实验：若 requested != effective，trial 标为 `CAPACITY_MISMATCH`，不能把“请求 5、实际 1”发布成五路实验；面向最终产品可以协商后排队，但状态接口必须明示。

适配器需先规范化计数：例如上游“未启用 batch/零 batch slot”应映射为 effective serial capacity=1，并保留原始报告值，不能把它误解为无法处理请求。

默认仍 S=1。所有 N=2/5 配置先通过阶段内存检查，不能照用上游 Qwen 0.56 GiB/slot、最多 8 行、半数专家驻显存等启发式常量。DeepSeek 每 slot 的 KV/index/SWA/压缩器/Engram 历史必须由实际后端给出。S=5,C=1 用来测闲置槽位挤占专家缓存的代价。

### 5.2 调度契约

```
QUEUED -> PREFILLING -> DECODING_SOLO <-> DECODING_BATCH -> FINISHED
                 ↕                    ↕
          YIELDED/PARKED          CANCELLING -> CANCELLED
```

每次转移绑定 owner、request_id、generation、expected_position 和 last-emitted token。模型状态 copy 成功不等于 SSE 已发送成功，两者要有独立 watermark。连接中断只终止本请求，不能把下一个请求读到旧队列 token；slot 复用等待 I/O/GPU/event/输出回调全部解除引用。

长 prompt 在可恢复块边界让行；短请求可插队但要做 aging/有界配额，持续短请求不能饿死长请求。记录最长不可打断 chunk 的时间，不能仅记录 chunk token 数。上游的“半个 chunk 时间给 decode”“短于一半”仅是一个策略，不是 DS 合理延迟 SLO。[U3]

prefill 每次只处理一个 admission 可作为起步策略，但已有 decode 必须在边界获得运行机会。先不叠加 speculation。以后启用 DSpark 时，solo/batch 的 draft 状态和拒绝回滚单独验收；不要照搬上游 MTP 的接受率经验。

### 5.3 惩罚和随机数必须一致

上游 batch 暂不应用 repetition/frequency/presence penalties，默认数值路径也不保证与 solo 完全相同。[U3] 我们不能静默丢这些参数：

- 已实现：按各自会话历史应用，并验证 solo/batch 一致。
- 未实现：在请求入队前明确拒绝，或者显式选择串行执行并报告 effective mode；不能忽略。
- 采样 RNG 必须与 request seed、逻辑 token 位置绑定，不由 batch 行号或调度次数决定。请求换 slot 不重置随机状态；失败重试不多消费随机数。
- 基线 speculation=false；temperature=0 用于定位，不声称它消除浮点/路由归约差异。

### 5.4 相同输出声明的正确范围

上游 exact 测试需要固定专家层、固定 CPU/GPU 选择和特定 CPU multi-token 数学路径；默认设置下作者也报告过输出分歧。[U3]

DS 验收拆三层：

1. **bit-exact storage**：文件、行/尺度、缓存返回字节必须完全相同，无容差。
2. **controlled-math parity**：固定 kernel、router、缓存/placement 与采样状态，检验 solo/batch/迁移/取消恢复；能相同的必须相同。
3. **optimized-math qualification**：不同分组/GEMM/设备归约造成差别时，使用独立同量化 oracle 的 teacher-forced logits 和预先规定阈值，接着跑业务质量；不能发现误差后放宽门槛。

Strata 的 `STRATA_IQ_MT_MIN`、`--pcie-frac` 等是 Qwen 引擎开关。这里只把它们作为上游 exact 条件记录，不给 DS 构造不存在的同名参数。

## 6. P0 协议 / P1 E2E：Responses 与 Codex

### 6.1 开发工具与被测客户端是两个角色

本地 Codex 可以先用现有商业模型开发 strata-ds；不要求它先靠未完成的 DeepSeek 服务运行。与此同时，一个固定版本的 Codex CLI 将作为**被测客户端**，通过独立测试配置连接未来的 `/v1/responses`。不要修改用户现有默认 provider 或让不稳定后端接管开发会话。

服务结构：

```
Responses / Chat Completions / Anthropic adapters
                      ↓
 CanonicalConversationIR + CapabilityValidator
                      ↓
 DeepSeekTemplateEncoder (fixed version/digest)
                      ↓
 RequestScheduler -> EXL3 or Q2 RuntimeBridge
                      ↓
 ModelEvents -> typed SSE / non-streaming assembler
```

可以审阅并保留许可地移植上游事件装配和测试思路；不能直接绑定 `serve.frontend.effort_kwargs`、Qwen 模板或其私有 tool result 顺序约定。发布中的 Codex 0.160.0 是上游一次测试条件，不是我们客户端版本已验证；本地记录实际 `codex --version`。[U1,U9,U10]

### 6.2 首版协议范围

先做 stateless full-history `input`、`store:false`、instructions/developer/message、function_call 与 call_id 关联的 tool output、namespace、自定义文本工具、reasoning effort 映射、text.format、HTTP SSE、终态与 usage。每项都需要 capability 声明；未实现明确拒绝。

这里的 stateless 是 **不提供 previous_response_id 的服务端 response store**，不是不允许内部 prefix cache。缓存必须绑定 owner 和模型/模板/tokenizer/量化/KV 身份，并对实际 token 前缀验证，不能信任客户端 cache key 就复用状态。

`previous_response_id`、服务器 conversation store、background、hosted tools、图像与 reasoning summaries 不在初版。显式 `store:true` 也不能假装持久化成功，应拒绝或取得客户端明确的 stateless 协商；响应中的 store/status 必须如实反映。不要像上游那样静默丢弃 hosted/未知工具；返回明确 400 及 unsupported_parameter，或先协商客户端能力。工具集合、name 和 call_id 不能因为 flatten 或 reorder 丢语义。

JSON schema 有两件不同的事：**接受 schema 字段**不等于**生成受硬约束保证的结果**。strict/grammar 未有对应执行能力时明确拒绝；如选择输出后校验路线，应标记 validate-only，失败发 terminal error，不把无效 JSON 当正常完成。绝不把“加进提示词”称为 grammar constraint。

### 6.3 encrypted_content 与信息安全

上游源码明说自己的 `encrypted_content` 为 Base64 编码，没有加密；这是兼容回放方案，不提供机密性。[U9]

我们不复制这一命名下的安全假设。首个互操作实现可选择经过被测客户端验证的明文 reasoning replay，并明确它是模型输出；若客户端确实要求 opaque replay，需使用标准库之外经过维护的 AEAD 实现、受管理密钥、nonce、版本、模型/租户绑定和轮换规则。**本包没有实现密码学或 AEAD**；相关能力在通过测试前保持 disabled/unsupported，不能输出假 encrypted 字段。

服务端不能把回放内容作为受信系统指令；这是用户提供的会话输入，即使附带合法密文也仅证明来源/完整性，不提升指令权限。日志和请求监控默认不记录完整提示和思考文本；测试数据公开/合成。

### 6.4 协议测试先于真实 GPU

使用 deterministic scripted backend 测试，不冒充模型质量：

- response.created → in_progress → output_item.added → delta/done → item.done → 单一 terminal；单调 sequence_number、稳定 item_id/call_id。
- 中文 UTF-8、任意网络分片、工具参数增量、多个工具结果逆序返回、缺失/重复 call_id 的拒绝策略。
- 客户端断流后取消、server error、超时、输出 token 上限；只发一个正确终态，不吞错误返回 completed。
- 流与非流的最终 items/usage 一致；SSE chunk 不是 token。
- 重复工具请求重放不由模型服务器执行外部工具。测试客户端工具在隔离目录运行，副作用由 harness 管理。
- 初次请求、保留 thinking 的 follow-up、客户端删减 reasoning 的 follow-up：能证明 exact-prefix 才命中缓存，否则安全重算。
- loopback 默认、API key 校验、Host/Origin/CORS、监控接口权限。云端通过用户批准的隧道或受保护入口，不公开裸服务。

真实模型通过短序列后，补“读文件→改文件→执行隐藏测试→处理失败→再次修改”的实际 Codex 工具链验收。上游 ~96% prefix reuse 不作为 DS 的目标值或实测。[U1,U9]

## 7. 后置优化和不采用的自动回退

### 7.1 减少 host roundtrip / graph

#646 的优化方向值得借鉴，但 上游 zero-doorbell 的关键前提是该层全部专家均已在 GPU，而不是“主存足够大”。EXL3 / 24 GiB 卡不能根据 256 GiB RAM 就宣称满足。[U1]

先 profile eager 的 SSD 等待、H2D、GEMM、route、CPU launch、attention，再实现特定子图。要求：所有 graph 指针生命周期稳定，captured buffers 不被错误复用；动态 residency table 带 generation；确切 GPU-ready 数据和 events 已满足才能走快路径，否则回正确 fallback。测试覆盖 batch row 数和每 token top-k 的乘积边界，不能复制 Qwen 固定 80/128 个条目的计划数组。[U13]

### 7.2 Linux 和 CPU

采集 cpuset.cpus.effective、CPU affinity、NUMA 距离、有效线程数、THP/AnonHugePages、mlock/pinned 实际结果。THP only opt-in A/B，失败退普通页并报告；不修改共享宿主机配置，也不把 mlock 成功等同 CUDA pin 成功。

Qwen IQ CPU gather 对 GGUF 可能有参考价值，对初版 GPU-only EXL3 不直接消除 H2D。V100/Pascal 的 CUDA12、SYCL、AMD、跨机与多 GPU 另列后续能力，不能因便宜先让首轮算子适配扩散。[U1]

### 7.3 finite 与重复终止

对意外的 NaN/+Inf 以及未经该算子契约允许的 -Inf，第一时间记录 layer/expert/位置并失败，不用无条件 nan_to_num 掩盖错误。合法 attention/token mask 的 -Inf 必须单独声明，不能被数值守卫误判。启发式重复终止另列 `heuristic_repeat_stop`，记录阈值、是否启用和截断位置。合法长重复文本要有负例；模型在此终止不能算“完成业务任务”。上游 256 次单 token 只是参考默认，不是 DS 已认可产品策略。[U1]

### 7.4 不采用 HDD→整表 RAM 的回退

DeepSeek 首版部署要求符合目标 I/O 的本地 SSD。检测 HDD、未知远程存储或不支持所需 I/O 模式时给出原因，允许明确批准的低速实验但不改变模型数据驻留。不能像小一些的 Qwen PLE 策略那样悄悄把全 Engram 读入主存。[U1]

## 8. 新的实验覆盖：不扩张硬件数量，先扩张执行证据

硬件 recipes 文件逐字保留。新增 runtime profiles，是**未来实现要接受的实验契约**，不是 Strata 或当前 strata-ds 可直接读取的运行配置。

| profile | S 槽位 | C 请求 | 目的 |
|---|---:|---:|---|
| serial-fixed | 1 | 1 | 固定 chunk/ring 数学路径基线 |
| serial-byte | 1 | 1 | byte ring + 联合预算对照 |
| slots2-idle | 2 | 1 | 仅槽位预留造成的缓存代价 |
| slots2-load | 2 | 2 | 两路吞吐与尾延迟 |
| slots2-queue | 2 | 5 | 超出容量排队，真实 effective 容量 |
| slots5-idle | 5 | 1 | 五槽位空闲代价 |
| slots5-load | 5 | 5 | 真 batch 去重、带宽与公平性 |
| long-short-yield | 2 | 3 | 一路背景 decode + 后到长/短请求交错的预定到达序列 |

long-short-yield 的 C=3 表示同时最多三个未完成请求：开始一个 decode、加入长 prefill，再加入短请求；不是“同时三份模型状态都塞进两槽位”。具体调度可能需要把被让出的 prefill 状态 park 到 host/SSD；这份存储和恢复峰值必须纳入计划，容量不足则显式排队。

初轮只有 serial-fixed/serial-byte 在 EXL3 256/24、1K/8K 上有资格启动。batch profiles 由真实 batch 和状态迁移门槛解锁，再测试 128/24 和 Q2 对照。不要把 11×8×上下文×重复一口气租完。

3 种 I/O 变体、3 种 ring 大小、THP 和 graph 均为单因素 ablation，不与全部维度盲目全排列。每轮至少 3 次试验用于初筛，最终候选交叉随机顺序、更多重复和公共任务/隐藏任务拆分；不能拿单次极值作配置推荐。

每次报告新增：requested/effective/allocated slots、active rows、是否真实 batch、chunk、ring bytes/slots/stride、cache loan/refill、GPU hit、RAM conditional hit、逻辑/物理 SSD bytes、H2D bytes、queue wait、preemption wait、最大未服务间隔、TTFT-thinking/answer、TPOT 分位、末请求 TTFT、end-to-end task success/cost。未拿到的指标写 unknown，不能用 0。

还要给出 S=1,C=1、S>1,C=1 和 S>1,C>1 的同表结果。改善最后请求的开始时间、但总吞吐降低，是可能合理的产品取舍；不能只挑其中一个数字。[U3]

## 9. 本地 Codex 的实施顺序和 PR 拆分

完整依赖见 `specs/upstream-v0139.backlog.json`。保留原 SD-000…017，并追加 U39-000…011 和明确的 prerequisite augmentation；不是把旧任务推倒重编号。

| 批次 | 主任务 | 完成证据 |
|---|---|---|
| A：离线设计与接口 | U39-000/001/004；SD-000/001 | 标签锁、许可、真实 tensor schema；预算和 Responses 假后端测试 |
| B：单专家与 Engram | SD-002/003/004 + U39-002 | 目标卡真实 packed expert 与原始行 byte 对照；真实事件生命周期 |
| C：全模型与准入 | SD-005/006/007 + U39-005 | 256/24 短序列独立 oracle；phase peaks；受保护 Responses 接口 |
| D：低内存与 prefill | SD-008/010 + U39-003 | 256→192→128；H2D/SSD 分开；fixed/byte 联合预算 A/B |
| E：真实 Agent 与多会话 | SD-011/012/014 + U39-006/007/009 | 工具循环、采样/惩罚、迁移、断流/取消、长短公平性 |
| F：最终配方选择 | U39-008/010/011，按授权逐项 | CPU/THP、可选 graph、统一结果与费用；不把估计当实测 |

U39-008/010 是可选调优，不能阻止正确性基线开展；U39-011 的主表也必须能在它们 disabled 时完成。第一轮保留无 speculation、无 vision、无多 GPU。

每个 PR：小范围变更、来源文件与许可、输入/输出身份、测试命令和原始结果、实际 GPU 型号、未覆盖项、回滚开关。二进制/模型/客户端分别锁定版本。不要用 `--allow-unvalidated-checkpoint` 的存在替代验收，也不要提前把 `BACKENDS.implemented` 改为 true。

### 与旧包的集成方式

本更新包已包含旧 v0.2 的离线工具和 11 个 recipe，不需要手工把多份 zip 混合。原 v0.1 engine 仍另行保留。若本地已经修改旧设计目录，先比较差异再合并；不直接覆盖用户代码或已有 manifest。

旧 `docs/OVERALL_DESIGN.zh-CN.md` 等作为详细模型设计保留；在内存准入、IO 策略、服务前置、槽位/负载分离和 exactness 声明上，本增量文档优先。新 profile 不替代 quant/hardware recipe。`results/` 的旧记录是历史，当前验证位于 `results/upgrade0139/`。

## 10. 本包能执行什么、不能证明什么

已提供离线数学/契约工具：阶段峰值计算、post-residency I/O 候选、按字节 ring 容量、试验槽位校验、任务 DAG 验证、runtime profile 矩阵生成。它们不分配 CUDA、不做 mmap 性能测试、不创建 HTTP 服务、不执行工具或租机。设计矩阵所有项保持 `can_launch=false`。

这些工具输入中的字节数由调用方提供；例如 ring slot stride 并非已从真实 EXL3 manifest 自动生成。通过数学测试只能证明所给预算的算术与边界处理，不证明漏项不存在或模型能运行。原 75 个测试继续保留，新测试独立统计。本轮在 Python 3.13.5 上执行了原 75 + 新 52 = 127 个 unittest 案例，全部通过、0 跳过；C++17 只做接口语法检查。最终压缩包在独立目录解压后，也重复通过同套测试、设计契约、C++17 接口语法和文件 hash 检查。

本次并未将 Strata 源码复制进入包内；源文件只以路径、blob SHA 和链接作为阅读依据。后续复制源码时必须保留许可，尤其 EXL3/Engram 集成中 AGPL 与 MIT 不可混标。

## 11. 固定来源和阅读入口

U1. Strata v0.1.39 release（发布时间、发布声明、硬件条件和限制）：
https://github.com/Niko1221/Strata/releases/tag/v0.1.39

U2. 标签对应 commit API：
https://api.github.com/repos/Niko1221/Strata/git/ref/tags/v0.1.39

U3. 该标签 BATCHING（exact 条件、惩罚限制、slots/solo、让行和测试工具）：
https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/BATCHING.md

U4. 原 v0.2 完整设计和工具（本更新包中保留的 docs/、specs/、recipes/）。用作现有方案证据，不作为上游模型正确性证据。

U5. 新预填充源码（本次读到第 1–190 行，含 ring 逻辑和架构常量）：
https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp

U6. 旧基线预填充源码（本次比较第 110–175 行）：
https://github.com/Niko1221/Strata/blob/99f3dbd0b21d1401b3769e0c0d963913607f380b/src/prefill/prefill.cpp

U7. 内存 file_cache_keeps 契约：
https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/platform/memory.hpp

U8. 专家源（本次读第 1–180 行，并定位 #577 注释；不是整文件审计）：
https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/expert_source.cpp

U9. Responses（本次读第 1–190、200–355 行；不是完整服务安全审计）：
https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/responses.py

U10. Codex 官方 provider/config 文档（真实实验时再次检查固定客户端版本）：
https://developers.openai.com/codex/config-reference/

U11. #583 PR 元数据与设计；closed/merged=false，以标签内实现为准：
https://github.com/Niko1221/Strata/pull/583

U12. Linux cgroup v2 文档（内存/页缓存/限额）：
https://docs.kernel.org/admin-guide/cgroup-v2.html

U13. verify 计划数组边界（本次 search 返回的 #646 片段，需本地进一步读完整 kernel）：
https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/verify_kernels.cu

所有模型/硬件性能数值引用上游时必须保留原条件。本文没有把 Qwen 的增速或双 Spark 的 EXL3 成绩转换成 DS 单卡预测。
