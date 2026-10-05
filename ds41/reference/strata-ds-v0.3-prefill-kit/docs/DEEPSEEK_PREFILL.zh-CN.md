# Strata-DS v0.3：DeepSeek V4.1 Flash 真实几何、预填充字节预算与有界执行设计

核查日期：2026-10-04。主线：coolbho3k EXL3 3 bpw；单张 16/24 GiB 独显；128/192/256/384 GiB 主存；Engram 留 SSD。本文件覆盖 v0.2.1 第四节中尚未落到具体模型的部分。旧配方、质量门槛、Vast 权限策略保留。

**本版交付的是已经实现并测试的模型专用预算器、真实权重头部审计器、路由分组/有界执行参考、实测配置选择器，以及可在本地目标 GPU 上执行的组件探针。它不是已经跑通完整模型的 CUDA 引擎。**

## 1. 这次的具体结论

不再让 Codex 自己猜 `slot_stride` 或把 Qwen 的 384 个 ring slots 照搬进来。对所锁定的全 3-bit MUL1 TP1 格式：

| 项目 | 具体值 |
|---|---:|
| 主干层数 / 每层路由专家 / top-k | 40 / 384 / 6 |
| 主干宽度 H / 专家中间宽度 F | 5120 / 2304 |
| 每个专家完整三矩阵参数量 | 35,389,440 |
| 三比特 trellis 本体 | 13,271,040 字节 |
| 三矩阵 suh/svh 尺度 | 44,544 字节 |
| 三个 MUL1 标记 | 12 字节 |
| **完整专家源载荷** | **13,315,596 字节 = 12.698742 MiB** |
| **本设计 GPU ring slot stride（组件按 256 B 对齐）** | **13,316,352 字节 = 12.699463 MiB** |
| 一层 384 专家源载荷 | 5,113,188,864 字节 = 4.762028 GiB |
| 40 层全部路由专家源载荷 | 204,527,554,560 字节 = 190.481129 GiB |
| 单个矩阵 FP16 重建临时空间 | **23,592,960 字节 = 22.5 MiB** |

这些是固定模型配置和上游真实 EXL3 分配接口的推导值，不是以“平均 bpw”猜测。来源与核查等级见第 2 节。[PF-S01–03]

**初始实施参数现在确定如下：**

| 首轮参数 | 16 GiB GPU | 24 GiB GPU |
|---|---:|---:|
| 长提示外层 chunk | 1024 token | 2048 token |
| GPU 专家 ring 的预算 | 256 MiB | 512 MiB |
| ring 可容纳专家数 | 20 | 40 |
| ring 实际分配 | 253.989 MiB | 507.979 MiB |
| 同一专家 wave 的最多专家数 G | 4 | 8 |
| 单专家行 tile R | 64 | 64 |
| 同时处理的专家行上限 G×R | 256 | 512 |
| attention query tile / key tile | 64 / 64 | 64 / 64 |
| indexer query tile / key tile / head tile | 16 / 4096 / 4 | 16 / 4096 / 4 |
| Engram 计算子块 | 最多 256 token | 最多 256 token |
| 首轮槽位 / speculation / graph | 1 / 关闭 / eager | 1 / 关闭 / eager |

这里“确定”指**实施与首轮验证的起点**，不是冒充已经找到最快参数。选择理由是：模型真实每专家约 12.70 MiB，双 wave 最低需要 8/16 槽；20/40 槽提供有界预读余量，同时给热专家保留显存。chunk=1024/2048 使均匀路由下平均每专家分别约 16/32 行，64 行 tile 足以切割偏斜热点而不要求全部激活同时驻留。最终性能由真实测量选择，不由这张表宣布。

## 2. 证据：核到了什么，没有核到什么

### 2.1 已核查的固定源码

主计算与缓存配置来自 `tpurtell/ds41rt` 固定提交中保存的官方配置镜像，内容包括 40 层、384 专家、top-6、5120/2304、CSA2 来源和 Engram 结构。EXL3 张量形状、dtype、MUL1 标记和运算顺序来自 **量化发布方自己的定制运行时**。运行时主提交为 `1d8ac64af01c6fec87f39eb1dd526ff183615c73`。[PF-S01–03]

还检查了真实 grouped-prefill、预填充工作区修正、组件测试脚本和完整预填充实验记录。[PF-S04–07]

### 2.2 不能偷换的证据等级

本轮直接访问 Hugging Face 固定 revision 的 config/头部失败；没有下载全部 checkpoint，没有逐个扫描真实模型的 184,320 个专家组件。故 `specs/prefill/deepseek-v41.facts.json` 明确标为 **PINNED_RUNTIME_SCHEMA_AND_CONFIG_MIRROR_NOT_HF_PAYLOAD_AUDIT**，而不是假装已经拥有完整真实 offset 清单。

本包新增 `prefill audit`，在本地拿到文件后直接读取真实 safetensors 头部与 4-byte MUL1 标记，检查形状、类型、完整专家束、覆盖范围和文件身份；它会确认上述布局，或明确拒绝不匹配的文件。**这是完成来源验证的机器可执行步骤，不是让 Codex 再设计布局。**全文件 SHA-256 与数值正确性仍分开验收。

预算中约 10.074 GiB 的非专家载荷来自此前发布方 active 载荷账本减去本次精确计算的路由载荷，仍包含视觉。该数只是首轮权重预算提示，不是已经测到的强制显存。不同计算格式、常驻转换、外部 runtime、临时缓冲都可能改变实际值；真实准入必须换用本地清单和峰值记录。

### 2.3 模型与代码版本锁不变

- 量化仓库：`coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw`。
- 量化 revision：`650cae2c13aaaec303871a35301503570889c0be`。
- 量化对应基座：`df42c109f1defefcbfcedbe7d905718a12266e40`。
- Engram 无损资产使用原锁文件单独 revision/manifest，不等于量化版本混装。
- Strata 参考：v0.1.39，`6f32ec070f23ced9f50e704d854d775da52591ab`。

GGUF Q2、SAGE 的布局不能输入本版 EXL3 专用预算器。本包保留它们的原硬件配方与通用规划工具，但专用 planner 遇到这些格式会拒绝；不得把错误布局套上去输出好看的容量表。

## 3. 一个 EXL3 专家到底由什么构成

上游加载顺序是 w1（gate）、w3（up）、w2（down）。TP1 形状如下：[PF-S02,03]

| 组件 | w1 / w3 | w2 | dtype |
|---|---|---|---|
| trellis | [320,144,48] | [144,320,48] | I16 |
| suh | [5120] | [2304] | F16 |
| svh | [2304] | [5120] | F16 |
| mul1 | scalar 或 [1] | scalar 或 [1] | I32 |

每个投影：

```
5120 × 2304 × 3 / 8          = 4,423,680 B trellis
(5120 + 2304) × 2           =    14,848 B scales
MUL1 marker                =         4 B
投影载荷                    = 4,438,532 B
三个投影                    =13,315,596 B
```

MUL1 标记按 uint32 为 `0x83DCD12D`。存成有符号 I32 时不能直接用正整数比较而误拒绝。每个 Hadamard 分组需保持 128 通道边界。[PF-S03]

### 3.1 文件布局和运行布局必须分开

本设计对 12 个组件起始地址分别做 256-byte 对齐。其额外 padding 共 756 B，最终 stride 为 13,316,352 B。

这 **不是 HF 源文件中“一个连续专家块”的声明**。原文件的 gate/up/down、scale 可能处于不同文件/offset；`audit` 输出实际 `file_offset`。读取器依这些区间收集，拷贝到对齐的目标 slot。若后续生成专家 pack，使用自己的版本、校验和、offset 表，并保留原源文件身份。

256-byte GPU 对齐也不是 NVMe 的 O_DIRECT 对齐。磁盘对齐应由文件系统/运行环境核验；4 KiB 读取可能发生读放大。必须分别统计源载荷、目标 slot 字节、H2D padding、逻辑读取和物理磁盘字节。

### 3.2 不可以直接复用双 Spark 的半尺寸

上游 grouped 运行时的专家中间维度是 **1152**，因为使用 TP2；单卡 TP1 应为 **2304**。它固定的 `MAX_ROWS=1056*6` 也只是该 serving 路径的容量，不是模型允许的通用最大 chunk。[PF-S04]

源码的 FatWorkspace 按 TP2 公式恰好为 **144,466,596 B**。把同样的全路由行布局换成 TP1、2048 token，则变成 **308,484,864 B**，还没有计入继承的 thin workspace、输出、主层 residual、attention 等。这是本版测试中的固定回归，防止再次把 TP2 数字搬到单卡。

### 3.3 运算顺序也是布局契约

上游参考：gate/up 输出先转 FP32；gate 只截上界 10，up 截到 [-10,10]；`silu(gate)*up*route_weight` 转 FP16 后送 w2。不能擅自把 route_weight 从 w2 前搬到 w2 后，再声称完全相同，因为舍入位置改变。[PF-S03]

上游分布式参考可以跳过本 rank 没有的专家；**我们的 TP1 执行器必须拒绝缺失专家**。不能把“缺权重跳过”当 offload fallback。

## 4. 三层分块：outer chunk、expert wave、row tile

### 4.1 外层提示块 T

完整模型按 T 个 token 推进，控制 residual、路由、候选索引、前缀状态及请求让行边界。首轮 T=1024（16G）/2048（24G）。尾块按实际 token 数，不额外执行 padding token；小 append 17/64/128/256 token 不要膨胀成 1024/2048。

### 4.2 专家 wave G

当前层 attention 完成、router 得到真实 top-6 后，按 expert ID 稳定分组。每次最多租用 G 个专家（首轮 4 或 8）。**整个专家在该层该 chunk 内的所有行 tile 完成前不释放**；这样热点专家有 1024 行时也只从 SSD/RAM 装载一次，而不是每 64 行重新加载一次。

GPU ring 至少容纳 2G 个完整专家：一个 wave 计算，另一个 wave 完成 I/O/H2D。20/40 槽余下部分只用于受限的下一 wave 预读；没有空槽就施加背压，不另分配临时无界 tensor。

### 4.3 单专家 row tile R

一个专家分到的 token 行数可能高度偏斜。按 R=64 切分；各段仍属于同一个专家 lease。gate/up Hadamard 输入、输出和 down 输入只为当前 G×R 行分配。

因此峰值与 T 的关系分开：每 token 的 residual、路由和贡献槽位随 T 增长；重的单专家运算临时缓冲随 **G×R** 增长，而不是随全部 T×6 增长。

### 4.4 路由统计不能用“均匀分配”假定替代

仅作几何理解，平均行数为 `6T/384=T/64`：

| T | 几何平均每专家行数 |
|---:|---:|
| 64 | 1 |
| 256 | 4 |
| 512 | 8 |
| 1024 | 16 |
| 2048 | 32 |
| 4096 | 64 |

实际专家数和偏斜必须来自 router。一个 64-token append 也可能触及很多不同专家，不能因为平均每专家一行就假定只读几个专家。`routes.py` 对真实路由列表生成直方图、稳定分组、行段和 SSD/H2D 逻辑流量；不使用均匀随机公式决定执行。

### 4.5 执行伪代码

```
for chunk in prompt.chunks(T):
    engram.issue_for_known_token_ids(chunk)      # 两个 Engram 层分别保有身份
    residual = embed_and_initialize_hc(chunk)
    for layer in range(40):                    # 首轮不裁剪 decoder
        residual = attention_with_bounded_query_key_tiles(layer, residual)
        ids, route_weights = exact_router(layer, residual)  # 384 experts, top-6
        grouped = stable_group_by_expert(ids, route_weights)
        for wave in grouped.waves(G):
            next_wave.prefetch_into_free_slots_only()
            leases = await_all_complete_bundles(wave)      # 缺少任一组件即失败
            for expert in wave:
                for rows in expert.rows.tiles(R):
                    values = expert_kernel(lease, rows, original_route_weights)
                    write_contribution[token, original_topk_slot] = values
            release_only_after_last_cuda_consumer(leases)
        combine_contributions_in_declared_order()
        update_hc_and_commit_layer_state()
    commit_chunk_and_service_ready_decode_requests()
```

上游 immutable expert bank 捕获了指针，不能直接放进可淘汰缓存。本设计的 slot descriptor 要带 expert identity、generation、12 个组件地址和完成 event；已捕获旧地址的工作没有结束前，slot 不得复用。

`routes.execute_reference` 已经实现相应 CPU 调度验证：打开不超过 G 个句柄、每专家跨 tile 只打开一次、异常关闭所有 lease、按原 top-k slot 放回结果。它使用合成数值，不是 EXL3 数学内核或异步 GPU 证据。

## 5. 传输环：具体容量和资源边界

| GPU ring 预算 | 完整专家槽 | 实际分配 MiB | 未用尾部字节 |
|---|---:|---:|---:|
| 256 MiB | 20 | 253.989 | 2,108,416 |
| 512 MiB | 40 | 507.979 | 4,216,832 |
| 1024 MiB | 80 | 1015.957 | 8,433,664 |

计算全用整数除法，不能把 `floor` 改成 `ceil` 再越界分配；预算不足 2G 槽就换更小 G 的独立已验证策略，或拒绝该候选。不能 clamp 到一个最低 16 槽后超支。

主存 staging 另有 **两个与实际 GPU ring 分配同大小的有界 pinned arena**：分别承接读取和尚未完成的 H2D 消费。16G 起点约 508 MiB，24G 起点约 1016 MiB，不锁住几十/几百 GiB 全专家池。主存温专家缓存保持普通主存页或明确批准的驻留策略。

GPU 热缓存是独立区域，不能把 ring 的 20/40 个 transient 槽当作长期热点缓存重复计数。第一轮并发 decode 期间不借热缓存；需要借用时必须加入 prefill 后回填的 I/O 和显存瞬时双副本。

## 6. 预填充工作区：逐项具体分配

下面描述**要实现的 TP1 有界 executor**，并不是宣称现成 vLLM 已按此分配。代码 `planner.workspace_budget()` 返回逐项字节和各阶段峰值；不同阶段只有在最后消费者 event 完成后才能复用 arena。

### 6.1 跨阶段存活的主要对象

- mHC residual 双缓冲 `[T,4,5120]` BF16，各一份。
- 标准化层输入 `[T,5120]` FP16。
- 层输出累积 `[T,5120]` FP32。
- candidate block IDs `[T,2048]` I32。
- selected key IDs `[T,512]` I32。
- token 位置与边界元数据。

本版为了先保证稳定累加顺序，MoE 阶段另保留贡献 `[T,6,5120]` FP32。例如 T=2048 时这项 **240 MiB**。后续若用更紧凑的固定顺序增量累加，可以减少它，但属于改变执行顺序的独立方案，不能让两个公式共用一个配方标识。

### 6.2 单专家运算临时区

记 G×R 为同时活跃的专家行数：

```
两个 gate/up Hadamard 输入：G×R×5120×2×2 bytes
两个 gate/up FP32 输出：   G×R×2304×2×4 bytes
down 输入+变换输入：       G×R×2304×2×2 bytes
down 输出 FP16：           G×R×5120×2 bytes
路由/行段元数据：           明确字节项
```

默认 packed kernel 路径禁用自动整矩阵 reconstruction；算子若需要其他 workspace，必须由 capability/测量报告给出。规划器先显式预留 native 64 MiB、sort 32 MiB 作为试验额度，**这两项是设计预留，不是已测上界**。

参考 fallback 允许顺序重建一个矩阵，并另留等大的 Hadamard 临时矩阵，共 **45 MiB**。不允许三个矩阵同时重建，更不允许整层/整模型展开。若本地 LinearEXL3 实际策略比此复杂，应先据 probe 调整合同，而非忽略峰值。

### 6.3 attention 不能展开成 `[T,heads,keys,dim]`

真实配置是 64 个 query heads、512 维、128-token 滑动窗和 512 个稀疏选中项。[PF-S01]

举例 T=2048 时，错误地给所有 head 复制 640 个 KV 项会产生约 **80 GiB** 的 gather 张量。即使正确共享 head，一次 gather 全 T 也约 1.25 GiB。我们不采用这两种布局。

固定 query slab=64，key slab=64：

| attention 临时项 | 设计大小 |
|---|---:|
| query `[64,64,512]` BF16 | 4 MiB |
| 共享 KV gather `[64,64,512]` BF16，第 2 维是 keys 不是 heads | 4 MiB |
| scores `[64,64,64]` FP32 | 1 MiB |
| online output `[64,64,512]` FP32 | 8 MiB |
| attention output BF16 | 4 MiB |
| online max / LSE | 32 KiB |

实现必须做 online softmax/sink/mask 的精确语义合并，滑动窗与压缩 KV 的位置和 RoPE 都依 V4.1，不能用简化注意力代替。改变 softmax 分块会改变浮点归约，先用同量化 oracle 比较，不借“无损存储”掩盖算术差异。

### 6.4 indexer 同样使用二维分块

query tile=16，key tile=4096，head tile=4；逐块累加完整 32 个 indexer head 的打分，随后统一执行正确的候选/Top-k 规则，不能每个 head tile 独立选 Top-k 后丢失正确候选。

不创建 `[T,完整context]` 的巨大 scores，更不多乘 query head 数。保留 top-512 和 candidate top-2048-block 所需的有界归并缓冲，明确 tie-break、合法 `-Inf` mask 和无效项处理。后续索引源复用层 20 的候选领域，不能把 candidate block=8 与 row tile=64 混为一回事。

### 6.5 输出头仅对需要输出的位置计算

普通 prompt 只需要最后有效位置的 logits：129,280×4=517,120 B。teacher-forced 对照若需要全部位置，应分块导出或单独增加 head 预算；不能在同一个“最后一行”计划下突然分配 `[T,129280]`。

### 6.6 首轮工作区计算结果

| 模式 | 外层 T | G/R | 跨阶段对象 + 最高阶段工作区 |
|---|---:|---:|---:|
| 16G 初始配方 | 1024 | 4/64 | **350.465 MiB** |
| 24G 初始配方 | 2048 | 8/64 | **604.921 MiB** |

这包含上面明确列出的 native/sort 试验预留，**不包含模型权重、KV 持久状态、ring、GPU 热专家以及外部 runtime**；它们在总账中分别相加。不能单看这两个数宣称峰值显存已实测。

## 7. KV/index 与 CED：不把层数和压缩算错

配置的 KV 来源为 2/8/14/20，index query/reselection 来源为 2/8/14/20/24/28/32/36。**query 重算/选候选的次数不能直接当成重复存放 KV key 的份数**。本有界存储设计以 4 个来源 bank 管理共享 main/index keys。[PF-S01,06]

前 3 个来源压缩比为 2，来源 20 压缩比为 1。BF16 参考布局的全局数据渐近每原始 token：

```
(512 main +128 index) ×2 bytes ×(1/2+1/2+1/2+1)
= 3200 bytes/token/session
```

每个 bank 分别按源的完整压缩行数和 256-row page 向上取整；不能先平均 token 大小，再忽略尾页。源码已强调“全局未压缩行数”的 workspace 上界不能又除一次 DCP/压缩率。[PF-S06]

SWA 先保守给 40 层各自 128×512 BF16，共 5 MiB/会话。压缩器的半组状态、页表、身份/位置另计。compressed-state 本版设计给了 main/index 两个 FP32 累积通道的空间；实际算子结构若不同，需要和 adapter 清单比对，不能把它当已核完的原版 buffer carve。

`packed_890_experimental` 是另一份明确的研究布局：main 288 B/行、index 68 B/行，渐近 890 B/token。它不作为第一基线，更不宣称 stock vLLM 自动支持。先用 `reference_bf16`，后者另过数值门槛。

### CED 不被用来虚减一半预算或计算量

前 20 层 encoder / 后 20 层 decoder 的结构，并不自动授权我们在首次 prefill 只跑前 20 层。来源 20、候选选择、滑动窗和 decoder replay 必须正确接续。本版先按**完整 40 层**执行和计算权重流量；要采用 decoder 尾部重放/裁剪，新增独立算法身份和 oracle 测试。

相关原生实现的公开文档也将 bounded-decoder-replay 的近似质量验收单独保留，不能把其高吞吐直接当成本项目精确算法证据。[PF-S09]

## 8. Engram：字节少但需独立 I/O 调度

两个层每层 24 行，每行 256B FP8 +8B E8M0，因此逻辑读取是 **12,672 B/token**。2048 token 的全部两个来源为 25,952,256 B；双缓冲约 49.5 MiB，原表不常驻。

本设计为 Engram 保留 2 GiB 主存行缓存，设备每次仅处理最多 256 token 的一个 Engram 层。行哈希、压缩 tokenizer、padding、bucket 边界完全保留。prefill 入口可以提前产生已知 token 的行请求；消费前必须按 generation 收齐，取消不能提前复用 DMA 仍引用的缓冲。

Engram 随机小页和专家大块读取使用逻辑上独立的优先级队列：短 decode 查表优先，其次当前层未完成专家，再是未来已知请求的预读。不能让专家 stream-all 塞满队列后让每个 decode 等整批磁盘工作。

缓存命中后 `pread` 读到的字节不等于物理 SSD 字节；底层 page15 对齐也可能放大。全部结果必须单列逻辑 row bytes、返回的 read bytes、physical io.stat bytes、缓存命中以及不可打断 I/O 等待。

## 9. 总账与不同主存配方

本版不再沿用“ring 永远4GiB”这种模糊预留。设计总账明确分开：

```
GPU = mandatory weights + per-session state + ring
    + max(simultaneously-live phase workspace)
    + runtime allowance + safety + hot experts

Host = OS/other/safety + CPU runtime/scratch + file-page-cache allowance
     + Engram row cache + two pinned expert arenas + Engram double buffer
     + warm experts + explicit GPU-hot backup (default0)
```

初始外部 GPU runtime=1 GiB，安全余量=1 GiB；它们是尚未标定前的起始额度，实际 `cudaMemGetInfo`、NVML、allocator、native scratch 必须验证。主存系统/安全18 GiB、runtime8 GiB、文件页缓存4 GiB、Engram cache2 GiB，staging 按实际 ring 计算。

8K 上下文、1 个槽位，以发布账本的 10.074 GiB 非专家载荷作设计权重预算，得到：

| 配方 | GPU 热专家缓存 GiB | 主存温专家缓存 GiB | 仍在 SSD 的专家载荷 GiB |
|---|---:|---:|---:|
| 128 RAM /16 GPU | 3.299 | 95.469 | 91.719 |
| 128 RAM /24 GPU | 10.802 | 94.948 | 84.737 |
| 192 RAM /24 GPU | 10.802 | 158.954 | 20.735 |
| 256 RAM /16 GPU | 3.299 | 187.193 | 0.000 |
| 256 RAM /24 GPU | 10.802 | 179.690 | 0.000 |
| 384 RAM /24 GPU | 10.802 | 179.690 | 0.000 |

**这张表是实现合同下的容量推导，不是 GPU 跑出来的分配数据。**缓存含 slot padding，最后一列是源 payload，所以三列相加不会精确等于190.481GiB。非专家未知转换和更大 native workspace 会挤掉热专家；页缓存/容器限制也可能挤掉温专家。`can_launch` 因真实 adapter 尚未实现而仍为 false。

同样容量不代表同样 hit-rate。初版 EXL3 主要在 GPU 算：全主存命中只能减少 SSD I/O，不能消除 H2D。每层每chunk最坏触及384专家时仅源载荷就4.762GiB；40层最坏190.481GiB。全层流式的算法最坏值不是每个实测chunk的实际I/O。必须取真实路由、缓存和prefetch/refill来计算。

本版给了全部6个EXL3硬件配方在8K/32K/128K的具体计划JSON与CSV，剩余Q2/原生/SAGE配方文件原样保留，供各自后端继续实现。

## 10. 自动调参是可执行流程，不是“以后再说”

### 10.1 候选枚举

`candidate_plans()` 已实现整数预算筛选。默认搜索 T=256/512/1024/2048/3072/4096，ring=256/512/1024MiB；超context、少于2G槽、超host/GPU预算的候选明确标记。

不靠二分假设工作区随 T 单调；后续 kernel 的 shape specialization 可能有跳跃。计算能装下只决定候选资格，绝不直接选“最大chunk”。

真实上游在双Spark的2048→3072实验中，3072约慢2%，多耗约0.45GiB主存空间；FAT_MIN从16降到2也没有稳定增益。这些是防止盲调的证据，不是我们PC的速度预测。[PF-S05]

### 10.2 调参目标

`choose_measured()` 已实现：同模型/量化/后端/硬件/数据/缓存热度/数学策略/槽位身份，每候选至少3个独立trial，并要求质量和内存门槛先过。

目标函数：

```
本轮总时间 = prefill_ms + cache_refill_ms + 固定后续decode工作负载_ms
约束 = decode 最大未服务间隔 <= 预先指定的限额
```

它会拒绝合成数据、只有组件测量、未知SSD/H2D计数、重复trial、身份不一致或质量未过。速度最快但decode长期得不到服务的候选不会胜出。选择器不自动改部署、不创建Vast实例。

测量记录自身的真实性仍由harness、原始日志、hash和人工审查建立；JSON里写了`MODEL_PERFORMANCE_MEASURED`不是证据的密码学证明。

### 10.3 首轮运行顺序

1. 16G seed与24G seed的真实EXL3单专家/shape测试；先证明可用的packed kernel，而非先全模型下载。
2. 256/24完整模型，fixed数学路径、1K/8K、reference BF16 KV、无speculation。
3. 固定T2048，对256/512/1024MiB ring单因素比较。
4. 固定胜出的ring，再比较T512/1024/2048/3072/4096；不能省去prefill后的缓存回填和decode。
5. 192/128主存限额和真实硬件复核；现有decode时缩短chunk与让行策略独立试验。
6. 小append64/256/1024独立评估，不使用“全层所有专家”的默认流式路径。
7. 最后才扩大2/5会话，单独记录allocated slots、offered concurrency、峰值状态和公平性。

每轮先绑定 exact offer、预算、最长时长、退出策略再租机。本包云权限仍关闭。

## 11. 本包新增代码：本地可以直接使用

### 11.1 已实现的普通Python模块（无新增第三方依赖）

| 文件 | 实际功能 |
|---|---|
| `prefill/layout.py` | 真实几何、TP1组件形状、整数载荷/对齐/环容量、config校验 |
| `prefill/audit.py` | 本地多shard头部与MUL1 marker核验、专家束完整性、实际offset目录 |
| `prefill/planner.py` | 7个工作阶段、四bank KV、host/GPU分别预算、具体seed与候选枚举 |
| `prefill/routes.py` | 稳定top-k分组、专家wave/row tile、H2D/SSD逻辑流量、可运行CPU调度参考 |
| `prefill/tune.py` | 从合格、匹配且重复的完整模型试验选择配置 |
| `prefill/__main__.py` | facts/plan/matrix/audit/select命令 |
| `tools/probe_exl3_prefill_gpu.py` | 本地真实权重小组件GPU探针；GPU阶段未在本次执行 |

路径前缀为 `strata_ds_lab/`。本版没有复制AGPL源码，以上是独立的合同计算、审计与参考工具。复制或改造上游真实GPU实现时仍须遵循原许可；本包的MIT标签不覆盖上游引擎代码。

### 11.2 不需要GPU即可运行

```bash
python tools/verify_bundle.py
python -m unittest discover -s tests -v
python -m strata_ds_lab.prefill facts
python -m strata_ds_lab.prefill plan --recipe exl3-3-r128-v16
python -m strata_ds_lab.prefill plan --recipe exl3-3-r192-v24 --context 131072
python -m strata_ds_lab.prefill plan --recipe exl3-3-r128-v24 --append-tokens 64
python -m strata_ds_lab.prefill matrix --recipe exl3-3-r256-v24 > /tmp/prefill-candidates.json
```

### 11.3 真实头部核验

在本地checkpoint准备好后生成显式shard列表，先少量完整专家，再全量：

```bash
python -m strata_ds_lab.prefill audit \
  --checkpoint /你的/模型目录 \
  --files-list /你的/显式shard列表.json \
  --config /你的/模型目录/config.json \
  --require-complete > /tmp/exl3-real-bundles.json
```

列表文件是JSON字符串数组，例如本地实际存在的文件相对路径；本包不编造某个发布revision的shard文件名。若只核验代表性小组件，去掉`--require-complete`；输出会明确表示没有完整覆盖。

全量要求40×384=15,360个专家，每个12个组件；每个投影形状、dtype与标记必须符合全3bit MUL1布局。不允许丢错组件后通过。这个入口不校验全部权重payload SHA，不完成非专家语义分组，也不执行GPU。

### 11.4 实际GPU组件探针

先从锁定来源构建并审查匹配的ExLlamaV3；本包不自动pip安装/运行远端源码。用已审查的`LinearEXL3`源文件SHA256确认安装身份，随后：

```bash
python tools/probe_exl3_prefill_gpu.py --help
# 以下参数由本地实际文件和审查结果填写；不是已经验证的机器命令：
python tools/probe_exl3_prefill_gpu.py \
  --checkpoint /你的/模型目录 \
  --files-list /你的/代表性shard列表.json \
  --layer 0 --expert 0 \
  --expected-linear-source-sha256 "$REVIEWED_LINEAR_SHA256" \
  --rows 1,8,16,32,64,128,256 --repeats 5 \
  --device cuda:0 --maximum-component-gpu-mib 1024 \
  --output /tmp/exl3-component.json \
  --allow-local-gpu-component-run
```

探针确实加载选中的12个组件、哈希选中载荷、记录H2D样本、比较上游default和forced-reconstruct的输出及峰值。输入是固定随机激活，不是完整模型的捕获激活；报告明确标为`COMPONENT_MEASURED_NOT_MODEL_QUALIFIED`。GPU预算是操作后的峰值守卫，不是绝对分配上限。它不是layer bandwidth benchmark，也不会因此让完整后端implemented=true。

Native/CUDA二进制和依赖仍要单独锁定；只有LinearEXL3源hash并不能证明整个扩展的可信性。使用自己的受信构建环境，不能从未知来源随便加载.so。

## 12. Codex现在具体改哪几块

详见 `specs/prefill/backlog.json`；保留SD/U39任务ID，不整体重新编号。

1. **PF-000来源落地**：运行config和真实header/marker审计，绑定15,360专家/184,320组件，完成text-only强制张量清单和payloadhash。这里不再需要重新研究专家束尺寸。
2. **PF-001受限packed executor**：按TP1 5120/2304实现或适配已有MUL1 GPU核。明确禁用自动大矩阵重建；fallback用45MiB单投影合同。先rows1/16/32/64/128。
3. **PF-002文件→ring→wave**：接真实I/O和CUDA event，最少2G slots、singleflight、背压、专家跨row tile lease、generation失效。TP1缺专家硬失败。
4. **PF-003 attention/index slabs**：分别接64×64在线attention和16×4096索引扫描；Top-k tie、压缩源、RoPE、candidate复用单独对照。
5. **PF-004真实阶段预算**：把原来的字段输入型“预算建议”替换为本版明确buffer公式和实际adapter额外项；动态shape、第一次warmup、取消恢复测峰值。不得只降低planner报告数来消除OOM。
6. **PF-005固定与实测选择**：固定数学路径、wholemodel+质量门槛、环/块单因素实验，再导入选择器；保留fast但不合格样本，不能只保存最好一轮。

这六项是下一步GPU/模型实现；相应CPU计算、审计、路由调度和测量记录处理代码已经在包内，不需要从空接口开始。

## 13. 验收不能省掉的反例

- 13.315MB专家误按Qwen1.38MB分槽导致超分配；256MiB只给20槽，不是384槽。
- TP2中间维1152误用于TP1；三比特trellis误当通用GGUF IQ3。
- 少一个suh/svh/mul1、混入4bit专家、marker有符号比较、shard截断/越界/别名重复。
- 一个热点专家分到整个chunk，切成多个row tile却被重复读取；当前chunk未选中的专家被无条件stream-all。
- 环不足2G时被强制最小槽位放大；GPU slot被下一wave覆盖而旧CUDA核还在读。
- 后续indexer query源被误当额外KV keybank；ratio=2/DCP2重复除法；尾部未完整压缩组丢失。
- 小append被补齐成2048做冗余权重读取；所有token的head logits突然与“最后一行”合同冲突。
- 更大chunk只报告prefill更快，却忽略cache refill、后续decode和最慢会话等待。
- CPU合成测试通过被误写成GPU已跑；probe四个复用权重的调度成绩被当作完整专家带宽。

## 14. 来源索引

所有链接仅作定位；提交/源文件Git blob身份见 `specs/prefill/sources.lock.json`。

- PF-S01：配置镜像，`tpurtell/ds41rt@3067d0684a0b572d88a17d38ca3f7d10599619b4`，`rust/crates/ds41rt-loader/src/official-v41-config.json`。
- PF-S02：发布方`release/runtime/ds41/vllm_exl3.py`，blob `b29d0ea806f9f27407b87b0544e98ed88551c5ad`。
- PF-S03：`release/runtime/ds41/exl3_moe.py`，blob `bcef8163078f9fb3e583412e120153a2f21374c4`。
- PF-S04：`release/runtime/serving/spark_grouped_prefill.py`，blob `81523ddb157a225b0e54a69fdc21d551d030e223`。
- PF-S05：`release/experimental/prefill/RESULTS.md`，blob `2db1a5b556a454a0cfe3f8b597a87d6ea2eb542e`。
- PF-S06：`release/runtime/ds41/vllm_prefill_workspace.py`，blob `5efee5b471317b7771d03eee5df1b38aa9fafbed`。
- PF-S07：`release/runtime/probes/check_exl3_prefill_bench.py`，blob `ec24cef4cbdcb53adc81c05a9a8c8fd208d7f026`。
- PF-S08：`Niko1221/Strata@6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp`。
- PF-S09：`tpurtell/ds41rt@3067d0684a0b572d88a17d38ca3f7d10599619b4/docs/ds41-prefill-tail.md`。

PF-S02至07均来自`coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark@1d8ac64af01c6fec87f39eb1dd526ff183615c73`。完整网页URL由锁文件生成在`docs/PREFILL_SOURCES.md`。未直接获取的HF内容不被标成已读取；引用上游测量保留双Spark/TP2条件，不能外推到我们的PC。

## 15. 本次验证记录

本次Linux/Python 3.13.5环境运行完整旧+新CPU测试；具体数量与日志见 `results/prefill/validation-summary.json` 和 `unittest.txt`。跨shard和真实形状测试使用合成、稀疏safetensors fixture，不是预训练模型权重。GPU探针仅做语法与help检查，未执行CUDA。
