> **v0.2.1 设计更新：**先读 [v0.1.39 增量方案](UPSTREAM_v0.1.39_DELTA.zh-CN.md)。本文件保留原模型与实验设计；阶段内存、I/O、Responses、slot/load 与 exactness 要求以新增文档为准。原 11 个配方和模型选择不变。

# Strata-DS 多配方整体方案 v0.2

日期：2026-10-04。本文是工程设计，不是已验收引擎。内存统一用 GiB；供应商标称的 GB/MB 必须在启动前转换/实测核验。

## 1. 直接回答：EXL3 3 bpw 不是必须 256G

上一轮把 128 GiB 作为唯一目标，因而把更小的 Q2 放在首位。现在用户允许多个主存档位，应该改为质量与成本两条主线，而不是继续淘汰 3 bpw。

coolbho3k 发布的 active payload 为 215,344,840,128 bytes，即约 200.56 GiB；原生 Engram 为 202,758,032,400 bytes，约 188.83 GiB，供 SSD 查表。另有保留的 draft，不能不加区分地把整包 426.06 十进制 GB 当作 RAM 需求。[S1]

200.56 GiB 的 active 口径包含视觉；第一阶段做纯文本，应依据完整 tensor manifest 去掉确实不依赖的视觉/draft组件。这里只用发布口径作保守设计提示，不能先扣一个猜测值。Engram 全表不计入常驻权重预算，但它的行缓存、I/O staging、页缓存和计算结果要计入内存。

结论：128 GiB + SSD 可以探索；192 GiB 值得作为折中点；256 GiB 是消除“大量冷专家 I/O”干扰的优先锚点。它们都不是源项目已经验证的独显 PC 配置。

## 2. 为什么两台 Spark 不是一台 PC 的直接安装配方

上游 recipe 是双 GB10/Spark、TP2/DCP2、定制 vLLM/EXL3/Engram 与 DSpark 的组合。模型卡明确不能用 stock vLLM 或普通 ExLlamaV3 直接加载。两台 128 GB UMA 的性能不能映射成 256 GiB DDR + 24 GiB PCIe 独显的速度。[S1,S2]

最重要的差别是专家运算位置。初版 EXL3 路线设计为：打包专家保存在主存/SSD，未在 GPU 的专家分块传入 GPU 执行。不能假定现成高效 EXL3 CPU 算子已经存在。主存增大可以降低 SSD miss，但仍有 H2D 流量。后续 CPU packed EXL3 backend 若实现，必须作为另一个明确实验变体。

当前 GPU-only 缺失流量模型：

```
H_GPU = 请求专家字节中已经在 GPU 的比例
H_RAM = GPU miss 中由主存命中的字节比例
E = 每 target token 访问的打包专家字节（以真实 trace 测量）
SSD_bytes = E * (1 - H_GPU) * (1 - H_RAM)
H2D_bytes = E * (1 - H_GPU)
```

“全部专家在 RAM”只使 SSD_bytes 为零，不使 H2D_bytes 为零。将两个命中率合并成一个命中率会遮住关键瓶颈。

## 3. 产品目标与非目标

目标是输出质量、单流交互、多会话效率和整机成本的可检查折中曲线。允许 16/24 GiB GPU 配 128/192/256/384 GiB 主存；首次先 24 GiB。模型身份固定 DeepSeek V4.1 Flash，不能悄悄改成 V4 Flash、0731、裁剪专家或改变路由 top-k。

第一阶段：纯文本、单 GPU、无投机、固定各后端 KV 格式、一个会话。第二阶段：CPU/GPU/SSD 专家缓存、prefill/append 优化。第三阶段：2/5 会话 continuous batching、会话持久化。DSpark/MTP、视觉、跨机并行另行验收。

没有真实模型对照前不宣传“完成移植”；没有本目标机器测量前不发布 token/s；单位测试通过不等于模型质量通过。

## 4. 11 个配方与含义

| 配方 ID | 模型 | RAM/GPU GiB | 阶段与问题 |
|---|---|---|---|
| exl3-3-r256-v24 | EXL3 3 bpw | 256/24 | 第一锚点：先验证质量和单卡执行；降低专家 SSD miss 干扰 |
| exl3-3-r192-v24 | 同上 | 192/24 | 主要折中候选：主存减小后的真实 I/O 与延迟 |
| exl3-3-r128-v24 | 同上 | 128/24 | 原目标机器：积极验证，不提前淘汰 |
| exl3-3-r128-v16 | 同上 | 128/16 | 显存与主存双紧预算；GPU mandatory tensors 必须单独核验 |
| exl3-3-r256-v16 | 同上 | 256/16 | 对照：把 RAM miss 与小显存/H2D 问题分离 |
| q2-r128-v24 | 校准混合 Q2 | 128/24 | 成本主线对照，比较真实任务质量与性能 |
| q2-r128-v16 | 同上 | 128/16 | 成本下沿兼容档 |
| q2-r192-v24 | 同上 | 192/24 | Q2 全计算权重驻留的潜在锚点 |
| exl3-3-r384-v24 | EXL3 3 bpw | 384/24 | 扩展：检验 256 是否已足够，及大主存限额实验 |
| native-r384-v24 | 原生 MXFP4/FP8 | 384/24 | 扩展：独立数值 oracle 候选；不是已能单卡运行的原生服务 |
| sage-1.59-r128-v24 | SAGE EXL3 1.59 bpw | 128/24 | 扩展容量试验；质量/方法复现性先单独过门槛 |

Q2 的现成 DwarfStar V4.1 路径作为独立端到端对照，不能拿 GGUF 文件直接输入 EXL3 runtime。[S3,S4] SAGE 不与 coolbho3k 的 EXL3 MUL1 固定布局视为同一种格式；必须审计每个张量的 subtype。[S5]

## 5. 初始预算：都是设计值

对每档 RAM，先保留：系统/其他进程/余量18 GiB，Engram 热行2 GiB，I/O staging4 GiB，runtime/KV/元数据8 GiB，共32 GiB。GPU 非权重 reserve6 GiB，剩余空间用于 mandatory计算权重及热专家。16 GiB 卡只剩10 GiB权重额度；若 mandatory 部分大于它，应报告该配方当前不可行，不能静默降精度。

| RAM/GPU | 主存权重池上限 | GPU全部权重上限 | 独占存放的合计上限 | 对 EXL3 200.56 GiB口径的预算差额 |
|---|---:|---:|---:|---:|
| 128/16 | 96 | 10 | 106 | 94.56 GiB |
| 128/24 | 96 | 18 | 114 | 86.56 GiB |
| 192/24 | 160 | 18 | 178 | 22.56 GiB |
| 256/16 | 224 | 10 | 234 | 0 |
| 256/24 | 224 | 18 | 242 | 0 |
| 384/24 | 352 | 18 | 370 | 0 |

上表是可复算的“发布 active 包口径预算”，不是张量放置证明。text-only裁剪会改变 active值；设备共享权重、临时反量化、graph地址稳定性及双份复制也会改变实际可用值。

GPU/RAM 合计只是互斥存放成功时的上界，不是统一内存。`plan` 永远 `can_launch=false`，直到真实 header分组、后端、硬件与正确性验收落地。零预算差额不代表一定放得下，更不代表权重请求命中率100%。

128/192/256G 在云端可用性不同；允许更大宿主机限额实验，但记录为 cap-emulated。容器需同时限制匿名内存、文件页缓存和 swap；只减小应用 cache 配置，不是严格的低内存实验。

## 6. 统一结构，分离量化后端

```
recipe.json + model/asset lock + actual hardware evidence
                       |
              Runtime/Benchmark Adapter
                       |
      SourceCatalog -- Tensor/Expert/Engram Providers
                       |
     PlacementPlanner + ByteBudget + Lease/Epoch ledger
                       |
 SSD immutable blobs -> CPU warm packed experts -> GPU hot packed experts
 SSD Engram rows ----> bounded row cache/staging -> Engram compute
                       |
    GGUF Q2 / EXL3 MUL1 / native oracle (separate format-aware backends)
                       |
   layer execution / grouped prefill / decode scheduler / session state
                       |
     correctness artifacts + per-layer I/O/compute timing + trial results
```

共同拥有的是资源契约、配方、证据格式与测试机制，不是强行让所有后端共用错误的 Qwen 图或同一二进制布局。第一阶段允许两个独立 runtime，之后根据量化核的实际进展逐渐收敛。

## 7. 存储方案

所有配方：Engram 留 SSD，保留 FP8/E8M0 原始行字节；最多缓存有限行，不全表 `.clone()`。已有 page15 格式可以作为单独、带许可证和版本锁的适配器；不能把 TP2 rank0/rank1资产拼接成 TP1文件，必须重建 ownership/行映射。[S1,S2]

专家：保存完整 immutable backing，CPU和GPU只放所需专家。GPU热点专家可移除主存冗余副本，但必须在H2D完成、文件backing可恢复、消费者lease安全后再回收。峰值双副本进入 staging预算。

每个配方先给1 TiB空闲磁盘作为实验预留，不当作模型最小磁盘要求。Engram无损重排会产生暂存副本；原表与重排表同时保留时需按manifest真实计费/检查空闲空间。模型多配方顺序测，避免同时下载数TB权重。

## 8. 实验分阶段，不做盲目全排列

A. 本地无需GPU：运行本包测试、审阅锁文件/许可证/预算。当前已能完成。

B. 一张目标GPU的组件验证：只提取少量真实EXL3专家、非专家样本、Engram行；验证MUL1数学、shape、反量化、路由加权、取消/事件生命周期。能避免先花钱下载几百GB后才发现kernel不支持。

C. 256/24 EXL3锚点：完整权重、纯文本、1K上下文、无speculation、对独立同量化oracle；再8K。完整模型 adapter 未完成之前不能跳到此步。

D. 同机器资源限额：256→192→128，保持权重、GPU、线程、KV、提示不变，辨认容量引起的差异。确认cgroup/page cache可控后才记为cap实验，不替代真实128G硬件验证。

E. 物理128/24、128/16与256/16复核；Q2同类场景对照。换机器同时记录CPU/通道/NUMA/PCIe/NVMe，不把所有变化归因于RAM。

F. 只有质量和单流通过后扩到32K/128K、2/5会话与coding-agent append。再决定是否引入DSpark，不能先假定1.6x倍率。

G. 384G/原生/SAGE是扩展，不在初轮自动执行。

## 9. 怎样选最终版本

不把“最高token/s”作为唯一胜者。输出Pareto表：成功完成隐藏测试的任务数、中文质量、工具调用可靠性、warm append首token、decode p95、端到端任务时间、5会话公平性、峰值内存、磁盘流量、每成功任务成本。

EXL3128可能因为SSD miss慢，但仍值得用户自行判断质量收益；EXL3192可能成为最佳折中；EXL3256若主要受H2D限制，继续加RAM未必值得。Q2若在任务评测上显著下降，不以速度补偿掩盖质量问题。

当前没有任何一档被宣布胜出。

## 10. 本次交付与以后开发的边界

已写好并执行测试：配方规划、矩阵生成、header审计、同步FileRows参考、cache lease控制参考、计时汇总、只读机器探测与Vast方案生成。没有整模型下载、GPU模型运行、CUDA事件验证、云端租赁、GitHub写入。详细状态见 IMPLEMENTATION_STATUS.md。

主线工作不是再提高 RAM 建议，而是完成 EXL3 单卡adapter、Engram文件provider、有界专家执行与独立正确性对照。

来源标识见 SOURCES.md；包内 specs/sources.lock.json 保存本轮实际核查的版本与未解析项。
