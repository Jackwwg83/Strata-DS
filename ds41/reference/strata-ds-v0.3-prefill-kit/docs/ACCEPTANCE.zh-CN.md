> **v0.2.1 设计更新：**先读 [v0.1.39 增量方案](UPSTREAM_v0.1.39_DELTA.zh-CN.md)。本文件保留原模型与实验设计；阶段内存、I/O、Responses、slot/load 与 exactness 要求以新增文档为准。原 11 个配方和模型选择不变。

# 验收与实验设计

## 1. 两道正确性门槛，不能合并

**量化质量门槛**：原生发布权重 → EXL3 3 bpw / Q2 / SAGE，在同一个base revision、tokenizer、模板、样本和生成策略下比较。低比特带来的误差在此评估。

**实现与offload门槛**：同一量化、相同张量 → 新后端、新缓存、SSD读取、snapshot、batch之后的结果。不能拿“量化本来就有误差”解释offload引入的错误。

同一token位置的teacher-forced logits是诊断工具；top1一致率不是代码pass率。不同作者的不同数据集KL不能直接横向排名。EXL3上游与原生对照是选择它为质量主线的证据，不替代本场景质量门槛。[S1]

## 2. 证据链

每次trial强制记录：recipe SHA256、model/base revision、quant manifest、Engram语义/布局manifest、tokenizer/template、engine commit+dirty diff hash、镜像digest、driver/runtime/CUDA版本、GPU型号/UUID/实际free bytes、CPU型号/物理和有效核数/cpuset/NUMA、RAM/cgroup祖先限制、NVMe挂载/文件系统、线程与PCIe/H2D实测。

结果分为 DESIGN_ESTIMATE、SYNTHETIC_TEST、COMPONENT_MEASURED、MODEL_CORRECTNESS_MEASURED、MODEL_PERFORMANCE_MEASURED。缺证据的项写null/unknown，不写0。模型文件hash通过不能自动升级为质量通过。

两类clock：客户端端到端使用单一控制机时钟；设备内部用同一worker的monotonic时钟+CUDA events。跨机不同monotonic起点不能直接相减。engine token event不是SSE chunk，chunk中可能含多个token。

## 3. 存储与内核单元验收

1. 原始与重排行bytes/scale bit-exact，覆盖首末row、跨page、重复row、pad、坏offset、短读、corrupt manifest、TP owner错误；不只看dequant近似输出。
2. EXL3真实单expert：完整MUL1组件、gate/up/down、不同batch，和独立上游参考比较；包含实际model形状及异常shape拒绝。把量化数学与SwiGLU/clipping/weighting分别测试。
3. CUDA H2D/event：模拟迟到与取消，stage源和device slot在所有消费者完成前不释放；新generation不接收旧completion；多stream真实事件trace，不只CPU布尔值。
4. 内存：稳态、加载峰值、prefill峰值、第一次graph capture、context增长、5会话都记录；检查显存和host/cgroup，不只PyTorch allocated。
5. 有限缓存未命中时必须取回正确专家，不能跳过、换专家、改top-k或读未就绪slot。

本包75项离线测试仅覆盖其中CPU参考组件的子集，不能据此宣布以上GPU验收完成。

## 4. 模型数值边界

先1/2/3/127/128/129 token、压缩组奇偶、候选block边界、KV/index复用层、Engram首token/归一化、中文和标点，再1K/8K。先eager后graph；先无热缓存后热缓存；先1session后2session交错。

独立oracle可以来自经过固定的上游EXL3或原生runtime。不同TP/矩阵核可能有浮点归约差异，先记录oracle自身重复误差，再设阈值。不可为了通过而在看到结果后放宽阈值。相同storage bytes能bit-exact的部分不接受浮点容差。

失败时保存最早错误layer、row/专家身份、输入token位置、tensor hash、上游与本地logits，不泄露生产客户prompt。

## 5. 性能工作负载

| 家族 | 输入与状态 | 输出/并发 | 衡量目标 |
|---|---|---|---|
| 简短交互 | 1K/8K，model cold与warm分离 | 256，1人 | 基本交互与加载成本 |
| 新仓库阅读 | 8K/32K/128K新prefix | 256，1人 | 真正prefill而非cache复用 |
| Coding-agent工具追加 | 固定8K/32K前缀后追加64/256/1024 tokens | 128或256，1/2/5人 | 工具返回后的TTFT、前缀命中与长链延迟 |
| 跨领域热点切换 | 代码→中文→长文摘要→代码 | 256，1人 | expert cache抖动，不只稳态最佳命中 |
| 并发突发 | 2/5个独立session，同时ready | 256每人 | aggregate goodput、最慢请求、公平性 |
| staggered agent | 用户时间错开、短工具请求 | 多轮 | 长prefill不能饿死decode |
| 恢复/取消 | 压缩半组、KV边界、中途取消/重试 | 固定tokens | 正确状态与恢复时延 |

prompt长度由锁定tokenizer/模板实际生成并复核，不以字符数估算。模型max context、allocated context、有效prompt长度、输出tokens分别记录。`matrix.py`输出的prompt_target是待实现的精确token目标，不表示已生成了这些数据。

性能replay可以使用公开/合成代码与确定性工具结果，以隔离模型时间；真正Agent评测必须另跑可执行工具、隐藏测试与失败恢复。离线replay不能被称为Agent任务成功率。

## 6. 四种热度不可混淆

- model cold：进程与权重初始化，文件缓存状态另外记录。
- kernel warm/new prefix：JIT/graph已warm，但没有该prompt的prefix cache。
- prefix warm/agent append：同prefix状态可复用。
- storage cold：底层文件页未在OS cache；需证明，不因重启进程就假定。

禁止在Vast共享宿主机全局drop_caches、重置GPU或改其他租户服务。容器内posix_fadvise只是提示，仍需物理I/O证据。不能制造privileged容器来得到“干净成绩”。

大主存机器的应用cache限额不等于真实128G：OS可能在页缓存中持有全部冷专家。确认cgroup、私有文件/缓存charge与io.stat；无法确认时只报告“应用缓存限制实验”。同一GPU设allocator上限也不是16G显卡，因为计算和带宽没变。

## 7. 指标定义

TTFT = 第一枚真实输出token时间 - 客户端提交时间。另分queue、prefill、decode/首token处理时间。存在thinking时分别报告首thinking token、首可见答案token与答案完成时间。

TPOT分位使用相邻真实token事件差值。Aggregate goodput = 成功请求输出tokens总数 / trial从最早提交到最后结束的墙钟时间；错误与取消消耗的时间仍在分母。不能加总每用户token/s冒充总吞吐。per-session rates、slowest request与failure rate单列。

SSD专家bytes、SSD Engram bytes、逻辑read bytes、物理device bytes、H2D packed bytes、activation bytes分别统计。命中率按请求bytes，不按缓存容量比例。GPU hit，RAM conditional hit，disk miss分别输出。缓存miss次数相同但expert大小不同仍会有不同流量。

成本记录实际租用wall time（含下载/JIT/warmup/失败）、磁盘、出入网；每成功任务成本 = 本trial完整成本 / 成功任务数，无成功任务时null而不是0。云端实际报价和账单不同分别保留。

## 8. 公平比较与统计

主存sweep优先同一个GPU/CPU/NVMe的受控cap，再换真实物理档复核。量化比较允许不同backend作为“整套产品配方”比较，但不能把性能差全部归因于量化位数。单量化的offload比较保持backend/engine/kv/模板不变。

screen同条件至少3次独立trial，报告所有值与median；最终候选增加重复、随机交叉运行顺序、记录时序降温/共享负载，报告波动范围。p99样本少时注明不稳定，不把3个请求的p99当可靠SLO。

缓存profile建立集、参数调优集与最终验收集分开；最终阶段才揭示hidden tests。公开任务明确dataset version、license、runner和timeout。量化测试使用同一基座的固定prompt；不得用不可复现的API“最新模型”作为混杂基线。

## 9. 质量任务与建议决策门槛

质量集合至少包含：中文指令/长文事实，代码修复隐藏测试，工具参数JSON schema，连续多轮工具执行，跨文件更改，长输出重复/循环，长上下文依赖。可从30个公开代码修复+20中文任务+20工具任务做初筛，再按实际失败类型扩充；这是建议样本，不是已经提供或执行的题库。

硬门槛：无数据错配、无跳过专家、无不可解释offload数值差；没有以swap抖动掩盖超内存；无串session；固定caps下无OOM。

质量非劣阈值、TTFT/token/s产品目标由团队在看最终结果前预先签定。建议比较相对EXL3或原生的任务成功率与paired差异置信区间，不直接设“93.63%一致率就是合格”。specs/quality-policy.json保留未批准项null，因此不会自动宣布质量通过。

如果低内存性能不够，报告SSD/H2D/CPU/计算哪项限制，不隐瞒失败，也不自动切到更低bit/更大GPU。
