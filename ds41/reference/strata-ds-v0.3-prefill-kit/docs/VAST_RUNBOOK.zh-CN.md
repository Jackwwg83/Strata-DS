> **v0.2.1 设计更新：**先读 [v0.1.39 增量方案](UPSTREAM_v0.1.39_DELTA.zh-CN.md)。本文件保留原模型与实验设计；阶段内存、I/O、Responses、slot/load 与 exactness 要求以新增文档为准。原 11 个配方和模型选择不变。

# 本地 Codex → Vast 实验运行手册（设计阶段）

本次没有调用Vast账户或租赁机器。本包仅能生成只读搜索参数、离线预算和实验矩阵；真正的生命周期控制器列为待开发任务。请不要把本手册理解为已提供一键自动跑整模型的程序。

## 1. 权限与预算

API key仅放本地凭据管理/环境，不上传GitHub，不交给模型服务器，不复制个人SSH私钥。使用明确scope的key；实验只运行公开/合成任务，不上传客户代码。

specs/cloud-policy.json默认allow_paid_create=false，下载也未授权，max_total/hourly/duration为null。用户在本地批准后，由控制器绑定recipe、offer、image digest、run deadline和最大实例数。最初最多1个活跃实例，不做整张矩阵并行购买。

未经批准不能把“预算建议”当作有权消费。当前包即使读取到批准的policy也只输出proposal，不执行create。

## 2. 搜索与筛选

先读当前CLI help与Vast官方字段，因为远端可更新。推荐只读流程：[S7,S8]

```bash
vastai --help
vastai search offers --help
python -m strata_ds_lab vast-search exl3-3-r256-v24
```

最后一条打印argv数组，不执行。其cpu_ram/gpu_ram是按Vast文档MB做的粗筛；实际GiB、容器allocation和free VRAM以租机后预检为准。不能因为host总内存256G就认为租1卡能用全部256G。当前available实例和价格必须由Codex在实际租赁时查询，本包没有过时报价表。

首次使用一个24GiB GPU档，并验证具体架构上的EXL3 kernel；RTX 3090/4090这类同容量GPU也不是相同性能。16GiB另作真实硬件验证，不拿24G卡限额替代最终结论。

关注CPU effective cores/cpuset、NUMA、PCIe通道和实际H2D、内存通道/带宽、local disk挂载和读性能；不要只按GPU FLOPS/$排序。advertised disk_bw不是本模型随机行/专家大块混合I/O的性能。

## 3. 只先运行轻量preflight

先构建/组件验证用实例或进入已批准实例，在下载完整模型前：

```bash
python -m strata_ds_lab probe --path /workspace > hardware.private.json
python -m unittest discover -s tests -v
```

人工/控制器进一步核验nvidia-smi、CUDA小矩阵、BF16支持、目标packed kernel、pinned H2D/DtoH、小范围本地NVMe读取、可用磁盘、cgroup祖先limit、swap.current、memory.events、CPU affinity/NUMA。probe是只读证据，不是自动合格判决。

每次临时I/O测试只用本实例新建文件，绝不读写raw磁盘设备或其他路径。共享host不全局drop_caches，不卸载/升级驱动，不停别人的服务，不开privileged。

只有GPU算子兼容、实际分配达到配方预算、磁盘和I/O合格才允许完整模型下载。失败则立即取回日志并结束该次实验，而不是先花下载流量再排障。

## 4. 权重/软件准备

仅下载锁定revision中的必要文件，保留许可证/manifest；默认不开trust_remote_code，不直接执行下载包中的脚本。校验大小和SHA256、分片完整性后发布本地complete标记。

EXL3模型pin与Engram pin可以不同，但必须用来源明确的配套manifest和语义digest。不能套用双Spark预编译镜像作为x86独显镜像：CPU架构、GPU架构、TP2 ownership、编译依赖不同。先审计源代码许可，创建专用amd64/目标CUDA镜像，记录最终digest。

预留原始Engram与转换输出同时存在的空间；有TP1文件行reader时可不做page15转换。首次不下载DSpark/视觉附加包，也不同时下载所有quant。只选一个主配方完成后，再审查是否下载Q2对照。

## 5. 首轮固定顺序

1. 本地离线测试和来源锁定（免费、不租GPU）。
2. 代表性EXL3真实expert GPU smoke（小数据）+ Engram行byte对照。
3. EXL3256/24，全模型短输入+同量化oracle。失败不得直接扩128K。
4. EXL3256/24 8K单流，然后192/24、128/24的受控cap。
5. 真实128/24验证；再16GiB GPU变体、Q2对照。
6. 胜出候选才做32K/128K、2/5并发、Agent多轮，最后optional配方。

`matrix --stage smoke`生成8个核心配方的单个计划case；screen每配方18个计划case；full每配方72个计划case。它们不是同时租赁指令。指定--recipes只生成批准的子集。

## 6. 大主存限额实验注意

以物理档128GiB为例，本方案预留系统/余量18，worker本身预算110GiB（96权重+2行cache+4staging+8runtime）。同一384G宿主机做限额时，要在经授权可用的cgroup设置等价worker预算并禁止swap，同时仍报告它是cap实验。

cgroup只控capacity，不模拟较小主机的内存通道/频率/NUMA。文件页cache的charge可能来自此前不同cgroup；必须用私有文件/可解释的warmup与物理I/O证明，不将page cache隐藏在宿主机剩余内存。

没有权限或无法证明limiting就记录APP_CACHE_CAP_ONLY，不能伪造physical128G结果。不能使用cuda allocator fraction来声称真16G卡测试。

## 7. 生命周期控制器待实现状态机

```
PLAN -> APPROVED -> OFFER_RECHECKED -> CREATE_REQUESTED
 -> INSTANCE_KNOWN -> PREFLIGHT -> WEIGHTS_VERIFIED -> TESTING
 -> COLLECTING -> ARTIFACTS_SAVED -> DESTROY_REQUESTED -> DESTROY_VERIFIED
```

每个状态持久化本地journal：run_id、recipe/offer/image digest、instance_id、授权上限、deadline、最后provider响应、result hashes。API key不入journal。

create超时属于UNKNOWN：先通过唯一label/本地记录与provider查明是否已创建，不自动重复create。明确绑定创建的instance_id，销毁只允许本次且经用户授权的实例，不使用destroy all。没有实例label/查询能力时由人工恢复，不猜测实例。

设置本地独立watchdog、总预算/时长检查和阶段timeout；训练/推理job timeout不等于provider实例销毁。网络断开时本地watchdog也可能无法销毁，因此它不是provider侧硬费用上限；需记录alert与人工处理路径，不能保证任何情况下绝不超费。

## 8. 成本与结束

成本包含GPU/CPU租用、磁盘、下载/上传、JIT/warmup、失败重试。`dph_total`与storage cost不要重复计算；cost-plan接收的是明确包含磁盘的normalized hourly，下载费另列。examples/normalized-quote.synthetic.json所有价格都是合成数据，不是Vast报价。

Vast停止实例仍可能继续收磁盘费；结束后先取回最小结果包并核验本地hash，再destroy并确认provider状态。[S7]只停Python程序、Docker容器或SSH断开都不代表停止收费。

预算耗尽但复制日志很慢时先保全最小JSON/journal；权重可重下，不应为了等待复制几百GB缓存而继续收费。销毁会删除实例数据，必须有明确的结果保全/批准策略。

## 9. 结果交付结构

```
private-results/<run_id>/
  authorization.redacted.json
  offer-snapshot.redacted.json
  hardware.json
  lock.json
  preflight.json
  trial-<id>/request-events.jsonl
  trial-<id>/counters.jsonl
  correctness.json
  task-results.jsonl
  cost-estimate-and-actual.json
  artifact-sha256.json
  destroy-receipt.json
```

模型/用户prompt可包含隐私，默认private。公开报告只发布聚合数、复现公共任务与脱敏硬件条件。CI secrets和API key永不写入stdout/截图/markdown。
