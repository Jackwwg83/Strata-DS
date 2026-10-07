> **v0.2.1 设计更新：**先读 [v0.1.39 增量方案](UPSTREAM_v0.1.39_DELTA.zh-CN.md)。本文件保留原模型与实验设计；阶段内存、I/O、Responses、slot/load 与 exactness 要求以新增文档为准。原 11 个配方和模型选择不变。

# 代码设计：从 v0.1 reference 到多配方执行架构

本设计中的“必须”是开发验收要求，不意味着本次已经实现。现有可执行参考模块与尚缺的GPU后端详见 IMPLEMENTATION_STATUS.md。

## 1. 先锁定身份，不按模型字符串猜兼容

`ModelIdentity` 分离五个身份：base checkpoint revision、量化张量manifest digest、tokenizer digest、模板/编码器digest、Engram哈希/压缩token语义digest。资源snapshot还带engine commit、backend/layout version、KV format、session generation。

已确认coolbho3k权重pin为650cae2c13aaaec303871a35301503570889c0be，基座源为df42c109f1defefcbfcedbe7d905718a12266e40；其无损Engram资产使用另一revision93a185...，这不是错误混装，因为上游显式绑定相同原始行字节。验证模型manifest和Engrammanifest各自hash，并验证语义身份，不能只比较仓库revision字符串。[S1,S2]

旧v0.1 provenance的DeepSeek revision是2cba9e42...，不同于上述基座。旧tiny测试继续可用，但不能直接作为新量化的独立oracle。不同版本必须先确认config、tensor、tokenizer及模板差异，不能静默复用。

Q2及SAGE尚未在本次锁定精确发布revision，配置保留null；真正下载/运行前必须解析和审阅。`main`不作为可重现运行身份。模型权重、Engram布局、CUDA二进制、辅助draft分别锁定。

## 2. SourceCatalog / manifest

目标结构：

```text
CheckpointManifest
  identity, source_license, source_repo, revision
  files[]: relative_path, length, sha256
  tensors[]: source_file, offset, length, dtype, shape, quant_layout
  partitions:
    text_mandatory_nonrouted
    routed_experts[layer][expert] -> component tensor references
    engram_values / engram_scales
    vision_only / draft_only / unsupported
  estimated_peak_buffers by decode/prefill/context/concurrency
```

`catalog.py`只实现safe safetensors头部读取：前8字节长度有上限、JSON重复键拒绝、dtype白名单、维度/offset/payload一致、无overlap/hole/trailing数据、禁止路径越界及跨文件重名。它没有用名称猜EXL3专家分组，也没有实现GGUF loader。

真实GGUF metadata使用固定版本的官方/上游parser，保留原始量化block布局、tensor strides与分片关系。EXL3每个专家的code、scale、旋转/排列、校正和padding必须一起建manifest。禁止仅凭3bpw乘参数量推断文件offset。

headers不证明payload完整；完整下载使用预期SHA256、长度、原子完成标记。统计文件hash应流式读取，不全文件read。活动worker使用不可变inode/目录，不允许原位更新权重。

## 3. TensorProvider替换整包resident clone

旧 `checkpoint.py::load_checkpoint` 对get_tensor()再clone；旧`Model.__init__`保留CPU canonical weight并把非专家权重搬GPU。迁移不是删除内存检查，而是换成provider。

接口分三类：

- DenseProvider：保护的router/shared/attention等张量；manifest决定native格式、执行格式与device峰值。每个tensor复制完成后可按策略释放不需要的CPU来源。
- ExpertProvider：按完整ExpertKey租用packed components；SSD是权威backing，RAM/GPU是缓存。不能把一次加载扩成整层全部专家反量化。
- EngramProvider：根据确定性的token历史得到row IDs，只取values/scales行，返回带生命周期的buffer；不持有全表Torch tensor。

Peak ledger = 已驻留bytes + 正在读取/复制bytes + 正在反量化scratch + event仍引用的旧buffer + allocator/graph预留。常驻预算和瞬时预算不同。shared_source_memmap占虚拟地址不等于零物理内存，应观测page cache与cgroup。

不要一次性pin几十GiB权重。只pin已明确限额的staging环；若要zero-copy访问主存，需要单独的硬件/驱动及带宽验证。

## 4. Engram文件后端

### 4.1 原始字节正确性优先

源格式FP8行与E8M0尺度可能分开存放。先支持双区域bounded pread；后续做数据+尺度共置的无损布局。无损意味着原始字节保持、不是先dequant再requant。哈希、prime桶、token归一化、padding、layer ownership按固定官方源码实现。

`FileRows`是同步参考：给定file/offset/rows/row_bytes/stride，小批读取，有界行payload cache，重复row复用，短读/越界/文件变更拒绝。它不认识DeepSeek张量名，不做FP8计算，不是O_DIRECT，不声称硬RSS限制。返回buffer、Python对象、OS页缓存在cache预算之外。

### 4.2 生产I/O设计

Linux首选单独实现`DirectFile`后端：显式查询/验证对齐，4KiB页读取、有限在途数、batch去重和排序、两条优先级队列（decode Engram优先；bulk experts/prefill其次）。是否用io_uring按目标容器能力选择，不要求为了性能取得宿主机特权。

接口 `issue(rows, destination, generation) -> Ticket` 与 `collect(ticket)`。token已知即可预取；层消费前校验完成。buffer在I/O和GPU最后消费event完成前不可复用。取消不等于设备停止引用：先排空/标记generation，再回收。

小行物理I/O存在读放大；必须分别记录logical row bytes、pread return bytes、block device/cgroup io.stat物理bytes。不能把FileRows的pread_bytes作为SSD实测。

### 4.3 page15与许可证

coolbho3k page15资产按TP2 rank ownership组织，不能直接拼接为单卡布局。可审计并适配其格式，也可独立实现保留原始字节的TP1布局；两者都需全量/抽样行对照与manifest约束。源reader/integration含AGPL-3.0-only，复制改造必须保留并满足相应许可，不能当作MIT重新标记。进程隔离不是自动消除许可证义务的证明。[S2]

## 5. 专家缓存：字节预算 + lease + 代际

共同状态：ABSENT → LOADING_HOST → HOST_READY → COPYING_GPU → GPU_READY；中途失败进入FAILED后清理。实际同一专家可能同时有host/gpu两份，但复制期间必须明确计费。

`CacheLedger`是状态机控制参考，实际仅LOADING/READY，带唯一generation、容量预占和lease。它不分配device内存；`copy_complete=True`只是CPU测试的授权信号，生产必须来自真的CUDA event，不准把enqueue成功当完成。

生产要求：

1. reserve之前先确认强制权重、KV、staging和安全余量；reserve即占容量。
2. 相同key并发miss singleflight；引用同一future，而不是重复下载。
3. 完整数据校验后再发布HOST_READY；CUDA事件确认后再发布GPU_READY。
4. lease覆盖拷贝/核计算最后消费者；reclaim不能与消费者并行。
5. 失败的旧generation完成回调不得发布到新slot；cancel后重试必须被fencing。
6. cache admission、预取和eviction可以改变性能，不能改变router选择、权重或top-k。
7. decode与prefill的缓存策略分离：prefill一次见全体专家不应污染所有长期decode热点。
8. per-layer最低份额防止前几层抢光；其上动态LFU/LRU/带衰减热度经held-out trace选择。

不要用测试语料先训练缓存profile再在相同输入上声称可泛化命中率。trace只需(layer,expert,bytes,phase)，不默认保存用户原文。

## 6. EXL3 GPU执行路径：第一优先级

### 6.1 路线A：直接packed kernels

复用/适配经过审计的EXL3 MUL1格式与GPU矩阵核。小批decode至少覆盖1/2/5 token行；prefill覆盖递增batch。验证padding、布局、activation precision、旋转/置换、scale/codebook、gate/up/down的关联。实际专家形状由manifest读取，不能直接使用Qwen的640 hidden等假设。

先独立测试1个真实专家对上游，再MoE层（包括shared expert、routing correction、weighted combine），最后接入官方V4.1层图。算子支持GPU架构要逐个编译/运行；不拿GB10的sm_121专用binary当成sm_89/sm_120通用包。

### 6.2 路线B：小块反量化fallback

若packed kernel尚不能在目标GPU运行，可做单expert/单tile BF16 scratch以证明数值。必须设反量化峰值上限、禁止全模型/全层反量化，结果标明`exl3_tile_dequant_reference`，不得与packed性能混报。保留所有量化信息，不转换为另一种更易跑的量化而仍沿用原配方ID。

### 6.3 CPU路径

首版EXL3由GPU计算，RAM只作为packed权重缓存。因此全RAM命中仍有H2D。EXL3 CPU packed算子未来作为可选backend，验证吞吐后才考虑split；通用Python反量化CPU实现仅作为参考，不作为性能默认fallback。

GGUF路径可以参考已有CPU quant内核，但V4.1整图与CPU/GPU分工仍需验证。Q2使用哪个gate/up/down精度要逐tensor确认，不能将“Q2”理解为统一2bit。

## 7. RuntimeBridge与数学图

初期两个外部backend：EXL3 custom runtime分支，DwarfStar GGUF对照分支；共享实验格式。后续统一模型math adapter前，不强行复制全部服务栈。

V4.1必须保留CSA2来源与复用、Engram、HC、RoPE、router语义。旧v0.1仅是tiny reference，不能作为上游替代。实现backend必须报告：base/quant/tokenizer/template身份、支持的context/并发/KV/vision/spec、mandatory GPU bytes、当前instance hardware、数值qual evidence。

本包`contracts.py`四个backend全部implemented=false；`require_backend()`明确拒绝真实执行。不要为了通过测试把它改成true，必须以真实接口和artifact证据替换。

## 8. Layer调度与重叠

当前层路由出来后才能知道其确切专家。不能假定可以跨所有层无依赖流水。下一层专家预取只能是提示/预测，未命中走正确fallback，不能修改模型选择。

同层可重叠：已驻GPU专家计算、CPU适配器专家计算（若存在）、缺失专家读取/H2D，以及不依赖结果的shared branch。combine必须等所有被路由专家完成。只把“实际隐藏掉的等待时间”计为overlap，不用时间线理想叠加替代测量。

CUDA graph使用稳定slot或受控indirection。被graph捕获的指针不可在eviction后指向另一个expert而不刷新绑定/generation。prefill graph与decode graph可能需要不同scratch及形状capture；初版先eager正确性，再graph。

## 9. Prefill与agent append

prefill按token块、layer、expert分组，对同一个expert的多token聚合GEMM，限制一次staged专家数与总scratch。批量fetch去重，有界pipeline，记录每块的读取字节。禁止逐token重复加载整套专家；也禁止为了吞吐把所有层都展开驻留。

Coding-agent关键不是只有初次32K prefill，还有8K/32K前缀后的64/256/1024-token工具结果追加。shared prefix命中与kernel warm分开记录。换配方、quant或engine layout时旧prefix/SSD状态默认失效，除非显式兼容且数值验证。

## 10. 多会话与持久化

所有会话独立的KV/index/SWA/压缩器/Engram token历史。专家及immutable权重可共享。真实batch的不同session token进入同一forward，不能把串行round_robin叫并行推理。

队列分离prefill/decode，限制长prefill挤占decode；取消按request generation失效。单session token append按expected_position串行；重试不得重复消费。cache lease可跨batch但session KV不可互串。

snapshot包含model/quant/tokenizer/template/Engram/KV backend identity与generation。写临时文件、校验、fsync、原子完成标记后才发布；恢复先全部校验，再绑定。支持部分压缩组、环形window边界、取消中途恢复。私有目录/加密是部署职责；默认不公开真实聊天数据。

## 11. 旧文件迁移清单

| v0.1文件/函数 | v0.2方向 | 保留与验收 |
|---|---|---|
| checkpoint.py::load_checkpoint | SourceCatalog + Providers | 保留shape/安全校验；删除的应是全量clone，不是检查 |
| weights.py::Weight | DenseHandle / PackedExpert / RowSource | 不再假定所有权重都是常驻Torch对象 |
| model.py::__init__ | RuntimeBridge / measured placement | 不全量copy非专家；检查实际GPU峰值 |
| model.py::_engram路径 | EngramProvider.issue/collect | 完全相同哈希/原始行字节；只换存储 |
| cache.py::ExpertPool | backend-specific executor + shared lease registry | 原FP4 CPU核不是EXL3核，不换名冒充 |
| planner.py | 两阶段：设计hint → exact admission | 旧payload+largest_tensor基于旧clone，不能继续当硬门槛 |
| runtime.py | SessionScheduler/CheckpointStore | old round_robin保留为单线程正确性对照，不当batch |
| cli.py | lab工具与真实engine CLI分离 | 不新增空壳serve命令假装兼容API |

建议把v0.1原样放在`reference/v01`或保留tag，别在第一个commit同时大改所有文件。本包放入`labs/recipe-design-v02`，随后小PR逐个垂直切片替换。

## 12. 最小可验证开发顺序

L0身份/manifest → L1真实EXL3单专家GPU对照 → L2 Engram file行对照 → L3单层MoE与两层依赖 → L4全模型短序列与独立oracle → L5三层缓存和取消 → L6预算与真实128/192/256实验 → L7 grouped prefill/append → L8多会话 → L9 optional speculation。

每一阶段给出changed files、测试、真实证据路径、失败项与回滚方式。详见specs/backlog.json。
