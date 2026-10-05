# 本地 Codex 入口：Strata-DS v0.3 Prefill Kit

用户要求的核心不是继续写泛化计划，而是完成真实 V4.1 EXL3 单卡分层推理。当前包已提供模型专用预算/审计/调度参考；GPU与全模型实现仍需你逐个垂直切片落地。

## 第一步：不租机

阅读 `docs/DEEPSEEK_PREFILL.zh-CN.md`、`AGENTS.md`、`specs/prefill/sources.lock.json`、`specs/prefill/backlog.json`，再看旧整体设计与验收规程。优先级：v0.3 prefill文档 > v0.2.1增量文档 > 旧整体文档。不要覆盖已有用户分支；先git diff与manifest。

```bash
python tools/verify_bundle.py
python -m unittest discover -s tests -v
python -m strata_ds_lab.prefill facts
python -m strata_ds_lab.prefill plan --recipe exl3-3-r128-v16
python -m strata_ds_lab.prefill plan --recipe exl3-3-r256-v24
python -m strata_ds_lab.upgrade0139 validate
python -m strata_ds_lab backend-status
```

## 第二步：按现成合同接真实组件，不再猜slot大小

1. PF-000：核对真实HF config/量化身份；运行prefill audit，完整TP1专家13,315,596B，组件256B对齐stride13,316,352B。HF本轮未直连成功，所以本地必须完成marker/header/payload审核；不要把源码推导的尺寸当checksum。
2. PF-001：按5120/2304和上游MUL1运行真实单expert GPU probe，核验目标GPU的kernel与峰值；不直接复用GB10的sm_121 binary或跳过min-capability门槛。
3. PF-002：将同步CPU参考替换为真实 I/O→pinned ring→CUDA event→expert wave，20/40slots、G4/G8、R64；已选专家的多个row tile只加载一次。Missing expert在TP1必须失败。
4. PF-003：bounded attention query64/key64；index query16/key4096/head4。四个keybank共享关系、RoPE、Top-k、candidate和跨chunk半组状态照原V4.1语义；别裁剪decoder来拿快结果。
5. PF-004：allocator分配逐项对账。planner中的native64MiB、sort32MiB、runtime1GiB/安全1GiB是明确试验预留，不是实测上限；实际峰值超出就报告/重计划，不能改报告掩盖。
6. PF-005：256/24全模型正确性通过后，以T2048、ring512MiB为初始基线，再单因素ring/chunk和192/128RAM扫描。质量不合格、SSD/H2D未知、只有1次结果不进自动选择器。

## 与原计划衔接

11个硬件配方原样保留，EXL3主线不能偷偷换Q2。Q2/SAGE尚无本版精确bundle适配时，专用planner拒绝是正确行为；别为“全绿”按EXL3数据计算它们。SD/U39任务继续保留，新增PF依赖写明。

Responses脚本化后端测试可并行推进；开发用Codex仍使用用户现有provider。模型通过短序列后才用独立配置接新服务的Codex工具循环。先single-slot/无speculation，再2/5用户。

## 租机前

`specs/cloud-policy.json`仍不授权花费。获得本地用户对offer、镜像、上限和结束策略授权后才能租。先小组件，不先下载几百GB；不改共享宿主机sysctl、driver，不全局drop_caches，不destroy-all。结束保全最小证据并确认销毁，stop不等于所有费用结束。

每个PR附：exact源身份、changed files、执行/跳过项、真实GPU/模型是否运行、原始结果、未通过项和回滚。不可把CPU测试当成GPU测试；新测试最终数量以results/prefill实际日志为准。
