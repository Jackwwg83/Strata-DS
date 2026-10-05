# Strata-DS v0.3 — DeepSeek V4.1 专用预填充设计与可执行工具

**EXL3 3 bpw 主线、Engram SSD、16/24 GiB GPU、128/192/256/384 GiB RAM。**

本版把 v0.2.1 “字节预算不能照搬Qwen”的原则，落实为 **5120×2304、384专家、top-6、13,315,596 B完整专家载荷、13,316,352 B对齐slot** 的模型专用实现。16G起点1024-token/20slots/G4；24G起点2048-token/40slots/G8。

## 先读

[完整模型专用预填充设计](docs/DEEPSEEK_PREFILL.zh-CN.md) → [本地Codex任务](CODEX_START_HERE.md) → [来源和证据等级](docs/PREFILL_SOURCES.md)。

这是**源码研究、明确执行合同和可运行CPU工具**，不是完整DeepSeek GPU引擎。本轮没有下载全部真实checkpoint、没有运行CUDA或Vast。真实header审计代码已给出，固定source-shape推导不冒充整库payload验证。GPU组件探针可在本地受信环境运行，本轮只做语法/help检查。

## 马上运行（Python >=3.11，无新增依赖）

```bash
python tools/verify_bundle.py
python -m unittest discover -s tests -v
python -m strata_ds_lab.prefill facts
python -m strata_ds_lab.prefill plan --recipe exl3-3-r128-v16
python -m strata_ds_lab.prefill plan --recipe exl3-3-r192-v24 --context 131072
python -m strata_ds_lab.prefill plan --recipe exl3-3-r128-v24 --append-tokens 64
python -m strata_ds_lab.prefill matrix --recipe exl3-3-r256-v24
python -m strata_ds_lab backend-status
```

## 新增可执行代码

`strata_ds_lab/prefill/`：格式/对齐、真实本地头部+标记审计、7阶段buffer/四bank状态预算、stable routes/waves/row tiles及CPU执行参考、实测候选选择。`tools/probe_exl3_prefill_gpu.py`：明确授权后加载少量真实EXL3组件的GPU探针，不执行云操作。

预算器输出 `fit_under_declared_design_contract`，绝不把它当成实际CUDA启动成功；真实adapter仍然未实现。探针的合成激活也不等于模型质量或真实路由分布。

## 完整包内容

原 v0.2.1 的全部离线工具、11个硬件配方、8个S/C运行profile、SD与U39任务、Vast运行规程保留。不需下载旧ZIP来拼接。冲突以 `docs/DEEPSEEK_PREFILL.zh-CN.md` 为预填充设计最新依据。

`results/prefill/` 包含本次测试日志和计算JSON/CSV。其他 `results/` 是历史证据，不与本轮GPU或新代码混算。旧 v0.1 experimental模型引擎不在本设计包内，继续作为另行保存的历史参考。

MIT 只覆盖本包独立工具；没有把上游AGPL GPU源码复制或改标为MIT。所有付费云权限保持关闭，所有完整推理后端仍为未实现。
