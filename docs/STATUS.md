# 项目状态

更新日期：2026-08-12

## 远程仓库

- GitHub：https://github.com/Eternity1212/rl_gate
- `origin` 已配置；默认分支 `main`
- 自动同步：`bash scripts/install_autosync_hook.sh`
- 手动同步：`bash scripts/sync_to_github.sh --also-main`

## 当前阶段

**训练塌缩诊断（2026-08-12）→ 暂停扩跑矩阵，优先修 trainer**

集群上已完成部分 full run（如 B1.orm.s1、Ours.triage.s0），但对 final checkpoint 抽样生成发现：
- 空输出 / `WHY WHY WHY` 重复 / 无 `\boxed{}`
- `mean_outcome≈0` 是因为**模型生成已崩**，不是 reward 误判
- **现有 completed run 不能当论文有效结果**；须先修 GRPO/TRL 训练实现（KL/reference、loss、decode），再重跑

已合入重写版 `scripts/trl_train_entry.py`（官方 TRL GRPOTrainer + KL + 退化熔断）。  
**旧集群 collapsed checkpoints 全部作废**；须先通过 `docs/TRAINING_COLLAPSE.md` 验收再扩矩阵。

## 已完成

- [x] SPEC / 实验计划 / 相关工作文档  
- [x] 核心 advantage 库 + 单测（12 passed）  
- [x] **全部实验矩阵** `docs/EXPERIMENT_MATRIX.md`  
- [x] **怎么跑** `docs/HOW_TO_RUN.md`  
- [x] **一键入口** `./run.sh`  
- [x] **一键下载** `scripts/download_assets.py`  
- [x] **实验注册表** `configs/experiment_registry.yaml`  
- [x] 消融配置 `configs/ablations/*`  
- [x] matrix / summarize / smoke / contamination 脚本  
- [x] 算力预算 `docs/COMPUTE_BUDGET.md`（H100·h + LoRA 论文可行性）  
- [x] **主文锁定 LoRA \(r=64\)**：`docs/LORA_EXPERIMENTS.md` + 全部 train yaml / registry  


## 未完成（阻塞真·全流程 GPU 训练）

- [ ] 安装并锁定 veRL 版本  
- [ ] `scripts/verl_train_entry.py` 实机 GRPO 入口  
- [ ] reward manager 写入 r_o/r_p/confidence  
- [ ] 真实 eval harness 出分到 `outputs/tables/*.csv`  

当前可用：`./run.sh all --dry-run` 验证编排；`./run.sh download` 拉资产。

## 下次工作

1. 在有 GPU 机器：`./run.sh download && ./run.sh verify-assets`  
2. 接入 veRL trainer  
3. 去掉 dry-run 跑 S2 smoke（200 steps）  
