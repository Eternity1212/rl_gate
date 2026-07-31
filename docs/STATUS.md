# 项目状态

更新日期：2026-07-31

## 远程仓库

- GitHub：https://github.com/Eternity1212/rl_gate
- `origin` 已配置；默认分支 `main`
- 自动同步：`bash scripts/install_autosync_hook.sh`
- 手动同步：`bash scripts/sync_to_github.sh --also-main`

## 当前阶段

**Phase A 完成 + 实验编排/下载脚本完成 → 下一步 Phase B（veRL 真训练接线）**

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
