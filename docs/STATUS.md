# 项目状态

更新日期：2026-07-31

## 远程仓库

- GitHub：https://github.com/Eternity1212/rl_gate
- `origin` 已配置；默认分支 `main`
- 自动同步：运行 `bash scripts/install_autosync_hook.sh` 后，每次 `git commit` 会 push 当前分支
- 手动同步：`bash scripts/sync_to_github.sh`（可选 `--also-main` 同步更新 main）

## 当前阶段

**Phase A — Spec Coding（已完成）→ 下一步 Phase B**

## 已完成

- [x] 项目初始化与 git  
- [x] README / SPEC / EXPERIMENT_PLAN / RELATED_WORK / ROADMAP / API  
- [x] 核心库：四格分类、优势、过滤、指标、基线 method  
- [x] 配置 YAML 模板（主配置 + 6 基线）  
- [x] 单元测试与手算脚本  
- [x] veRL adapter 接口草稿（未接实机）  

## 进行中

- [ ] Phase B：veRL 实机接线与 ORM-GRPO 小步复现  

## 已知风险

1. 神经 PRM 选择未最终钉死 → 主文选定前用 `rule` 跑通管线。  
2. veRL 版本 API 可能漂移 → adapter 保持薄封装。  
3. TRIAGE 被审稿视为「拼装」→ 消融必须完整。  

## 下次工作

1. `pip install -e ".[dev]"` + `pytest`  
2. 准备 GPU 环境与 DAPO-Math-17k  
3. 实现数学 outcome reward 与 veRL hook  
