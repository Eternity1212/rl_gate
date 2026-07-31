# TRIAGE-GRPO

**T**axonomy-aware **R**eward **I**nconsistency **A**daptive **G**ating for **E**stimators in **GRPO**

面向可验证数学推理（RLVR）的在线强化学习方法：按「结果对错 × 过程高低」将每条 rollout 分为 2×2 四格，并施加**四种不同待遇**。

仓库：https://github.com/Eternity1212/rl_gate

---

## 怎么跑（最短路径）

```bash
cd /Users/bytedance/projects/triage-grpo
chmod +x run.sh
./run.sh setup
source .venv/bin/activate

# 无 GPU：单测 + 下载编排演练
./run.sh unit
./run.sh download          # 模型 + DAPO-Math-17k + 评测集
./run.sh all --dry-run     # 列出并演练全部实验矩阵

# 有 GPU + veRL 后（Phase B 接线完成）：
# ./run.sh all
```

**完整说明：** [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)  
**全部实验清单：** [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md)  
**机器注册表：** [configs/experiment_registry.yaml](configs/experiment_registry.yaml)

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | **怎么运行（优先读）** |
| [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md) | **所有要跑的实验** |
| [docs/ASSETS.md](docs/ASSETS.md) | **模型/数据 ID 与磁盘占用** |
| [docs/SPEC.md](docs/SPEC.md) | 方法公式与四格算子 |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | 锁定模型/数据/成功标准 |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | 与 PAPO/P-GRPO/PROF/LENS 差异 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 分期计划 |
| [docs/STATUS.md](docs/STATUS.md) | 当前进度 |

---

## `./run.sh` 命令一览

| 命令 | 作用 |
|------|------|
| `./run.sh help` | 帮助 |
| `./run.sh setup` | 创建 venv 并安装依赖 |
| `./run.sh unit` | pytest |
| `./run.sh download` | 一键下载模型与数据集 |
| `./run.sh verify-assets` | 校验本地资产 |
| `./run.sh list` | 列出全部 run_id |
| `./run.sh train --run <id>` | 跑单个实验 |
| `./run.sh matrix --stage main` | 跑某一阶段全部实验 |
| `./run.sh all --dry-run` | 全流程编排（可无 GPU） |
| `./run.sh summarize` | 汇总 `outputs/tables/` |

---

## 锁定配置

- **主模型**：`Qwen/Qwen2.5-Math-1.5B`
- **训练数据**：`DAPO-Math-17k`
- **评测**：MATH-500, AMC23, AIME24, AIME25, OlympiadBench, MinervaMath
- **框架**：veRL + GRPO（G=8）
- **主文必跑**：S0+S1+S2+S3(main)+S4(ablation) ≈ 33 runs（详见矩阵文档）

---

## 当前状态（诚实）

| 项 | 状态 |
|----|------|
| 实验是否列清 | ✅ |
| 一键下载 | ✅ `./run.sh download` |
| 一键编排 | ✅ `./run.sh all` |
| 核心库单测 | ✅ |
| veRL 真训练接线 | ⏳ Phase B（现支持 `--dry-run` 跑通流程） |

---

## 远程同步

```bash
bash scripts/install_autosync_hook.sh   # commit 后自动 push
bash scripts/sync_to_github.sh --also-main
```
