# TRIAGE-GRPO

**T**axonomy-aware **R**eward **I**nconsistency **A**daptive **G**ating for **E**stimators in **GRPO**

面向可验证数学推理（RLVR）的在线强化学习方法：按「结果对错 × 过程高低」将每条 rollout 分为 2×2 四格，并施加**四种不同待遇**，统一对比并超越单一的门控（P-GRPO/PAPO）、过滤（PROF）与负组惩罚（LENS）。

> 本仓库遵循 **Spec → 实现 → 实验** 流程。请先读文档再改代码。

---

## 文档导航（必读顺序）

| 顺序 | 文档 | 内容 |
|------|------|------|
| 1 | [docs/SPEC.md](docs/SPEC.md) | 方法定义、公式、四格算子、接口契约 |
| 2 | [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | 锁定的模型/数据/评测/基线/成功标准 |
| 3 | [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | 与 PAPO / P-GRPO / PROF / LENS 等差异 |
| 4 | [docs/ROADMAP.md](docs/ROADMAP.md) | 12 周实验 + 代码分期 |
| 5 | [docs/API.md](docs/API.md) | 模块 API 与 veRL 接入点 |
| 6 | [docs/STATUS.md](docs/STATUS.md) | 当前进度与已知问题 |

---

## 一句话方法

```
对每个 prompt 的 G 条回答：
  算 r_o ∈ {0,1}（答案对错）、r_p ∈ [0,1]（过程分）
  分到四格：C↑ / C↓ / W↓ / W↑
  分别：强化 / 正确子集抑制 / 置信负惩罚 / 滤冲突+降权
  合成 advantage → GRPO 更新
```

---

## 锁定配置（不要随意改）

- **主模型**：`Qwen/Qwen2.5-Math-1.5B`
- **训练数据**：`DAPO-Math-17k`
- **评测**：MATH-500, AMC23, AIME24, AIME25, OlympiadBench, MinervaMath
- **框架**：veRL + GRPO（G=8）
- **必比基线**：ORM-GRPO, P-GRPO, PAPO, PROF, LENS, Naive-PRM/Hybrid, TRIAGE

详见 [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)。

---

## 远程仓库与同步

- 仓库：https://github.com/Eternity1212/rl_gate
- 安装提交后自动推送：`bash scripts/install_autosync_hook.sh`
- 手动推送当前分支：`bash scripts/sync_to_github.sh`
- 同时更新 GitHub `main`：`bash scripts/sync_to_github.sh --also-main`

## 快速开始（当前阶段：核心单元可测）

```bash
cd /Users/bytedance/projects/triage-grpo
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 四格分类 + 优势函数单测（不依赖 GPU / veRL）
pytest -q

# 手算对齐演示
python scripts/unit_check_advantage.py
```

完整 veRL 训练依赖 GPU 与外部数据，见 [docs/ROADMAP.md](docs/ROADMAP.md) Phase B。

---

## 仓库结构

```
triage-grpo/
├── README.md
├── docs/                 # 规划与说明书
├── configs/              # 实验 YAML
├── src/triage_grpo/      # 核心库（四格/优势/过滤/指标）
├── scripts/              # 检查与训练入口
└── tests/                # 单元测试
```

---

## 明确不做（范围外）

- 原 GatePO 单点「高过程分+错答案再罚一项」（与 PAPO/P-GRPO 增量过薄）
- 同期上 SWE 多轮 Agent / 70B 刷榜
- 同时更换 PRM、数据、算法（一次只动一个旋钮）

---

## 状态

见 [docs/STATUS.md](docs/STATUS.md)。**Phase A 已完成**：SPEC 文档 + 核心 advantage 库 + 配置模板 + `pytest` 12 passed。下一步 Phase B：veRL 接线。
