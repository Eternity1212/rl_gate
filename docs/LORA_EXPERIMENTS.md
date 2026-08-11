# LoRA 训练实验说明（已锁定）

版本：v1.0  
状态：**主文默认训练模式 = LoRA，\(r=64\)**；所有对比/消融必须同一套 LoRA。

相关： [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) · [COMPUTE_BUDGET.md](COMPUTE_BUDGET.md) · [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

---

## 1. 为什么主文用 LoRA？

| 原因 | 说明 |
|------|------|
| 硬件 | 目标机器 **2×H100-80GB**，全参 G=8、长生成偏紧 |
| 成本 | 主文总卡时大约从 ~600 H100·h 降到 ~200 H100·h |
| 科学问题 | TRIAGE 改的是 **优势/奖励怎么算**，不是「必须全参才能成立」 |
| 公平性 | 基线与 Ours **同一 LoRA** → 相对排序仍可解释 |

默认共享块：`configs/lora_defaults.yaml`（已写入各 train yaml）。

---

## 2. 锁定超参（勿中途改）

```yaml
lora:
  enabled: true
  r: 64
  alpha: 128          # 2 * r
  dropout: 0.05
  bias: none
  target_modules: [q_proj, k_proj, v_proj, o_proj]

trainer:
  n_gpus: 2
```

| 项 | 锁定值 | 为什么定这个 |
|----|--------|--------------|
| **\(r\)** | **64** | 1.5B 数学 RL 的常用甜点：容量够分方法差异，又明显省显存 |
| \(\alpha\) | 128 | 惯例 \(\alpha=2r\)，缩放稳定 |
| dropout | 0.05 | 轻度正则，避免 LoRA 过拟合短跑 |
| target | 注意力 q/k/v/o | 改策略分布的主路径；不加 MLP 以控显存 |
| 卡数 | 2 | 匹配 2×H100 |

### 为何不选别的 \(r\)？

| \(r\) | 何时用 | 风险 |
|-------|--------|------|
| 32 | 显存仍紧 / 附录对照 | 方法差距可能被压扁 |
| **64（主文）** | **默认** | — |
| 128 | 附录「更大容量」 | 更贵；若只给 Ours 用不公平 |

**禁止**：主表中途把某方法改成全参或不同 \(r\)。

---

## 3. 对比实验怎么做（LoRA 设定下）

原则不变：**同一模型、同一数据、同一 LoRA、同一 steps，只换 method。**

### 3.1 主对比（论文 Tab.1）

| run_id | 方法 | LoRA | 要回答的问题 |
|--------|------|------|--------------|
| B0.eval | 基座无 RL | — | 起点分 |
| B1.orm.* | ORM-GRPO | r=64 | 默认强基线 |
| B2.pgrpo.* | P-GRPO | r=64 | 错答过程门控 |
| B3.papo.* | PAPO | r=64 | **最关键竞品** |
| B4.prof.* | PROF | r=64 | 冲突过滤 |
| B5.lens.* | LENS | r=64 | 置信惩罚 |
| B6.hybrid.* | Naive hybrid | r=64 | 反面：易 hack |
| B6.prm.* | PRM-direct | r=64 | 反面：易 hack |
| Ours.triage.* | **TRIAGE** | r=64 | 本文 |

seeds：各方法 0、1。  
命令：`./run.sh matrix --stage main`

### 3.2 消融（论文 Tab.2，仍全部 LoRA r=64）

| run_id | 只改 TRIAGE 哪一项 | 目的 |
|--------|-------------------|------|
| A1.no_ah | 关 W↑ 特殊待遇 | 错误高过程格不能当普通错 |
| A2.merge_wrong | W↑/W↓ 合并 | 错误侧分流必要 |
| A3.merge_correct | α=0 | 正确侧过程必要 |
| A4.no_lens | γ=0 | LENS 项有用 |
| A5.no_filter | filter→downweight | 过滤 vs 降权 |
| A6.papo_only | 退回 PAPO | 相对 PAPO 增益来源 |

### 3.3 LoRA 专用附录（可选，不阻塞主文）

| run_id | 内容 | 何时跑 |
|--------|------|--------|
| L.r32.triage / L.r32.papo | \(r=32\) 各 1 seed | 主结果成立后 |
| L.r128.triage / L.r128.papo | \(r=128\) 各 1 seed | 预算有余 |
| L.ft.triage / L.ft.papo | **全参核对** 各 1 seed | 强烈建议若还能挤 ~50–100 H100·h |

附录目的：证明「相对排序不依赖某一个 \(r\)」；全参核对回答审稿「LoRA 会不会反转」。

---

## 4. 具体情况分析：LoRA 下实验会怎样？

### 4.1 预期会发生什么

| 现象 | 预期 | 含义 |
|------|------|------|
| 绝对分略低于文献全参数字 | 常见 | **不要**和别人全参绝对分硬比 |
| TRIAGE vs PAPO **相对序** | 多可保留 | 方法文核心 |
| vs Hybrid/PRM hack_rate | TRIAGE 应明显更好 | 故事最稳的一块 |
| 方法间分差变小 | 可能 | 用 2 seed + 报均值；强调 hack/稳定性 |
| 四格占比仍随训练变化 | 应出现 | 证明算子真的在工作 |

### 4.2 对创新点有没有伤害？

| 问题 | 结论 |
|------|------|
| LoRA 会不会改掉 TRIAGE 的创新定义？ | **不会**。创新在 \(A_o,A_p,A_n,A_h\) 四格待遇，与是否全参正交。 |
| 审稿会不会说「只是 LoRA 技巧」？ | 若文中把 LoRA 当贡献 → 危险。正确写法：**LoRA 是训练效率设定，贡献是 triage 算子**。 |
| 会不会让消融变无效？ | 若 \(r\) 太小（如 8）可能；**\(r=64\)** 一般够承载消融差异。 |
| 会不会让「打不过 PAPO」更常发生？ | **略增风险**（容量限制缩小差距）。缓解：统一 LoRA、报 hack、可选全参核对。 |

**一句话：LoRA 不削弱创新点的「定义」，但会略提高「相对增益变小」的实验风险；应用叙事与核对实验对冲。**

### 4.3 对论文产出的影响

| 投稿目标 | LoRA r=64 主文 | 说明 |
|----------|----------------|------|
| AAAI / ACL Findings | **可支撑** | 写清 LoRA；同设定比方法；消融+hack |
| 顶会「强实验/绝对 SOTA」线 | 偏紧 | 最好加 L.ft.* 全参核对 |
| 伪装全参 SOTA | **禁止** | 会被审稿打穿 |

正文建议句（可直接用）：

> All RL methods are trained with the same LoRA configuration (\(r=64\), \(\alpha=128\), attention projections) on Qwen2.5-Math-1.5B for a fair, compute-matched comparison. Our contribution is the triage advantage operators, not the adapter parameterization.

---

## 5. 成功 / 失败怎么判（LoRA 主文）

**仍用 EXPERIMENT_PLAN 成功标准**，额外加一条：

5. **透明披露**：摘要或实验设置明确写 LoRA \(r=64\)；Tab.3 报 GPU-h。

熔断（建议）：

- S2 后四格几乎恒空或 `hack_rate` 全方法无差异 → 先修 \(r_p\)/τ，勿开主矩阵。  
- 主表 TRIAGE 全面弱于 PAPO 且消融不掉点 → 优先排查实现，再考虑 \(r=128\) 或全参核对，而不是改创新故事。

---

## 6. 命令（2×H100）

```bash
export CUDA_VISIBLE_DEVICES=0,1
./run.sh setup && source .venv/bin/activate
./run.sh download
./run.sh verify-assets

# 编排演练
./run.sh all --dry-run

# veRL 就绪后
./run.sh matrix --stage smoke
./run.sh matrix --stage main
./run.sh matrix --stage ablation
./run.sh summarize
```

配置自检：任意 `configs/**/*.yaml` 应含 `lora.r: 64` 与 `n_gpus: 2`。
