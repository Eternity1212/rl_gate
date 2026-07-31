# 实验计划（锁定）

版本：v0.1 — **配置已锁定，禁止无记录地更改。**

---

## 1. 目标投稿

- 主投：AAAI / ACL Findings  
- 备选：ICLR（需更强消融或 7B 全表）

---

## 2. 锁定资源

| 项 | 值 |
|----|-----|
| 主模型 | `Qwen/Qwen2.5-Math-1.5B` |
| 扩展模型 | `Qwen/Qwen2.5-Math-7B`（Phase C） |
| 训练数据 | DAPO-Math-17k（仅含整数答案数学题） |
| 评测 | MATH-500, AMC23, AIME24, AIME25, OlympiadBench, MinervaMath |
| 框架 | veRL |
| 算法骨架 | GRPO |
| Group size \(G\) | 8 |
| Max new tokens | 2048（显存不足可改为 1536，需记 STATUS） |
| 硬件 | 4×A100 80G（1.5B）；7B 用 8 卡 |
| Seeds | 2（主表）；关键结论补第 3 个 |
| 过程分 | 主文固定一种 PRM；附录 rule |

---

## 3. 方法对照表

| ID | 方法 | 配置文件 |
|----|------|----------|
| B0 | Base（无 RL） | — |
| B1 | ORM-GRPO | `configs/baselines/orm_grpo.yaml` |
| B2 | P-GRPO | `configs/baselines/p_grpo.yaml` |
| B3 | PAPO | `configs/baselines/papo.yaml` |
| B4 | PROF-GRPO | `configs/baselines/prof.yaml` |
| B5 | LENS | `configs/baselines/lens.yaml` |
| B6 | Naive hybrid / PRM-direct | `configs/baselines/naive_hybrid.yaml` |
| Ours | TRIAGE-GRPO | `configs/triage_grpo_1.5b.yaml` |

---

## 4. 超参默认（TRIAGE）

```yaml
tau: 0.5
alpha: 1.0          # A_p 权重
beta: 1.0           # W↑ downweight 时使用
gamma: 0.1          # LENS 惩罚
w_high_mode: filter # filter | downweight
lens_normalize: true
eps: 1.0e-6
```

敏感性：\(\tau \in \{0.3,0.5,0.7\}\)，\(\alpha \in \{0.5,1.0,1.5\}\)，\(\gamma \in \{0.05,0.1,0.2\}\) — 仅在主结果成立后扫。

---

## 5. 成功标准

1. **准确性**：六集平均 ≥ 最强单基线（B2–B5 中最优），且显著高于 B1  
2. **可靠性**：`hack_rate`（W↑ 占比）明显低于 B6（PRM-direct / naive hybrid）  
3. **消融**：合并 W↑∪W↓、或合并 C↑∪C↓、或去掉 filter、或去掉 LENS → 至少一项主指标下降  
4. **公平算力**：与 B3/B4 相同 steps / 近似相同 GPU-h  

失败熔断见 ROADMAP。

---

## 6. 主图表清单

| 编号 | 内容 |
|------|------|
| Fig.1 | 训练中四格占比随 step 变化 |
| Fig.2 | TRIAGE 方法示意图 |
| Fig.3 | 同算力训练曲线（准确率） |
| Tab.1 | 六 benchmark 主结果 |
| Tab.2 | 消融 |
| Tab.3 | hack_rate / 长度 / GPU-h |
| Fig.4 | \(\tau\) 或 \(\alpha\) 敏感性 |

---

## 7. 数据与去污染

- 训练：DAPO-Math-17k  
- 评测集不得进入训练  
- 启动实验前运行 `scripts/check_contamination.py`（Phase B 实现）记录重叠为 0 或披露残留  

---

## 8. 明确不做

- GatePO 旧方案  
- 同期 SWE/Web Agent  
- 为刷 AIME SOTA 无限加私有数据  
