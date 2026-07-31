# 模块 API

---

## 包路径

```
src/triage_grpo/
  types.py          # Cell, RolloutRecord, AdvantageOutput
  classify.py       # assign_cell
  advantage.py      # compute_advantages(method=...)
  filter_mask.py    # build_sample_mask
  metrics.py        # cell_stats, hack_rate
  rewards/
    outcome.py      # OutcomeReward（接口 + 简易数学解析）
    process_base.py # ProcessScorer Protocol
    rule_process.py # 规则过程分
  baselines.py      # method 名称常量
```

---

## 核心调用

```python
from triage_grpo.advantage import compute_group_advantages
from triage_grpo.types import GroupInput

out = compute_group_advantages(
    GroupInput(
        outcome_rewards=[1, 1, 0, 0],
        process_scores=[0.9, 0.2, 0.1, 0.85],
        confidences=[0.8, 0.7, 0.3, 0.9],
    ),
    method="triage",
    tau=0.5,
    alpha=1.0,
    beta=1.0,
    gamma=0.1,
    w_high_mode="filter",
)
# out.advantages: List[float]
# out.cells: List[Cell]
# out.masks: List[bool]  # False = 不进入梯度
```

---

## veRL 接入点（Phase B）

推荐挂钩位置（名称随 veRL 版本微调）：

1. **Reward Manager**：对每条 response 写 `r_o`, `r_p`, `confidence` 到 `non_tensor_batch`  
2. **Advantage Estimator**：替换默认 GRPO advantage，调用 `compute_group_advantages`  
3. **采样 mask**：`masks=False` 的样本 loss 权重为 0  

适配器草稿：`src/triage_grpo/verl_adapter.py`（提供函数签名与注释，完整训练在 Phase B 接线）。

---

## 配置键

见 `configs/triage_grpo_1.5b.yaml`：

```yaml
triage:
  method: triage
  tau: 0.5
  alpha: 1.0
  beta: 1.0
  gamma: 0.1
  w_high_mode: filter
  lens_normalize: true
  process_scorer: rule   # rule | prm
```
