# 训练塌缩诊断（2026-08-12）

## 结论（已坐实）

`mean_outcome` 长期为 0 **主要不是** reward 把对题判错，而是 **训后模型生成退化**（空串 / 重复词 / 无答案）。  
因此：当前 completed full run **不能写入论文主表**；应 **停扩矩阵，先修 trainer**。

## 证据

对 `B1.orm.s1` / `Ours.triage.s0` 的 `final` checkpoint，用训练题抽样生成：

| run | 现象 | extract / score |
|-----|------|-----------------|
| B1.orm.s1 | response 为空 | null / 0 |
| Ours.triage.s0 | `WHY WHY WHY...` 等崩文 | null / 0 |

`score_outcome_exact` 给 0 合理。

## 更可能根因（实现侧）

1. GRPO/TRL 入口是不稳定近似：policy loss、token mask、labels 构造错误  
2. 缺少或过弱的 **reference / KL** → 策略漂出可解码区域  
3. 全组 reward=0 时仍大力更新，或 advantage 广播错误  
4. 生成配置异常（EOS、max_new_tokens、prompt 截断）导致空输出再回灌  
5. LoRA lr 过大或更新步把分布推崩  

## 正确下一步（不要做 / 要做）

**不要：** 继续盲跑 B2/B3/A*/H.* 扩矩阵。  

**要：**

1. 审查并重写训练入口（推荐官方 **TRL GRPOTrainer** + 自定义 reward，再挂 TRIAGE advantage）  
2. 冒烟验收标准（任一失败则不算修好）：  
   - 基座未训：能输出含数字/`\\boxed{}` 的正常解答  
   - 训 50～200 step 后：仍非空、非重复词；`mean_outcome` 偶发 >0  
   - ORM 与 TRIAGE 同设定下生成可读  
3. 通过后再重跑 main（至少 B0/B1/B3/Ours）+ 评测 MATH-500  

## 与创新点关系

TRIAGE **公式库仍可用**；塌缩说明 **训练引擎不可信**，不是「四格想法被证伪」。修好后端后应用同一矩阵重做实验。

---

## 修复入口（2026-08-12）

仓库已新增/重写：

- `scripts/trl_train_entry.py` — 基于官方 **`trl.GRPOTrainer`**
- `scripts/run_train.py` — 优先调用上述入口（不再依赖未接线 veRL）

### 抗塌缩默认

| 项 | 默认 | 环境变量可改 |
|----|------|----------------|
| KL β | 0.04 | `TRIAGE_KL_BETA` |
| LR | 1e-6 | `TRIAGE_LR` |
| 温度 | 0.7 | `TRIAGE_TEMPERATURE` |
| 生成质量熔断 | 近窗 ≥60% 空/重复则 abort | 代码内 `DegeneracyMonitor` |

### 验收（修好的定义）

在 GPU 上：

```bash
pip install -r requirements.txt
pip install -e .
# 按机器安装匹配的 torch CUDA 轮子

export TRIAGE_SKIP_ENSURE=1
./run.sh train --run S2.orm --steps 50
./run.sh train --run S2.triage --steps 50
```

通过标准：

1. 不出现 DEGENERACY_ABORT  
2. `metrics.jsonl` 里 `pct_degenerate` 明显低于 0.6  
3. 偶发 `mean_outcome > 0`  
4. 对 checkpoint 抽样生成：**非空、非 WHY 重复、可抽答案**

通过后再重跑 ORM / PAPO / TRIAGE 全量；**旧 collapsed checkpoints 作废。**
