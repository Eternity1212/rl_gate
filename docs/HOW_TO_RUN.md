# 怎么跑（从零到全流程）

面向：不会配环境也能按命令执行。  
总入口：**`./run.sh`**

---

## 0. 诚实说明（当前能力边界）

| 能力 | 状态 |
|------|------|
| 实验清单是否列清 | ✅ `docs/EXPERIMENT_MATRIX.md` + `configs/experiment_registry.yaml` |
| 一键单测 / 四格冒烟 | ✅ `./run.sh unit` 等 |
| 一键下载模型与数据 | ✅ `./run.sh download`（需网络与 HF 访问） |
| 一键编排全部实验 | ✅ `./run.sh all` / `./run.sh matrix --stage ...` |
| **真·GPU GRPO 训练** | ⚠️ 编排与 dry-run 已通；**veRL 实机 trainer 接线仍是 Phase B** |

没有 GPU / 没装 veRL 时：用 `./run.sh all --dry-run` 可跑通编排、产出 `outputs/*/job.json`。  
有 GPU 后：装 veRL，去掉 `--dry-run`，并完成 `scripts/verl_train_entry.py` 接线（见下文 Phase B）。

---

## 1. 第一次使用（5 分钟）

```bash
cd /Users/bytedance/projects/triage-grpo

# 1) 环境
chmod +x run.sh
./run.sh setup
source .venv/bin/activate

# 2) 无 GPU 自检
./run.sh unit
./run.sh check-advantage
./run.sh smoke-cells

# 3) 看全部实验 ID
./run.sh list
```

---

## 2. 一键下载模型 / 数据集

```bash
# 默认：Qwen2.5-Math-1.5B + DAPO-Math-17k + 6 个评测集
./run.sh download

# 同时下 7B
./run.sh download --with-7b

# 校验
./run.sh verify-assets
./run.sh check-contamination
```

下载落盘位置：

```text
assets/
  models/Qwen2.5-Math-1.5B/
  data/DAPO-Math-17k/
  data/eval/MATH-500/ ...
  manifest.json
```

> 国内若 HF 慢：设置镜像，例如  
> `export HF_ENDPOINT=https://hf-mirror.com`  
> 然后再 `./run.sh download`。

---

## 3. 全部实验怎么跑

完整清单见 [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md)。

### 3.1 只跑某一阶段

```bash
./run.sh matrix --stage smoke --dry-run
./run.sh matrix --stage main --dry-run
./run.sh matrix --stage ablation --dry-run
./run.sh matrix --stage sensitivity --dry-run
./run.sh matrix --stage extend --dry-run
```

### 3.2 只跑某一个 run_id

```bash
./run.sh train --run S2.triage --steps 200 --dry-run
./run.sh train --run Ours.triage.s0 --dry-run
./run.sh train --run B3.papo.s0 --dry-run
```

### 3.3 一条龙（推荐）

```bash
# 无 GPU：验证编排
./run.sh all --dry-run

# 有 GPU + veRL 就绪后（Phase B）：
./run.sh all
./run.sh all --with-sensitivity
./run.sh all --with-extend
```

### 3.4 汇总

```bash
./run.sh summarize
# 产出 outputs/tables/runs.csv 等
```

---

## 4. 主文最低必须跑哪些

按优先级：

1. **S0**：unit / advantage / smoke-cells  
2. **S1**：download + verify  
3. **S2 smoke**：`S2.orm`, `S2.triage`（200 steps）  
4. **S3 main**：B1–B6 + Ours（两 seed）→ 论文主表  
5. **S4 ablation**：A1–A6 → 消融表  
6. **summarize**

敏感性（Stage 5）与 7B（Stage 6）是加分项。

---

## 5. Phase B：真训练还差什么

当 `import verl` 成功后，`scripts/run_train.py` 仍会提示需要接线。需要补：

1. 安装 veRL（官方文档）  
2. 实现 `scripts/verl_train_entry.py`：读 config + 调用 GRPO  
3. 在 reward manager 写入 `r_o/r_p/confidence`  
4. advantage 处调用 `triage_grpo.verl_adapter.compute_advantages_for_verl_group`  

完成前，**请用 `--dry-run` 跑通实验编排与资源下载**，不要空等。

---

## 6. 常见问题

**Q: download 某个评测集失败？**  
A: 脚本会写 `DOWNLOAD_FAILED.txt` 并继续。主训练只强依赖 DAPO-Math-17k + 1.5B 模型。评测集可稍后补。

**Q: 没有 4 卡？**  
A: 改各 yaml 里 `trainer.n_gpus`，或先 `--dry-run`。

**Q: 如何同步到 GitHub？**  
A: 已配 `origin` → https://github.com/Eternity1212/rl_gate  
`bash scripts/sync_to_github.sh --also-main`

---

## 7. 命令速查

| 你想做的事 | 命令 |
|------------|------|
| 看帮助 | `./run.sh help` |
| 装环境 | `./run.sh setup` |
| 单测 | `./run.sh unit` |
| 下数据模型 | `./run.sh download` |
| 列实验 | `./run.sh list` |
| 跑主对比（演练） | `./run.sh matrix --stage main --dry-run` |
| 全流程演练 | `./run.sh all --dry-run` |
| 出汇总表 | `./run.sh summarize` |
