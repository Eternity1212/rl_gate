# 全部实验清单（必须跑完才算主文齐）

版本：v0.1  
原则：**一个实验 = 一个 run_id = 一份配置 = 一个输出目录**。  
执行入口：`./run.sh`（见 [HOW_TO_RUN.md](HOW_TO_RUN.md)）。

---

## 0. 资源依赖（所有实验共用）

| 资源 | 来源 | 本地路径（下载后） |
|------|------|-------------------|
| 主模型 | `Qwen/Qwen2.5-Math-1.5B` | `assets/models/Qwen2.5-Math-1.5B` |
| 扩展模型（可选） | `Qwen/Qwen2.5-Math-7B` | `assets/models/Qwen2.5-Math-7B` |
| 训练数据 | `BytedTsinghua-SIA/DAPO-Math-17k`（或 `open-r1/DAPO-Math-17k-Processed`） | `assets/data/DAPO-Math-17k` |
| 评测数据 | MATH-500 / AMC23 / AIME24 / AIME25 / OlympiadBench / MinervaMath | `assets/data/eval/` |
| 框架 | veRL（Phase B 安装） | `third_party/verl` 或环境包 |

一键下载：

```bash
./run.sh download          # 1.5B + 训练数据 + 评测数据（约 10–15 GB 含缓存）
./run.sh download --with-7b
```

磁盘与 HF ID 细节：[ASSETS.md](ASSETS.md)。

---

## 1. Stage 0 — 本地无 GPU（必跑，CI 级）

| run_id | 内容 | 命令 | 成功标准 |
|--------|------|------|----------|
| S0.unit | 单元测试 | `./run.sh unit` | pytest 全绿 |
| S0.adv | 手算优势对齐 | `./run.sh check-advantage` | 打印 OK |
| S0.cells | 合成数据四格统计 | `./run.sh smoke-cells` | 产出 `outputs/smoke/cell_stats.json` |

---

## 2. Stage 1 — 数据与模型就绪检查

| run_id | 内容 | 命令 | 成功标准 |
|--------|------|------|----------|
| S1.dl | 下载资产 | `./run.sh download` | `assets/manifest.json` 存在且校验通过 |
| S1.verify | 校验本地文件 | `./run.sh verify-assets` | 模型/数据路径存在 |
| S1.contam | 训练–评测去污染 | `./run.sh check-contamination` | 报告重叠为 0 或已披露 |

---

## 3. Stage 2 — 训练冒烟（GPU，短跑）

固定：模型 1.5B，数据 DAPO-Math-17k，G=8，**200 optimizer steps**，seed=0。

| run_id | method | 配置 | 命令 |
|--------|--------|------|------|
| S2.orm | orm_grpo | `configs/baselines/orm_grpo.yaml` | `./run.sh train --run S2.orm --steps 200` |
| S2.triage | triage | `configs/triage_grpo_1.5b.yaml` | `./run.sh train --run S2.triage --steps 200` |

成功标准：无崩溃；日志含 `triage/pct_*`；checkpoint 写入 `outputs/<run_id>/`。

---

## 4. Stage 3 — 主对比实验（论文 Tab.1）

固定：1.5B，DAPO-Math-17k，**同 steps / 同 GPU-h**，seeds ∈ {0,1}。

| run_id | method | 配置 | seeds |
|--------|--------|------|-------|
| B0.eval | base（无 RL，只评测） | — | 0 |
| B1.orm.s0 / B1.orm.s1 | orm_grpo | `configs/baselines/orm_grpo.yaml` | 0,1 |
| B2.pgrpo.s0 / B2.pgrpo.s1 | p_grpo | `configs/baselines/p_grpo.yaml` | 0,1 |
| B3.papo.s0 / B3.papo.s1 | papo | `configs/baselines/papo.yaml` | 0,1 |
| B4.prof.s0 / B4.prof.s1 | prof | `configs/baselines/prof.yaml` | 0,1 |
| B5.lens.s0 / B5.lens.s1 | lens | `configs/baselines/lens.yaml` | 0,1 |
| B6.hybrid.s0 / B6.hybrid.s1 | naive_hybrid | `configs/baselines/naive_hybrid.yaml` | 0,1 |
| B6.prm.s0 / B6.prm.s1 | prm_direct | 改 `triage.method=prm_direct` | 0,1 |
| Ours.triage.s0 / Ours.triage.s1 | triage | `configs/triage_grpo_1.5b.yaml` | 0,1 |

一键跑主对比（两 seed）：

```bash
./run.sh matrix --stage main
```

评测集（每个 checkpoint 都跑）：MATH-500, AMC23, AIME24, AIME25, OlympiadBench, MinervaMath。

```bash
./run.sh eval --run Ours.triage.s0
./run.sh eval --stage main   # 批量评已完成的 main runs
```

---

## 5. Stage 4 — 消融（论文 Tab.2）

基于 TRIAGE 默认，一次只改一个旋钮；seed=0。

| run_id | 改动 | 说明 |
|--------|------|------|
| A1.no_ah | `w_high_mode=downweight` 且 `beta=0` 或关闭 W↑ 特殊待遇 | 合并错误侧 |
| A2.merge_wrong | W↑ 与 W↓ 同用 LENS | 证明分流必要 |
| A3.merge_correct | C↑/C↓ 不做 A_p（α=0） | 证明正确子集过程必要 |
| A4.no_lens | `gamma=0` | 去掉 W↓ 惩罚 |
| A5.no_filter | `w_high_mode=downweight` | 对比 filter |
| A6.papo_only | 等价 method=papo | 对照 |

```bash
./run.sh matrix --stage ablation
```

---

## 6. Stage 5 — 敏感性（论文 Fig.4，主结果成立后）

| run_id | 网格 |
|--------|------|
| H.tau.{0.3,0.5,0.7} | τ |
| H.alpha.{0.5,1.0,1.5} | α |
| H.gamma.{0.05,0.1,0.2} | γ |

```bash
./run.sh matrix --stage sensitivity
```

---

## 7. Stage 6 — 扩展（加分，非阻塞）

| run_id | 内容 |
|--------|------|
| E.7b.orm / E.7b.triage | Qwen2.5-Math-7B |
| E.rule_rp | 全过程用 rule process（稳健性） |
| E.prm_rp | 固定神经 PRM（主文过程分） |

```bash
./run.sh matrix --stage extend
```

---

## 8. 汇总与论文表

| run_id | 命令 | 产出 |
|--------|------|------|
| R.tables | `./run.sh summarize` | `outputs/tables/main.csv`, `ablation.csv`, `hack_rate.csv` |

---

## 9. 推荐执行顺序（最短路径）

```text
./run.sh unit
./run.sh download
./run.sh verify-assets
./run.sh matrix --stage smoke        # S2
./run.sh matrix --stage main         # Tab.1
./run.sh matrix --stage ablation     # Tab.2
./run.sh summarize
# 可选：
./run.sh matrix --stage sensitivity
./run.sh matrix --stage extend
```

或一条龙（有 GPU 时）：

```bash
./run.sh all
```

`all` = unit → download → verify → smoke → main → ablation → summarize。  
敏感性与 7B 不进 `all`，需显式加 `--with-sensitivity` / `--with-extend`。

---

## 10. 计数核对

| Stage | run 数量（约） |
|-------|----------------|
| S0 | 3 |
| S1 | 3 |
| S2 smoke | 2 |
| S3 main | 1 + 8×2 + 2 = **19**（含 B0 与 B6 两条） |
| S4 ablation | 6 |
| S5 sensitivity | 9 |
| S6 extend | 4 |
| **主文最低必须** | **S0+S1+S2+S3+S4+R ≈ 33 runs** |

机器注册表：`configs/experiment_registry.yaml`（`./run.sh matrix` 读取）。
