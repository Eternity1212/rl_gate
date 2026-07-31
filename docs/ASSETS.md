# 需要下载的模型 / 数据集与磁盘占用

一键下载：`./run.sh download`（可选 `--with-7b`）  
落盘根目录：`assets/`（**不进 Git**，只提交代码与文档）

---

## 1. 清单（写清楚要下什么）

| 类型 | Hugging Face ID | 用途 | 是否必须 |
|------|-----------------|------|----------|
| 主模型 | `Qwen/Qwen2.5-Math-1.5B` | 全部主实验 | **必须** |
| 扩展模型 | `Qwen/Qwen2.5-Math-7B` | Stage extend | 可选 |
| 训练数据 | `BytedTsinghua-SIA/DAPO-Math-17k` | RL 训练 | **必须** |
| 训练数据（备用） | `open-r1/DAPO-Math-17k-Processed` | 官方失败时回退 | 自动回退 |
| 评测 | `HuggingFaceH4/MATH-500` | 主表 | **必须** |
| 评测 | `HuggingFaceH4/aime_2024` | 主表 | 建议 |
| 评测 | `MathArena/aime_2025`（若失效需替换） | 主表 | 建议 |
| 评测 | `math-ai/amc23` | 主表 | 建议 |
| 评测 | `Hothan/OlympiadBench` | 主表 | 建议 |
| 评测 | `math-ai/minervamath` | 主表 | 建议 |
| 框架代码 | veRL（单独安装，非 HF 权重） | 真训练 | Phase B 必须 |

本地路径约定：

```text
assets/models/Qwen2.5-Math-1.5B/
assets/models/Qwen2.5-Math-7B/          # 仅 --with-7b
assets/data/DAPO-Math-17k/
assets/data/eval/<BENCH>/
assets/manifest.json
```

---

## 2. 大概占用空间（按 HF 元数据估算，2026-07 测）

| 资源 | 下载体积（约） | 备注 |
|------|----------------|------|
| Qwen2.5-Math-1.5B | **~3.1 GB** | FP16 safetensors |
| Qwen2.5-Math-7B | **~15.2 GB** | 可选 |
| DAPO-Math-17k（官方） | **~0.3 GB** | 训练主数据 |
| DAPO-Math-17k-Processed | ~0.01 GB | 备用，很小 |
| MATH-500 | <10 MB | |
| AIME24 / AMC23 / Minerva | 各 <10 MB | |
| OlympiadBench | **~0.1 GB** | 评测里最大的一块 |
| AIME25 | 视镜像而定；失败则脚本写 `DOWNLOAD_FAILED.txt` | |
| HF 缓存 / 解压冗余 | 额外 **+20–50%** | `~/.cache/huggingface` 可能另占一份 |

### 推荐预留

| 场景 | 建议预留磁盘 |
|------|----------------|
| **最低可跑主文（1.5B + 训练 + 评测）** | **10–15 GB**（含缓存余量） |
| **加上 7B** | **30–40 GB** |
| **真训练（checkpoint + rollout 缓存）** | 再另加 **50–200+ GB**（看 steps/保存频率；不在 download 里） |

> 上面是「下载资产」空间，不是 GPU 显存。1.5B GRPO 训练通常仍建议 **多卡 A100/H100**（见实验计划）。

---

## 3. 命令

```bash
./run.sh setup
source .venv/bin/activate

# 国内可先：
# export HF_ENDPOINT=https://hf-mirror.com

./run.sh download              # ~3.5–5 GB 有效文件 + 缓存
./run.sh download --with-7b    # 再加 ~15 GB
./run.sh verify-assets
```

检查占用：

```bash
du -sh assets assets/models assets/data 2>/dev/null
du -sh ~/.cache/huggingface 2>/dev/null
```

---

## 4. 不进 GitHub 的内容

以下**故意不上传**（体积大 / 许可）：

- `assets/models/**`
- `assets/data/**`
- `outputs/**`、checkpoint、wandb

GitHub 只放：代码、配置、MD 说明书、下载脚本。
