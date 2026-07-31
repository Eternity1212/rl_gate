# TRIAGE-GRPO 方法规格（SPEC）

版本：v0.1  
状态：已锁定，实现必须遵守本文；变更需改版本号并更新 STATUS。

---

## 1. 问题定义

在 GRPO + 可验证结果奖励（ORM）设定下，过程信号（PRM / 规则过程分）与结果信号会出现四类组合。已有工作通常只处理其中一部分：

| 工作 | 处理方式 | 盲点 |
|------|----------|------|
| ORM-GRPO | 只看对错 | 忽略过程；全对组梯度消失 |
| P-GRPO | 答错则过程奖励=0 | W↑ 与 W↓ 一视同仁 |
| PAPO | 过程优势只在正确子集归一 | 同上，不区分错误侧 |
| PROF | 过滤冲突样本 | C↓ 与 W↑ 一视同仁地丢 |
| LENS | 错答按置信度惩罚 | 不看过程分 |

**Claim**：过程–结果不一致性不是单一现象；fluent-wrong（W↑）与 sloppy-correct（C↓）必须用不同算子。

---

## 2. 符号

对固定 prompt \(x\)，策略采样 \(G\) 条回答 \(\{o_i\}_{i=1}^G\)：

- \(r_o^{(i)} \in \{0,1\}\)：结果奖励（答案可验证正确=1）
- \(r_p^{(i)} \in [0,1]\)：过程分（PRM 轨迹聚合或规则过程）
- \(c^{(i)} \in (0,1]\)：回答置信度代理（默认：序列平均 token 概率；可替换）
- \(\tau \in (0,1)\)：过程高低阈值（默认 0.5，实验可扫）
- \(\alpha, \beta, \gamma\)：过程优势、W↑ 降权、LENS 惩罚系数

---

## 3. 四格分类（必须实现）

```
if r_o == 1 and r_p >= τ  →  C_HIGH   # C↑
if r_o == 1 and r_p <  τ  →  C_LOW    # C↓
if r_o == 0 and r_p <  τ  →  W_LOW    # W↓
if r_o == 0 and r_p >= τ  →  W_HIGH   # W↑  黑客签名
```

枚举名在代码中固定为：`C_HIGH | C_LOW | W_LOW | W_HIGH`。

---

## 4. 四种待遇（算子）

### 4.1 结果优势 \(A_o\)（全组）

标准 GRPO 组归一：

\[
A_o^{(i)} = \frac{r_o^{(i)} - \mu_o}{\sigma_o + \varepsilon}
\]

其中 \(\mu_o,\sigma_o\) 在 \(G\) 条上计算。若 \(\sigma_o\approx 0\)（全对或全错），\(A_o^{(i)}=0\)。

### 4.2 过程优势 \(A_p\)（仅正确子集，PAPO 式）

令 \(C = \{i: r_o^{(i)}=1\}\)。若 \(|C|<2\)，则全体 \(A_p^{(i)}=0\)。否则仅在 \(C\) 上：

\[
A_p^{(i)} =
\begin{cases}
\dfrac{r_p^{(i)} - \mu_p^{C}}{\sigma_p^{C}+\varepsilon} & i\in C \\
0 & i\notin C
\end{cases}
\]

效果：C↑ 相对强化，C↓ 相对抑制；错误样本不获得正过程优势。

### 4.3 W↓：LENS 式置信负惩罚 \(A_n\)

仅对 `W_LOW`：

\[
A_n^{(i)} = -\gamma \cdot \tilde{c}^{(i)}
\quad\text{若 cell}=W\_LOW,\ \text{否则 }0
\]

\(\tilde{c}\) 为组内归一后的置信度（或原始 \(c\)，由配置 `lens_normalize` 控制）。  
**不做**对 W↑ 的 LENS 惩罚（避免与 4.4 双重计数）；W↑ 走过滤/降权。

### 4.4 W↑：冲突过滤 + 降权 \(A_h\)

对 `W_HIGH`：

1. **过滤模式**（`w_high_mode=filter`，对齐 PROF 精神）：该样本不进入策略梯度（mask=0），或优势置 0 且不参与过程相关项。
2. **降权模式**（`w_high_mode=downweight`，默认用于消融）：

\[
A_h^{(i)} = -\beta \cdot r_p^{(i)}
\quad\text{若 cell}=W\_HIGH
\]

主文默认：`filter`（更干净）；附录报告 `downweight`。

### 4.5 最终优势

\[
A^{(i)} = A_o^{(i)} + \alpha A_p^{(i)} + A_n^{(i)} + A_h^{(i)}
\]

再按 GRPO/veRL 惯例广播到 token（本库只负责序列级 \(A\)）。

---

## 5. 基线必须可开关复现

同一代码路径通过 `method` 字段切换：

| method | 行为 |
|--------|------|
| `orm_grpo` | 仅 \(A_o\) |
| `p_grpo` | \(A = \mathrm{norm}(r_o + r_o\cdot r_p)\)（答错过程门控为 0） |
| `papo` | \(A_o + \alpha A_p\)（无 \(A_n,A_h\)） |
| `prof` | 过滤 C↓ 与 W↑（或按 PROF 原设定过滤冲突），剩余做 ORM-GRPO |
| `lens` | \(A_o\) + 对所有 \(r_o=0\) 的置信惩罚 |
| `naive_hybrid` | \(\mathrm{norm}(\lambda r_o+(1-\lambda)r_p)\) |
| `prm_direct` | \(\mathrm{norm}(r_p)\) |
| `triage` | 完整四算子 |

---

## 6. 过程分 \(r_p\) 契约

接口：`ProcessScorer.score(prompt, response) -> float in [0,1]`

实现档位：

1. **rule**（稳妥）：格式合法 + 可解析答案 + 可选逐步规则（见 `rewards/rule_process.py`）
2. **prm**（主文）：外部 PRM / rubric；路径与模型名写入配置，**全文实验固定一个**

禁止在同一主表中途更换 PRM。

---

## 7. 监控指标（训练必须 log）

- 四格占比：`pct_c_high, pct_c_low, pct_w_low, pct_w_high`
- 黑客代理：`hack_rate = pct_w_high`；`mean_rp_given_wrong`
- 长度：`mean_response_tokens`
- 优势：`mean_abs_Ao, mean_abs_Ap, frac_zero_advantage`
- 过滤：`frac_masked_w_high`

---

## 8. 非目标（Out of scope v0.1）

- 多轮 Agent / SWE
- 训练新的大型 PRM
- 修改 Transformer 结构
- 与 DAPO/CISPO 同时比算法族（附录可选）

---

## 9. 验收（实现层）

1. 给定手算小例子，四格标签与 \(A_o,A_p,A_n,A_h\) 误差 \(< 1e-5\)
2. `method=papo` 时与手写 PAPO 一致
3. `method=triage` + `w_high_mode=filter` 时 W↑ 的 `mask=False`
4. 单测不依赖 GPU
