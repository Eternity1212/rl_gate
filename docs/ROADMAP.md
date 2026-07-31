# 路线图：文档 → 代码 → 实验

---

## Phase A — Spec Coding（已完成，Week 0–1）

目标：不依赖 GPU，完成可测核心库。

- [x] README + SPEC + 实验锁定文档  
- [x] `triage_grpo` 四格分类 / 优势 / 过滤 / 指标  
- [x] 基线 method 开关  
- [x] 单元测试 + `unit_check_advantage.py`（12 passed）  
- [x] YAML 配置模板  
- [x] veRL 接入伪代码 / adapter 接口（不强制跑通全训练）

**完成标准**：`pytest` 全绿；手算脚本通过。 ✅

---

## Phase B — veRL 接入（Week 1–3）

- [ ] 安装 veRL，拉取 DAPO-Math-17k  
- [ ] Outcome reward（数学答案核对）  
- [ ] ProcessScorer（rule 先，prm 后）  
- [ ] 自定义 advantage 钩子接入 GRPO  
- [ ] 跑通 ORM-GRPO 小步（200 steps）  
- [ ] 日志：四格占比  

**熔断**：小步都跑不通 → 不进入主实验，先修数据/奖励。

---

## Phase C — 基线与主实验（Week 4–9）

按 EXPERIMENT_PLAN：B1→B6→TRIAGE→消融→主表。

---

## Phase D — 写作与复现包（Week 10–12）

- 主图主表  
- related work 对齐 RELATED_WORK.md  
- 配置、种子、脚本打包  

---

## 代码分期原则

1. **先纯函数，后框架**：advantage 库零依赖 torch 也可测（可用 numpy）。  
2. **配置驱动 method**：禁止为每个基线复制粘贴一整份 trainer。  
3. **一次一个旋钮**：改 \(\tau\) 时不改数据。  
4. **每阶段更新 STATUS.md**。  
