"""Python-3.9-compatible GRPO loop (no trl.GRPOTrainer import).

Uses transformers + peft only. Includes reference KL and degeneracy guards
to avoid the empty / repetitive collapse seen in earlier shims.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from triage_grpo.advantage import compute_group_advantages
from triage_grpo.classify import assign_cells
from triage_grpo.metrics import cell_stats
from triage_grpo.rewards.outcome import score_outcome_exact
from triage_grpo.rewards.rule_process import RuleProcessScorer
from triage_grpo.types import GroupInput
from triage_grpo.verl_adapter import TriageConfig

_WORD_RE = re.compile(r"\w+")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repetition_ratio(text: str) -> float:
    toks = _WORD_RE.findall((text or "").lower())
    if len(toks) < 8:
        return 0.0
    return 1.0 - (len(set(toks)) / len(toks))


def is_degenerate(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return True
    if _repetition_ratio(t) >= 0.75:
        return True
    if re.search(r"(WHY\s+){5,}", t, re.I):
        return True
    if re.search(r"(.{1,20})\1{8,}", t):
        return True
    return False


class DegeneracyMonitor:
    def __init__(self, window: int = 64, bad_frac: float = 0.6):
        self.window = window
        self.bad_frac = bad_frac
        self.flags: List[int] = []

    def update(self, completions: Sequence[str]) -> None:
        for c in completions:
            self.flags.append(1 if is_degenerate(c) else 0)
        if len(self.flags) > self.window * 4:
            self.flags = self.flags[-self.window * 4 :]

    def should_abort(self) -> bool:
        if len(self.flags) < self.window:
            return False
        recent = self.flags[-self.window :]
        return (sum(recent) / float(len(recent))) >= self.bad_frac


def _completion_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: List[int],
) -> torch.Tensor:
    """Per-sequence mean logprob over completion tokens only."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    seq_lp = []
    for i, plen in enumerate(prompt_lens):
        # labels index j corresponds to token j+1; completion starts at plen
        start = max(plen - 1, 0)
        end = attention_mask[i, 1:].sum().item()
        if end <= start:
            seq_lp.append(torch.tensor(0.0, device=token_lp.device))
            continue
        seq_lp.append(token_lp[i, start:end].mean())
    return torch.stack(seq_lp)


@torch.no_grad()
def _generate_group(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, int]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(os.environ.get("TRIAGE_MAX_PROMPT", "1024")),
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = int(input_ids.shape[1])

    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=max(temperature, 1e-5),
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=group_size,
    )
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_cfg,
    )
    texts = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_len:]
        texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    # pad batch for later logprob
    pad_id = tokenizer.pad_token_id
    max_len = out.shape[1]
    # out already padded equally by generate
    attn = (out != pad_id).long()
    return texts, out, attn, prompt_len


def run_grpo39(
    *,
    run_id: str,
    root: Path,
    model_name: str,
    records: List[dict],
    triage_cfg: TriageConfig,
    method: str,
    seed: int,
    steps: int,
    lora: dict,
    smoke: bool = False,
    max_completion: int = 1024,
    group_size: int = 8,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = root / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if smoke and metrics_path.exists():
        metrics_path.unlink()

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    policy = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()

    if lora.get("enabled", True):
        peft_cfg = LoraConfig(
            r=int(lora.get("r", 64)),
            lora_alpha=int(lora.get("alpha", 128)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(
                lora.get("target_modules")
                or ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
        )
        policy = get_peft_model(policy, peft_cfg)

    policy.to(device)
    ref.to(device)

    lr = float(os.environ.get("TRIAGE_LR", "1e-6"))
    beta_kl = float(os.environ.get("TRIAGE_KL_BETA", "0.04"))
    temperature = float(os.environ.get("TRIAGE_TEMPERATURE", "0.7"))
    opt = AdamW((p for p in policy.parameters() if p.requires_grad), lr=lr)

    scorer = RuleProcessScorer()
    monitor = DegeneracyMonitor()
    gold_by_prompt = {r["prompt"]: r["answer"] for r in records}

    resolved = {
        "run_id": run_id,
        "backend": "grpo39_transformers_peft",
        "python_target": "3.9+",
        "model": model_name,
        "method": method,
        "seed": seed,
        "steps": steps,
        "lora": lora,
        "triage": asdict(triage_cfg),
        "anti_collapse": {
            "kl_beta": beta_kl,
            "learning_rate": lr,
            "temperature": temperature,
        },
        "created_at": _utc(),
    }
    (out_dir / "resolved_config.yaml").write_text(
        __import__("yaml").safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    policy.train()
    best_outcome = -1.0

    for step in range(1, int(steps) + 1):
        batch = random.choice(records)
        prompt = batch["prompt"]
        gold = gold_by_prompt[prompt]

        texts, full_ids, full_attn, prompt_len = _generate_group(
            policy,
            tok,
            prompt,
            group_size=group_size,
            max_new_tokens=max_completion,
            temperature=temperature,
            device=device,
        )
        monitor.update(texts)
        if monitor.should_abort():
            raise RuntimeError(
                "DEGENERACY_ABORT: recent completions mostly empty/repetitive. "
                "Lower TRIAGE_LR or raise TRIAGE_KL_BETA; do not expand matrix."
            )

        r_o = [score_outcome_exact(c, gold) for c in texts]
        r_p = [scorer.score(prompt, c) for c in texts]
        conf = [
            0.99 if is_degenerate(c) else min(0.95, 0.4 + 0.0005 * len(c)) for c in texts
        ]

        if method in ("orm_grpo", "base_eval"):
            from triage_grpo.advantage import group_normalize
            import numpy as np

            adv = group_normalize(np.asarray(r_o, dtype=np.float64)).tolist()
            masks = [True] * len(texts)
            cells = assign_cells(r_o, r_p, tau=triage_cfg.tau)
        else:
            out_adv = compute_group_advantages(
                GroupInput(
                    outcome_rewards=r_o, process_scores=r_p, confidences=conf
                ),
                method=method,
                tau=triage_cfg.tau,
                alpha=triage_cfg.alpha,
                beta=triage_cfg.beta,
                gamma=triage_cfg.gamma,
                w_high_mode=triage_cfg.w_high_mode,
                lens_normalize=triage_cfg.lens_normalize,
                hybrid_lambda=triage_cfg.hybrid_lambda,
                eps=triage_cfg.eps,
            )
            adv = list(out_adv.advantages)
            masks = list(out_adv.masks)
            cells = list(out_adv.cells)

        # Drop masked / degenerate from gradient (advantage 0)
        for i, (m, t) in enumerate(zip(masks, texts)):
            if (not m) or is_degenerate(t):
                adv[i] = 0.0

        prompt_lens = [prompt_len] * full_ids.shape[0]
        logp_pi = _completion_logprobs(policy, full_ids, full_attn, prompt_lens)
        with torch.no_grad():
            logp_ref = _completion_logprobs(ref, full_ids, full_attn, prompt_lens)

        adv_t = torch.tensor(adv, device=device, dtype=logp_pi.dtype)
        # GRPO-style: maximize adv * logpi, plus KL(pi||ref) ≈ mean(logpi - logref)
        pg_loss = -(adv_t * logp_pi).mean()
        kl = (logp_pi - logp_ref).mean()
        loss = pg_loss + beta_kl * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in policy.parameters() if p.requires_grad), 1.0
        )
        opt.step()

        mean_outcome = float(sum(r_o) / max(1, len(r_o)))
        best_outcome = max(best_outcome, mean_outcome)
        stats = cell_stats(cells)
        row = {
            "t": step,
            "loss": float(loss.detach().cpu()),
            "pg_loss": float(pg_loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "mean_outcome": mean_outcome,
            "mean_advantage": float(sum(adv) / max(1, len(adv))),
            "pct_degenerate": float(
                sum(1 for c in texts if is_degenerate(c)) / max(1, len(texts))
            ),
            "hack_rate": float(
                sum(
                    1
                    for o, p in zip(r_o, r_p)
                    if o < 0.5 and p >= triage_cfg.tau
                )
                / max(1, len(r_o))
            ),
            **{f"pct_{k}": float(v) for k, v in stats.items()},
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if step == 1 or step % 10 == 0:
            print(json.dumps(row, ensure_ascii=False), flush=True)

        save_every = int(os.environ.get("TRIAGE_SAVE_STEPS", "50"))
        if step % save_every == 0 or step == steps:
            step_dir = ckpt_root / f"ckpt-step-{step:06d}"
            if step_dir.exists():
                shutil.rmtree(step_dir, ignore_errors=True)
            step_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(step_dir)
            tok.save_pretrained(step_dir)
            # retention: keep newest only + best/final later
            for old in sorted(ckpt_root.glob("ckpt-step-*")):
                if old.name != step_dir.name:
                    shutil.rmtree(old, ignore_errors=True)

    final_dir = ckpt_root / "final"
    best_dir = ckpt_root / "best"
    if final_dir.exists():
        shutil.rmtree(final_dir, ignore_errors=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    if best_dir.exists():
        shutil.rmtree(best_dir, ignore_errors=True)
    shutil.copytree(final_dir, best_dir)

    if smoke:
        shutil.rmtree(ckpt_root, ignore_errors=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)
        (ckpt_root / "SMOKE_CHECKPOINTS_DELETED.txt").write_text(
            "smoke: checkpoints deleted\n", encoding="utf-8"
        )

    result = {
        "run_id": run_id,
        "status": "finished",
        "backend": "grpo39_transformers_peft",
        "steps": steps,
        "best_mean_outcome_seen": best_outcome,
        "finished_at": _utc(),
    }
    (out_dir / "train_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
