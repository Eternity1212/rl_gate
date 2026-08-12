#!/usr/bin/env python3
"""Paper-oriented TRL GRPO entry (replaces collapsed local training shim).

Design goals
------------
1. Use HuggingFace ``trl.GRPOTrainer`` (real GRPO + reference KL), not a homemade
   policy-loss toy loop that can collapse to empty / repetitive text.
2. Keep TRIAGE/baseline *reward shaping* in-repo via ``triage_grpo``.
3. Fail fast if generations degenerate (empty / high repetition).

Python:
  - 3.10+ : try official trl.GRPOTrainer
  - 3.9   : automatic fallback to ``triage_grpo.grpo39_trainer`` (no trl.GRPOTrainer)

Force 3.9 path anytime with: ``export TRIAGE_FORCE_GRPO39=1``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triage_grpo.advantage import compute_group_advantages  # noqa: E402
from triage_grpo.classify import assign_cells  # noqa: E402
from triage_grpo.metrics import cell_stats  # noqa: E402
from triage_grpo.rewards.outcome import score_outcome_exact  # noqa: E402
from triage_grpo.rewards.rule_process import RuleProcessScorer  # noqa: E402
from triage_grpo.types import GroupInput  # noqa: E402
from triage_grpo.verl_adapter import TriageConfig  # noqa: E402


def _prefer_grpo39() -> bool:
    if os.environ.get("TRIAGE_FORCE_GRPO39") == "1":
        return True
    if sys.version_info < (3, 10):
        return True
    # Probe TRL GRPO import (fails on py3.9 with newer transformers/trl)
    try:
        from trl import GRPOTrainer  # noqa: F401
        return False
    except Exception as e:
        print(f"[warn] trl.GRPOTrainer unavailable ({e}); using grpo39 backend", flush=True)
        return True


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_model_path(cfg: dict) -> str:
    p = _deep_get(cfg, "model", "path", default="Qwen/Qwen2.5-Math-1.5B")
    local = ROOT / "assets" / "models" / "Qwen2.5-Math-1.5B"
    if local.exists() and "1.5B" in str(p):
        return str(local)
    return str(p)


def _guess_prompt_answer(ex: dict) -> Optional[tuple[str, str]]:
    """Map heterogeneous math datasets to (prompt, gold_answer)."""
    prompt_keys = ("prompt", "problem", "question", "query", "input")
    answer_keys = ("answer", "gt_answer", "ground_truth", "label", "target", "solution")
    prompt = None
    for k in prompt_keys:
        if k in ex and ex[k] is not None:
            prompt = str(ex[k])
            break
    # DAPO / some HF rows nest under reward_model / extra_info
    if prompt is None and isinstance(ex.get("extra_info"), dict):
        for k in prompt_keys:
            if k in ex["extra_info"]:
                prompt = str(ex["extra_info"][k])
                break
    gold = None
    for k in answer_keys:
        if k in ex and ex[k] is not None:
            gold = str(ex[k])
            break
    if gold is None and isinstance(ex.get("reward_model"), dict):
        rm = ex["reward_model"]
        if "ground_truth" in rm:
            gold = str(rm["ground_truth"])
    if prompt is None or gold is None:
        return None
    # Prefer chat-style user content if already messages
    if isinstance(ex.get("prompt"), list):
        try:
            prompt = "\n".join(
                m.get("content", "") for m in ex["prompt"] if isinstance(m, dict)
            )
        except Exception:
            pass
    instruct = (
        "Solve the following math problem. Put the final integer answer in "
        "\\boxed{}.\n\n" + prompt.strip()
    )
    return instruct, gold.strip()


def load_train_records(max_samples: Optional[int] = None) -> List[dict]:
    """Load local parquet/jsonl under assets, else stream HF dataset."""
    local_dir = ROOT / "assets" / "data" / "DAPO-Math-17k"
    records: List[dict] = []

    if local_dir.exists():
        try:
            from datasets import load_from_disk

            ds = load_from_disk(str(local_dir))
            split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
            for ex in split:
                pa = _guess_prompt_answer(dict(ex))
                if pa:
                    records.append({"prompt": pa[0], "answer": pa[1]})
        except Exception:
            # try raw files
            for fp in list(local_dir.rglob("*.jsonl"))[:1]:
                for line in fp.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    pa = _guess_prompt_answer(json.loads(line))
                    if pa:
                        records.append({"prompt": pa[0], "answer": pa[1]})

    if not records:
        from datasets import load_dataset

        for repo in (
            "BytedTsinghua-SIA/DAPO-Math-17k",
            "open-r1/DAPO-Math-17k-Processed",
        ):
            try:
                ds = load_dataset(repo)
                split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
                for ex in split:
                    pa = _guess_prompt_answer(dict(ex))
                    if pa:
                        records.append({"prompt": pa[0], "answer": pa[1]})
                if records:
                    break
            except Exception as e:
                print(f"[warn] load {repo} failed: {e}")

    if not records:
        raise RuntimeError(
            "No training records found. Run ./run.sh download first, "
            "or ensure HF access to DAPO-Math-17k."
        )
    if max_samples is not None:
        records = records[: max(1, int(max_samples))]
    return records


_WORD_RE = re.compile(r"\w+")


def _repetition_ratio(text: str) -> float:
    toks = _WORD_RE.findall((text or "").lower())
    if len(toks) < 8:
        return 0.0
    return 1.0 - (len(set(toks)) / len(toks))


def _is_degenerate(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return True
    if _repetition_ratio(t) >= 0.75:
        return True
    # classic collapse patterns
    if re.search(r"(WHY\s+){5,}", t, re.I):
        return True
    if re.search(r"(.{1,20})\1{8,}", t):
        return True
    return False


class DegeneracyMonitor:
    """Abort training if a moving window of completions is mostly garbage."""

    def __init__(self, window: int = 64, bad_frac: float = 0.6):
        self.window = window
        self.bad_frac = bad_frac
        self.flags: List[int] = []

    def update(self, completions: Sequence[str]) -> None:
        for c in completions:
            self.flags.append(1 if _is_degenerate(c) else 0)
        if len(self.flags) > self.window * 4:
            self.flags = self.flags[-self.window * 4 :]

    def should_abort(self) -> bool:
        if len(self.flags) < self.window:
            return False
        recent = self.flags[-self.window :]
        return (sum(recent) / len(recent)) >= self.bad_frac


def build_reward_fn(
    method: str,
    triage_cfg: TriageConfig,
    gold_by_prompt: Dict[str, str],
    monitor: DegeneracyMonitor,
    metrics_path: Path,
):
    """TRL reward_funcs callable.

    For ORM: scalar outcome.
    For TRIAGE/baselines: compute in-repo group advantages and return them as
    the reward vector so GRPO's group norm is applied on a triage-shaped signal.
    (We keep ``scale_rewards`` configurable; advantages already carry method logic.)
    """
    scorer = RuleProcessScorer()
    step_box = {"n": 0}

    def reward_func(completions: List[str], prompts: List[str], **kwargs):
        del kwargs
        n = len(completions)
        monitor.update(completions)
        # Group size G: consecutive completions share a prompt
        # TRL typically passes flattened [p0]*G + [p1]*G ...
        rewards = [0.0] * n
        i = 0
        while i < n:
            p = prompts[i]
            j = i
            while j < n and prompts[j] == p:
                j += 1
            group_comp = completions[i:j]
            gold = gold_by_prompt.get(p, "")
            r_o = [score_outcome_exact(c, gold) for c in group_comp]
            r_p = [scorer.score(p, c) for c in group_comp]
            # cheap confidence proxy: inverse degeneracy / length
            conf = []
            for c in group_comp:
                if _is_degenerate(c):
                    conf.append(0.99)  # confident nonsense → LENS can punish
                else:
                    conf.append(min(0.95, 0.4 + 0.0005 * len(c)))

            if method in ("orm_grpo", "base_eval"):
                for k, v in enumerate(r_o):
                    rewards[i + k] = float(v)
            else:
                out = compute_group_advantages(
                    GroupInput(
                        outcome_rewards=r_o,
                        process_scores=r_p,
                        confidences=conf,
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
                # Use advantages as reward signal; mask → strongly negative
                for k, (a, m) in enumerate(zip(out.advantages, out.masks)):
                    rewards[i + k] = float(a) if m else -1.0

            # metrics line
            step_box["n"] += 1
            cells = assign_cells(r_o, r_p, tau=triage_cfg.tau)
            stats = cell_stats(cells)
            row = {
                "t": step_box["n"],
                "mean_outcome": float(sum(r_o) / max(1, len(r_o))),
                "mean_reward": float(sum(rewards[i:j]) / max(1, j - i)),
                "hack_rate": float(
                    sum(
                        1
                        for ro, rp in zip(r_o, r_p)
                        if ro < 0.5 and rp >= triage_cfg.tau
                    )
                    / max(1, len(r_o))
                ),
                "pct_degenerate": float(
                    sum(1 for c in group_comp if _is_degenerate(c))
                    / max(1, len(group_comp))
                ),
                **{f"pct_{k}": float(v) for k, v in stats.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if monitor.should_abort():
                raise RuntimeError(
                    "DEGENERACY_ABORT: >={:.0%} completions in recent window "
                    "are empty/repetitive. Check LR/KL/decode; do not continue "
                    "the experiment matrix.".format(monitor.bad_frac)
                )
            i = j
        return rewards

    return reward_func


def retain_checkpoints(ckpt_root: Path, keep_steps: int = 1) -> None:
    """Keep best/, final/, and the newest step dir only."""
    if not ckpt_root.exists():
        return
    step_dirs = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("ckpt-step-")],
        key=lambda p: p.name,
    )
    for p in step_dirs[:-keep_steps]:
        shutil.rmtree(p, ignore_errors=True)


def run_training(
    *,
    run_id: str,
    config_path: Optional[Path],
    method: str,
    seed: int,
    steps: int,
    overrides: Optional[dict] = None,
    smoke: bool = False,
) -> dict:
    overrides = overrides or {}
    cfg = _load_yaml(config_path) if config_path and config_path.exists() else {}
    # apply flat overrides like triage.tau
    for k, v in overrides.items():
        parts = str(k).split(".")
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v

    triage_cfg = TriageConfig.from_mapping(cfg.get("triage") or {"method": method})
    if method:
        triage_cfg.method = method

    out_dir = ROOT / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists() and smoke:
        metrics_path.unlink()

    resolved = {
        "run_id": run_id,
        "method": triage_cfg.method,
        "seed": seed,
        "steps": steps,
        "smoke": smoke,
        "model": _resolve_model_path(cfg),
        "lora": cfg.get("lora")
        or {
            "enabled": True,
            "r": 64,
            "alpha": 128,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "triage": asdict(triage_cfg),
        "trainer_backend": "auto",
        "anti_collapse": {
            "kl_beta": float(os.environ.get("TRIAGE_KL_BETA", "0.04")),
            "learning_rate": float(os.environ.get("TRIAGE_LR", "1.0e-6")),
            "degeneracy_abort": True,
        },
        "created_at": _utc(),
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
    }

    max_samples = int(os.environ.get("TRIAGE_MAX_SAMPLES", "0")) or None
    if smoke and max_samples is None:
        max_samples = 256
    records = load_train_records(max_samples=max_samples)
    model_name = resolved["model"]
    lora_cfg = resolved["lora"]
    max_completion = int(
        _deep_get(cfg, "data", "max_response_length", default=1024) or 1024
    )
    max_completion = int(os.environ.get("TRIAGE_MAX_COMPLETION", str(min(max_completion, 1024))))
    group_size = int(_deep_get(cfg, "algorithm", "group_size", default=8) or 8)

    # -------- Python 3.9 / incompatible TRL: local GRPO --------
    if _prefer_grpo39():
        from triage_grpo.grpo39_trainer import run_grpo39

        resolved["trainer_backend"] = "grpo39_transformers_peft"
        (out_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(
            "[info] Using grpo39 backend (Python 3.9-safe; no trl.GRPOTrainer).",
            flush=True,
        )
        return run_grpo39(
            run_id=run_id,
            root=ROOT,
            model_name=model_name,
            records=records,
            triage_cfg=triage_cfg,
            method=triage_cfg.method,
            seed=seed,
            steps=steps,
            lora=lora_cfg,
            smoke=smoke,
            max_completion=max_completion,
            group_size=group_size,
        )

    # -------- Python 3.10+ : official TRL --------
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    resolved["trainer_backend"] = "trl.GRPOTrainer"
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    set_seed(seed)
    gold_by_prompt = {r["prompt"]: r["answer"] for r in records}
    train_ds = Dataset.from_list([{"prompt": r["prompt"]} for r in records])

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    peft_config = None
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 64)),
            lora_alpha=int(lora_cfg.get("alpha", 128)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(
                lora_cfg.get("target_modules")
                or ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
        )

    n_gpus = int(_deep_get(cfg, "trainer", "n_gpus", default=1) or 1)
    monitor = DegeneracyMonitor(window=64, bad_frac=0.6)
    reward_fn = build_reward_fn(
        triage_cfg.method, triage_cfg, gold_by_prompt, monitor, metrics_path
    )

    cfg_kwargs = dict(
        output_dir=str(ckpt_root),
        seed=seed,
        max_steps=int(steps),
        per_device_train_batch_size=int(os.environ.get("TRIAGE_BATCH", "1")),
        gradient_accumulation_steps=int(os.environ.get("TRIAGE_GRAD_ACCUM", "4")),
        learning_rate=float(resolved["anti_collapse"]["learning_rate"]),
        logging_steps=1,
        save_steps=int(os.environ.get("TRIAGE_SAVE_STEPS", "50")),
        save_total_limit=2,
        num_generations=group_size,
        max_completion_length=max_completion,
        beta=float(resolved["anti_collapse"]["kl_beta"]),
        temperature=float(os.environ.get("TRIAGE_TEMPERATURE", "0.7")),
        report_to=[],
        bf16=bool(torch.cuda.is_available()),
        remove_unused_columns=False,
    )
    for k, v in (
        ("max_prompt_length", int(_deep_get(cfg, "data", "max_prompt_length", default=1024) or 1024)),
        ("scale_rewards", False),
    ):
        cfg_kwargs[k] = v
    try:
        grpo_args = GRPOConfig(**cfg_kwargs)
    except TypeError:
        for opt in ("max_prompt_length", "scale_rewards", "temperature"):
            cfg_kwargs.pop(opt, None)
        grpo_args = GRPOConfig(**cfg_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=grpo_args,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    try:
        trainer = GRPOTrainer(processing_class=tok, **trainer_kwargs)
    except TypeError:
        trainer = GRPOTrainer(tokenizer=tok, **trainer_kwargs)

    train_result = trainer.train()
    final_dir = ckpt_root / "final"
    best_dir = ckpt_root / "best"
    trainer.save_model(str(final_dir))
    if best_dir.exists():
        shutil.rmtree(best_dir, ignore_errors=True)
    shutil.copytree(final_dir, best_dir)
    step_snap = ckpt_root / f"ckpt-step-{int(steps):06d}"
    if step_snap.exists():
        shutil.rmtree(step_snap, ignore_errors=True)
    shutil.copytree(final_dir, step_snap)
    retain_checkpoints(ckpt_root, keep_steps=1)

    if smoke:
        shutil.rmtree(ckpt_root, ignore_errors=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)
        (ckpt_root / "SMOKE_CHECKPOINTS_DELETED.txt").write_text(
            "smoke run: checkpoints deleted after train\n", encoding="utf-8"
        )

    result = {
        "run_id": run_id,
        "status": "finished",
        "steps": steps,
        "train_loss": float(getattr(train_result, "training_loss", math.nan) or math.nan),
        "backend": "trl.GRPOTrainer",
        "n_gpus_config": n_gpus,
        "finished_at": _utc(),
    }
    (out_dir / "train_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", default="triage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--overrides-json",
        default="{}",
        help='JSON dict of overrides, e.g. {"triage.tau": 0.3}',
    )
    args = ap.parse_args()
    overrides = json.loads(args.overrides_json)
    cfg_path = Path(args.config) if args.config else None
    if cfg_path and not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    try:
        result = run_training(
            run_id=args.run_id,
            config_path=cfg_path,
            method=args.method,
            seed=args.seed,
            steps=args.steps,
            overrides=overrides,
            smoke=args.smoke,
        )
    except Exception as e:
        out_dir = ROOT / "outputs" / args.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        err = {
            "run_id": args.run_id,
            "status": "error",
            "message": str(e),
            "finished_at": _utc(),
        }
        (out_dir / "train_result.json").write_text(
            json.dumps(err, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(err, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
