"""Train one recipe.  Usage:  apr-train configs/recipes/baseline.yaml [key.sub=value ...]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer
from trl import GRPOConfig

from apr.config import deep_merge, load_config, load_config_overrides
from apr.data import build_rl_dataset, load_anchor
from apr.rewards import build_rewards
from apr.trainer import AlignedGRPOTrainer


def _dtype(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32  # MPS/CPU: fp32 is the safe default for RL numerics
    return getattr(torch, name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. grpo.beta=0.04")
    ap.add_argument("--scale", help="scale config merged on top of the recipe, e.g. configs/scale/llama8b.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.scale:
        cfg = deep_merge(cfg, load_config(a.scale))
    cfg = load_config_overrides(cfg, a.overrides)

    out_dir = Path(cfg.get("output_dir") or f"runs/{cfg['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = build_rl_dataset(cfg, tok)
    funcs, weights = build_rewards(cfg)
    print(f"[apr] {len(train_ds)} prompts; rewards={[f.__name__ for f in funcs]} weights={weights}")

    grpo = dict(cfg.get("grpo", {}))
    dtype = _dtype(cfg["model"].get("dtype", "auto"))
    args = GRPOConfig(
        output_dir=str(out_dir),
        reward_weights=weights,
        model_init_kwargs={"dtype": dtype},
        bf16=dtype == torch.bfloat16,
        seed=cfg.get("seed", 0),
        report_to=cfg.get("report_to", "none"),
        run_name=cfg["name"],
        **grpo,
    )

    peft_config = None
    if cfg.get("lora"):
        from peft import LoraConfig

        l = cfg["lora"]
        peft_config = LoraConfig(
            r=l.get("r", 16), lora_alpha=l.get("alpha", 32), lora_dropout=l.get("dropout", 0.0),
            target_modules=l.get("target_modules", "all-linear"), task_type="CAUSAL_LM",
        )

    rc = cfg.get("recipe", {})
    anchor, sd = rc.get("anchor", {}), rc.get("self_distill", {})
    trainer = AlignedGRPOTrainer(
        model=model_name,
        reward_funcs=funcs,
        args=args,
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=peft_config,
        anchor_data=load_anchor(anchor["path"]) if anchor.get("coef", 0) > 0 else None,
        anchor_coef=anchor.get("coef", 0.0),
        anchor_every=anchor.get("every", 1),
        anchor_batch_size=anchor.get("batch_size", 4),
        sd_coef=sd.get("coef", 0.0),
        sd_clip=sd.get("clip", 5.0),
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    with open(out_dir / "log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    print(f"[apr] done -> {out_dir}")


if __name__ == "__main__":
    main()
