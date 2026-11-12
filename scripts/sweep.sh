#!/usr/bin/env bash
# Full experiment: train every recipe x seed, evaluate every checkpoint, plot the tradeoff.
#   SCALE=configs/scale/qwen1p5b.yaml SEEDS="0 1 2" PRESET=full bash scripts/sweep.sh
set -euo pipefail
SEEDS=${SEEDS:-"0"}
PRESET=${PRESET:-quick}
SCALE_ARG=${SCALE:+--scale $SCALE}
MODEL=$(python -c "import yaml,sys;c=yaml.safe_load(open('${SCALE:-configs/base.yaml}'));print(c.get('model',{}).get('name','Qwen/Qwen2.5-0.5B-Instruct'))")
RECIPES=${RECIPES:-"baseline kl anchor_mix interleaved_sft targeted_env self_distill self_distill_nospec rm_ensemble combined"}
KL_BETAS=${KL_BETAS:-"0.01 0.04 0.1 0.3"}

# 1. Anchor data from the *initial* policy (once per base model).
ANCHOR=data/anchor/anchor_constitutional.jsonl
[ -f "$ANCHOR" ] || apr-build-anchor --model "$MODEL" --n 1000 --constitutional --out "$ANCHOR"

# 2. Train.
for s in $SEEDS; do
  for r in $RECIPES; do
    if [ "$r" = kl ]; then
      for b in $KL_BETAS; do
        n=kl_${b}_s$s; [ -d runs/$n/final ] || apr-train configs/recipes/kl.yaml $SCALE_ARG grpo.beta=$b seed=$s name=$n
      done
    else
      n=${r}_s$s; [ -d runs/$n/final ] || apr-train configs/recipes/$r.yaml $SCALE_ARG seed=$s name=$n
    fi
  done
done

# 3. Evaluate step 0 + every checkpoint (results are cached per step/suite).
for d in runs/*_s*/; do apr-eval --run "$d" --preset "$PRESET" ${JUDGE:+--judge $JUDGE}; done

# 4. Tradeoff curve + summary table.
apr-plot --results results --baseline baseline --out figures
