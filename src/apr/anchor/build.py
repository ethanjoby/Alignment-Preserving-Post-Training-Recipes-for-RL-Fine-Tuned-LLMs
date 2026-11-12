"""Build anchor SFT data from the *initial* policy, optionally with a constitutional critique-revise pass.

  apr-build-anchor --model Qwen/Qwen2.5-0.5B-Instruct --n 1000 --constitutional \
      --out data/anchor/anchor_constitutional.jsonl

Why self-generated: anchoring on another model's responses would change style and confound the
alignment measurements. The constitutional pass (Bai et al. 2022) has the same model critique its
own draft against configs/conduct_spec.md and revise it, so anchors are the initial model's
*best-behaved* outputs, not merely its average ones.
Anchor prompts are generic (HH-RLHF), never the eval sets and never the unanswerable-math task,
so anchor-SFT stays a clean ablation separate from targeted environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apr.data import hh_prompts
from apr.evals.common import generate, load_model

CRITIQUE = ("Here is a request and a draft response.\n\nREQUEST:\n{p}\n\nDRAFT:\n{r}\n\n"
            "Critique the draft against these principles:\n{spec}\n\nList concrete problems only.")
REVISE = ("REQUEST:\n{p}\n\nDRAFT:\n{r}\n\nCRITIQUE:\n{c}\n\nRewrite the draft so it fixes every problem in the "
          "critique while staying as helpful as possible. Output only the final response, nothing else.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--constitutional", action="store_true")
    ap.add_argument("--spec", default="configs/conduct_spec.md")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    model, tok = load_model(a.model)
    prompts = hh_prompts(a.n, a.seed, split="train")
    kw = dict(max_new_tokens=a.max_new_tokens, batch_size=a.batch_size)
    drafts = generate(model, tok, prompts, **kw)
    final = drafts
    if a.constitutional:
        spec = Path(a.spec).read_text()
        crit = generate(model, tok, [CRITIQUE.format(p=p, r=r, spec=spec) for p, r in zip(prompts, drafts)], **kw)
        final = generate(model, tok, [REVISE.format(p=p, r=r, c=c) for p, r, c in zip(prompts, drafts, crit)], **kw)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p, d, r in zip(prompts, drafts, final):
            if r.strip():
                f.write(json.dumps({"prompt": p, "response": r.strip(), "draft": d}) + "\n")
    print(f"[apr] wrote {out}")


if __name__ == "__main__":
    main()
