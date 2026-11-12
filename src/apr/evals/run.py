"""Evaluate a model, or every checkpoint of a run (including step 0 = the initial model).

  apr-eval --run runs/baseline --preset quick            # step 0, checkpoint-*, final
  apr-eval --model Qwen/Qwen2.5-0.5B-Instruct --suites gsm8k truthfulqa
Results: results/<run>/<step>.json  (summary metrics)  +  <step>.<suite>.jsonl (per-item, for audits)
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import torch
import yaml

from apr.evals.common import load_model
from apr.evals.suites import SUITES

PRESETS = {
    # ~10-20 min per checkpoint for a 0.5B model on an M-series laptop.
    "quick": {"gsm8k": {"limit": 200, "unanswerable_n": 100}, "truthfulqa": {"limit": 200},
              "bbq": {"limit_per_cat": 30}, "harmbench": {"limit": 100}, "xstest": {}},
    "full": {"gsm8k": {"limit": None, "unanswerable_n": 400}, "truthfulqa": {},
             "bbq": {"limit_per_cat": 400}, "harmbench": {"limit": 400}, "xstest": {}},
    "smoke": {"gsm8k": {"limit": 8, "unanswerable_n": 4, "max_new_tokens": 128}, "truthfulqa": {"limit": 8},
              "bbq": {"limit_per_cat": 2}, "harmbench": {"limit": 4, "max_new_tokens": 64},
              "xstest": {"max_new_tokens": 48, "limit": 8}},
}


def checkpoints(run: Path) -> list[tuple[int, str]]:
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    out = [(0, cfg["model"]["name"])]
    for d in run.glob("checkpoint-*"):
        out.append((int(d.name.split("-")[1]), str(d)))
    if (run / "final").exists():
        steps = [s for s, _ in out]
        last = cfg["grpo"].get("max_steps", max(steps) + 1)
        if last not in steps:
            out.append((last, str(run / "final")))
    return sorted(out)


def evaluate(path: str, suites: list[str], preset: str, out_json: Path, judge: str | None, batch_size: int):
    done = json.loads(out_json.read_text()) if out_json.exists() else {}
    todo = [s for s in suites if s not in done.get("_suites", [])]
    if not todo:
        print(f"[apr-eval] cached {out_json}")
        return done
    model, tok = load_model(path)
    for s in todo:
        kw = dict(PRESETS[preset].get(s, {}))
        kw.setdefault("batch_size", batch_size)
        if s == "harmbench":
            kw["judge"] = judge
        print(f"[apr-eval] {path} :: {s} {kw}")
        metrics, items = SUITES[s](model, tok, **kw)
        done.update(metrics)
        done.setdefault("_suites", []).append(s)
        with open(out_json.with_suffix(f".{s}.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps(it, default=str) + "\n")
        out_json.write_text(json.dumps(done, indent=1, default=list))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if not k.endswith("_ci")})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="training output dir; evaluates step 0 + all checkpoints")
    g.add_argument("--model", help="single HF id or local path")
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--preset", default="quick", choices=list(PRESETS))
    ap.add_argument("--results", default="results")
    ap.add_argument("--judge", default=None, help="HF id of a HarmBench classifier (default: string match)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", default=None, help="regex filter on step numbers, e.g. '^(0|100|300)$'")
    a = ap.parse_args(argv)

    if a.model:
        name = re.sub(r"[^\w.-]", "_", a.model)
        out = Path(a.results) / "_models" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        evaluate(a.model, a.suites, a.preset, out, a.judge, a.batch_size)
        return
    run = Path(a.run)
    rdir = Path(a.results) / run.name
    rdir.mkdir(parents=True, exist_ok=True)
    for step, path in checkpoints(run):
        if a.steps and not re.search(a.steps, str(step)):
            continue
        res = evaluate(path, a.suites, a.preset, rdir / f"{step:06d}.json", a.judge, a.batch_size)
        res.update({"_step": step, "_path": path, "_preset": a.preset})
        (rdir / f"{step:06d}.json").write_text(json.dumps(res, indent=1, default=list))


if __name__ == "__main__":
    main()
