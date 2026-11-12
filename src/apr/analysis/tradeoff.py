"""Capability-vs-alignment tradeoff analysis across recipes.

  apr-plot --results results --baseline baseline --out figures

Produces
  figures/tradeoff.png         capability gain (x) vs alignment composite (y), one trajectory per recipe
  figures/metrics_by_step.png  small multiples: every alignment metric vs training step
  figures/summary.md / .csv    headline table

Headline numbers (per recipe, relative to the baseline = capability-maximizing GRPO):
  gain_retained   = (cap_final - cap_0) / (cap_baseline_final - cap_0)
  align_tax       = align_0 - align_final               (composite; positive = alignment lost)
  tax_reduction   = 1 - align_tax / align_tax_baseline
  align_at_match  = alignment composite interpolated where the recipe's trajectory first reaches
                    cap_0 + MATCH * baseline_gain   (NaN if it never gets there)
The thesis is supported when some recipe has gain_retained ~>= 0.9 and tax_reduction >> 0,
with CIs that do not overlap the baseline (run >= 3 seeds; the table averages run dirs named
<recipe>_s<seed>).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAPABILITY = "gsm8k_acc"
# metric -> (orientation, family). Orientation -1 means lower raw value is better.
ALIGN = {
    "truthfulqa_mc2": (1, "chat"),
    "bbq_unbiased": (1, "chat"),
    "bbq_acc_ambig": (1, "chat"),
    "harmbench_asr": (-1, "chat"),
    "xstest_safe_compliance": (1, "chat"),
    "xstest_unsafe_refusal": (1, "chat"),
    "abstain_on_unanswerable": (1, "rl_misbehavior"),
    "fabrication_rate": (-1, "rl_misbehavior"),
    "hack_rate": (-1, "rl_misbehavior"),
}
MATCH = 0.9

# Reference categorical palette (fixed order, light surface); baseline is the neutral reference.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE_COLOR = "#52514e"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def oriented(m: dict, k: str) -> float | None:
    v = m.get(k)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v if ALIGN[k][0] > 0 else 1 - v


def composite(m: dict, family: str | None = None) -> float | None:
    vals = [oriented(m, k) for k, (_, f) in ALIGN.items() if family in (None, f)]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def load(results: Path) -> dict[str, list[dict]]:
    """recipe -> list (over seeds) of trajectories [{step, metrics}] sorted by step."""
    runs: dict[str, list[list[dict]]] = defaultdict(list)
    for d in sorted(p for p in results.iterdir() if p.is_dir() and not p.name.startswith("_")):
        traj = []
        for f in sorted(d.glob("[0-9]*.json")):
            if f.name.count(".") > 1:
                continue
            m = json.loads(f.read_text())
            traj.append({"step": m.get("_step", int(f.stem)), "m": m})
        if traj:
            recipe = re.sub(r"_s\d+$", "", d.name)
            runs[recipe].append(sorted(traj, key=lambda t: t["step"]))
    return runs


def avg_traj(seeds: list[list[dict]]) -> list[dict]:
    """Average seeds step-by-step on the steps all seeds share."""
    common = set.intersection(*[{t["step"] for t in s} for s in seeds])
    out = []
    for step in sorted(common):
        ms = [next(t["m"] for t in s if t["step"] == step) for s in seeds]
        keys = set.intersection(*[set(m) for m in ms])
        agg = {k: sum(m[k] for m in ms) / len(ms) for k in keys if all(isinstance(m[k], (int, float)) for m in ms)}
        agg["_n_seeds"] = len(ms)
        out.append({"step": step, "m": agg})
    return out


def align_at_capability(traj: list[dict], target: float) -> float:
    pts = [(t["m"].get(CAPABILITY), composite(t["m"])) for t in traj]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    for (c0, a0), (c1, a1) in zip(pts, pts[1:]):
        if c0 < target <= c1:
            w = (target - c0) / (c1 - c0)
            return a0 + w * (a1 - a0)
    if pts and pts[0][0] >= target:
        return pts[0][1]
    return float("nan")


def summarize(runs: dict, baseline: str) -> list[dict]:
    trajs = {r: avg_traj(s) for r, s in runs.items()}
    b = trajs.get(baseline)
    if not b:
        raise SystemExit(f"baseline run '{baseline}' not found in results; have {sorted(trajs)}")
    cap0, al0 = b[0]["m"].get(CAPABILITY), composite(b[0]["m"])
    b_gain = b[-1]["m"].get(CAPABILITY) - cap0
    b_tax = al0 - composite(b[-1]["m"])
    target = cap0 + MATCH * b_gain
    rows = []
    for r, t in trajs.items():
        fin = t[-1]["m"]
        gain = fin.get(CAPABILITY, float("nan")) - cap0
        tax = al0 - (composite(fin) or float("nan"))
        rows.append({
            "recipe": r,
            "seeds": fin.get("_n_seeds", 1),
            "final_step": t[-1]["step"],
            "capability": fin.get(CAPABILITY),
            "cap_gain": gain,
            "gain_retained": gain / b_gain if b_gain else float("nan"),
            "align_composite": composite(fin),
            "align_chat": composite(fin, "chat"),
            "align_rl_misbehavior": composite(fin, "rl_misbehavior"),
            "align_tax": tax,
            "tax_reduction": 1 - tax / b_tax if b_tax > 0 else float("nan"),
            f"align_at_{int(MATCH*100)}pct_gain": align_at_capability(t, target),
            **{k: fin.get(k) for k in ALIGN},
        })
    rows.sort(key=lambda x: (x["recipe"] != baseline, -(x["align_composite"] or 0)))
    return rows


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(points, key=lambda p: (-p[0], -p[1]))
    front, best = [], -1e9
    for x, y in pts:
        if y > best:
            front.append((x, y))
            best = y
    return sorted(front)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def colors_for(recipes: list[str], baseline: str) -> dict[str, tuple[str, str]]:
    out, i = {}, 0
    for r in sorted(recipes):
        if r == baseline:
            out[r] = (BASELINE_COLOR, "-")
        elif i < len(SERIES):
            out[r] = (SERIES[i], "-")
            i += 1
        else:  # never generate a 9th hue; extra runs are drawn as muted dotted lines
            out[r] = ("#a3a29c", ":")
    return out


def plot_tradeoff(runs, baseline, out: Path):
    trajs = {r: avg_traj(s) for r, s in runs.items()}
    cap0 = trajs[baseline][0]["m"][CAPABILITY]
    col = colors_for(list(trajs), baseline)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    finals = []
    for r, t in sorted(trajs.items()):
        xs = [p["m"].get(CAPABILITY, float("nan")) - cap0 for p in t]
        ys = [composite(p["m"]) for p in t]
        c, ls = col[r]
        ax.plot(xs, ys, ls, color=c, linewidth=2, alpha=0.9, zorder=2)
        ax.scatter(xs[-1:], ys[-1:], s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(r, (xs[-1], ys[-1]), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=INK)
        finals.append((xs[-1], ys[-1]))
    ax.scatter([0], [composite(trajs[baseline][0]["m"])], s=70, marker="D", color=INK, zorder=5)
    ax.annotate("initial model", (0, composite(trajs[baseline][0]["m"])), xytext=(6, -12),
                textcoords="offset points", fontsize=8.5, color=INK2)
    fr = pareto(finals)
    if len(fr) > 1:
        ax.plot(*zip(*fr), color=INK2, linewidth=1, linestyle="--", zorder=1, label="Pareto frontier (final points)")
        ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)
    b_gain = trajs[baseline][-1]["m"][CAPABILITY] - cap0
    ax.axvline(MATCH * b_gain, color=GRID, linewidth=1.5, zorder=0)
    ax.text(MATCH * b_gain, ax.get_ylim()[0], f"{int(MATCH*100)}% of baseline gain ", fontsize=8, color=INK2,
            va="bottom", ha="right")
    ax.set_xlabel(f"Capability gain over initial model ({CAPABILITY}, strict)", color=INK, fontsize=10)
    ax.set_ylabel("Alignment composite (higher is better)", color=INK, fontsize=10)
    ax.set_title("Capability vs. alignment during RL fine-tuning", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_metrics(runs, baseline, out: Path):
    trajs = {r: avg_traj(s) for r, s in runs.items()}
    col = colors_for(list(trajs), baseline)
    keys = [CAPABILITY] + [k for k in ALIGN if any(k in p["m"] for t in trajs.values() for p in t)]
    n = len(keys)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows), dpi=140, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, k in zip(axes.flat, keys):
        _style(ax)
        for r, t in sorted(trajs.items()):
            pts = [(p["step"], p["m"][k]) for p in t if k in p["m"]]
            if pts:
                c, ls = col[r]
                ax.plot(*zip(*pts), ls, color=c, linewidth=2, marker="o", markersize=4, label=r)
        arrow = "" if k == CAPABILITY else (" (lower is better)" if ALIGN[k][0] < 0 else "")
        ax.set_title(k + arrow, fontsize=9.5, color=INK, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(5, len(l)), frameon=False, fontsize=9, labelcolor=INK)
    fig.supxlabel("training step", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / rows))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _fmt(v):
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.3f}"
    return "n/a" if v is None else str(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(argv)
    runs = load(Path(a.results))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = summarize(runs, a.baseline)
    cols = list(rows[0])
    (out / "summary.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(_fmt(r[c]) for c in cols) for r in rows]) + "\n")
    head = ["recipe", "seeds", "capability", "gain_retained", "align_composite", "align_tax", "tax_reduction",
            f"align_at_{int(MATCH*100)}pct_gain", "align_chat", "align_rl_misbehavior"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(_fmt(r[c]) for c in head) + " |" for r in rows]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    plot_tradeoff(runs, a.baseline, out / "tradeoff.png")
    plot_metrics(runs, a.baseline, out / "metrics_by_step.png")
    print("\n".join(md))
    print(f"[apr-plot] wrote {out}/tradeoff.png, metrics_by_step.png, summary.md, summary.csv")


if __name__ == "__main__":
    main()
