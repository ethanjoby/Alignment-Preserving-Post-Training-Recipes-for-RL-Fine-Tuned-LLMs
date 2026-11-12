"""Reward functions in TRL's ``f(prompts, completions, **dataset_columns) -> list[float|None]`` form.

Capability rewards (what a capability-maximizing run optimizes):
  correctness  - strict or *lenient* (hackable) GSM8K grading
  format       - small shaping reward for producing a \\boxed{} answer

Alignment-preserving rewards (recipe components):
  honesty      - targeted environment: reward abstaining on unanswerable problems, penalize fabrication
  hack_monitor - general misbehavior penalty from a programmatic monitor (multi-answer grader gaming)
  rm_ensemble  - general alignment reward from an ensemble of reward models, conservatively aggregated
"""

from __future__ import annotations

import math

import torch

from apr.answers import grade


def _text(completion) -> str:
    if isinstance(completion, list):  # conversational: [{"role": "assistant", "content": ...}]
        return completion[-1]["content"]
    return completion


def _named(fn, name):
    fn.__name__ = name
    return fn


def correctness_reward(grader: str = "lenient"):
    key = {"strict": "strict_correct", "lenient": "lenient_correct"}[grader]

    def f(prompts, completions, gold, answerable, kind, **_):
        out = []
        for c, g, a, k in zip(completions, gold, answerable, kind):
            if k != "math":
                out.append(None)
                continue
            # Capability-only objective: an honest abstention on an unanswerable item earns the same 0
            # as a wrong guess, so guessing is weakly dominant. This is the "pursue the task at all
            # costs" pressure we want to reproduce.
            out.append(float(grade(_text(c), g if a else None)[key]))
        return out

    return _named(f, f"correctness_{grader}")


def format_reward():
    def f(prompts, completions, kind, **_):
        return [float(grade(_text(c), None)["has_box"]) if k == "math" else 0.0 for c, k in zip(completions, kind)]

    return _named(f, "format")


def honesty_reward():
    def f(prompts, completions, gold, answerable, kind, **_):
        out = []
        for c, g, a, k in zip(completions, gold, answerable, kind):
            if k != "math":
                out.append(None)
                continue
            r = grade(_text(c), g if a else None)
            if not a:
                out.append(1.0 if r["abstained"] else (-1.0 if r["fabricated"] else 0.0))
            else:
                out.append(-0.5 if r["abstained"] else 0.0)  # discourage blanket abstention
        return out

    return _named(f, "honesty")


def hack_monitor_reward():
    def f(prompts, completions, kind, **_):
        return [(-1.0 if grade(_text(c), None)["multi_answer"] else 0.0) if k == "math" else None
                for c, k in zip(completions, kind)]

    return _named(f, "hack_monitor")


class RMEnsembleReward:
    """Ensemble of sequence-classification reward models with conservative aggregation.

    Each RM's scores are z-normalized with running statistics, then combined as
    ``mean - k * std`` across RMs (uncertainty-weighted optimization, Coste et al. 2023), which
    penalizes completions on which RMs disagree, typically where a single RM is being exploited.
    """

    def __init__(self, model_names: list[str], k: float = 1.0, device: str | None = None, max_length: int = 1024):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.__name__ = "rm_ensemble"
        self.k = k
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.models, self.toks = [], []
        for name in model_names:
            tok = AutoTokenizer.from_pretrained(name)
            m = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.bfloat16 if self.device != "cpu" else torch.float32)
            self.models.append(m.to(self.device).eval())
            self.toks.append(tok)
        self.stats = [[0, 0.0, 0.0] for _ in model_names]  # count, mean, M2 (Welford)

    def _format(self, tok, prompt, completion):
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(prompt + [{"role": "assistant", "content": completion}], tokenize=False)
        user = prompt[-1]["content"] if isinstance(prompt, list) else prompt
        return f"Question: {user}\n\nAnswer: {completion}"

    @torch.no_grad()
    def __call__(self, prompts, completions, **_):
        per_rm = []
        for i, (m, tok) in enumerate(zip(self.models, self.toks)):
            texts = [self._format(tok, p, _text(c)) for p, c in zip(prompts, completions)]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            s = m(**enc).logits[:, 0].float().cpu().tolist()
            st = self.stats[i]
            for x in s:
                st[0] += 1
                d = x - st[1]
                st[1] += d / st[0]
                st[2] += d * (x - st[1])
            std = math.sqrt(st[2] / max(st[0] - 1, 1)) or 1.0
            per_rm.append([(x - st[1]) / std for x in s])
        t = torch.tensor(per_rm)  # (n_rm, B)
        agg = t.mean(0) - (self.k * t.std(0, unbiased=False) if t.size(0) > 1 else 0.0)
        return agg.tolist()


def build_rewards(cfg: dict):
    """Return (reward_funcs, reward_weights) from cfg['rewards'] and cfg['recipe']."""
    rw, rc = cfg.get("rewards", {}), cfg.get("recipe", {})
    funcs, weights = [], []

    def add(fn, w):
        if w:
            funcs.append(fn)
            weights.append(float(w))

    c = rw.get("correctness", {"weight": 1.0, "grader": "lenient"})
    add(correctness_reward(c.get("grader", "lenient")), c.get("weight", 1.0))
    add(format_reward(), rw.get("format", {}).get("weight", 0.0))
    add(honesty_reward(), rc.get("honesty_reward", {}).get("weight", 0.0))
    add(hack_monitor_reward(), rc.get("hack_monitor", {}).get("weight", 0.0))
    ens = rc.get("rm_ensemble", {})
    if ens.get("weight", 0.0):
        add(RMEnsembleReward(ens["models"], k=ens.get("k", 1.0)), ens["weight"])
    return funcs, weights
