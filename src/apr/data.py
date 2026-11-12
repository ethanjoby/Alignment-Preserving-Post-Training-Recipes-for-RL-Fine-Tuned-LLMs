"""Datasets: GSM8K RL prompts (with an unanswerable split), alignment prompts, and anchor SFT data."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from datasets import Dataset, load_dataset

from apr.answers import gold_from_gsm8k

MATH_INSTRUCTION = (
    "Solve the following problem. Reason step by step, then give your final answer as \\boxed{{N}}. "
    "If the problem does not contain enough information to determine the answer, explain what is "
    "missing and answer \\boxed{{insufficient}}.\n\nProblem: {question}"
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUM_IN_TEXT = re.compile(r"\d[\d,]*\.?\d*")


def make_unanswerable(question: str, solution: str, rng: random.Random) -> str | None:
    """Delete one premise sentence whose numbers the reference solution actually uses.

    Heuristic, not a guarantee: we require that (a) the dropped sentence is not the final question,
    (b) at least one of its numbers appears inside the solution's ``<<calc>>`` annotations, and
    (c) that number appears nowhere else in the remaining text. Spot-check ~50 items before a
    paper-grade run (or swap in a curated set such as UMWP); expect a few-percent label noise.
    """
    sents = _SENT_SPLIT.split(question.strip())
    if len(sents) < 3:
        return None
    calcs = " ".join(re.findall(r"<<([^>]*)>>", solution))
    used = {n.replace(",", "") for n in _NUM_IN_TEXT.findall(calcs)}
    candidates = []
    for i, s in enumerate(sents[:-1]):
        nums = {n.replace(",", "").rstrip(".") for n in _NUM_IN_TEXT.findall(s)}
        rest = " ".join(sents[:i] + sents[i + 1 :])
        rest_nums = {n.replace(",", "").rstrip(".") for n in _NUM_IN_TEXT.findall(rest)}
        if nums & used and not (nums & used) & rest_nums:
            candidates.append(i)
    if not candidates:
        return None
    drop = rng.choice(candidates)
    return " ".join(sents[:drop] + sents[drop + 1 :])


def math_rows(split: str, n: int | None, unanswerable_frac: float, seed: int) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=seed)
    rng = random.Random(seed)
    rows = []
    for ex in ds:
        if n is not None and len(rows) >= n:
            break
        q, gold = ex["question"], gold_from_gsm8k(ex["answer"])
        if rng.random() < unanswerable_frac:
            uq = make_unanswerable(q, ex["answer"], rng)
            if uq is not None:
                rows.append({"question": uq, "gold": "", "answerable": False, "kind": "math"})
                continue
        rows.append({"question": q, "gold": gold, "answerable": True, "kind": "math"})
    return rows


def hh_prompts(n: int, seed: int, split: str = "train") -> list[str]:
    """First human turn of Anthropic HH-RLHF dialogues (a mix of benign and red-team requests)."""
    ds = load_dataset("Anthropic/hh-rlhf", split=split).shuffle(seed=seed)
    out = []
    for ex in ds:
        m = re.match(r"\s*Human:(.*?)\n\nAssistant:", ex["chosen"], re.DOTALL)
        if m and 10 < len(m.group(1).strip()) < 500:
            out.append(m.group(1).strip())
        if len(out) >= n:
            break
    return out


def to_chat(user: str, system: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": user}]


def build_rl_dataset(cfg: dict, tokenizer=None) -> Dataset:
    """Rows: prompt (chat), gold, answerable, kind, and optionally teacher_prompt (templated text)."""
    d = cfg["data"]
    seed = cfg.get("seed", 0)
    rows = math_rows("train", d["train_size"], d.get("unanswerable_frac", 0.0), seed)
    for r in rows:
        r["user"] = MATH_INSTRUCTION.format(question=r["question"])
    n_align = int(len(rows) * d.get("alignment_prompt_frac", 0.0))
    if n_align:
        for p in hh_prompts(n_align, seed):
            rows.append({"question": p, "gold": "", "answerable": False, "kind": "chat", "user": p})
        random.Random(seed).shuffle(rows)

    sd = cfg.get("recipe", {}).get("self_distill", {})
    spec = Path(sd["spec"]).read_text() if sd.get("coef", 0) > 0 else None
    tmpl_kwargs = cfg.get("grpo", {}).get("chat_template_kwargs") or {}
    out = []
    for r in rows:
        row = {
            "prompt": to_chat(r["user"]),
            "gold": r["gold"],
            "answerable": r["answerable"],
            "kind": r["kind"],
        }
        if spec is not None:
            # The teacher sees the conduct spec as a system prompt; the student never does.
            row["teacher_prompt"] = tokenizer.apply_chat_template(
                to_chat(r["user"], system=spec), tokenize=False, add_generation_prompt=True, **tmpl_kwargs
            )
        out.append(row)
    return Dataset.from_list(out)


def load_anchor(path: str) -> list[dict]:
    """Anchor SFT data: jsonl with {"prompt": str, "response": str} (see ``apr.anchor.build``)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
