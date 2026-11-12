"""YAML configs with ``inherit:`` chaining and ``a.b.c=value`` CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parents = cfg.pop("inherit", None)
    if parents:
        parents = [parents] if isinstance(parents, str) else parents
        merged: dict = {}
        for p in parents:
            merged = deep_merge(merged, load_config(path.parent / p))
        cfg = deep_merge(merged, cfg)
    return load_config_overrides(cfg, overrides)


def load_config_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        node = cfg
        *head, last = key.split(".")
        for h in head:
            node = node.setdefault(h, {})
        node[last] = yaml.safe_load(raw)
    return cfg
