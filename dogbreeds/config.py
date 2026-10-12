"""Učitavanje YAML konfiguracije + override-ovi iz komandne linije.

Primer:
    python -m dogbreeds.train --config configs/baseline.yaml --set train.epochs=5 model.name=efficientnet_b0
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Podržava nasleđivanje: `base: baseline.yaml` u YAML-u učitava osnovni fajl pa ga dopunjuje."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        cfg = deep_update(load_config(path.parent / cfg.pop("base")), cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_by_dotted_key(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_by_dotted_key(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def merged(cfg: dict, updates: dict[str, Any]) -> dict:
    """Vraća kopiju konfiguracije sa izmenama (koristi Optuna za svaki trial)."""
    out = copy.deepcopy(cfg)
    for k, v in updates.items():
        set_by_dotted_key(out, k, v)
    return out
