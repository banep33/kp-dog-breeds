"""Podaci: podela na train/val/test, Dataset klasa i transformacije.

Pokretanje podele (jednom, pre treninga):
    python -m dogbreeds.data --config configs/baseline.yaml

Zašto pravimo SOPSTVENI test skup?
Kaggle test skup nema labele, pa nam treba skup koji model NIKADA ne vidi tokom
treninga ni tokom izbora hiperparametara. val = za izbor modela/HP, test = jednom, na kraju.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import load_config


# ---------------------------------------------------------------- podela

def prepare_splits(labels_csv: str, out_dir: str, val_size: float, test_size: float, seed: int) -> dict:
    df = pd.read_csv(labels_csv)
    assert {"id", "breed"} <= set(df.columns), "labels.csv mora imati kolone id,breed"

    # Opciono: kolona 'group' (npr. klaster gotovo identičnih slika istog psa).
    # Ako postoji, cela grupa ide u isti split -> nema curenja (leakage) između train i test.
    if "group" in df.columns:
        # StratifiedGroupKFold: grupe se ne cepaju I raspodela rasa ostaje ujednačena.
        from sklearn.model_selection import StratifiedGroupKFold

        def strat_group_split(d: pd.DataFrame, frac: float):
            k = max(2, round(1 / frac))  # npr. 10% -> 10 foldova, uzimamo jedan
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            a, b = next(sgkf.split(d, d["breed"], groups=d["group"]))
            return d.iloc[a], d.iloc[b]

        trval, test = strat_group_split(df, test_size)
        train, val = strat_group_split(trval, val_size / (1 - test_size))
    else:
        # Stratifikacija: svaka rasa je zastupljena u istom odnosu u svim skupovima.
        trval, test = train_test_split(df, test_size=test_size, stratify=df["breed"], random_state=seed)
        train, val = train_test_split(
            trval, test_size=val_size / (1 - test_size), stratify=trval["breed"], random_state=seed
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(df["breed"].unique())
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)
    (out / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    stats = {
        "n_total": len(df),
        "n_classes": len(classes),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "min_per_class": int(df["breed"].value_counts().min()),
        "max_per_class": int(df["breed"].value_counts().max()),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# ---------------------------------------------------------------- dataset

class BreedDataset(Dataset):
    def __init__(self, csv_path: str | Path, images_dir: str | Path, classes: list[str], transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def path_of(self, i: int) -> Path:
        return self.images_dir / f"{self.df.iloc[i]['id']}.jpg"

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.path_of(i)).convert("RGB")  # neke slike su grayscale/CMYK
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[row["breed"]]


def load_classes(splits_dir: str | Path) -> list[str]:
    return json.loads((Path(splits_dir) / "classes.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- transformacije

def build_transforms(data_config: dict, train_cfg: dict):
    """Koristimo timm transformacije da preprocessing TAČNO odgovara pretreniranom modelu
    (ista rezolucija, mean/std, interpolacija, crop_pct)."""
    from timm.data import create_transform

    train_tf = create_transform(
        **data_config,
        is_training=True,
        auto_augment=train_cfg.get("auto_augment"),
        re_prob=train_cfg.get("random_erasing", 0.0),
        scale=(0.35, 1.0),  # psi su često mali deo slike -> agresivniji crop pomaže
    )
    eval_tf = create_transform(**data_config, is_training=False)
    return train_tf, eval_tf


def build_loaders(cfg: dict, data_config: dict):
    d = cfg["data"]
    classes = load_classes(d["splits_dir"])
    train_tf, eval_tf = build_transforms(data_config, cfg["train"])
    sd = Path(d["splits_dir"])
    ds = {
        "train": BreedDataset(sd / "train.csv", d["images_dir"], classes, train_tf),
        "val": BreedDataset(sd / "val.csv", d["images_dir"], classes, eval_tf),
    }
    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(ds["train"], batch_size=d["batch_size"], shuffle=True, drop_last=True,
                            num_workers=d["num_workers"], pin_memory=pin, persistent_workers=d["num_workers"] > 0),
        "val": DataLoader(ds["val"], batch_size=d["batch_size"] * 2, shuffle=False,
                          num_workers=d["num_workers"], pin_memory=pin),
    }
    return loaders, classes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Napravi train/val/test podelu")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    c = load_config(a.config, a.set)
    s = prepare_splits(c["data"]["labels_csv"], c["data"]["splits_dir"],
                       c["data"]["val_size"], c["data"]["test_size"], c["seed"])
    print(json.dumps(s, indent=2))
