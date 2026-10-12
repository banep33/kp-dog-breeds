"""Evaluacija na test skupu + kalibracija + analiza grešaka.

    python -m dogbreeds.evaluate --checkpoint runs/<run>/best.pt --config configs/baseline.yaml

Izlaz (runs/<run>/eval/):
    metrics.json        top1, top3, log-loss, ECE pre i posle kalibracije, temperatura
    per_class.csv       tačnost po rasi (gde je model slab?)
    confusions.csv      najčešći parovi zamena (npr. Siberian husky <-> Malamute)
    test_probs.npy      verovatnoće (za cleanlab - traženje pogrešnih labela, Dan 3)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import BreedDataset, build_transforms
from .metrics import expected_calibration_error, fit_temperature, log_loss, topk_accuracy
from .model import create_model


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_model({**ckpt["config"]["model"], "pretrained": False}, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_logits(model, loader, device):
    out, ts = [], []
    for x, y in loader:
        out.append(model(x.to(device)).float().cpu())
        ts.append(y)
    return torch.cat(out), torch.cat(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(a.checkpoint, device)
    classes = ckpt["classes"]
    _, eval_tf = build_transforms(ckpt["data_config"], cfg["train"])
    sd, d = Path(cfg["data"]["splits_dir"]), cfg["data"]

    def loader(split):
        ds = BreedDataset(sd / f"{split}.csv", d["images_dir"], classes, eval_tf)
        return ds, DataLoader(ds, batch_size=d["batch_size"] * 2, num_workers=d["num_workers"])

    # 1) temperatura se fituje na VAL skupu (nikad na test!)
    _, val_loader = loader("val")
    val_logits, val_t = predict_logits(model, val_loader, device)
    T = fit_temperature(val_logits, val_t)

    # 2) test
    test_ds, test_loader = loader(a.split)
    logits, t = predict_logits(model, test_loader, device)
    probs_raw = logits.softmax(1).numpy()
    probs_cal = (logits / T).softmax(1).numpy()
    tn = t.numpy()

    metrics = {
        **topk_accuracy(logits, t, ks=(1, 3, 5)),
        "log_loss": log_loss(logits, t),
        "log_loss_calibrated": log_loss(logits / T, t),
        "ece": expected_calibration_error(probs_raw, tn),
        "ece_calibrated": expected_calibration_error(probs_cal, tn),
        "temperature": T,
        "n": int(len(tn)),
        "model": ckpt["model_name"],
    }

    out = Path(a.checkpoint).parent / "eval"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.save(out / f"{a.split}_probs.npy", probs_cal)

    pred = probs_cal.argmax(1)
    per_class = (pd.DataFrame({"breed": [classes[i] for i in tn], "correct": pred == tn})
                 .groupby("breed")["correct"].agg(["mean", "count"])
                 .rename(columns={"mean": "accuracy", "count": "n"}).sort_values("accuracy"))
    per_class.to_csv(out / "per_class.csv")

    pairs = Counter((classes[a_], classes[p]) for a_, p in zip(tn, pred) if a_ != p)
    pd.DataFrame([{"true": k[0], "pred": k[1], "count": v} for k, v in pairs.most_common(30)]) \
        .to_csv(out / "confusions.csv", index=False)

    # sačuvaj temperaturu u checkpoint -> koristi je inference/ONNX
    ckpt["temperature"] = T
    torch.save(ckpt, a.checkpoint)

    print(json.dumps(metrics, indent=2))
    print("\nNajslabije rase:\n", per_class.head(10).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
