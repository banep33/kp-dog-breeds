"""Fine-tuning pretreniranog modela za 120 rasa.

Pokretanje:
    python -m dogbreeds.train --config configs/baseline.yaml
    python -m dogbreeds.train --config configs/baseline.yaml --set model.name=convnext_tiny.dinov3_lvd1689m train.lr=5e-4

Tehnike (svaku treba da umeš da objasniš na intervjuu - vidi docs/INTERVIEW.md):
  * transfer learning + zamrzavanje backbone-a u prvoj epohi
  * discriminative LR (glava uči brže od backbone-a)
  * AdamW + cosine raspored sa warmup-om
  * label smoothing, RandAugment, Random Erasing, (opciono) Mixup/CutMix
  * EMA težina, mixed precision (AMP), early stopping
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import load_config
from .data import build_loaders
from .metrics import SELECTION_METRIC, selection_key, topk_accuracy
from .model import count_params, create_model, get_data_config, set_backbone_trainable, split_params


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(model, loader, device, max_batches=None) -> dict:
    model.eval()
    all_logits, all_t = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        all_logits.append(logits.float().cpu())
        all_t.append(y)
    logits, t = torch.cat(all_logits), torch.cat(all_t)
    out = topk_accuracy(logits, t, ks=(1, 3))
    out["loss"] = nn.functional.cross_entropy(logits, t).item()
    return out


def run(cfg: dict, trial=None) -> dict:
    """Jedan trening. Vraća najbolje metrike. `trial` je Optuna trial (za HPO, Dan 4)."""
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc, mc = cfg["train"], cfg["model"]

    # --- model i podaci
    from .data import load_classes
    classes = load_classes(cfg["data"]["splits_dir"])
    model = create_model(mc, num_classes=len(classes)).to(device)
    data_config = get_data_config(model, mc.get("img_size"))
    loaders, _ = build_loaders(cfg, data_config)
    print(f"[model] {mc['name']} | {count_params(model):.1f}M param | input {data_config['input_size']} | device {device}")

    # --- optimizer: dve param grupe (discriminative learning rate)
    backbone, head = split_params(model)
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": tc["lr"] * tc["backbone_lr_mult"]},
         {"params": head, "lr": tc["lr"]}],
        weight_decay=tc["weight_decay"],
    )
    steps_per_epoch = len(loaders["train"])
    if tc.get("max_train_batches"):
        steps_per_epoch = min(steps_per_epoch, tc["max_train_batches"])
    from timm.scheduler import CosineLRScheduler
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=tc["epochs"] * steps_per_epoch,
        warmup_t=tc["warmup_epochs"] * steps_per_epoch,
        warmup_lr_init=1e-6,
        lr_min=1e-6,
        t_in_epochs=False,
    )

    # --- loss (+ opciono Mixup/CutMix)
    mixup_fn = None
    if tc.get("mixup_alpha", 0) > 0 or tc.get("cutmix_alpha", 0) > 0:
        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy
        mixup_fn = Mixup(mixup_alpha=tc["mixup_alpha"], cutmix_alpha=tc["cutmix_alpha"],
                         label_smoothing=tc["label_smoothing"], num_classes=len(classes))
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])

    ema = None
    if tc.get("ema_decay"):
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=tc["ema_decay"], use_warmup=True)

    use_amp = tc.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- izlazni folder
    run_name = cfg["output"].get("run_name") or f"{mc['name'].split('.')[0]}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(cfg["output"]["dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    wandb_run = None
    if cfg.get("logging", {}).get("wandb"):
        import wandb
        wandb_run = wandb.init(project=cfg["logging"]["wandb_project"], name=run_name, config=cfg)

    best = {SELECTION_METRIC: -1.0, "top1": -1.0}
    history, bad_epochs, step = [], 0, 0
    for epoch in range(tc["epochs"]):
        set_backbone_trainable(model, epoch >= tc.get("freeze_epochs", 0))
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(loaders["train"]):
            if tc.get("max_train_batches") and i >= tc["max_train_batches"]:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            scheduler.step_update(step)
            if ema is not None:
                ema.update(model, step=step)
            running += loss.item() * x.size(0)
            n += x.size(0)

        eval_model = ema.module if ema is not None else model
        val = evaluate_loader(eval_model, loaders["val"], device)
        row = {"epoch": epoch + 1, "train_loss": running / max(n, 1), "val_loss": val["loss"],
               "val_top1": val["top1"], "val_top3": val["top3"],
               "lr": optimizer.param_groups[1]["lr"], "sec": time.time() - t0}
        history.append(row)
        print(f"[ep {epoch + 1:02d}] train_loss {row['train_loss']:.3f} | val_loss {val['loss']:.3f} "
              f"| top1 {val['top1']:.4f} | top3 {val['top3']:.4f} | {row['sec']:.0f}s")
        if wandb_run:
            wandb_run.log(row)

        if selection_key(val) > selection_key(best):
            best = {**val, "epoch": epoch + 1}
            bad_epochs = 0
            torch.save({
                "state_dict": eval_model.state_dict(),
                "model_name": mc["name"],
                "classes": classes,
                "data_config": data_config,
                "metrics": best,
                "config": cfg,
            }, run_dir / "best.pt")
        else:
            bad_epochs += 1

        if trial is not None:  # Optuna: prekini loše trial-ove rano (pruning)
            import optuna
            trial.report(val[SELECTION_METRIC], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if bad_epochs >= tc.get("early_stopping_patience", 999):
            print(f"[early stop] nema poboljšanja {bad_epochs} epohe")
            break

    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    (run_dir / "best_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    if wandb_run:
        wandb_run.summary.update({f"best_{k}": v for k, v in best.items()})
        wandb_run.finish()
    print(f"[done] best val top1 {best['top1']:.4f} top3 {best['top3']:.4f} (ep {best['epoch']}) -> {run_dir}")
    return {**best, "run_dir": str(run_dir)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="override, npr. train.epochs=5")
    a = ap.parse_args()
    run(load_config(a.config, a.set))


if __name__ == "__main__":  # obavezno na Windows-u zbog DataLoader workera
    main()
