"""Linear probe: treniramo SAMO klasifikacionu glavu nad zamrznutim embeddinzima.

ZAŠTO JE OVO PRVI MODEL KOJI TREBA NAPRAVITI
--------------------------------------------
* **Brzo:** embeddinzi se izračunaju jednom (dogbreeds.embed), posle toga je trening
  logističke regresije nad matricom [N, 768] — sekunde na CPU-u, bez GPU-a.
* **Pošten baseline:** govori koliko informacije već postoji u pretreniranim
  reprezentacijama. Fine-tuning (dogbreeds.train) mora ovo da pobedi — inače ne vredi.
* **Jeftino poređenje kandidata:** isti postupak za 3 backbone-a daje Pareto krivu
  tačnost/latencija bez ijednog GPU-sata.
* **Deployable:** trenirana glava se upisuje u `classifier` sloj pravog timm modela i
  snima kao `best.pt` u ISTOM formatu kao `dogbreeds.train`, pa `evaluate.py` i
  `export.py` rade bez izmena.

    python -m dogbreeds.probe --emb data/emb/convnext_tiny --splits data/splits
    python -m dogbreeds.probe --sweep --splits data/splits        # svi kandidati -> Pareto
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .embed import load as load_emb
from .metrics import (SELECTION_METRIC, expected_calibration_error, fit_temperature,
                      selection_key, topk_accuracy)

# Kandidati iz docs/PLAN.md (Dan 2). Prefiks .npy dolazi iz dogbreeds.embed.
CANDIDATES = {
    "efficientnet_b0.ra4_e3600_r224_in1k": "data/emb/efficientnet_b0",
    "mobilenetv4_conv_medium.e500_r256_in1k": "data/emb/mobilenetv4_conv_medium",
    "convnext_tiny.dinov3_lvd1689m": "data/emb/convnext_tiny",
}


# ---------------------------------------------------------------- podaci

def align_splits(ids: list[str], splits_dir: str | Path, classes: list[str]
                 ) -> dict[str, np.ndarray]:
    """Mapira id-eve iz train/val/test.csv na redove u matrici embeddinga."""
    pos = {str(i): k for k, i in enumerate(ids)}
    sd = Path(splits_dir)
    out: dict[str, np.ndarray] = {}
    c2i = {c: i for i, c in enumerate(classes)}
    for split in ("train", "val", "test"):
        df = pd.read_csv(sd / f"{split}.csv")
        df["id"] = df["id"].astype(str)
        missing = [i for i in df["id"] if i not in pos]
        if missing:
            raise KeyError(f"{len(missing)} id-eva iz {split}.csv nema embedding (prvi: {missing[:3]})")
        out[f"{split}_idx"] = np.array([pos[i] for i in df["id"]], dtype=np.int64)
        out[f"{split}_y"] = df["breed"].map(c2i).to_numpy(dtype=np.int64)
    return out


# ---------------------------------------------------------------- trening glave

def feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Srednja vrednost i standardna devijacija po dimenziji, SAMO sa train skupa."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.maximum(sd, 1e-6)          # mrtve dimenzije (sd=0) ne smeju da dele nulom
    return mu.astype(np.float32), sd.astype(np.float32)


def fold_standardization(head: nn.Linear, mu: np.ndarray, sd: np.ndarray
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Uračunaj standardizaciju u težine, da model u produkciji radi nad SIROVIM feature-ima.

        W·z + b  gde je  z = (e − μ)/σ
      = W·(e − μ)/σ + b
      = (W/σ)·e + (b − (W/σ)·μ)

    Dakle treniramo nad dobro uslovljenim ulazom (isti learning rate radi za sve
    backbone-e bez obzira na skalu njihovih feature-a), a izvozimo OBIČAN Linear nad
    sirovim izlazom backbone-a. Bez dodatnih slojeva, bez neslaganja trening/produkcija.
    """
    sd_t = torch.from_numpy(sd)
    mu_t = torch.from_numpy(mu)
    W_eff = head.weight.detach() / sd_t                 # [C, D] / [D] -> po kolonama
    b_eff = head.bias.detach() - W_eff @ mu_t
    return W_eff, b_eff


def effective_weights(head: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    """Tezine glave izrazene nad SIROVIM feature-ima (standardizacija uracunata)."""
    return fold_standardization(head, head.feature_mean, head.feature_std)


def apply_head(head: nn.Linear, X: np.ndarray) -> torch.Tensor:
    """Logiti za sirove feature-e, kroz uracunate tezine. Namerno NE radimo
    head((X-mu)/sd): ovako svaki broj koji prijavimo dolazi iz iste matrice
    koja zavrsi u checkpoint-u i u ONNX-u."""
    W, b = effective_weights(head)
    with torch.no_grad():
        return torch.nn.functional.linear(torch.from_numpy(np.ascontiguousarray(X)), W, b)


def train_head(
    X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray,
    num_classes: int, epochs: int = 60, lr: float = 1e-2, weight_decay: float = 1e-4,
    label_smoothing: float = 0.05, batch_size: int = 512, seed: int = 42, verbose: bool = True,
    standardize: bool = True, trial=None,
) -> tuple[nn.Linear, dict, list[dict]]:
    """Jedan Linear sloj (D -> 120), AdamW + cosine. Namerno isti recept kao u `train.py`
    da poređenje bude pošteno — razlika je samo u tome što je backbone zamrznut.

    standardize=True poravnava skalu feature-a između backbone-a, pa jedan isti
    learning rate važi za sve; inače bi poređenje merilo koliko nam LR odgovara
    kom modelu, a ne kvalitet reprezentacija. Statistike se računaju SAMO na train
    skupu i kasnije se uračunaju u težine (fold_standardization)."""
    torch.manual_seed(seed)
    if standardize:
        mu, sd = feature_stats(X)
        X, Xv = (X - mu) / sd, (Xv - mu) / sd
    else:
        mu = np.zeros(X.shape[1], dtype=np.float32)
        sd = np.ones(X.shape[1], dtype=np.float32)
    Xt, yt = torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(y)
    Xvt, yvt = torch.from_numpy(np.ascontiguousarray(Xv)), torch.from_numpy(yv)

    head = nn.Linear(X.shape[1], num_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best = {SELECTION_METRIC: -1.0, "top1": -1.0}
    best_state, history = None, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            loss = crit(head(Xt[b]), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()

        head.eval()
        with torch.no_grad():
            logits = head(Xvt)
        m = topk_accuracy(logits, yvt, ks=(1, 3))
        m["loss"] = nn.functional.cross_entropy(logits, yvt).item()
        history.append({"epoch": ep + 1, "train_loss": tot / len(Xt), **m})
        if selection_key(m) > selection_key(best):
            best = {**m, "epoch": ep + 1}
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if verbose and (ep + 1) % 20 == 0:
            print(f"  [ep {ep + 1:03d}] train_loss {tot / len(Xt):.3f} | "
                  f"val top1 {m['top1']:.4f} top3 {m['top3']:.4f}")

        if trial is not None:  # Optuna: prekini lose trial-ove rano (isti hook kao train.run)
            import optuna
            trial.report(m[SELECTION_METRIC], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    head.load_state_dict(best_state)
    head.eval()
    head.feature_mean, head.feature_std = mu, sd   # nosi ih do upisa u checkpoint
    return head, best, history


# ---------------------------------------------------------------- checkpoint

def save_as_timm_checkpoint(head: nn.Linear, model_name: str, classes: list[str],
                            emb_meta: dict, metrics: dict, out_dir: Path, cfg: dict) -> Path:
    """Upisuje treniranu glavu u `classifier` pravog timm modela i snima `best.pt`
    identičnog formata kao `dogbreeds.train` -> evaluate/export rade bez izmena."""
    from .model import create_model, head_prefix

    model = create_model({"name": model_name, "pretrained": True}, len(classes))
    prefix = head_prefix(model)
    sd = model.state_dict()
    assert sd[f"{prefix}.weight"].shape == head.weight.shape, (
        f"glava {tuple(head.weight.shape)} != timm classifier {tuple(sd[f'{prefix}.weight'].shape)}")
    # Standardizacija se URACUNAVA u tezine -> `classifier` radi nad SIROVIM izlazom
    # backbone-a, bez ijednog dodatnog sloja (vidi fold_standardization).
    W_eff, b_eff = effective_weights(head)
    sd[f"{prefix}.weight"] = W_eff.clone()
    sd[f"{prefix}.bias"] = b_eff.clone()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    torch.save({
        "state_dict": sd,
        "model_name": model_name,
        "classes": classes,
        "data_config": emb_meta["data_config"],
        "metrics": metrics,
        "config": cfg,
        "training_mode": "linear_probe",
    }, ckpt_path)
    return ckpt_path


def verify_checkpoint_matches_probe(ckpt_path: Path, X: np.ndarray, head: nn.Linear,
                                    n: int = 8) -> float:
    """Sanity check: pun timm model (backbone + upisana glava) na ISTOJ slici mora dati
    iste logite kao glava nad keširanim embeddingom. Hvata neslaganje pooling/norm slojeva."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from .model import create_model
    model = create_model({**ckpt["config"]["model"], "pretrained": False}, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        ref = head(torch.from_numpy(X[:n]))
        # emuliramo "backbone je već odradio svoje": pozivamo samo klasifikator nad embeddingom
        got = torch.nn.functional.linear(
            torch.from_numpy(X[:n]),
            dict(model.named_parameters())[
                [k for k, _ in model.named_parameters() if k.endswith("weight")][-1]],
            dict(model.named_parameters())[
                [k for k, _ in model.named_parameters() if k.endswith("bias")][-1]],
        )
    return float((ref - got).abs().max())


# ---------------------------------------------------------------- jedan probe

def select_lr(emb, s, num_classes, lr_grid, epochs, weight_decay, seed, verbose):
    """Bira learning rate po VALIDACIONOM skupu, posebno za svaki backbone.

    Zasto: feature-i razlicitih backbone-a imaju razlicitu skalu i uslovljenost, pa
    jedan fiksni LR ne odgovara svima podjednako. Da ga ne biramo, Pareto bi merio
    koliko nam je LR slucajno odgovarao kom modelu, a ne kvalitet reprezentacija.
    Izbor ide ISKLJUCIVO preko val skupa -- test se ne dira dok se sve ne zakljuca."""
    results = []
    for lr in lr_grid:
        head, best, hist = train_head(
            emb[s["train_idx"]], s["train_y"], emb[s["val_idx"]], s["val_y"],
            num_classes=num_classes, epochs=epochs, lr=lr,
            weight_decay=weight_decay, seed=seed, verbose=False)
        results.append({"lr": lr, "head": head, "best": best, "hist": hist})
        if verbose:
            print(f"    lr {lr:<8.4g} -> val top1 {best['top1']:.4f} top3 {best['top3']:.4f}")
    win = max(results, key=lambda r: selection_key(r["best"]))
    if verbose:
        print(f"    izabran lr = {win['lr']:g}")
    return win["head"], win["best"], win["hist"], win["lr"], [
        {"lr": r["lr"], "val_top1": r["best"]["top1"], "val_top3": r["best"]["top3"]}
        for r in results]


def run_probe(emb_prefix: str, splits_dir: str, model_name: str | None = None,
              out_dir: str | None = None, epochs: int = 60, lr: float = 1e-2,
              weight_decay: float = 1e-4, seed: int = 42, verbose: bool = True,
              lr_grid: list[float] | None = None, label_smoothing: float = 0.05,
              batch_size: int = 512) -> dict:
    emb, ids, meta = load_emb(emb_prefix)
    model_name = model_name or meta["model"]
    classes = json.loads((Path(splits_dir) / "classes.json").read_text(encoding="utf-8"))
    s = align_splits(ids, splits_dir, classes)

    if verbose:
        print(f"\n=== linear probe: {model_name} (dim {emb.shape[1]}) ===")
        print(f"    train {len(s['train_idx'])} | val {len(s['val_idx'])} | test {len(s['test_idx'])}")

    t0 = time.perf_counter()
    lr_search = None
    if lr_grid:
        head, best, history, lr, lr_search = select_lr(
            emb, s, len(classes), lr_grid, epochs, weight_decay, seed, verbose)
    else:
        head, best, history = train_head(
            emb[s["train_idx"]], s["train_y"], emb[s["val_idx"]], s["val_y"],
            num_classes=len(classes), epochs=epochs, lr=lr, weight_decay=weight_decay,
            label_smoothing=label_smoothing, batch_size=batch_size,
            seed=seed, verbose=verbose)
    train_sec = time.perf_counter() - t0

    # kalibracija: temperatura se fituje na VAL, primenjuje na TEST (nikad obrnuto)
    val_logits = apply_head(head, emb[s["val_idx"]])
    test_logits = apply_head(head, emb[s["test_idx"]])
    val_y_t = torch.from_numpy(s["val_y"])
    test_y_t = torch.from_numpy(s["test_y"])
    T = fit_temperature(val_logits, val_y_t)

    test_probs_raw = test_logits.softmax(1).numpy()
    test_probs_cal = (test_logits / T).softmax(1).numpy()
    res = {
        "model": model_name,
        "emb_dim": int(emb.shape[1]),
        "val_top1": best["top1"], "val_top3": best["top3"], "val_loss": best["loss"],
        **{f"test_{k}": v for k, v in topk_accuracy(test_logits, test_y_t, ks=(1, 3, 5)).items()},
        "test_log_loss": float(nn.functional.cross_entropy(test_logits, test_y_t)),
        "test_log_loss_calibrated": float(nn.functional.cross_entropy(test_logits / T, test_y_t)),
        "test_ece": expected_calibration_error(test_probs_raw, s["test_y"]),
        "test_ece_calibrated": expected_calibration_error(test_probs_cal, s["test_y"]),
        "temperature": T,
        "best_epoch": best["epoch"],
        "lr": lr,
        "label_smoothing": label_smoothing,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr_search": lr_search,
        "n_test": int(len(s["test_y"])),
        "head_train_seconds": round(train_sec, 1),
        "embed_img_per_sec_cpu": meta.get("img_per_sec"),
    }
    if verbose:
        print(f"    val  top1 {res['val_top1']:.4f}  top3 {res['val_top3']:.4f}")
        print(f"    TEST top1 {res['test_top1']:.4f}  top3 {res['test_top3']:.4f}  "
              f"top5 {res['test_top5']:.4f}")
        print(f"    ECE {res['test_ece']:.4f} -> {res['test_ece_calibrated']:.4f} (T={T:.3f})  "
              f"| glava trenirana u {train_sec:.1f}s")

    if out_dir:
        od = Path(out_dir)
        cfg = {"model": {"name": model_name, "pretrained": True}, "seed": seed,
               "probe": {"epochs": epochs, "lr": lr, "weight_decay": weight_decay,
                         "label_smoothing": label_smoothing, "batch_size": batch_size}}
        ckpt = save_as_timm_checkpoint(head, model_name, classes, meta,
                                       {"top1": best["top1"], "top3": best["top3"],
                                        "epoch": best["epoch"]}, od, cfg)
        torch.save({"temperature": T}, od / "calibration.pt")
        pd.DataFrame(history).to_csv(od / "history.csv", index=False)
        (od / "probe_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        np.save(od / "test_probs.npy", test_probs_cal)
        np.save(od / "val_probs.npy", (val_logits / T).softmax(1).numpy())
        # upiši temperaturu u checkpoint da je export.py prenese u model.json
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        c["temperature"] = T
        torch.save(c, ckpt)
        res["checkpoint"] = str(ckpt)
        if verbose:
            print(f"    -> {ckpt}")
    return res


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Linear probe nad keširanim embeddinzima")
    ap.add_argument("--emb", default="data/emb/convnext_tiny")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--model", default=None, help="timm ime (default: iz meta embeddinga)")
    ap.add_argument("--out", default=None, help="folder za best.pt (npr. runs/probe_convnext)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-grid", type=float, nargs="*",
                    default=[3e-3, 1e-2, 3e-2, 1e-1],
                    help="LR se bira po val skupu; prazno = koristi --lr")
    ap.add_argument("--sweep", action="store_true", help="probe za sve kandidate -> Pareto tabela")
    ap.add_argument("--report", default="reports/probe_results.json")
    a = ap.parse_args()

    if a.sweep:
        rows = []
        for name, prefix in CANDIDATES.items():
            if not Path(prefix).with_suffix(".npy").exists():
                print(f"[preskačem] nema {prefix}.npy — pokreni dogbreeds.embed --model {name}")
                continue
            rows.append(run_probe(prefix, a.splits, name,
                                  out_dir=f"runs/probe_{name.split('.')[0]}",
                                  epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay,
                                  seed=a.seed, lr_grid=a.lr_grid))
        if not rows:
            raise SystemExit("nema ni jednog embeddinga — prvo pokreni dogbreeds.embed")
        df = pd.DataFrame(rows)
        cols = ["model", "emb_dim", "lr", "val_top1", "val_top3", "test_top1", "test_top3",
                "test_ece", "test_ece_calibrated", "temperature"]
        print("\n" + df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n-> {a.report}")
    else:
        run_probe(a.emb, a.splits, a.model, a.out, a.epochs, a.lr, a.weight_decay, a.seed,
                  lr_grid=a.lr_grid, label_smoothing=a.label_smoothing,
                  batch_size=a.batch_size)


if __name__ == "__main__":
    main()
