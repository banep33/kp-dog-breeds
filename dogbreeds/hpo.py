"""Pretraga hiperparametara (Optuna TPE + pruning).

Dva prostora, isti modul:

    # glava nad zamrznutim embeddinzima -- ~15 s po trial-u, radi na CPU-u
    python -m dogbreeds.hpo --space probe --emb data/emb/efficientnet_b0 --splits data/splits --trials 30

    # pun fine-tuning -- minuti po trial-u, trazi GPU (Kaggle)
    python -m dogbreeds.hpo --space finetune --config configs/kaggle.yaml --trials 25 --epochs 4

ZASTO BAS OVAKO
---------------
* **TPE umesto grid search-a.** Grid trosi budzet ravnomerno po celom prostoru, a
  vecina parametara nije podjednako vazna. TPE gradi model "koje vrednosti su do sada
  davale dobre rezultate" i uzorkuje iz njega -- isti broj trial-ova, bolje pokriven
  deo prostora koji nesto znaci.
* **MedianPruner.** Trial koji je posle k epoha ispod medijane dotadasnjih trial-ova u
  istoj epohi se prekida. Kod fine-tuninga to je razlika izmedju 3 h i 5 h.
* **Optimizujemo top-3** (`metrics.SELECTION_METRIC`), jer zadatak trazi prikaz tri
  odgovora. Top-1 se pamti kao `user_attr` i razresava izjednacenje (`selection_key`).
* **SAMO val skup.** Nijedan trial ne vidi test -- inace bi prijavljeni broj bio
  rezultat pretrage nad testom, a ne postena procena.
* **`sqlite:///optuna.db`.** Study prezivljava prekid: Kaggle sesija koja pukne na
  17. trial-u nastavlja od 18., ne od nule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, merged
from .metrics import SELECTION_METRIC, selection_key


# ---------------------------------------------------------------- prostori pretrage

def suggest_probe(trial) -> dict:
    """Glava nad zamrznutim embeddinzima.

    Opseg za `lr` ide do 3e-1 jer je Dan 2 pokazao da optimum za probe lezi visoko
    (izabrano 0.003-0.01 zavisno od backbone-a), a ne u opsegu za fine-tuning."""
    return {
        "lr": trial.suggest_float("lr", 1e-4, 3e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "epochs": trial.suggest_int("epochs", 20, 80, step=20),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
    }


def suggest_finetune(trial) -> dict:
    """Pun fine-tuning. Kljucevi su tackasti override-i za config (vidi config.merged).

    Opsezi iz docs/PLAN.md, Dan 4. `backbone_lr_mult` je tu jer je u Kaggle prolazu
    bio nagadjan (0.1), a `label_smoothing` jer je dokazano pokvario kalibraciju
    (ECE 0.108, temperatura 0.738)."""
    return {
        "train.lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "train.weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "train.label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "train.backbone_lr_mult": trial.suggest_float("backbone_lr_mult", 0.05, 1.0, log=True),
        "model.drop_path_rate": trial.suggest_float("drop_path_rate", 0.0, 0.3),
        "train.mixup_alpha": trial.suggest_categorical("mixup_alpha", [0.0, 0.2]),
        "model.img_size": trial.suggest_categorical("img_size", [224, 256, 288]),
        # svaki trial u svoj folder, da checkpointi ne gaze jedan drugi
        "output.run_name": f"hpo_trial_{trial.number:03d}",
    }


# ---------------------------------------------------------------- ciljne funkcije

def make_probe_objective(emb_prefix: str, splits_dir: str, seed: int):
    """Ucitava embeddinge JEDNOM, pa svaki trial samo trenira glavu (sekunde)."""
    from .embed import load as load_emb
    from .probe import align_splits, train_head

    emb, ids, meta = load_emb(emb_prefix)
    classes = json.loads((Path(splits_dir) / "classes.json").read_text(encoding="utf-8"))
    s = align_splits(ids, splits_dir, classes)
    Xtr, ytr = emb[s["train_idx"]], s["train_y"]
    Xva, yva = emb[s["val_idx"]], s["val_y"]
    print(f"[hpo] {meta['model']} | dim {emb.shape[1]} | train {len(ytr)} | "
          f"val {len(yva)} | {len(classes)} klasa")

    def objective(trial) -> float:
        params = suggest_probe(trial)
        _, best, _ = train_head(Xtr, ytr, Xva, yva, num_classes=len(classes),
                                seed=seed, verbose=False, trial=trial, **params)
        trial.set_user_attr("top1", best["top1"])
        trial.set_user_attr("top3", best["top3"])
        trial.set_user_attr("best_epoch", best["epoch"])
        return best[SELECTION_METRIC]

    return objective, {"model": meta["model"], "emb_dim": int(emb.shape[1]),
                       "n_train": int(len(ytr)), "n_val": int(len(yva))}


def make_finetune_objective(config_path: str, overrides: list[str], epochs: int):
    """Svaki trial je kratak trening (`epochs` epoha). Pobednik se posle trenira do kraja."""
    from . import train as train_mod

    base = load_config(config_path, overrides)

    def objective(trial) -> float:
        cfg = merged(base, {**suggest_finetune(trial), "train.epochs": epochs})
        best = train_mod.run(cfg, trial=trial)
        trial.set_user_attr("top1", best["top1"])
        trial.set_user_attr("top3", best["top3"])
        trial.set_user_attr("run_dir", best.get("run_dir"))
        return best[SELECTION_METRIC]

    return objective, {"config": config_path, "trial_epochs": epochs}


# ---------------------------------------------------------------- izbor pobednika

def best_trial_by_selection(study):
    """Najbolji trial po (top-3, top-1).

    Optuna optimizuje jedan skalar (top-3), pa kod izjednacenja vraca prvi po redu.
    Posto nam je top-1 drugi kriterijum (vidi metrics.selection_key), biramo sami nad
    zavrsenim trial-ovima koristeci user_attrs."""
    import optuna

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return None
    return max(done, key=lambda t: selection_key({
        SELECTION_METRIC: t.value,
        "top1": t.user_attrs.get("top1", 0.0),
    }))


# ---------------------------------------------------------------- grafici

def save_plots(study, out_dir: Path) -> list[str]:
    """plot_optimization_history i plot_param_importances.

    Matplotlib backend, jer plotly nije u zavisnostima -- isti grafici, bez dodatnog
    paketa."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, fn in (("hpo_history", ovm.plot_optimization_history),
                     ("hpo_param_importances", ovm.plot_param_importances)):
        try:
            ax = fn(study)
            path = out_dir / f"{name}.png"
            ax.figure.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(ax.figure)
            saved.append(str(path))
        except Exception as e:          # importances pada ako je premalo zavrsenih trial-ova
            print(f"[hpo] grafik {name} preskocen: {e}")
    return saved


# ---------------------------------------------------------------- CLI

def main() -> None:
    import optuna

    ap = argparse.ArgumentParser(description="Optuna pretraga hiperparametara")
    ap.add_argument("--space", choices=["probe", "finetune"], required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=None, help="sekundi, gornja granica")
    ap.add_argument("--storage", default="sqlite:///optuna.db",
                    help="prezivljava prekid sesije; prazno = study samo u memoriji")
    ap.add_argument("--study-name", default=None, help="default: hpo_<space>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default=None, help="default: reports/hpo_<space>.json")
    ap.add_argument("--figures", default="reports/figures")
    # --- probe
    ap.add_argument("--emb", default="data/emb/efficientnet_b0")
    ap.add_argument("--splits", default="data/splits")
    # --- finetune
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="override, npr. data.images_dir=...")
    ap.add_argument("--epochs", type=int, default=4, help="epoha po trial-u (fine-tuning)")
    a = ap.parse_args()

    if a.space == "probe":
        objective, meta = make_probe_objective(a.emb, a.splits, a.seed)
    else:
        objective, meta = make_finetune_objective(a.config, a.set, a.epochs)

    study_name = a.study_name or f"hpo_{a.space}"
    study = optuna.create_study(
        study_name=study_name,
        storage=a.storage or None,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=10),
        # ne secemo pre 5. trial-a (nema sa cim da se poredi) ni pre 3. epohe (jos se zagreva)
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    already = len([t for t in study.trials if t.state.is_finished()])
    if already:
        print(f"[hpo] nastavljam study '{study_name}' -- vec zavrseno {already} trial-ova")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout, show_progress_bar=True)

    best = best_trial_by_selection(study)
    states: dict[str, int] = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1

    res = {
        "space": a.space,
        "study_name": study_name,
        "metric": SELECTION_METRIC,
        "n_trials_total": len(study.trials),
        "trial_states": states,
        "best_trial_number": best.number if best else None,
        "best_params": best.params if best else None,
        "best_val_top1": best.user_attrs.get("top1") if best else None,
        "best_val_top3": best.user_attrs.get("top3") if best else None,
        **meta,
    }

    out = Path(a.report or f"reports/hpo_{a.space}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== HPO {a.space}: {len(study.trials)} trial-ova, {states} ===")
    if best:
        print(f"    najbolji #{best.number}: val top1 {res['best_val_top1']:.4f} "
              f"top3 {res['best_val_top3']:.4f}")
        for k, v in best.params.items():
            print(f"      {k:20} {v}")
    print(f"    -> {out}")
    for f in save_plots(study, Path(a.figures)):
        print(f"    -> {f}")


if __name__ == "__main__":
    main()
