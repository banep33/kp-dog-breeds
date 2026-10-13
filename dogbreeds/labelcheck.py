"""Dan 3 — traženje POGREŠNIH LABELA (cleanlab + confident learning).

ZAŠTO
-----
Stanford Dogs je skupljan sa interneta i labele nisu savršene: isti pas je negde
upisan kao `siberian_husky`, negde kao `eskimo_dog`; štenci se mešaju; neke slike
sadrže dva psa različitih rasa. Svaka pogrešna labela truje dvaput:
  * u treningu — model uči netačnu vezu
  * u testu    — kažnjavamo model kad je ZAPRAVO u pravu

Zato je popravljanje podataka često jeftinije od svakog daljeg podešavanja modela.

KAKO (confident learning, Northcutt et al.)
-------------------------------------------
Da bismo procenili da li je labela slike X pogrešna, treba nam predikcija modela
koji X **nije video u treningu** — inače bi model samo ponovio (možda pogrešnu)
labelu koju je zapamtio. Rešenje je unakrsna validacija:

  1. podeli podatke na K delova (grupno! duplikati moraju ostati zajedno)
  2. za svaki deo: treniraj glavu na ostalih K-1, predvidi za taj deo
  3. svaka slika tako dobija **out-of-sample** verovatnoće
  4. cleanlab iz njih rangira slike kod kojih se model ubedljivo ne slaže sa labelom

Nezavisna potvrda: `dogbreeds.dedup.cross_label_pairs` nalazi skoro identične slike
sa RAZLIČITIM labelama. Slike koje prijave OBA metoda su najsigurniji kandidati.

    python -m dogbreeds.labelcheck --emb data/emb/efficientnet_b0 \
        --labels data/raw/labels_grouped.csv --folds 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .embed import load as load_emb
from .probe import apply_head, train_head


def out_of_sample_probs(emb: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        n_classes: int, folds: int = 5, seed: int = 42,
                        epochs: int = 40, verbose: bool = True) -> np.ndarray:
    """Verovatnoće za SVAKU sliku, dobijene od modela koji tu sliku nije video.

    StratifiedGroupKFold, a ne obican KFold: grupe skoro identicnih slika moraju
    ostati zajedno, inace model "vidi" sliku kroz njenog blizanca i out-of-sample
    verovatnoce vise nisu out-of-sample.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    probs = np.zeros((len(y), n_classes), dtype=np.float32)
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(emb, y, groups=groups), 1):
        head, best, _ = train_head(
            emb[tr], y[tr], emb[te], y[te], num_classes=n_classes,
            epochs=epochs, seed=seed, verbose=False)
        probs[te] = apply_head(head, emb[te]).softmax(1).numpy()
        if verbose:
            print(f"  fold {k}/{folds}: {len(tr)} train / {len(te)} held-out "
                  f"| top1 {best['top1']:.4f}", flush=True)
    return probs


def find_issues(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indeksi sumnjivih labela, rangirani (najsumnjiviji prvi)."""
    from cleanlab.filter import find_label_issues

    return find_label_issues(labels=y, pred_probs=probs,
                             return_indices_ranked_by="self_confidence")


def self_confidence(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Koliku verovatnoću model daje baš UPISANOJ labeli. Nisko = sumnjivo."""
    return probs[np.arange(len(y)), y]


def main() -> None:
    ap = argparse.ArgumentParser(description="Nađi verovatno pogrešne labele")
    ap.add_argument("--emb", default="data/emb/efficientnet_b0")
    ap.add_argument("--labels", default="data/raw/labels_grouped.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup-report", default="reports/dedup.json",
                    help="za ukrštanje sa cross-label signalom")
    ap.add_argument("--out", default="reports/label_issues.csv")
    ap.add_argument("--probs-out", default="data/emb/oos_probs.npy")
    a = ap.parse_args()

    emb, ids, meta = load_emb(a.emb)
    df = pd.read_csv(a.labels)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id").loc[ids].reset_index()
    classes = sorted(df["breed"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y = df["breed"].map(c2i).to_numpy(dtype=np.int64)
    groups = df["group"].to_numpy() if "group" in df.columns else np.arange(len(df))

    print(f"[labelcheck] {len(df)} slika, {len(classes)} rasa, backbone {meta['model']}")
    print(f"[labelcheck] {a.folds}-struka grupna unakrsna validacija:")
    probs = out_of_sample_probs(emb, y, groups, len(classes), a.folds, a.seed, a.epochs)

    Path(a.probs_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.probs_out, probs)

    idx = find_issues(probs, y)
    conf = self_confidence(probs, y)
    pred = probs.argmax(1)
    print(f"\n[labelcheck] cleanlab prijavljuje {len(idx)} sumnjivih labela "
          f"({100 * len(idx) / len(df):.2f}% dataseta)")

    out = pd.DataFrame({
        "id": df["id"].to_numpy()[idx],
        "labela": [classes[c] for c in y[idx]],
        "model_kaze": [classes[c] for c in pred[idx]],
        "p_labela": conf[idx].round(4),
        "p_model": probs[idx, pred[idx]].round(4),
        "group": groups[idx],
    })

    # Ukrstanje sa nezavisnim signalom iz deduplikacije.
    dedup_ids: set[str] = set()
    rep = Path(a.dedup_report)
    if rep.exists():
        d = json.loads(rep.read_text(encoding="utf-8"))
        for p in d.get("cross_label_suspects", []):
            dedup_ids.update([str(p.get("id_i")), str(p.get("id_j"))])
    out["i_dedup"] = out["id"].isin(dedup_ids)
    both = int(out["i_dedup"].sum())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    print(f"[labelcheck] od toga {both} potvrdjuje i deduplikacija "
          f"(skoro identicne slike, razlicite labele) -> najsigurniji kandidati")
    print("\nTop 20 najsumnjivijih:")
    print(out.head(20).to_string(index=False))

    # Koje su rase najcesce zamenjene -- otkriva sistemske probleme u labelama
    pairs = (out.groupby(["labela", "model_kaze"]).size()
             .sort_values(ascending=False).head(15))
    print("\nNajcesci parovi (upisano -> model tvrdi):")
    print(pairs.to_string())
    print(f"\n-> {a.out}\n-> {a.probs_out}")


if __name__ == "__main__":
    main()
