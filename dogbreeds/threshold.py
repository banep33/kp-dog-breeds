"""Prag sigurnosti: kada servis uz tri odgovora prikazuje upozorenje "nisam siguran".

    python -m dogbreeds.threshold --run runs/probe_efficientnet_b0 --splits data/splits

KOJI SIGNAL
-----------
`docs/PLAN.md` predlaze `max_prob`, ali merenje pokazuje da je **zbir verovatnoca prva
tri odgovora** (`top3_mass`) znatno bolji: pri istoj sigurnosti odgovorenih (~99,5%)
daje 96% pokrivenosti umesto 73,5%. Razlog je jednostavan -- proizvod prikazuje TRI
odgovora, pa treba meriti poverenje u ono sto se stvarno prikazuje, a ne u prvi odgovor.
Modul racuna oba i bira po podacima.

KOJE PRAVILO
------------
Upozorenje nije odbijanje: korisnik i dalje vidi tri rase, samo uz napomenu. Zato cena
laznog upozorenja nije velika, a cena propustene greske jeste. Pravilo je zato
proizvodno: **upozori na najvise `max_warn_rate` zahteva, i u tom budzetu uhvati sto
vise gresaka.** Prag se fituje ISKLJUCIVO na val skupu; test se meri jednom, na kraju.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = ("top3_mass", "max_prob")


def confidence(probs: np.ndarray, signal: str) -> np.ndarray:
    """Skalarno poverenje po slici, iz matrice verovatnoca [N, C]."""
    srt = -np.sort(-probs, axis=1)
    if signal == "max_prob":
        return srt[:, 0]
    if signal == "top3_mass":
        return srt[:, :3].sum(1)
    raise ValueError(f"nepoznat signal: {signal}")


def hit_topk(probs: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    return (np.argsort(-probs, axis=1)[:, :k] == y[:, None]).any(1)


def sweep(conf: np.ndarray, hit3: np.ndarray, grid: np.ndarray) -> list[dict]:
    """Risk-coverage kriva: za svaki prag koliko se odgovara i koliko se gresi."""
    n_err = int((~hit3).sum())
    rows = []
    for tau in grid:
        warn = conf < tau
        n_warn = int(warn.sum())
        caught = int((warn & ~hit3).sum())
        rows.append({
            "tau": float(tau),
            "warn_rate": n_warn / len(conf),
            "coverage": 1.0 - n_warn / len(conf),
            "top3_on_answered": float(hit3[~warn].mean()) if (~warn).any() else float("nan"),
            "n_warned": n_warn,
            "errors_caught": caught,
            "error_recall": caught / n_err if n_err else 0.0,
        })
    return rows


def select(conf: np.ndarray, hit3: np.ndarray, max_warn_rate: float = 0.05,
           grid: np.ndarray | None = None) -> tuple[dict, list[dict]]:
    """Najveci prag koji ostaje u budzetu upozorenja -- on hvata najvise gresaka.

    `error_recall` monotono raste sa pragom, a `warn_rate` takodje; zato je optimum
    u budzetu uvek na njegovoj ivici."""
    grid = np.round(np.arange(0.00, 1.0001, 0.005), 4) if grid is None else grid
    rows = sweep(conf, hit3, grid)
    ok = [r for r in rows if r["warn_rate"] <= max_warn_rate]
    best = max(ok, key=lambda r: (r["error_recall"], r["tau"])) if ok else rows[0]
    return best, rows


def load_split(run_dir: Path, splits_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    probs = np.load(run_dir / f"{split}_probs.npy")
    classes = json.loads((splits_dir / "classes.json").read_text(encoding="utf-8"))
    df = pd.read_csv(splits_dir / f"{split}.csv")
    y = df["breed"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    if len(y) != len(probs):
        raise ValueError(f"{split}: {len(probs)} verovatnoca, {len(y)} labela")
    return probs, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Izbor praga sigurnosti na val skupu")
    ap.add_argument("--run", default="runs/probe_efficientnet_b0",
                    help="folder sa val_probs.npy i test_probs.npy")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--max-warn-rate", type=float, default=0.05,
                    help="na koliko najvise zahteva smemo da prikazemo upozorenje")
    ap.add_argument("--report", default="reports/threshold.json")
    a = ap.parse_args()

    run_dir, splits_dir = Path(a.run), Path(a.splits)
    val_probs, val_y = load_split(run_dir, splits_dir, "val")
    val_hit3 = hit_topk(val_probs, val_y)

    chosen, per_signal = None, {}
    for sig in SIGNALS:
        best, rows = select(confidence(val_probs, sig), val_hit3, a.max_warn_rate)
        # kriva se cuva prorediena (korak 0,02 od 0,70): izvestaj se cita ocima,
        # a pun sweep od 200 tacaka samo pravi sum u gitu
        curve = [r for r in rows if r["tau"] >= 0.70 and round(r["tau"] * 1000) % 20 == 0]
        per_signal[sig] = {"best": best, "curve": curve}
        print(f"[{sig:10}] tau {best['tau']:.3f} | upozorenje na {best['warn_rate']:.1%} "
              f"| top3 na odgovorenim {best['top3_on_answered']:.2%} "
              f"| uhvaceno gresaka {best['errors_caught']}/{int((~val_hit3).sum())} "
              f"({best['error_recall']:.1%})")
        if chosen is None or best["error_recall"] > chosen[1]["error_recall"]:
            chosen = (sig, best)

    sig, best = chosen
    res = {
        "signal": sig,
        "threshold": best["tau"],
        "rule": f"najveci tau sa warn_rate <= {a.max_warn_rate:.0%}, fitovan na val",
        "val": best,
        "n_val": int(len(val_y)),
        "per_signal": per_signal,
    }

    # jednokratna provera na TEST -- prag se NE menja prema ovim brojevima
    test_path = run_dir / "test_probs.npy"
    if test_path.exists():
        test_probs, test_y = load_split(run_dir, splits_dir, "test")
        test_hit3 = hit_topk(test_probs, test_y)
        t_conf = confidence(test_probs, sig)
        res["test"] = sweep(t_conf, test_hit3, np.array([best["tau"]]))[0]
        res["n_test"] = int(len(test_y))
        t = res["test"]
        print(f"\n[TEST, prag {best['tau']:.3f}] upozorenje na {t['warn_rate']:.1%} "
              f"| top3 na odgovorenim {t['top3_on_answered']:.2%} "
              f"| uhvaceno gresaka {t['errors_caught']}/{int((~test_hit3).sum())} "
              f"({t['error_recall']:.1%})")

    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
