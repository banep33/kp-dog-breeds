"""Deduplikacija: nalaženje gotovo identičnih slika i grupisanje u klastere.

ZAŠTO JE OVO NAJVAŽNIJI KORAK U DANU 1
--------------------------------------
Stanford Dogs / Kaggle Dog Breed sadrže serije fotografija istog psa (ista sesija slikanja,
nekoliko kadrova). Ako slika A padne u `train` a njena skoro-identična sestra A' u `test`,
model na testu "pamti", ne generalizuje. Voxel51 je našao ~11% takvih slika u Stanford Dogs.
Rezultat: prijavljena tačnost je optimistična, a u produkciji model radi lošije.

Rešenje u tri koraka:
  1. embedding svake slike (dogbreeds.embed)
  2. parovi sa kosinusnom sličnošću > prag  ->  "ista scena"
  3. tranzitivno zatvaranje (A~B, B~C  =>  A,B,C u istoj grupi) preko **Union-Find**
     -> kolona `group` u labels.csv -> `dogbreeds.data` automatski radi StratifiedGroupKFold

    python -m dogbreeds.dedup --emb data/emb/convnext_tiny --labels data/raw/labels.csv \
        --threshold 0.95 --out data/raw/labels_grouped.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ Union-Find

class UnionFind:
    """Disjoint Set Union sa path compression + union by size.

    Zašto ne samo "svi parovi u isti bucket"? Jer je relacija tranzitivna preko lanaca:
    A~B i B~C znači da A i C moraju u istu grupu iako njihova direktna sličnost možda
    nije prešla prag. To je problem **povezanih komponenti grafa**, a Union-Find ga rešava
    u praktično O(n) (amortizovano O(alfa(n)), gde je alfa inverzna Ackermannova funkcija).
    """

    __slots__ = ("parent", "size")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        # path compression: iterativno, da ne pukne rekurzija na dugim lancima
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:  # union by size: manje drvo pod veće
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def build_groups(pairs, n: int) -> np.ndarray:
    """pairs: iterable parova (i, j). Vraća niz dužine n sa ID-om grupe za svaku sliku.
    Slike bez duplikata dobijaju svoju grupu (veličine 1)."""
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(int(i), int(j))
    roots = np.fromiter((uf.find(k) for k in range(n)), dtype=np.int64, count=n)
    # renumeracija u 0..G-1 da ID-evi budu kompaktni i stabilni
    _, groups = np.unique(roots, return_inverse=True)
    return groups.astype(np.int64)


# ------------------------------------------------------------------ parovi

def find_similar_pairs(emb: np.ndarray, min_sim: float = 0.85, block: int = 1024
                       ) -> tuple[pd.DataFrame, np.ndarray]:
    """Svi parovi (i < j) sa kosinusnom sličnošću >= min_sim + sličnost sa najbližim susedom.

    Embeddinzi su L2-normalizovani, pa je `emb @ emb.T` matrica kosinusnih sličnosti.
    Celu matricu 20.580 x 20.580 ne pravimo odjednom (1.7 GB) -> idemo blok po blok.
    """
    n = emb.shape[0]
    nn_sim = np.full(n, -1.0, dtype=np.float32)
    nn_idx = np.zeros(n, dtype=np.int64)
    rows, cols, sims = [], [], []

    for s in range(0, n, block):
        e = min(s + block, n)
        S = emb[s:e] @ emb.T                      # [blok, n]
        np.fill_diagonal(S[:, s:e], -1.0)         # ne poredi sliku sa samom sobom
        nn_idx[s:e] = S.argmax(1)
        nn_sim[s:e] = S[np.arange(e - s), nn_idx[s:e]]
        r, c = np.where(S >= min_sim)
        keep = (r + s) < c                        # samo gornji trougao -> svaki par jednom
        rows.append(r[keep] + s)
        cols.append(c[keep])
        sims.append(S[r[keep], c[keep]])

    pairs = pd.DataFrame({
        "i": np.concatenate(rows) if rows else np.array([], dtype=np.int64),
        "j": np.concatenate(cols) if cols else np.array([], dtype=np.int64),
        "sim": np.concatenate(sims) if sims else np.array([], dtype=np.float32),
    }).sort_values("sim", ascending=False, ignore_index=True)
    return pairs, nn_sim


def threshold_sweep(pairs: pd.DataFrame, n: int,
                    thresholds=(0.85, 0.90, 0.93, 0.95, 0.97, 0.99)) -> pd.DataFrame:
    """Koliko parova / grupa / "izgubljenih" slika dobijamo za razne pragove.
    Prag je inženjerska odluka: previše nizak -> spajamo različite pse (gubimo podatke),
    previše visok -> curenje ostaje. Ovu tabelu stavljamo u dnevnik odluka."""
    out = []
    for t in thresholds:
        sel = pairs.loc[pairs["sim"] >= t, ["i", "j"]].to_numpy()
        g = build_groups(sel, n)
        sizes = pd.Series(g).value_counts()
        multi = sizes[sizes > 1]
        out.append({
            "threshold": t,
            "pairs": int(len(sel)),
            "groups_total": int(sizes.size),
            "groups_with_dupes": int(multi.size),
            "images_in_dupe_groups": int(multi.sum()),
            "redundant_images": int(multi.sum() - multi.size),  # koliko bi "ostalo" posle brisanja
            "largest_group": int(sizes.max()),
        })
    return pd.DataFrame(out)


def cross_label_pairs(pairs: pd.DataFrame, breeds: np.ndarray, min_sim: float = 0.93) -> pd.DataFrame:
    """Vrlo slične slike sa RAZLIČITIM labelama = kandidati za pogrešnu labelu.
    Besplatan signal o kvalitetu podataka, pre nego što uopšte istreniramo model (Dan 3)."""
    s = pairs[pairs["sim"] >= min_sim].copy()
    s["breed_i"] = breeds[s["i"].to_numpy()]
    s["breed_j"] = breeds[s["j"].to_numpy()]
    return s[s["breed_i"] != s["breed_j"]].reset_index(drop=True)


# ------------------------------------------------------------------ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Nađi duplikate i dodaj kolonu 'group' u labels.csv")
    ap.add_argument("--emb", default="data/emb/convnext_tiny", help="prefiks .npy/.json iz dogbreeds.embed")
    ap.add_argument("--labels", default="data/raw/labels.csv")
    ap.add_argument("--out", default="data/raw/labels_grouped.csv")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--min-sim", type=float, default=0.85, help="dokle skupljamo kandidate za sweep")
    ap.add_argument("--report", default="reports/dedup.json")
    a = ap.parse_args()

    from .embed import load

    emb, ids, meta = load(a.emb)
    df = pd.read_csv(a.labels)
    df["id"] = df["id"].astype(str)
    # embeddinzi su u redosledu iz svog .json-a -> poravnaj labels.csv na taj redosled
    df = df.set_index("id").loc[ids].reset_index()
    assert len(df) == emb.shape[0], f"{len(df)} labela vs {emb.shape[0]} embeddinga"

    print(f"[dedup] {emb.shape[0]} slika, backbone {meta['model']}, dim {meta['dim']}")
    pairs, nn_sim = find_similar_pairs(emb, a.min_sim)
    print(f"[dedup] {len(pairs)} parova sa sim >= {a.min_sim}")

    sweep = threshold_sweep(pairs, len(df))
    print("\n" + sweep.to_string(index=False))

    breeds = df["breed"].to_numpy()
    cross = cross_label_pairs(pairs, breeds)
    print(f"\n[dedup] {len(cross)} vrlo slicnih parova sa RAZLICITIM labelama (sim >= 0.93)")

    sel = pairs.loc[pairs["sim"] >= a.threshold, ["i", "j"]].to_numpy()
    df["group"] = build_groups(sel, len(df))
    sizes = df["group"].value_counts()
    chosen = {
        "threshold": a.threshold,
        "n_images": int(len(df)),
        "n_pairs": int(len(sel)),
        "n_groups": int(sizes.size),
        "groups_with_dupes": int((sizes > 1).sum()),
        "images_in_dupe_groups": int(sizes[sizes > 1].sum()),
        "pct_images_affected": round(100 * sizes[sizes > 1].sum() / len(df), 2),
        "largest_group": int(sizes.max()),
        "nn_sim_median": float(np.median(nn_sim)),
        "nn_sim_p99": float(np.percentile(nn_sim, 99)),
        "embedding_model": meta["model"],
    }
    print("\n[dedup] izabrano:", json.dumps(chosen, indent=2))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df[["id", "breed", "group"]].to_csv(a.out, index=False)
    print(f"[dedup] -> {a.out}")

    rep = Path(a.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({
        "chosen": chosen,
        "sweep": sweep.to_dict("records"),
        "top_pairs": pairs.head(40).assign(
            id_i=df["id"].to_numpy()[pairs.head(40)["i"].to_numpy()],
            id_j=df["id"].to_numpy()[pairs.head(40)["j"].to_numpy()],
        ).to_dict("records"),
        "cross_label_suspects": cross.head(60).assign(
            id_i=df["id"].to_numpy()[cross.head(60)["i"].to_numpy()],
            id_j=df["id"].to_numpy()[cross.head(60)["j"].to_numpy()],
        ).to_dict("records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"[dedup] -> {rep}")


if __name__ == "__main__":
    main()
