"""Detekcija drifta: da li produkcija i dalje lici na ono na cemu je model meren.

Zasto je ovo potrebno uopste
----------------------------
Tacnost u produkciji se ne moze meriti, jer tacne odgovore niko ne zna -- korisnik
salje sliku psa i ne kaze koja je rasa. Zato se prati ono sto se MOZE meriti: da li
se raspodela onoga sto model vidi i izbacuje promenila u odnosu na dan kad je izmeren.

Ako se raspodela poverenja pomeri, prag "nisam siguran" (fitovan na val skupu) vise
ne znaci ono sto je znacio, a prijavljenih 98,25% top-3 vise ne vazi za ovaj promet.
To je najranije upozorenje koje se moze dobiti bez labela.

Dve mere, namerno razlicite po prirodi
--------------------------------------
**PSI** (*Population Stability Index*) poredi UDELE po binovima:

    PSI = sum_i (p_prod[i] - p_ref[i]) * ln(p_prod[i] / p_ref[i])

Simetrican je, uvek >= 0, i industrijski su se ustalile granice koje i mi koristimo:
< 0,10 stabilno | 0,10-0,25 pomak | > 0,25 ozbiljan drift. Radi i za brojeve (binovani
po kvantilima reference) i za kategorije (120 rasa).

**KS** (*Kolmogorov-Smirnov*) je najveca razlika dve kumulativne raspodele. Ne zavisi
od izbora binova -- sto je tacno slabost PSI-ja -- i daje p-vrednost. Vazi samo za
brojeve.

Kada se ne slazu, PSI je prijavio artefakt binovanja. Zato se racunaju oba.

Referenca se ZAMRZAVA
---------------------
Referentna raspodela se pravi jednom, iz **val** skupa, kroz isti produkcioni put
(ONNX + kapija), i cuva u `reports/drift_reference.json`. Ne sme da se racuna iz
produkcije "pre nedelju dana" -- tako bi drift koji dolazi polako prosao nezapazeno,
jer bi referenca klizila zajedno sa njim.

    python -m dogbreeds.drift reports/drift_reference.json data/prod/service.db
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Granice su industrijski standard za PSI; drzimo ih na jednom mestu da ih i servis,
# i alarmi, i izvestaj citaju isto.
PSI_SHIFT = 0.10        # ispod ovoga: stabilno
PSI_DRIFT = 0.25        # iznad ovoga: ozbiljan drift

# Ispod ovoliko uzoraka PSI je sum, ne signal: sa 50 zahteva i 10 binova po bin padne 5.
MIN_SAMPLES = 200

N_BINS = 10             # binovi za brojeve, po kvantilima reference
REF_SAMPLE_CAP = 1000   # koliko referentnih vrednosti cuvamo za KS

NUMERIC_FEATURES = ("top3_mass", "max_prob", "entropy", "dog_score")


# ---------------------------------------------------------------- mere

def psi(ref_props: np.ndarray, prod_props: np.ndarray) -> float:
    """Population Stability Index. Oba ulaza su udeli koji sabrani daju 1."""
    # Prazan bin daje ln(0) = -inf. Podizemo pod na vrednost manju od svakog realnog
    # udela, pa renormalizujemo -- standardna praksa i jedini nacin da mera ostane konacna.
    eps = 1e-6
    r = np.clip(np.asarray(ref_props, dtype=np.float64), eps, None)
    p = np.clip(np.asarray(prod_props, dtype=np.float64), eps, None)
    r = r / r.sum()
    p = p / p.sum()
    return float(np.sum((p - r) * np.log(p / r)))


def _ks_p_value(d: float, n: int, m: int) -> float:
    """Asimptotska p-vrednost za dvouzoracki KS (Kolmogorovljeva raspodela)."""
    if n == 0 or m == 0:
        return 1.0
    en = math.sqrt(n * m / (n + m))
    x = (en + 0.12 + 0.11 / en) * d
    if x < 1e-6:
        return 1.0
    s = sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * x * x) for j in range(1, 101))
    return float(min(1.0, max(0.0, 2.0 * s)))


def ks_two_sample(ref: np.ndarray, prod: np.ndarray) -> tuple[float, float]:
    """Vraca (D, p). D je najveca razlika empirijskih kumulativa."""
    a = np.sort(np.asarray(ref, dtype=np.float64))
    b = np.sort(np.asarray(prod, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    return d, _ks_p_value(d, a.size, b.size)


def status_of(value: float) -> str:
    if value >= PSI_DRIFT:
        return "drift"
    if value >= PSI_SHIFT:
        return "pomak"
    return "stabilno"


# ---------------------------------------------------------------- referenca

def _bin_edges(values: np.ndarray, n_bins: int = N_BINS) -> list[float]:
    """Ivice po kvantilima reference, ne po jednakoj sirini.

    top3_mass je nagnut ka 1,0 -- jednaka sirina bi sve strpala u poslednji bin i mera
    ne bi videla nista. Kvantili daju binove sa priblizno jednakim brojem uzoraka.
    """
    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    # Vezane vrednosti mogu dati ponovljene ivice -> prazan bin. Izbacujemo duplikate.
    return [float(e) for e in np.unique(edges)]


def _hist_props(values: np.ndarray, edges: list[float]) -> np.ndarray:
    idx = np.digitize(np.asarray(values, dtype=np.float64), np.array(edges[1:-1]))
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def build_reference(numeric: dict[str, list[float]], top1: list[str],
                    classes: list[str], meta: dict | None = None) -> dict:
    """Pravi zamrznutu referencu iz val skupa."""
    rng = np.random.default_rng(0)
    out: dict = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "psi_shift": PSI_SHIFT, "psi_drift": PSI_DRIFT,
        "min_samples": MIN_SAMPLES,
        "numeric": {}, "categorical": {},
        **(meta or {}),
    }
    for name, vals in numeric.items():
        v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
        if v.size == 0:
            continue
        edges = _bin_edges(v)
        # Uzorak za KS: cuvamo najvise REF_SAMPLE_CAP vrednosti da fajl ostane mali.
        sample = v if v.size <= REF_SAMPLE_CAP else rng.choice(v, REF_SAMPLE_CAP, replace=False)
        out["numeric"][name] = {
            "n": int(v.size),
            "edges": edges,
            "props": [float(x) for x in _hist_props(v, edges)],
            "sample": sorted(round(float(x), 6) for x in sample),
            "mean": float(v.mean()), "std": float(v.std()),
            "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
        }
    counts = Counter(top1)
    total = max(sum(counts.values()), 1)
    out["categorical"]["top1_breed"] = {
        "n": total,
        "classes": list(classes),
        "props": [counts.get(c, 0) / total for c in classes],
    }
    return out


# ---------------------------------------------------------------- poredjenje

def compare(reference: dict, numeric: dict[str, list[float]],
            top1: list[str]) -> dict:
    """Poredi produkcijski prozor sa referencom. Vraca izvestaj spreman za JSON."""
    n = max((len(v) for v in numeric.values()), default=0)
    report: dict = {
        "computed_at": time.time(),
        "n_samples": n,
        "min_samples": reference.get("min_samples", MIN_SAMPLES),
        "features": {},
    }
    if n < report["min_samples"]:
        report["status"] = "nedovoljno podataka"
        report["note"] = (f"{n} uzoraka u prozoru, potrebno bar {report['min_samples']}. "
                          "PSI na manjem uzorku meri sum uzorkovanja, ne drift.")
        return report

    worst = 0.0
    for name, ref in reference.get("numeric", {}).items():
        vals = np.asarray([x for x in numeric.get(name, []) if x is not None], dtype=np.float64)
        if vals.size == 0:
            continue
        prod_props = _hist_props(vals, ref["edges"])
        p = psi(np.asarray(ref["props"]), prod_props)
        d, pval = ks_two_sample(np.asarray(ref["sample"]), vals)
        worst = max(worst, p)
        report["features"][name] = {
            "psi": round(p, 4), "status": status_of(p),
            "ks": round(d, 4), "ks_p": round(pval, 6),
            "ks_significant": bool(pval < 0.01),
            "n": int(vals.size),
            "prod_mean": round(float(vals.mean()), 4),
            "ref_mean": round(float(ref["mean"]), 4),
            "prod_p50": round(float(np.percentile(vals, 50)), 4),
            "ref_p50": round(float(ref["p50"]), 4),
        }

    ref_cat = reference.get("categorical", {}).get("top1_breed")
    if ref_cat and top1:
        classes = ref_cat["classes"]
        counts = Counter(top1)
        total = max(len(top1), 1)
        prod_props = np.array([counts.get(c, 0) / total for c in classes])
        p = psi(np.asarray(ref_cat["props"]), prod_props)
        worst = max(worst, p)
        # Rase koje su najvise doprinele PSI-ju -- da izvestaj kaze i STA se pomerilo.
        #
        # Razdvojeno po smeru namerno. Doprinos je (q-r)*ln(q/r) i on je veci za rasu
        # koja je NESTALA nego za onu koja je porasla: za r=1/12, q->0 daje ~0,94, a
        # q=0,5 daje ~0,75. Jedna lista sortirana po doprinosu zato pokaze samo nestale
        # rase i sakrije pravi uzrok ("korisnici salju samo dve rase"). Izmereno na
        # testu test_promenjena_raspodela_rasa_se_vidi.
        eps = 1e-6
        r = np.clip(np.asarray(ref_cat["props"]), eps, None)
        q = np.clip(prod_props, eps, None)
        contrib = (q - r) * np.log(q / r)

        def _lista(mask: np.ndarray, k: int = 5) -> list[dict]:
            idx = [i for i in np.argsort(contrib)[::-1] if mask[i] and contrib[i] > 0.001]
            return [{"breed": classes[i], "psi_part": round(float(contrib[i]), 4),
                     "ref": round(float(r[i]), 5), "prod": round(float(q[i]), 5)}
                    for i in idx[:k]]

        cesce, redje = _lista(q > r), _lista(q < r)
        report["features"]["top1_breed"] = {
            "psi": round(p, 4), "status": status_of(p),
            "n": total, "n_classes": len(classes),
            "cesce_nego_u_referenci": cesce,
            "redje_nego_u_referenci": redje,
            # Zadrzano zbog kompatibilnosti: obe liste spojene i sortirane po doprinosu.
            "top_contributors": sorted(cesce + redje,
                                       key=lambda d: -d["psi_part"])[:5],
        }

    report["worst_psi"] = round(worst, 4)
    report["status"] = status_of(worst)
    return report


# ---------------------------------------------------------------- CLI

def _main(argv: list[str]) -> int:
    import argparse
    import sqlite3

    ap = argparse.ArgumentParser(description="Izvestaj o driftu iz dnevnika predikcija.")
    ap.add_argument("reference", type=Path, help="reports/drift_reference.json")
    ap.add_argument("db", type=Path, help="SQLite baza servisa")
    ap.add_argument("--hours", type=float, default=24.0, help="prozor unazad (sati)")
    ap.add_argument("--out", type=Path, default=None, help="gde snimiti izvestaj")
    a = ap.parse_args(argv)

    reference = json.loads(a.reference.read_text(encoding="utf-8"))
    conn = sqlite3.connect(str(a.db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT top3_mass, max_prob, entropy, dog_score, top1 FROM predictions"
        " WHERE ts >= ? ORDER BY ts", (time.time() - a.hours * 3600,)).fetchall()
    conn.close()

    numeric = {f: [r[f] for r in rows] for f in NUMERIC_FEATURES}
    rep = compare(reference, numeric, [r["top1"] for r in rows])
    rep["window_hours"] = a.hours

    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
