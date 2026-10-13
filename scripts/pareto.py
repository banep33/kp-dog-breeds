"""Dan 2 — Pareto kriva: tačnost naspram latencije na CILJNOM hardveru.

Spaja dva nezavisna merenja:
  * tačnost        reports/probe_results.json     (dogbreeds.probe --sweep)
  * latencija      reports/latency_candidates.json (scripts/benchmark_candidates.py)

Zašto Pareto, a ne "najbolji model"? Zato što "dovoljno mali da radi brzo" i
"dovoljno veliki za visoku preciznost" iz zadatka nisu jedan cilj nego dva, i
između njih postoji razmena. Model je **dominiran** ako postoji drugi koji je
istovremeno i brži i tačniji — takve odbacujemo bez diskusije. Ostatak je
Pareto granica i tu odluka postaje poslovna, ne tehnička.

    python scripts/pareto.py --budget-p95 100 --budget-mb 50

Izlaz: reports/pareto.json + reports/figures/pareto.png + tabela u terminalu
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("reports/figures")


def arch_of(timm_name: str) -> str:
    """'efficientnet_b0.ra4_e3600_r224_in1k' -> 'efficientnet_b0' (latencija zavisi
    od arhitekture, ne od toga na kom skupu su težine pretrenirane)."""
    return timm_name.split(".")[0]


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Pola širine 95% intervala poverenja za tačnost. Bez ovoga ne znaš da li je
    91,30% stvarno bolje od 91,25% ili je razlika jedna slika."""
    if n <= 0:
        return 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(abs(centre + half - p), abs(p - (centre - half)))


def pareto_front(rows: list[dict], acc_key: str, lat_key: str) -> list[dict]:
    """Model je dominiran ako postoji drugi sa >= tačnošću I <= latencijom
    (uz bar jednu strogu nejednakost)."""
    for r in rows:
        r["dominated_by"] = [
            o["arch"] for o in rows
            if o is not r
            and o[acc_key] >= r[acc_key] and o[lat_key] <= r[lat_key]
            and (o[acc_key] > r[acc_key] or o[lat_key] < r[lat_key])
        ]
        r["on_pareto"] = not r["dominated_by"]
    return rows


def plot(rows: list[dict], acc_key: str, budget_p95: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    front = sorted([r for r in rows if r["on_pareto"]], key=lambda r: r["p95_ms"])
    if len(front) > 1:
        ax.plot([r["p95_ms"] for r in front], [100 * r[acc_key] for r in front],
                "-", color="#0d6b68", lw=1.4, alpha=0.6, zorder=1, label="Pareto granica")
    for r in rows:
        on = r["on_pareto"]
        ax.scatter(r["p95_ms"], 100 * r[acc_key], s=70 + 9 * r.get("params_M", 5),
                   color="#0d6b68" if on else "#b9c6c4",
                   edgecolor="#0d6b68" if on else "#8a9a98", zorder=3)
        ax.annotate(f"{r['arch']}\n{r.get('params_M', '?')}M",
                    (r["p95_ms"], 100 * r[acc_key]),
                    textcoords="offset points", xytext=(9, -4),
                    fontsize=8.5, color="#121a19" if on else "#6b7b79")
    if budget_p95:
        ax.axvline(budget_p95, color="#a8430d", ls="--", lw=1.2)
        ax.text(budget_p95, ax.get_ylim()[0], f" budžet {budget_p95:.0f} ms p95",
                color="#a8430d", fontsize=9, va="bottom")
    ax.set_xlabel("CPU latencija p95, batch 1, 4 niti  [ms]  →  manje je bolje")
    ax.set_ylabel(f"{acc_key}  [%]  →  više je bolje")
    ax.set_title("Pareto: tačnost naspram brzine (linear probe, Ryzen 7 5700U)")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="reports/probe_results.json")
    ap.add_argument("--latency", default="reports/latency_candidates.json")
    ap.add_argument("--acc-key", default="test_top3", help="metrika za y-osu")
    ap.add_argument("--budget-p95", type=float, default=100.0, help="granica latencije [ms]")
    ap.add_argument("--out", default="reports/pareto.json")
    a = ap.parse_args()

    probe = json.loads(Path(a.probe).read_text(encoding="utf-8"))
    lat = {r["model"]: r for r in json.loads(Path(a.latency).read_text(encoding="utf-8"))}

    rows = []
    for p in probe:
        arch = arch_of(p["model"])
        if arch not in lat:
            print(f"[preskačem] nema merenja latencije za {arch}")
            continue
        n = p.get("n_test", 2058)
        rows.append({
            "arch": arch, "timm_name": p["model"],
            "params_M": lat[arch]["params_M"], "img": lat[arch]["img"],
            "p50_ms": lat[arch]["p50_ms"], "p95_ms": lat[arch]["p95_ms"],
            "test_top1": p["test_top1"], "test_top3": p["test_top3"],
            "test_top1_ci95": wilson_halfwidth(p["test_top1"], n),
            "test_ece_calibrated": p["test_ece_calibrated"],
            "temperature": p["temperature"],
            "within_budget": lat[arch]["p95_ms"] <= a.budget_p95,
        })

    rows = pareto_front(rows, a.acc_key, "p95_ms")
    rows.sort(key=lambda r: -r[a.acc_key])

    hdr = f"{'model':<26}{'par':>6}{'p95 ms':>9}{'top-1':>9}{'±95%':>7}{'top-3':>9}{'ECE':>8}  status"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        status = "PARETO" if r["on_pareto"] else "dominiran: " + ", ".join(r["dominated_by"])
        if not r["within_budget"]:
            status += " | VAN BUDZETA"
        print(f"{r['arch']:<26}{r['params_M']:>5.1f}M{r['p95_ms']:>9.1f}"
              f"{100 * r['test_top1']:>8.2f}%{100 * r['test_top1_ci95']:>6.2f}"
              f"{100 * r['test_top3']:>8.2f}%{r['test_ece_calibrated']:>8.4f}  {status}")

    # Da li su dva najbolja stvarno razlicita? Preklapanje intervala poverenja = nisu.
    best = [r for r in rows if r["on_pareto"]]
    if len(best) >= 2:
        a1, a2 = best[0], best[1]
        gap = abs(a1["test_top1"] - a2["test_top1"])
        tied = gap < (a1["test_top1_ci95"] + a2["test_top1_ci95"])
        print(f"\n{a1['arch']} vs {a2['arch']}: razlika u top-1 je "
              f"{100 * gap:.2f}pp -> {'STATISTICKI NERAZLUCIVO' if tied else 'stvarna razlika'}"
              f" (intervali {'se preklapaju' if tied else 'se ne preklapaju'})")
        if tied:
            faster = min(best[:2], key=lambda r: r["p95_ms"])
            print(f"   => odluka pada na brzinu/velicinu: {faster['arch']} "
                  f"({faster['p95_ms']:.1f} ms, {faster['params_M']}M)")

    plot(rows, a.acc_key, a.budget_p95, FIG / "pareto.png")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n-> {a.out}\n-> {FIG / 'pareto.png'}")


if __name__ == "__main__":
    main()
