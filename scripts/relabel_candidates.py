"""Petlja kontinualnog unapredjivanja: iz produkcije izvlaci slike koje vredi doznaciti.

    .venv\\Scripts\\python.exe scripts\\relabel_candidates.py

Zasto ovo postoji
-----------------
Oglas trazi "kontinualno unapredjivanje". Najskuplja stvar u tom poslu je ljudsko vreme
za labeliranje. Nasumicno uzimanje 500 novih slika skoro nista ne donosi -- model vec
pogadja 98,25% u top-3, pa bi 490 od njih bile slike koje vec zna.

Vredi doznaciti samo ono gde je model **slab ili pogresan**. Ovde se taj izbor pravi iz
dnevnika produkcije, po prioritetu:

  1  korisnik ispravio, tacne rase NIJE bilo ni u prva tri   -- najskuplja greska
  2  korisnik ispravio, bila je u top-3 ali ne prva          -- preklapanje klasa
  3  nisko poverenje (top3_mass < prag), bez ispravke        -- model sam prijavio sumnju
  4  kapija rekla "nije pas", a korisnik nije reagovao       -- ili OOD, ili kapija gresi

Izlaz je CSV za Label Studio i sazetak po rasama -- koja rasa trosi najvise gresaka,
tj. gde dodatni podaci zaista nesto menjaju.

Veza sa Danom 3: ako se ispravke gomilaju na parovima tipa toy_poodle / miniature_poodle,
to u produkciji potvrdjuje nalaz da granicu drzi preklapanje klasa, a ne nedostatak
podataka -- i tada doznacavanje NE treba raditi, nego spojiti ili razdvojiti klase.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAZLOG = {
    1: "ispravka: nije bilo ni u top-3",
    2: "ispravka: bilo u top-3, ali ne prvo",
    3: "nisko poverenje",
    4: "kapija: nije pas",
}


def _prioritet(r: dict, prag: float) -> int | None:
    if r["correct_breed"] is not None:
        return 1 if not r["was_top3"] else (2 if not r["was_top1"] else None)
    if not r["is_dog"]:
        return 4
    if not r["confident"] or r["top3_mass"] < prag:
        return 3
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "data/prod/service.db")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/relabel_candidates.csv")
    ap.add_argument("--summary", type=Path, default=ROOT / "reports/relabel_summary.json")
    ap.add_argument("--days", type=float, default=30.0, help="prozor unazad")
    ap.add_argument("--limit", type=int, default=1000, help="najvise redova u CSV-u")
    ap.add_argument("--threshold", type=float, default=None,
                    help="prag poverenja; podrazumevano iz reports/threshold.json")
    a = ap.parse_args()

    if not a.db.exists():
        print(f"[GRESKA] nema baze {a.db}")
        print("         Podigni stek (docker compose up -d) i pusti promet (scripts/replay.py).")
        return 1

    prag = a.threshold
    if prag is None:
        try:
            prag = float(json.loads((ROOT / "reports/threshold.json")
                                    .read_text(encoding="utf-8"))["threshold"])
        except (OSError, ValueError, KeyError):
            prag = 0.915

    since = time.time() - a.days * 86400
    conn = sqlite3.connect(str(a.db))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT p.*, f.correct_breed, f.was_top1, f.was_top3, f.note AS fb_note"
        "  FROM predictions p LEFT JOIN feedback f USING (request_id)"
        " WHERE p.ts >= ? ORDER BY p.ts", (since,)).fetchall()]
    total = len(rows)
    conn.close()

    if not total:
        print(f"[GRESKA] nema predikcija u poslednjih {a.days:.0f} dana.")
        return 1

    kandidati = []
    for r in rows:
        p = _prioritet(r, prag)
        if p is None:
            continue
        top3 = json.loads(r["top3_json"])
        kandidati.append({
            "prioritet": p,
            "razlog": RAZLOG[p],
            "request_id": r["request_id"],
            "vreme": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])),
            "model_predvideo": r["top1"],
            "korisnik_kaze": r["correct_breed"] or "",
            "top1": top3[0]["breed"] if len(top3) > 0 else "",
            "p1": round(top3[0]["probability"], 4) if len(top3) > 0 else "",
            "top2": top3[1]["breed"] if len(top3) > 1 else "",
            "p2": round(top3[1]["probability"], 4) if len(top3) > 1 else "",
            "top3": top3[2]["breed"] if len(top3) > 2 else "",
            "p3": round(top3[2]["probability"], 4) if len(top3) > 2 else "",
            "top3_mass": round(r["top3_mass"], 4),
            "entropija": round(r["entropy"], 4),
            "dog_score": round(r["dog_score"], 4) if r["dog_score"] is not None else "",
            "napomena": r["fb_note"] or "",
        })

    # Prioritet raste nagore; unutar istog prioriteta prvo najnesigurnije.
    kandidati.sort(key=lambda k: (k["prioritet"], k["top3_mass"]))
    kandidati = kandidati[:a.limit]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kandidati[0].keys()))
        w.writeheader()
        w.writerows(kandidati)

    po_razlogu = Counter(k["razlog"] for k in kandidati)
    # Koja rasa trosi najvise gresaka: kad korisnik ispravi, racunamo njegovu rasu;
    # inace prvu predvidjenu. To je rasa na koju bi dodatni podaci isli.
    po_rasi = Counter(k["korisnik_kaze"] or k["model_predvideo"] for k in kandidati)
    # Parovi zamena -- direktna provera nalaza iz Dana 3.
    parovi = Counter(
        tuple(sorted((k["model_predvideo"], k["korisnik_kaze"])))
        for k in kandidati if k["korisnik_kaze"] and k["korisnik_kaze"] != k["model_predvideo"])

    sazetak = {
        "generisano": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baza": str(a.db),
        "prozor_dana": a.days,
        "prag_poverenja": prag,
        "predikcija_ukupno": total,
        "kandidata": len(kandidati),
        "udeo_kandidata": round(len(kandidati) / total, 4),
        "po_razlogu": dict(po_razlogu),
        "rase_sa_najvise_gresaka": po_rasi.most_common(15),
        "parovi_zamena": [{"par": list(p), "n": n} for p, n in parovi.most_common(15)],
        "csv": str(a.out),
    }
    a.summary.write_text(json.dumps(sazetak, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nPredikcija u prozoru od {a.days:.0f} dana: {total}")
    print(f"Kandidata za doznacavanje: {len(kandidati)}  "
          f"({len(kandidati) / total:.1%} prometa)\n")
    for p in sorted(RAZLOG):
        n = po_razlogu.get(RAZLOG[p], 0)
        if n:
            print(f"  {p}. {RAZLOG[p]:<36} {n:>5}")
    print(f"\nRase koje trose najvise gresaka:")
    for breed, n in po_rasi.most_common(10):
        print(f"  {breed:<32} {n:>4}")
    if parovi:
        print(f"\nParovi zamena (model -> korisnik), provera nalaza iz Dana 3:")
        for (x, y), n in parovi.most_common(8):
            print(f"  {x:<26} <-> {y:<26} {n:>3}")
    print(f"\n-> {a.out}")
    print(f"-> {a.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
