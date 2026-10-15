"""Pusta slike kroz ZIV servis, kao da ih salju korisnici.

    .venv\\Scripts\\python.exe scripts\\replay.py --url http://127.0.0.1:8001

Cemu sluzi -- dve stvari odjednom:

1. **Puni dnevnik produkcije.** Monitoring nad praznom bazom je slika bez sadrzaja.
   Drift, Grafana paneli i izbor kandidata za doznacavanje labela traze stvaran promet.

2. **Nezavisno proverava servis od kraja do kraja.** Posto za test skup znamo tacne
   odgovore, racunamo top-1 i top-3 kroz ceo HTTP + Docker put i poredimo sa brojevima
   izmerenim lokalno (90,43% / 98,25%). Ako se razidju, negde postoji training/serving
   skew -- a to je greska koju nijedan unit test ne hvata.

Povratne informacije se salju POSTENO: tacan odgovor dolazi iz test.csv, a salje se samo
za deo zahteva, jer ni pravi korisnici ne prijavljuju svaku gresku. Nista se ne izmislja.

NAPOMENA: ovo je simuliran promet, ne pravi korisnici. Tako i treba da stoji u izvestaju.
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.smoke_service import _post_image, _post_json  # noqa: E402


def degrade(path: Path, mode: str, out_dir: Path) -> Path:
    """Namerno kvari sliku -- za PROVERU da detektor drifta i alarmi zaista opale.

    Detektor koji nikad nije opalio nije proveren detektor. Svaki rezim oponasa stvaran
    uzrok drifta u produkciji, ne nasumican sum:

      siva     korisnici predju na kameru koja salje crno-bele snimke
      mutno    losiji objektiv ili pomeren fokus
      sitno    agresivnija kompresija na strani klijenta (slanje preko mobilne mreze)
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        if mode == "siva":
            im = im.convert("L").convert("RGB")
        elif mode == "mutno":
            im = im.filter(ImageFilter.GaussianBlur(radius=3.0))
        elif mode == "sitno":
            w, h = im.size
            buf = io.BytesIO()
            im.resize((max(w // 6, 32), max(h // 6, 32)), Image.BILINEAR).save(
                buf, format="JPEG", quality=20)
            buf.seek(0)
            im = Image.open(buf).convert("RGB").resize((w, h), Image.BILINEAR)
        else:
            raise ValueError(f"nepoznat rezim '{mode}'")
        out = out_dir / f"{path.stem}__{mode}.jpg"
        im.save(out, format="JPEG", quality=92)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Simuliran produkcioni promet ka servisu.")
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--split", type=Path, default=ROOT / "data/splits/test.csv")
    ap.add_argument("--images", type=Path, default=ROOT / "data/raw/train")
    ap.add_argument("--limit", type=int, default=0, help="0 = ceo skup")
    ap.add_argument("--workers", type=int, default=4,
                    help="paralelnih klijenata; servis ima 4 ONNX niti")
    ap.add_argument("--feedback-rate", type=float, default=0.30,
                    help="udeo POGRESNIH predikcija za koje korisnik salje ispravku")
    ap.add_argument("--feedback-rate-ok", type=float, default=0.02,
                    help="udeo TACNIH predikcija koje korisnik ipak potvrdi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degrade", choices=("siva", "mutno", "sitno"), default=None,
                    help="namerno pokvari slike -- provera da detektor drifta opali")
    a = ap.parse_args()

    base = a.url.rstrip("/")
    rng = random.Random(a.seed)

    df = pd.read_csv(a.split)
    if a.limit:
        df = df.head(a.limit)
    rows = [(r.id, r.breed) for r in df.itertuples(index=False)
            if (a.images / f"{r.id}.jpg").exists()]
    if not rows:
        print(f"[GRESKA] nijedna slika iz {a.split.name} nije nadjena u {a.images}")
        return 1

    print(f"\nServis:  {base}")
    print(f"Skup:    {a.split.name}, {len(rows)} slika, {a.workers} paralelnih klijenata")
    print(f"Ispravke: {a.feedback_rate:.0%} gresaka + {a.feedback_rate_ok:.0%} tacnih")
    if a.degrade:
        print(f"KVARENJE: rezim '{a.degrade}' -- ovo je PROVERA detektora drifta, "
              "ne merenje tacnosti")
    print()

    results: list[dict] = []
    errors: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="replay_")) if a.degrade else None
    t0 = time.perf_counter()
    done = 0

    def one(item: tuple[str, str]) -> dict | None:
        img_id, truth = item
        src = a.images / f"{img_id}.jpg"
        if a.degrade:
            src = degrade(src, a.degrade, tmp)
        code, d = _post_image(f"{base}/predict", src, timeout=60)
        if a.degrade:
            src.unlink(missing_ok=True)
        if code != 200:
            errors.append(f"{img_id}: HTTP {code} {d.get('detail')}")
            return None
        breeds = [p["breed"] for p in d["predictions"]]
        return {
            "request_id": d["request_id"], "truth": truth,
            "top1": breeds[0], "top3": breeds,
            "hit1": breeds[0] == truth, "hit3": truth in breeds,
            "latency_ms": d["latency_ms"],
            "top3_mass": d["signals"]["top3_mass"],
            "confident": d["confident"], "is_dog": d["is_dog"],
        }

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one, rows):
            done += 1
            if r:
                results.append(r)
            if done % 250 == 0:
                el = time.perf_counter() - t0
                print(f"  {done}/{len(rows)}   {el:.0f} s   {done / el:.1f} zahteva/s")

    wall = time.perf_counter() - t0
    if not results:
        print("[GRESKA] nijedan zahtev nije uspeo.")
        for e in errors[:5]:
            print("  ", e)
        return 1

    # --- povratne informacije: samo tamo gde bi ih pravi korisnik i poslao ---
    # Kod kvarenja se ne salju: to je dijagnostika sistema, a ne promet korisnika, i ne
    # sme da zavrsi u kandidatima za doznacavanje labela.
    sent_fb = 0
    for r in (results if not a.degrade else []):
        p = a.feedback_rate if not r["hit1"] else a.feedback_rate_ok
        if rng.random() < p:
            code, _ = _post_json(f"{base}/feedback",
                                 {"request_id": r["request_id"],
                                  "correct_breed": r["truth"],
                                  "note": "simuliran promet (scripts/replay.py)"})
            sent_fb += int(code == 200)

    lat = np.array([r["latency_ms"] for r in results])
    n = len(results)
    top1 = sum(r["hit1"] for r in results) / n
    top3 = sum(r["hit3"] for r in results) / n
    warn = sum(not r["confident"] for r in results) / n
    notdog = sum(not r["is_dog"] for r in results) / n
    # Normalna aproksimacija binomnog intervala; isti racun kao u evaluate.py.
    ci3 = 1.96 * (top3 * (1 - top3) / n) ** 0.5 * 100

    print(f"\n{'=' * 62}")
    print(f"  poslato            {n} / {len(rows)}   ({len(errors)} gresaka)")
    print(f"  trajalo            {wall:.0f} s   ({n / wall:.1f} zahteva/s)")
    print(f"{'-' * 62}")
    print(f"  top-1              {top1:.2%}")
    print(f"  top-3              {top3:.2%}   (+/- {ci3:.2f}pp)")
    print(f"  'nisam siguran'    {warn:.2%}")
    print(f"  'nije pas'         {notdog:.2%}")
    print(f"{'-' * 62}")
    print(f"  latencija p50      {np.percentile(lat, 50):.1f} ms")
    print(f"  latencija p95      {np.percentile(lat, 95):.1f} ms")
    print(f"  latencija max      {lat.max():.1f} ms")
    print(f"{'-' * 62}")
    print(f"  ispravki poslato   {sent_fb}")
    print(f"{'=' * 62}")
    if a.degrade:
        print(f"\n  Rezim kvarenja '{a.degrade}'. Pad tacnosti i poverenja je OCEKIVAN --")
        print("  proveravamo da li ga detektor drifta vidi:")
        print(f"      curl \"{base}/drift?refresh=true\"\n")
    else:
        print("\n  Ocekivano iz evaluacije lokalno: top-1 90,33% | top-3 98,15% "
              "| upozorenje 3,9%")
        print("  (brojevi isporucenog artefakta iz Dana 2; HPO config je merio 90,43/98,25)")
        print("  Veca odstupanja od intervala poverenja znacila bi training/serving skew.\n")

    if tmp:
        try:
            tmp.rmdir()
        except OSError:
            pass

    if errors:
        print(f"  Prvih {min(5, len(errors))} gresaka:")
        for e in errors[:5]:
            print("   ", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
