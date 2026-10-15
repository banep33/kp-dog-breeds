"""Pravi ZAMRZNUTU referentnu raspodelu za detekciju drifta, iz val skupa.

    .venv\\Scripts\\python.exe scripts\\make_drift_reference.py

Zasto val a ne train: na val skupu su fitovani prag sigurnosti i temperatura, pa je on
definicija "ovako je izgledalo kad smo merili". Zasto ne test: test se ne dira.

Zasto kroz produkcioni put (ONNX + kapija), a ne kroz PyTorch: referenca mora da sadrzi
tacno one brojeve koje servis racuna, inace bi razlika ONNX/PyTorch izgledala kao drift.

Rezultat: reports/drift_reference.json (~80 KB). Ulazi i u Docker image.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dogbreeds.drift import NUMERIC_FEATURES, build_reference  # noqa: E402
from dogbreeds.inference import Predictor  # noqa: E402
from dogbreeds.notdog import DogGate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=ROOT / "data/splits/val.csv")
    ap.add_argument("--images", type=Path, default=ROOT / "data/raw/train")
    ap.add_argument("--model", type=Path, default=ROOT / "artifacts/model.onnx")
    ap.add_argument("--gate", type=Path, default=ROOT / "artifacts/notdog.onnx")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/drift_reference.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = ceo skup")
    a = ap.parse_args()

    df = pd.read_csv(a.split)
    if a.limit:
        df = df.head(a.limit)
    print(f"Referenca iz {a.split.name}: {len(df)} slika")

    pred = Predictor(a.model, threads=a.threads)
    gate = DogGate(a.gate, threads=a.threads) if a.gate.exists() else None
    if gate is None:
        print("  [upozorenje] nema notdog.onnx -- dog_score nece biti u referenci")

    numeric: dict[str, list[float]] = {f: [] for f in NUMERIC_FEATURES}
    top1: list[str] = []
    missing = 0
    t0 = time.perf_counter()

    for i, row in enumerate(df.itertuples(index=False), 1):
        path = a.images / f"{row.id}.jpg"
        if not path.exists():
            missing += 1
            continue
        with Image.open(path) as img:
            img = img.convert("RGB")
            r = pred.predict(img, k=3)
            score = gate.score(img) if gate else None
        numeric["top3_mass"].append(sum(p["probability"] for p in r["predictions"]))
        numeric["max_prob"].append(r["max_prob"])
        numeric["entropy"].append(r["entropy"])
        if score is not None:
            numeric["dog_score"].append(score)
        top1.append(r["predictions"][0]["breed"])
        if i % 250 == 0:
            el = time.perf_counter() - t0
            print(f"  {i}/{len(df)}   {el:.0f} s   ({el / i * 1000:.0f} ms/slika)")

    if missing:
        print(f"  [upozorenje] {missing} slika nije nadjeno u {a.images}")
    if not top1:
        print("[GRESKA] nijedna slika nije obradjena -- proveri --images putanju")
        return 1

    ref = build_reference(
        numeric, top1, pred.classes,
        meta={
            "source": f"{a.split.name} (n={len(top1)})",
            "model_version": pred.model_version,
            "gate": bool(gate),
            "gate_threshold": gate.threshold if gate else None,
        },
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nGotovo za {time.perf_counter() - t0:.0f} s -> {a.out} "
          f"({a.out.stat().st_size / 1024:.0f} KB)")
    for name, d in ref["numeric"].items():
        print(f"  {name:<12} medijana {d['p50']:.4f}   p05 {d['p05']:.4f}   "
              f"p95 {d['p95']:.4f}   ({d['n']} uzoraka, {len(d['edges']) - 1} binova)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
