"""Pravi mali lažni dataset u Kaggle formatu (labels.csv + train/*.jpg) za brzi test pipeline-a
na laptopu, bez GPU-a i bez pravih podataka.

    python scripts/make_dummy_data.py --out data/dummy --classes 5 --per-class 12
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=12)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rows = []
    for c in range(a.classes):
        color = tuple(rng.randint(0, 255) for _ in range(3))
        for j in range(a.per_class):
            w, h = rng.randint(180, 400), rng.randint(180, 400)
            img = Image.new("RGB", (w, h), tuple(rng.randint(0, 60) for _ in range(3)))
            d = ImageDraw.Draw(img)
            x0, y0 = rng.randint(0, w // 2), rng.randint(0, h // 2)
            d.ellipse([x0, y0, x0 + w // 3, y0 + h // 3], fill=color)
            img_id = f"c{c}_{j:03d}"
            img.save(out / "train" / f"{img_id}.jpg", quality=90)
            rows.append({"id": img_id, "breed": f"breed_{c}"})
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    print(f"{len(rows)} slika -> {out}")


if __name__ == "__main__":
    main()
