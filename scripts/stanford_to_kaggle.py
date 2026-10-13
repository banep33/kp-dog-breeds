"""Preuzima Stanford Dogs (120 rasa, 20.580 slika) i pretvara ga u Kaggle format:
    <out>/labels.csv   (id,breed)
    <out>/train/<id>.jpg

Kaggle "Dog Breed Identification" je podskup ovog dataseta, pa je ostatak koda isti.
Ne treba Kaggle nalog ni prihvatanje pravila.

    python scripts/stanford_to_kaggle.py --out /content/data
"""
import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"


def convert(images_root: Path, out: Path) -> Path:
    """images_root sadrži foldere tipa 'n02085620-Chihuahua'."""
    (out / "train").mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(p for p in images_root.iterdir() if p.is_dir()):
        breed = d.name.split("-", 1)[1].lower()  # 'n02085620-Chihuahua' -> 'chihuahua' (kao u Kaggle-u)
        for f in d.glob("*.jpg"):
            shutil.move(str(f), out / "train" / f.name)
            rows.append({"id": f.stem, "breed": breed})
    labels = out / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    print(f"{len(rows)} slika, {len({r['breed'] for r in rows})} rasa -> {labels}")
    return labels


def download_and_convert(out: Path) -> Path:
    out = Path(out)
    labels = out / "labels.csv"
    if labels.exists():
        return labels
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / "images.tar"
    if not tar_path.exists():
        print("preuzimam ~760 MB sa Stanford servera...")
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(tar_path), URL], check=True)
    with tarfile.open(tar_path) as t:
        t.extractall(out)
    labels = convert(out / "Images", out)
    shutil.rmtree(out / "Images", ignore_errors=True)
    tar_path.unlink(missing_ok=True)
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/stanford")
    download_and_convert(Path(ap.parse_args().out))
