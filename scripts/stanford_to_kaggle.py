"""Preuzima Stanford Dogs (120 rasa, 20.580 slika) i pretvara ga u Kaggle format:
    <out>/labels.csv   (id,breed)
    <out>/train/<id>.jpg

Kaggle "Dog Breed Identification" je podskup ovog dataseta (10.222 slike, iste 120 rase),
pa je ostatak koda identičan. Ovde ne treba Kaggle nalog ni prihvatanje pravila takmičenja.

    python scripts/stanford_to_kaggle.py --out data/raw
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"


def convert(images_root: Path, out: Path, copy: bool = False) -> Path:
    """images_root sadrži foldere tipa 'n02085620-Chihuahua'.
    copy=True kad je izvor read-only (npr. /kaggle/input)."""
    labels = out / "labels.csv"
    if labels.exists():
        return labels
    (out / "train").mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(p for p in images_root.iterdir() if p.is_dir()):
        breed = d.name.split("-", 1)[1].lower()  # 'n02085620-Chihuahua' -> 'chihuahua' (kao u Kaggle-u)
        for f in d.glob("*.jpg"):
            (shutil.copy2 if copy else shutil.move)(str(f), out / "train" / f.name)
            rows.append({"id": f.stem, "breed": breed})
    pd.DataFrame(rows).to_csv(labels, index=False)
    print(f"{len(rows)} slika, {len({r['breed'] for r in rows})} rasa -> {labels}")
    return labels


def _download(url: str, dest: Path) -> None:
    """urllib umesto wget-a: radi i na Windows-u, bez spoljnih alata."""
    def hook(blocks: int, bs: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, blocks * bs * 100.0 / total)
            sys.stdout.write(f"\r  {pct:5.1f}%  ({blocks * bs / 1e6:7.1f} / {total / 1e6:.1f} MB)")
            sys.stdout.flush()

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)  # noqa: S310 (fiksni, poznati URL)
    sys.stdout.write("\n")
    tmp.replace(dest)


def download_and_convert(out: Path) -> Path:
    out = Path(out)
    labels = out / "labels.csv"
    if labels.exists():
        print(f"već postoji: {labels}")
        return labels
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / "images.tar"
    if not tar_path.exists():
        print(f"preuzimam ~760 MB: {URL}")
        _download(URL, tar_path)
    print("raspakujem...")
    with tarfile.open(tar_path) as t:
        t.extractall(out, filter="data")
    labels = convert(out / "Images", out)
    shutil.rmtree(out / "Images", ignore_errors=True)
    tar_path.unlink(missing_ok=True)
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    download_and_convert(Path(ap.parse_args().out))
