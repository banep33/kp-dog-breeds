"""Raspakuje Kaggle "Dog Breed Identification" u format koji projekat koristi.

Preuzimanje (treba ~/.kaggle/access_token):
    kaggle competitions download -c dog-breed-identification -p data/kaggle

Pa onda:
    python scripts/prepare_kaggle.py

Iz zip-a vadimo SAMO `labels.csv` i `train/` (10.222 slike sa labelama).
`test/` (10.357 slika) preskačemo: Kaggle ga ne objavljuje sa labelama, pa nam je
za evaluaciju bezvredan — mi pravimo sopstveni test skup iz train dela, grupnim
splitom (vidi dogbreeds.data). `sample_submission.csv` (25 MB) nam ne treba jer
ne šaljemo rešenje na takmičenje.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


def extract(zip_path: Path, out: Path, with_test: bool = False) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    labels = out / "labels.csv"
    wanted_prefix = ("train/",) if not with_test else ("train/", "test/")

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        todo = [n for n in names
                if n == "labels.csv" or (n.startswith(wanted_prefix) and n.endswith(".jpg"))]
        skipped = len(names) - len(todo)
        print(f"u zip-u {len(names)} stavki; vadim {len(todo)}, preskačem {skipped}")
        existing = {p.name for p in (out / "train").glob("*.jpg")} if (out / "train").exists() else set()
        done = 0
        for n in todo:
            if n != "labels.csv" and Path(n).name in existing:
                continue
            z.extract(n, out)
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print(f"raspakovano {done} novih fajlova -> {out}")

    df = pd.read_csv(labels)
    n_img = len(list((out / "train").glob("*.jpg")))
    print(f"labels.csv: {len(df)} redova, {df['breed'].nunique()} rasa | slika na disku: {n_img}")
    missing = set(df["id"].astype(str)) - {p.stem for p in (out / "train").glob("*.jpg")}
    if missing:
        print(f"UPOZORENJE: {len(missing)} id-eva iz labels.csv nema sliku")
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/kaggle/dog-breed-identification.zip")
    ap.add_argument("--out", default="data/kaggle")
    ap.add_argument("--with-test", action="store_true",
                    help="izvadi i test/ (nema labele — korisno samo za demo slike)")
    a = ap.parse_args()
    extract(Path(a.zip), Path(a.out), a.with_test)
