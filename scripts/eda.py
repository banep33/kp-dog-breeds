"""Dan 1 — EDA: šta je u podacima PRE nego što se bilo šta trenira.

Odgovara na 4 pitanja koja menjaju inženjerske odluke:
  1. Koliko slika po rasi?  -> da li treba class weighting / balansiran sampler?
  2. Kakve su rezolucije i odnosi stranica?  -> koji img_size i crop ima smisla?
  3. Ima li grayscale / CMYK / oštećenih fajlova?  -> .convert("RGB") je obavezan?
  4. Koliko je slika manje od ciljne rezolucije? -> da li upscale kvari kvalitet?

    python scripts/eda.py --labels data/raw/labels.csv --images data/raw/train

Izlaz: reports/eda.json (brojevi za dnevnik odluka) + reports/figures/*.png
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # bez GUI-ja: skript radi i u CI-ju / na serveru
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

FIG = Path("reports/figures")


def scan_metadata(ids: list[str], images_dir: Path) -> pd.DataFrame:
    """PIL čita samo JPEG header dok se ne pozove .load() -> skeniranje 20k fajlova traje sekunde."""
    rows = []
    for i in ids:
        p = images_dir / f"{i}.jpg"
        try:
            with Image.open(p) as im:
                rows.append({"id": i, "w": im.width, "h": im.height, "mode": im.mode,
                             "kb": p.stat().st_size / 1024, "error": None})
        except Exception as e:  # noqa: BLE001
            rows.append({"id": i, "w": np.nan, "h": np.nan, "mode": None,
                         "kb": np.nan, "error": f"{type(e).__name__}: {e}"})
    m = pd.DataFrame(rows)
    m["aspect"] = m["w"] / m["h"]
    m["short_edge"] = m[["w", "h"]].min(axis=1)
    m["megapixels"] = m["w"] * m["h"] / 1e6
    return m


def plot_class_distribution(counts: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    counts.plot.bar(ax=ax, width=0.9, color="#4C78A8")
    ax.set_xticks([])
    ax.set_ylabel("broj slika")
    ax.set_xlabel(f"{len(counts)} rasa, sortirano po broju slika")
    ax.set_title(f"Raspodela slika po rasi (min {counts.min()}, max {counts.max()}, "
                 f"medijana {counts.median():.0f})")
    ax.axhline(counts.median(), color="#E45756", ls="--", lw=1, label="medijana")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=120)
    plt.close(fig)


def plot_image_properties(meta: pd.DataFrame, target: int = 224) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.6))
    axs[0].hist(meta["aspect"].dropna(), bins=60, color="#4C78A8")
    axs[0].axvline(1.0, color="#E45756", ls="--", lw=1)
    axs[0].set_title("odnos stranica (w/h)")
    axs[1].hist(meta["short_edge"].dropna(), bins=60, color="#4C78A8")
    axs[1].axvline(target, color="#E45756", ls="--", lw=1, label=f"{target}px")
    axs[1].set_title("kraća ivica (px)")
    axs[1].legend()
    axs[2].hist(meta["kb"].dropna(), bins=60, color="#4C78A8")
    axs[2].set_title("veličina fajla (KB)")
    for a in axs:
        a.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG / "image_properties.png", dpi=120)
    plt.close(fig)


def plot_samples(df: pd.DataFrame, images_dir: Path, n_breeds: int = 5, n_imgs: int = 6,
                 seed: int = 0) -> None:
    """Gledanje slika očima je deo posla: tu se vidi da pas često nije u centru,
    da ima više pasa na slici, ljudi, vodenih žigova."""
    rng = np.random.default_rng(seed)
    breeds = rng.choice(np.sort(df["breed"].unique()), n_breeds, replace=False)
    fig, axs = plt.subplots(n_breeds, n_imgs, figsize=(2.2 * n_imgs, 2.4 * n_breeds))
    for r, b in enumerate(breeds):
        ids = df.loc[df["breed"] == b, "id"].sample(n_imgs, random_state=seed)
        for c, i in enumerate(ids):
            axs[r, c].imshow(Image.open(images_dir / f"{i}.jpg").convert("RGB"))
            axs[r, c].axis("off")
        axs[r, 0].set_title(b, loc="left", fontsize=10, color="#333")
    fig.suptitle("Primeri po rasi", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "samples.png", dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/labels.csv")
    ap.add_argument("--images", default="data/raw/train")
    ap.add_argument("--target-size", type=int, default=224)
    ap.add_argument("--out", default="reports/eda.json")
    a = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    images_dir = Path(a.images)
    df = pd.read_csv(a.labels)
    df["id"] = df["id"].astype(str)
    counts = df["breed"].value_counts()

    print(f"{len(df)} slika, {df['breed'].nunique()} rasa")
    print(f"po rasi: min {counts.min()} ({counts.idxmin()}), max {counts.max()} ({counts.idxmax()}), "
          f"medijana {counts.median():.0f}, neuravnoteženost max/min = {counts.max() / counts.min():.2f}x")

    meta = scan_metadata(df["id"].tolist(), images_dir)
    broken = meta[meta["error"].notna()]
    modes = Counter(meta["mode"].dropna())
    small = int((meta["short_edge"] < a.target_size).sum())

    print(f"\nrezolucija: kraća ivica medijana {meta['short_edge'].median():.0f}px, "
          f"min {meta['short_edge'].min():.0f}px, max {meta['short_edge'].max():.0f}px")
    print(f"odnos stranica: p1 {meta['aspect'].quantile(0.01):.2f}, "
          f"medijana {meta['aspect'].median():.2f}, p99 {meta['aspect'].quantile(0.99):.2f}")
    print(f"mode: {dict(modes)}")
    print(f"slika sa kraćom ivicom < {a.target_size}px: {small} ({100 * small / len(meta):.1f}%)")
    print(f"oštećenih fajlova: {len(broken)}")

    report = {
        "n_images": int(len(df)),
        "n_classes": int(df["breed"].nunique()),
        "per_class": {
            "min": int(counts.min()), "min_breed": str(counts.idxmin()),
            "max": int(counts.max()), "max_breed": str(counts.idxmax()),
            "median": float(counts.median()), "mean": round(float(counts.mean()), 1),
            "imbalance_max_over_min": round(float(counts.max() / counts.min()), 2),
            "std": round(float(counts.std()), 1),
        },
        "resolution": {
            "short_edge_median": float(meta["short_edge"].median()),
            "short_edge_min": float(meta["short_edge"].min()),
            "short_edge_max": float(meta["short_edge"].max()),
            "short_edge_p05": float(meta["short_edge"].quantile(0.05)),
            f"below_{a.target_size}px": small,
            f"below_{a.target_size}px_pct": round(100 * small / len(meta), 2),
            "megapixels_median": round(float(meta["megapixels"].median()), 2),
        },
        "aspect_ratio": {
            "p01": round(float(meta["aspect"].quantile(0.01)), 3),
            "median": round(float(meta["aspect"].median()), 3),
            "p99": round(float(meta["aspect"].quantile(0.99)), 3),
            "portrait_pct": round(100 * float((meta["aspect"] < 1).mean()), 1),
        },
        "file_size_kb": {
            "median": round(float(meta["kb"].median()), 1),
            "p99": round(float(meta["kb"].quantile(0.99)), 1),
            "total_mb": round(float(meta["kb"].sum() / 1024), 1),
        },
        "pil_modes": {str(k): int(v) for k, v in modes.items()},
        "broken_files": broken["id"].tolist(),
        "least_represented_breeds": counts.tail(10).to_dict(),
        "most_represented_breeds": counts.head(10).to_dict(),
    }

    plot_class_distribution(counts)
    plot_image_properties(meta, a.target_size)
    plot_samples(df, images_dir)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    meta.drop(columns=["error"]).to_csv(out.with_name("image_metadata.csv"), index=False)
    print(f"\n-> {out}\n-> {FIG}/ (3 grafika)")


if __name__ == "__main__":
    main()
