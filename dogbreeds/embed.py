"""Ekstrakcija embeddinga: slika -> vektor fiksne dužine, pomoću pretreniranog backbone-a
bez klasifikacione glave (`num_classes=0`).

Ovaj modul je "radni konj" celog projekta i koristi se na 4 mesta:
  1. Dan 1 — deduplikacija (slične slike = slični vektori)
  2. Dan 2 — linear probe baseline (logistička regresija nad zamrznutim embeddinzima)
  3. Dan 3 — cleanlab (traženje pogrešnih labela nad out-of-sample verovatnoćama)
  4. Dan 6 — embedding drift u produkciji (da li korisničke slike liče na trening podatke)

Zašto DINOv3 kao podrazumevani backbone? Treniran je self-supervised, **bez ImageNet labela**,
pa reprezentacije nisu "videle" tačne odgovore za ovih 120 rasa (vidi docs/PLAN.md, zamka #1).

    python -m dogbreeds.embed --labels data/raw/labels.csv --images data/raw/train \
        --model convnext_tiny.dinov3_lvd1689m --out data/emb/dinov3_convnext_tiny
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageOnlyDataset(Dataset):
    """Dataset bez labela — za ekstrakciju embeddinga nam trebaju samo slike.
    Oštećene fajlove ne prekidamo, vraćamo crnu sliku i beležimo indeks."""

    def __init__(self, ids: list[str], images_dir: str | Path, transform, ext: str = ".jpg"):
        self.ids = list(ids)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.ext = ext
        self.failed: list[int] = []

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        p = self.images_dir / f"{self.ids[i]}{self.ext}"
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            self.failed.append(i)
            img = Image.new("RGB", (256, 256))
        return self.transform(img), i


def build_backbone(model_name: str, img_size: int | None = None, device: str = "cpu"):
    """Vraća (model bez glave, data_config, dimenzija embeddinga)."""
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    dc = dict(resolve_data_config({}, model=model))
    if img_size:
        dc["input_size"] = (3, img_size, img_size)
    transform = create_transform(**dc, is_training=False)
    return model, dc, transform


@torch.no_grad()
def extract(
    ids: list[str],
    images_dir: str | Path,
    model_name: str,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str | None = None,
    img_size: int | None = None,
    log_every: int = 20,
) -> tuple[np.ndarray, dict]:
    """Vraća (embeddings [N, D] L2-normalizovani float32, meta dict).

    L2 normalizacija je namerna: posle nje je skalarni proizvod = kosinusna sličnost,
    pa deduplikacija postaje jedno matrično množenje.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, dc, transform = build_backbone(model_name, img_size, device)
    ds = ImageOnlyDataset(ids, images_dir, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=device == "cuda")

    # Dimenziju NE pogađamo iz model.num_features: kod MobileNetV3/V4 postoji `conv_head`
    # posle poolinga (960 -> 1280), pa je stvarni izlaz veći od num_features. Alociramo
    # na prvom batch-u -> radi za svaku arhitekturu bez specijalnih slučajeva.
    out: np.ndarray | None = None
    t0, n_batches = time.perf_counter(), len(dl)
    for b, (x, idx) in enumerate(dl):
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            z = model(x.to(device, non_blocking=True))
        z = torch.nn.functional.normalize(z.float(), dim=1)
        if out is None:
            out = np.empty((len(ids), z.shape[1]), dtype=np.float32)
        out[idx.numpy()] = z.cpu().numpy()
        if log_every and (b + 1) % log_every == 0:
            el = time.perf_counter() - t0
            done = (b + 1) * batch_size
            print(f"  [{b + 1}/{n_batches}] {done} slika | {done / el:.0f} img/s | "
                  f"preostalo ~{el / (b + 1) * (n_batches - b - 1) / 60:.1f} min", flush=True)

    if out is None:
        raise RuntimeError("prazan dataset — nema ni jedne slike")
    meta = {
        "model": model_name,
        "dim": int(out.shape[1]),
        "n": len(ids),
        "data_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dc.items()},
        "device": device,
        "seconds": round(time.perf_counter() - t0, 1),
        "img_per_sec": round(len(ids) / max(time.perf_counter() - t0, 1e-9), 1),
        "failed_idx": sorted(set(ds.failed)),
    }
    return out, meta


def save(out_prefix: str | Path, emb: np.ndarray, ids: list[str], meta: dict) -> Path:
    p = Path(out_prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p.with_suffix(".npy"), emb)
    p.with_suffix(".json").write_text(
        json.dumps({**meta, "ids": list(ids)}, indent=2), encoding="utf-8")
    print(f"[embed] {p.with_suffix('.npy')}  {emb.shape}  ({meta['img_per_sec']} img/s)")
    return p.with_suffix(".npy")


def load(prefix: str | Path) -> tuple[np.ndarray, list[str], dict]:
    p = Path(prefix)
    npy = p if p.suffix == ".npy" else p.with_suffix(".npy")
    meta = json.loads(npy.with_suffix(".json").read_text(encoding="utf-8"))
    ids = meta.pop("ids")
    return np.load(npy), ids, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Izvuci embeddinge za sve slike iz labels.csv")
    ap.add_argument("--labels", default="data/raw/labels.csv")
    ap.add_argument("--images", default="data/raw/train")
    ap.add_argument("--model", default="convnext_tiny.dinov3_lvd1689m")
    ap.add_argument("--out", default=None, help="prefiks bez ekstenzije; default data/emb/<model>")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="samo prvih N slika (za brzi test)")
    ap.add_argument("--log-every", type=int, default=20, help="ispis napretka svakih N batch-eva (0 = bez)")
    a = ap.parse_args()

    df = pd.read_csv(a.labels)
    ids = df["id"].astype(str).tolist()[: a.limit]
    out = a.out or f"data/emb/{a.model.split('.')[0]}"
    emb, meta = extract(ids, a.images, a.model, a.batch_size, a.num_workers,
                        img_size=a.img_size, log_every=a.log_every)
    meta["labels_csv"] = a.labels
    save(out, emb, ids, meta)
    if meta["failed_idx"]:
        print(f"UPOZORENJE: {len(meta['failed_idx'])} slika nije moglo da se pročita")


if __name__ == "__main__":  # obavezno na Windows-u (DataLoader workers)
    main()
