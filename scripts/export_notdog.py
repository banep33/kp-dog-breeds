"""Izvoz ImageNet kapije "nije pas" u ONNX + kalibracija praga na val skupu.

    python scripts/export_notdog.py --out artifacts --splits data/splits --images data/raw/train

Uzima ORIGINALNI, nefine-tunovani `efficientnet_b0` sa ImageNet-1k glavom od 1000 klasa.
Prag se bira jednostrano: tako da bar `--keep` udeo POZNATIH pasa prodje kapiju. Lazno
odbijanje psa je jedina greska koju ovde mozemo da izmerimo -- za drugu stranu (koliko
ne-pasa prodje) treba negativan skup, koji nemamo. To je svesno ogranicenje i tako je
i prijavljeno.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dogbreeds.inference import preprocess, softmax  # noqa: E402
from dogbreeds.notdog import DOG_CLASS_RANGE  # noqa: E402


def export(model_name: str, out_dir: Path, opset: int = 17) -> tuple[Path, dict]:
    import timm

    model = timm.create_model(model_name, pretrained=True)   # ImageNet glava, 1000 klasa
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "notdog.onnx"
    dummy = torch.randn(1, *cfg["input_size"])
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"input": dummy.numpy()})[0]
    print(f"[notdog] {onnx_path} | max |torch-onnx| = {np.abs(ref - got).max():.2e}")
    return onnx_path, cfg


def dog_scores(onnx_path: Path, cfg: dict, paths: list[Path]) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    lo, hi = DOG_CLASS_RANGE
    idx = np.arange(lo, hi + 1)
    out = np.empty(len(paths), dtype=np.float32)
    for i, p in enumerate(paths):
        x = preprocess(Image.open(p), cfg)
        out[i] = softmax(sess.run(None, {name: x})[0][0])[idx].sum()
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0.ra4_e3600_r224_in1k")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--images", default="data/raw/train")
    ap.add_argument("--split", default="val")
    ap.add_argument("--keep", type=float, default=0.99,
                    help="udeo poznatih pasa koji mora da prodje kapiju")
    ap.add_argument("--report", default="reports/notdog.json")
    a = ap.parse_args()

    out_dir = Path(a.out)
    onnx_path, cfg = export(a.model, out_dir)

    df = pd.read_csv(Path(a.splits) / f"{a.split}.csv")
    paths = [Path(a.images) / f"{i}.jpg" for i in df["id"].astype(str)]
    print(f"[notdog] kalibracija na {len(paths)} slika iz {a.split} skupa")
    s = dog_scores(onnx_path, cfg, paths)

    tau = float(np.quantile(s, 1.0 - a.keep))
    res = {
        "model": a.model,
        "dog_class_range": list(DOG_CLASS_RANGE),
        "threshold": round(tau, 4),
        "rule": f"kvantil {1 - a.keep:.2%} dog_score-a na {a.split} skupu poznatih pasa",
        "n_calibration": len(paths),
        "dog_score_quantiles": {q: round(float(np.quantile(s, q)), 4)
                                for q in (0.001, 0.005, 0.01, 0.05, 0.25, 0.5)},
        "pass_rate_at_threshold": float((s >= tau).mean()),
        "napomena": ("Jednostrana kalibracija. Koliko NE-pasa prodje kapiju nije izmereno "
                     "jer nemamo negativan skup -- za to treba 100-200 slika bez psa."),
    }

    meta = {"data_config": cfg, "threshold": res["threshold"],
            "dog_class_range": list(DOG_CLASS_RANGE), "model_name": a.model}
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rp = Path(a.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[notdog] prag {res['threshold']:.4f} | prolazi {res['pass_rate_at_threshold']:.2%} poznatih pasa")
    print(f"[notdog] kvantili dog_score-a: {res['dog_score_quantiles']}")
    print(f"-> {onnx_path}\n-> {rp}")


if __name__ == "__main__":
    main()
