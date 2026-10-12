"""Inference bez PyTorch-a: PIL + NumPy + ONNX Runtime.

Ovo je kod koji ide u produkciju (FastAPI servis, Dan 5). Namerno NE zavisi od torch-a,
pa je Docker image mali (~200MB umesto ~2GB) i radi brzo na običnom CPU-u.

    python -m dogbreeds.inference artifacts/model.onnx slika.jpg
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_INTERP = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "nearest": Image.NEAREST}


def preprocess(img: Image.Image, data_config: dict) -> np.ndarray:
    """Replicira timm eval transformaciju: resize kraće ivice -> center crop -> normalize.
    Mora biti IDENTIČNO kao u treningu, inače tačnost u produkciji pada (training/serving skew)."""
    _, h_out, w_out = data_config["input_size"]
    crop_pct = data_config.get("crop_pct", 0.875)
    scale = math.floor(h_out / crop_pct)
    img = img.convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = scale, int(scale * h / w)
    else:
        nw, nh = int(scale * w / h), scale
    img = img.resize((nw, nh), _INTERP.get(data_config.get("interpolation", "bicubic"), Image.BICUBIC))
    left = int(round((nw - w_out) / 2.0))
    top = int(round((nh - h_out) / 2.0))
    img = img.crop((left, top, left + w_out, top + h_out))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(data_config["mean"], dtype=np.float32)) / np.array(data_config["std"], dtype=np.float32)
    return x.transpose(2, 0, 1)[None]  # HWC -> NCHW


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Predictor:
    def __init__(self, onnx_path: str | Path, threads: int | None = None):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        meta = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes"]
        self.data_config: dict = meta["data_config"]
        self.temperature: float = meta.get("temperature", 1.0)
        self.model_version: str = meta.get("model_version", onnx_path.stem)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img: Image.Image, k: int = 3) -> dict:
        t0 = time.perf_counter()
        x = preprocess(img, self.data_config)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = softmax(logits / self.temperature)
        top = np.argsort(-probs)[:k]
        return {
            "predictions": [{"breed": self.classes[i], "probability": float(probs[i])} for i in top],
            # signali za monitoring (Dan 6): niska sigurnost / visoka entropija = sumnjiv ulaz
            "max_prob": float(probs[top[0]]),
            "entropy": float(-(probs * np.log(probs + 1e-12)).sum()),
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "model_version": self.model_version,
        }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("upotreba: python -m dogbreeds.inference <model.onnx> <slika.jpg> [slika2.jpg ...]")
    p = Predictor(sys.argv[1])
    for path in sys.argv[2:]:
        r = p.predict(Image.open(path))
        print(f"\n{path}  ({r['latency_ms']:.1f} ms)")
        for i, pr in enumerate(r["predictions"], 1):
            print(f"  {i}. {pr['breed']:<35} {pr['probability']:.1%}")
