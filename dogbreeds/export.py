"""Izvoz u ONNX (+ opciono INT8 kvantizacija) i merenje brzine na CPU-u.

    python -m dogbreeds.export --checkpoint runs/<run>/best.pt --out artifacts
    python -m dogbreeds.export --checkpoint runs/<run>/best.pt --out artifacts --int8 --config configs/baseline.yaml

Zašto ONNX? Model postaje nezavisan od PyTorch-a, ONNX Runtime je optimizovan za CPU
(fuzija slojeva, grafovske optimizacije), a INT8 kvantizacija obično daje ~2-4x manji
fajl i brže izvršavanje uz mali pad tačnosti -> "dovoljno mali da radi brzo".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import load_checkpoint


def export_onnx(ckpt_path: str, out_dir: str, opset: int = 17) -> Path:
    model, ckpt = load_checkpoint(ckpt_path, torch.device("cpu"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"
    dummy = torch.randn(1, *ckpt["data_config"]["input_size"])
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    meta = {
        "classes": ckpt["classes"],
        "data_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ckpt["data_config"].items()},
        "temperature": ckpt.get("temperature", 1.0),
        "model_name": ckpt["model_name"],
        "model_version": Path(ckpt_path).parent.name,
        "val_metrics": ckpt.get("metrics"),
    }
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # provera: ONNX mora davati iste izlaze kao PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"input": dummy.numpy()})[0]
    print(f"[export] {onnx_path} | max |torch-onnx| = {np.abs(ref - got).max():.2e}")
    return onnx_path


def quantize_int8(onnx_path: Path, calib_images: list[Path], data_config: dict) -> Path:
    """Statička INT8 kvantizacija (za CNN bolja od dinamičke). Kalibracija na ~100-300 slika iz val skupa."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from PIL import Image

    from .inference import preprocess

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_images)

        def get_next(self):
            p = next(self.it, None)
            return None if p is None else {"input": preprocess(Image.open(p), data_config)}

    pre = onnx_path.with_name("model.pre.onnx")
    quant_pre_process(str(onnx_path), str(pre))
    q = onnx_path.with_name("model.int8.onnx")
    quantize_static(str(pre), str(q), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink(missing_ok=True)
    q.with_suffix(".json").write_text(onnx_path.with_suffix(".json").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[int8] {q} ({q.stat().st_size / 1e6:.1f} MB vs {onnx_path.stat().st_size / 1e6:.1f} MB)")
    return q


def verify_against(candidate: Path, reference: Path, images: list[Path],
                   data_config: dict, max_drop_pp: float = 2.0) -> dict:
    """Slaze li se kandidat sa referentnim modelom na istim slikama.

    Postoji zbog Bug-a 4: INT8 kvantizacija je pravila model koji se UREDNO ucitava,
    radi brzo i daje gluposti. Artefakt koji se ne poklapa sa referencom ne sme tiho
    da zavrsi u `artifacts/` -- bolje da izvoz padne ovde nego u produkciji."""
    import onnxruntime as ort
    from PIL import Image

    from .inference import preprocess

    def logits_for(path: Path) -> np.ndarray:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return np.stack([sess.run(None, {name: preprocess(Image.open(p), data_config)})[0][0]
                         for p in images])

    ref, got = logits_for(reference), logits_for(candidate)
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    # SQNR izlaza u dB: koliko je signala u odnosu na sum kvantizacije (vise = bolje)
    noise = ((ref - got) ** 2).mean()
    sqnr = float(10 * np.log10((ref ** 2).mean() / noise)) if noise > 0 else float("inf")
    ok = agree >= 1.0 - max_drop_pp / 100.0
    print(f"[verify] {candidate.name}: slaganje top-1 sa {reference.name} {agree:.1%} "
          f"| SQNR {sqnr:.1f} dB | {'OK' if ok else 'NE VALJA'}")
    return {"candidate": candidate.name, "reference": reference.name,
            "top1_agreement": agree, "sqnr_db": round(sqnr, 2), "ok": ok, "n": len(images)}


def benchmark(onnx_path: Path, threads: int = 4, runs: int = 50) -> dict:
    import onnxruntime as ort

    meta = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, *meta["data_config"]["input_size"]).astype(np.float32)
    for _ in range(5):  # zagrevanje
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    r = {"model": onnx_path.name, "threads": threads, "p50_ms": float(np.percentile(times, 50)),
         "p95_ms": float(np.percentile(times, 95)), "size_mb": onnx_path.stat().st_size / 1e6}
    print(f"[bench] {r['model']:<18} p50 {r['p50_ms']:.1f} ms | p95 {r['p95_ms']:.1f} ms | {r['size_mb']:.1f} MB | {threads} niti")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--config", default="configs/baseline.yaml", help="potreban za --int8 (putanje do val slika)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    onnx_path = export_onnx(a.checkpoint, a.out)
    results = [benchmark(onnx_path, a.threads)]
    if a.int8:
        import pandas as pd

        from .config import load_config
        cfg = load_config(a.config, a.set)
        val = pd.read_csv(Path(cfg["data"]["splits_dir"]) / "val.csv").sample(frac=1, random_state=0).head(a.calib)
        imgs = [Path(cfg["data"]["images_dir"]) / f"{i}.jpg" for i in val["id"]]
        meta = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
        q = quantize_int8(onnx_path, imgs, meta["data_config"])
        results.append(benchmark(q, a.threads))

        # Bug 4: kvantizovan model se uredno ucitava i brz je, a daje nasumicne odgovore.
        # Zato provera pre nego sto artefakt ostane na disku.
        check = verify_against(q, onnx_path, imgs[:50], meta["data_config"])
        results.append(check)
        if not check["ok"]:
            q.unlink(missing_ok=True)
            q.with_suffix(".json").unlink(missing_ok=True)
            print(f"[int8] ODBACENO: slaganje {check['top1_agreement']:.1%} je premalo. "
                  f"Isporucuje se FP32. Uzrok i merenja: reports/int8_diagnosis.json")
    Path(a.out, "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
