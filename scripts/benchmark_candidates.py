"""Dan 1: izmeri brzinu kandidata na SVOM laptopu PRE treninga.

Za latenciju nisu potrebne istrenirane težine (arhitektura određuje brzinu), pa ovo radi
bez interneta i bez GPU-a. Rezultat ti govori koji modeli uopšte ulaze u "budžet" brzine.

    python scripts/benchmark_candidates.py --threads 4
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch

CANDIDATES = [
    ("mobilenetv3_large_100", 224),
    ("efficientnet_b0", 224),
    ("mobilenetv4_conv_medium", 256),
    ("efficientnet_b2", 256),
    ("convnext_nano", 224),
    ("convnext_tiny", 224),
    ("vit_small_patch16_224", 224),
]


def bench(sess, x, runs):
    for _ in range(5):
        sess.run(None, {"input": x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(t, 50)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=30)
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for name, size in CANDIDATES:
            m = timm.create_model(name, pretrained=False, num_classes=120).eval()
            x = torch.randn(1, 3, size, size)
            p = Path(d) / f"{name}.onnx"
            torch.onnx.export(m, (x,), str(p), input_names=["input"], output_names=["logits"],
                              opset_version=17, dynamo=False)
            so = ort.SessionOptions()
            so.intra_op_num_threads = a.threads
            sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
            p50, p95 = bench(sess, x.numpy(), a.runs)
            r = {"model": name, "img": size, "params_M": round(sum(q.numel() for q in m.parameters()) / 1e6, 1),
                 "p50_ms": round(p50, 1), "p95_ms": round(p95, 1)}
            rows.append(r)
            print(f"{name:<26} {size}px  {r['params_M']:>5}M  p50 {p50:6.1f} ms  p95 {p95:6.1f} ms")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latency_candidates.json").write_text(json.dumps(rows, indent=2))
    print("-> reports/latency_candidates.json")


if __name__ == "__main__":
    main()
