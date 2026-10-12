"""Tačnost ONNX modela (FP32 ili INT8) na test skupu, kroz ISTI kod koji ide u produkciju.

    python scripts/eval_onnx.py artifacts/model.int8.onnx --splits data/splits --images data/raw/train

Ako se rezultat razlikuje od evaluate.py za više od ~0.5%, imaš training/serving skew
(drugačiji preprocessing) ili je kvantizacija previše oštetila model.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dogbreeds.inference import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--images", default="data/raw/train")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    p = Predictor(a.model)
    df = pd.read_csv(Path(a.splits) / f"{a.split}.csv")
    top1 = top3 = 0
    lat = []
    for _, row in df.iterrows():
        r = p.predict(Image.open(Path(a.images) / f"{row['id']}.jpg"), k=3)
        names = [x["breed"] for x in r["predictions"]]
        top1 += names[0] == row["breed"]
        top3 += row["breed"] in names
        lat.append(r["latency_ms"])
    n = len(df)
    res = {"model": a.model, "n": n, "top1": top1 / n, "top3": top3 / n,
           "mean_latency_ms_incl_preprocess": sum(lat) / n}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
