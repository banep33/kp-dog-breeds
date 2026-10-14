"""Kapija "da li je ovo uopste pas" -- bez PyTorch-a, kao i `inference.py`.

Zasto postoji: realni korisnici salju i macke, ljude, pejzaze i slike bez psa. Nas
klasifikator zna SAMO 120 rasa pasa, pa ce za macku samouvereno reci neku rasu. Prag
sigurnosti (`threshold.py`) to ne hvata -- macka moze dati visoko poverenje u pogresnu
rasu, jer model nema klasu "nije pas".

Kako radi: ImageNet-1k klasifikator (originalni, ne fine-tunovan) i zbir verovatnoca
preko ImageNet klasa koje SU psi.

    python -m dogbreeds.notdog artifacts/notdog.onnx slika.jpg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .inference import preprocess, softmax

# ImageNet-1k: 151-268 su rase pasa (Chihuahua ... mexican_hairless).
# Opseg je prosiren do 275 jer NASIH 120 klasa ukljucuje dingo (273), dhole (274) i
# african_hunting_dog (275). Da smo stali na 268, kapija bi odbijala tri sopstvene rase.
DOG_CLASS_RANGE = (151, 275)


class DogGate:
    """Vraca `dog_score` u [0, 1] -- koliko ImageNet verovatnoce pada na pse."""

    def __init__(self, onnx_path: str | Path, threads: int | None = None):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        meta = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.data_config: dict = meta["data_config"]
        self.threshold: float = meta.get("threshold", 0.5)
        lo, hi = meta.get("dog_class_range", DOG_CLASS_RANGE)
        self.dog_idx = np.arange(lo, hi + 1)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def score(self, img: Image.Image) -> float:
        x = preprocess(img, self.data_config)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        return float(softmax(logits)[self.dog_idx].sum())

    def check(self, img: Image.Image) -> dict:
        s = self.score(img)
        return {"dog_score": s, "is_dog": bool(s >= self.threshold), "threshold": self.threshold}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("upotreba: python -m dogbreeds.notdog <notdog.onnx> <slika.jpg> [...]")
    g = DogGate(sys.argv[1])
    for path in sys.argv[2:]:
        r = g.check(Image.open(path))
        print(f"{path}  dog_score {r['dog_score']:.4f}  -> {'pas' if r['is_dog'] else 'NIJE PAS'}")
