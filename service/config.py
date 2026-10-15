"""Podesavanja servisa -- sve preko env promenljivih, da Docker ne trazi izmenu koda."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


MODEL_PATH = _path("DOG_MODEL_PATH", "artifacts/model.onnx")
NOTDOG_PATH = _path("DOG_NOTDOG_PATH", "artifacts/notdog.onnx")
THRESHOLD_REPORT = _path("DOG_THRESHOLD_REPORT", "reports/threshold.json")
DB_PATH = _path("DOG_DB_PATH", "data/service.db")

# Monitoring drifta (Dan 6). Referenca je zamrznuta raspodela sa val skupa; servis je
# periodicno poredi sa prometom iz poslednjih DRIFT_WINDOW_HOURS sati i izlaze PSI kao
# Prometheus gauge. Interval 0 iskljucuje racunanje.
DRIFT_REFERENCE = _path("DOG_DRIFT_REFERENCE", "reports/drift_reference.json")
DRIFT_INTERVAL_S = float(os.environ.get("DOG_DRIFT_INTERVAL_S", 300))
DRIFT_WINDOW_HOURS = float(os.environ.get("DOG_DRIFT_WINDOW_HOURS", 24))

# Ulaz: ogranicenja brane servis od prevelikih i pokvarenih fajlova
MAX_UPLOAD_BYTES = int(os.environ.get("DOG_MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("DOG_MAX_IMAGE_PIXELS", 50_000_000))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# Gornja granica vremena odgovora. Budzet iz zadatka je p95 < 100 ms; 5 s je zastita
# od patoloskog ulaza, ne radna vrednost.
PREDICT_TIMEOUT_S = float(os.environ.get("DOG_PREDICT_TIMEOUT_S", 5.0))
ORT_THREADS = int(os.environ.get("DOG_ORT_THREADS", 4))

# Kapija "nije pas" se moze iskljuciti ako notdog.onnx nije isporucen
ENABLE_DOG_GATE = os.environ.get("DOG_ENABLE_GATE", "1") not in ("0", "false", "False")
