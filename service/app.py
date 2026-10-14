"""FastAPI servis za prepoznavanje rase psa.

    uvicorn service.app:app --host 0.0.0.0 --port 8000

Rute:
    GET  /          mini frontend
    POST /predict   slika -> tri najverovatnije rase + poverenje
    POST /feedback  korisnik javlja tacnu rasu
    GET  /health    spremnost (za Docker healthcheck i orkestrator)
    GET  /metrics   Prometheus (Dan 6)

Namerno NE zavisi od torch-a -- ucitava samo ONNX. Izmerena velicina image-a: 605 MB.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from dogbreeds.inference import Predictor
from dogbreeds.notdog import DogGate

from . import config
from .store import Store, top3_to_json

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

# --- metrike (Dan 6 ih skuplja; ovde se samo izlazu)
REQUESTS = Counter("dog_requests_total", "Zahteva ka /predict", ["outcome"])
LATENCY = Histogram("dog_predict_latency_seconds", "Trajanje /predict",
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
CONFIDENCE = Histogram("dog_top3_mass", "Zbir verovatnoca prva tri odgovora",
                       buckets=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
DOG_SCORE = Histogram("dog_gate_score", "Izlaz kapije da li je pas",
                      buckets=(0.05, 0.25, 0.5, 0.675, 0.8, 0.9, 1.0))
LOW_CONF = Counter("dog_low_confidence_total", "Odgovora uz upozorenje nisam siguran")
NOT_DOG = Counter("dog_not_a_dog_total", "Ulaza odbijenih kao nije pas")
FEEDBACK = Counter("dog_feedback_total", "Primljenih povratnih informacija", ["kind"])
READY = Gauge("dog_service_ready", "1 ako su modeli ucitani")

STATE: dict = {}


def _load_threshold() -> tuple[str, float]:
    """Prag iz reports/threshold.json; ako ga nema, radimo bez upozorenja (prag 0)."""
    try:
        d = json.loads(config.THRESHOLD_REPORT.read_text(encoding="utf-8"))
        return d.get("signal", "top3_mass"), float(d.get("threshold", 0.0))
    except (OSError, ValueError, KeyError):
        print(f"[servis] nema {config.THRESHOLD_REPORT} -- upozorenje o sigurnosti iskljuceno")
        return "top3_mass", 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["predictor"] = Predictor(config.MODEL_PATH, threads=config.ORT_THREADS)
    STATE["gate"] = None
    if config.ENABLE_DOG_GATE and config.NOTDOG_PATH.exists():
        STATE["gate"] = DogGate(config.NOTDOG_PATH, threads=config.ORT_THREADS)
    elif config.ENABLE_DOG_GATE:
        print(f"[servis] nema {config.NOTDOG_PATH} -- kapija nije pas iskljucena")
    STATE["signal"], STATE["threshold"] = _load_threshold()
    STATE["store"] = Store(config.DB_PATH)
    STATE["started"] = time.time()
    READY.set(1)
    print(f"[servis] model {STATE['predictor'].model_version} | "
          f"{len(STATE['predictor'].classes)} rasa | "
          f"prag {STATE['signal']} >= {STATE['threshold']:.3f} | "
          f"kapija {'DA' if STATE['gate'] else 'ne'}")
    yield
    READY.set(0)
    STATE["store"].close()


app = FastAPI(title="Prepoznavanje rase psa", version="1.0", lifespan=lifespan)


# ---------------------------------------------------------------- modeli odgovora

class Prediction(BaseModel):
    breed: str
    probability: float


class PredictResponse(BaseModel):
    request_id: str
    predictions: list[Prediction]
    confident: bool
    message: str | None = None
    is_dog: bool
    dog_score: float | None = None
    signals: dict
    model_version: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    correct_breed: str = Field(..., min_length=1)
    note: str | None = None


# ---------------------------------------------------------------- pomocne

async def _read_image(file: UploadFile) -> tuple[Image.Image, int]:
    """Validacija ulaza pre bilo kakvog racunanja: tip, velicina, dekodljivost."""
    if file.content_type not in config.ALLOWED_TYPES:
        raise HTTPException(415, f"nepodrzan tip '{file.content_type}'; "
                                 f"dozvoljeno: {sorted(config.ALLOWED_TYPES)}")
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"fajl veci od {config.MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    if not data:
        raise HTTPException(400, "prazan fajl")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()        # dekodiramo odmah: pokvaren JPEG puca ovde, ne u modelu
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise HTTPException(400, f"slika se ne moze procitati: {type(e).__name__}") from e
    return img, len(data)


def _infer(img: Image.Image) -> dict:
    """Ceo sinhroni deo na jednom mestu, da ga lako pustimo u threadpool."""
    gate = STATE["gate"]
    dog_score = gate.score(img) if gate else None
    is_dog = True if dog_score is None else dog_score >= gate.threshold
    out = STATE["predictor"].predict(img, k=3)
    out["top3_mass"] = sum(p["probability"] for p in out["predictions"])
    out["dog_score"] = dog_score
    out["is_dog"] = is_dog
    return out


# ---------------------------------------------------------------- rute

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    t0 = time.perf_counter()
    img, nbytes = await _read_image(file)
    w, h = img.size
    try:
        r = await asyncio.wait_for(run_in_threadpool(_infer, img),
                                   timeout=config.PREDICT_TIMEOUT_S)
    except asyncio.TimeoutError:
        REQUESTS.labels(outcome="timeout").inc()
        raise HTTPException(504, f"predikcija duza od {config.PREDICT_TIMEOUT_S:.0f} s")

    signal_value = r["top3_mass"] if STATE["signal"] == "top3_mass" else r["max_prob"]
    confident = signal_value >= STATE["threshold"]
    total_ms = (time.perf_counter() - t0) * 1000

    if not r["is_dog"]:
        message = ("Na slici verovatno nije pas. Evo ipak tri najblize rase, "
                   "ali im nemoj verovati.")
        NOT_DOG.inc()
    elif not confident:
        message = "Nisam siguran. Evo tri moguce rase, poredjane po verovatnoci."
        LOW_CONF.inc()
    else:
        message = None

    request_id = uuid.uuid4().hex
    STATE["store"].log_prediction({
        "request_id": request_id, "ts": time.time(),
        "model_version": r["model_version"],
        "top1": r["predictions"][0]["breed"],
        "top3_json": top3_to_json(r["predictions"]),
        "max_prob": r["max_prob"], "top3_mass": r["top3_mass"], "entropy": r["entropy"],
        "dog_score": r["dog_score"], "is_dog": int(r["is_dog"]),
        "confident": int(confident), "latency_ms": total_ms,
        "image_bytes": nbytes, "image_w": w, "image_h": h,
    })

    REQUESTS.labels(outcome="ok").inc()
    LATENCY.observe(total_ms / 1000)
    CONFIDENCE.observe(r["top3_mass"])
    if r["dog_score"] is not None:
        DOG_SCORE.observe(r["dog_score"])

    return PredictResponse(
        request_id=request_id,
        predictions=[Prediction(**p) for p in r["predictions"]],
        confident=confident, message=message,
        is_dog=r["is_dog"], dog_score=r["dog_score"],
        signals={"max_prob": r["max_prob"], "top3_mass": r["top3_mass"],
                 "entropy": r["entropy"], "threshold": STATE["threshold"],
                 "signal": STATE["signal"]},
        model_version=r["model_version"], latency_ms=round(total_ms, 2),
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    """Korisnik javlja tacnu rasu. Osnova za petlju unapredjivanja (Dan 6)."""
    classes = STATE["predictor"].classes
    if req.correct_breed not in classes:
        raise HTTPException(422, f"nepoznata rasa '{req.correct_breed}'")
    pred = STATE["store"].get_prediction(req.request_id)
    if pred is None:
        raise HTTPException(404, f"nema predikcije sa request_id '{req.request_id}'")

    top3 = [p["breed"] for p in json.loads(pred["top3_json"])]
    was_top1 = int(req.correct_breed == pred["top1"])
    was_top3 = int(req.correct_breed in top3)
    STATE["store"].log_feedback({
        "request_id": req.request_id, "ts": time.time(),
        "correct_breed": req.correct_breed, "was_top1": was_top1,
        "was_top3": was_top3, "note": req.note,
    })
    FEEDBACK.labels(kind="top1" if was_top1 else ("top3" if was_top3 else "miss")).inc()
    return {"ok": True, "was_top1": bool(was_top1), "was_top3": bool(was_top3)}


@app.get("/health")
async def health() -> dict:
    p = STATE.get("predictor")
    return {
        "status": "ok" if p else "loading",
        "model_version": p.model_version if p else None,
        "n_classes": len(p.classes) if p else 0,
        "dog_gate": STATE.get("gate") is not None,
        "threshold": {"signal": STATE.get("signal"), "value": STATE.get("threshold")},
        "uptime_s": round(time.time() - STATE["started"], 1) if STATE.get("started") else 0,
        "counts": STATE["store"].counts() if STATE.get("store") else {},
    }


@app.get("/breeds")
async def breeds() -> dict:
    return {"breeds": STATE["predictor"].classes}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(config.ROOT / "service" / "static" / "index.html")


app.mount("/static", StaticFiles(directory=config.ROOT / "service" / "static"), name="static")
