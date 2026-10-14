"""Testovi servisa: ugovor ruta + validacija ulaza.

Ne mere tacnost (to radi evaluate.py) nego da servis ne pada na losem ulazu i da
odgovor ima oblik koji frontend ocekuje.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Env se postavlja PRE prvog importa servisa.

    Bez reload-a: `service.app` registruje Prometheus kolektore na nivou modula, pa bi
    ponovno ucitavanje pucalo na DuplicateTimeseries."""
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if not (root / "artifacts" / "model.onnx").exists():
        pytest.skip("nema artifacts/model.onnx")
    os.environ["DOG_DB_PATH"] = str(tmp_path_factory.mktemp("db") / "test.db")

    from service.app import app
    with TestClient(app) as c:
        yield c


def _jpeg(size=(300, 300), color=(120, 100, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["n_classes"] == 120


def test_predict_vraca_tri_odgovora(client):
    r = client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["predictions"]) == 3
    assert d["predictions"] == sorted(d["predictions"], key=lambda p: -p["probability"])
    assert 0.0 <= d["signals"]["top3_mass"] <= 1.0001
    assert d["request_id"] and d["model_version"]


def test_odbija_nepodrzan_tip(client):
    r = client.post("/predict", files={"file": ("a.txt", b"nije slika", "text/plain")})
    assert r.status_code == 415


def test_odbija_pokvarenu_sliku(client):
    r = client.post("/predict", files={"file": ("a.jpg", b"\xff\xd8\xff nije jpeg", "image/jpeg")})
    assert r.status_code == 400


def test_odbija_prazan_fajl(client):
    r = client.post("/predict", files={"file": ("a.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_odbija_preveliki_fajl(client, monkeypatch):
    from service import config as cfg
    monkeypatch.setattr(cfg, "MAX_UPLOAD_BYTES", 1024)
    r = client.post("/predict", files={"file": ("a.jpg", _jpeg((900, 900)), "image/jpeg")})
    assert r.status_code == 413


def test_feedback_tok(client):
    d = client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).json()
    top1 = d["predictions"][0]["breed"]
    r = client.post("/feedback", json={"request_id": d["request_id"], "correct_breed": top1})
    assert r.status_code == 200
    assert r.json()["was_top1"] is True


def test_feedback_odbija_nepoznatu_rasu(client):
    d = client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).json()
    r = client.post("/feedback", json={"request_id": d["request_id"], "correct_breed": "zmaj"})
    assert r.status_code == 422


def test_feedback_odbija_nepoznat_request_id(client):
    r = client.post("/feedback", json={"request_id": "nema-me", "correct_breed": "beagle"})
    assert r.status_code == 404


def test_metrics(client):
    client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "dog_requests_total" in r.text
    assert "dog_predict_latency_seconds" in r.text
