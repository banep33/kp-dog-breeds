"""Brza provera da li ceo servis radi -- bez pregledaca.

    .venv\\Scripts\\python.exe scripts\\smoke_service.py [--url http://127.0.0.1:8000] [slika ...]

Bez argumenata salje sve slike iz primeri/. Za svaku ispisuje tri odgovora,
poverenje i latenciju, pa na kraju proveri /feedback i /metrics.
Izlazni kod 0 = sve proslo, 1 = nesto ne radi.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _post_json(url: str, payload: dict, timeout: float = 10.0) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _post_image(url: str, path: Path, timeout: float = 30.0) -> tuple[int, dict]:
    """multipart/form-data rucno -- da skripta ne zavisi ni od cega sem standardne biblioteke."""
    boundary = "----dogbreeds" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("images", nargs="*", type=Path)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    images = args.images or sorted(
        p for p in (ROOT / "primeri").glob("*") if p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    if not images:
        print("Nema slika. Stavi nesto u primeri/ ili navedi putanju kao argument.")
        return 1

    print(f"\nServis: {base}\n" + "-" * 62)

    # --- /health ---
    try:
        code, text = _get(f"{base}/health", timeout=5)
    except Exception as e:                                   # noqa: BLE001
        print(f"[GRESKA] Servis ne odgovara: {type(e).__name__}")
        print("         Pokreni ga sa pokreni.bat pa probaj ponovo.")
        return 1
    if code != 200:
        print(f"[GRESKA] /health vratio {code}")
        return 1
    h = json.loads(text)
    print(f"health      ok  | model {h['model_version']} | {h['n_classes']} rasa "
          f"| kapija {'da' if h['dog_gate'] else 'ne'} "
          f"| prag {h['threshold']['signal']} >= {h['threshold']['value']:.3f}")

    # --- /breeds ---
    code, text = _get(f"{base}/breeds")
    n_breeds = len(json.loads(text)["breeds"]) if code == 200 else 0
    print(f"breeds      {'ok ' if n_breeds else 'PAD'} | {n_breeds} klasa")

    # --- /predict za svaku sliku ---
    print("-" * 62)
    fails = 0
    last_id, last_breed = None, None
    for p in images:
        t0 = time.perf_counter()
        code, d = _post_image(f"{base}/predict", p)
        wall = (time.perf_counter() - t0) * 1000
        if code != 200:
            print(f"\n{p.name}\n  [PAD] HTTP {code}: {d.get('detail')}")
            fails += 1
            continue
        ocena = d.get("message") or "siguran"
        print(f"\n{p.name}   ({d['latency_ms']:.0f} ms u servisu, {wall:.0f} ms ukupno)")
        for i, pr in enumerate(d["predictions"], 1):
            print(f"  {i}. {pr['breed']:<28} {pr['probability'] * 100:5.1f}%")
        ds = d.get("dog_score")
        print(f"     poverenje(top3) {d['signals']['top3_mass'] * 100:.1f}%"
              + (f" | pas {ds * 100:.0f}%" if ds is not None else "")
              + f" | {ocena}")
        last_id, last_breed = d["request_id"], d["predictions"][0]["breed"]

    # --- /feedback (javimo da je prvi odgovor bio tacan) ---
    print("-" * 62)
    if last_id:
        code, d = _post_json(f"{base}/feedback",
                             {"request_id": last_id, "correct_breed": last_breed,
                              "note": "smoke test"})
        ok = code == 200 and d.get("ok")
        print(f"feedback    {'ok ' if ok else 'PAD'} | was_top1={d.get('was_top1')} "
              f"was_top3={d.get('was_top3')}")
        fails += 0 if ok else 1

    # --- odbijanje losih ulaza ---
    code, d = _post_json(f"{base}/feedback",
                         {"request_id": "nepostojeci", "correct_breed": "cocker_spaniel"})
    print(f"validacija  {'ok ' if code == 404 else 'PAD'} | nepoznat request_id -> {code} (ocekivano 404)")
    fails += 0 if code == 404 else 1

    # --- /metrics ---
    code, text = _get(f"{base}/metrics")
    has = code == 200 and "dog_requests_total" in text
    print(f"metrics     {'ok ' if has else 'PAD'} | Prometheus izlaz "
          f"{'sadrzi dog_requests_total' if has else 'NE valja'}")
    fails += 0 if has else 1

    print("-" * 62)
    print("SVE PROSLO\n" if fails == 0 else f"PROBLEMA: {fails}\n")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
