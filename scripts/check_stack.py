"""Provera celog monitoring lanca: servis -> Prometheus -> Grafana.

    .venv\\Scripts\\python.exe scripts\\check_stack.py [--promet]

Proverava redom ono sto moze da se pokvari, od najnizeg sloja nagore. `--promet` pre
provere posalje slike iz primeri/, da panel ne bude prazan i da odnosi imaju sta da dele.

Izlazni kod 0 = sve u redu, 1 = nesto ne valja.

Zasto postoji: "da li je sve normalno" je pitanje na koje se odgovara proverom, ne
gledanjem da li su kontejneri zeleni. Kontejner moze biti `healthy` a Prometheus ga ne
vidi; Prometheus moze da ga vidi a Grafana da gleda pogresan izvor podataka.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dogbreeds  # noqa: F401,E402  -- forsira UTF-8 na stdout (Windows konzola je cp1252)

SERVIS = "http://127.0.0.1:8001"
PROM = "http://127.0.0.1:9090"
GRAFANA = "http://127.0.0.1:3000"

OK, PAD, UPOZ = "ok ", "PAD", " ! "
_greske: list[str] = []


def red(status: str, naslov: str, detalj: str = "") -> None:
    print(f"  [{status}] {naslov:<34} {detalj}")
    if status == PAD:
        _greske.append(naslov)


def dohvati(url: str, timeout: float = 6.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _starost_s(iso: str) -> float | None:
    """Koliko je sekundi proslo od ISO vremena koje Prometheus vraca.

    Prometheus salje npr. '2026-10-15T14:12:33.123456789Z' -- devet decimala i 'Z',
    sto `datetime.fromisoformat` ne prima. Zato se decimale skrate na sest, a 'Z'
    prevede u '+00:00'. Racunanje preko `time.mktime` je ovde pogresno jer tretira
    UTC string kao lokalno vreme.
    """
    try:
        s = re.sub(r"\.(\d{6})\d*", r".\1", iso.strip()).replace("Z", "+00:00")
        return (datetime.now(timezone.utc) - datetime.fromisoformat(s)).total_seconds()
    except (ValueError, AttributeError):
        return None


def promql(izraz: str):
    import urllib.parse
    d = dohvati(f"{PROM}/api/v1/query?query={urllib.parse.quote(izraz)}")
    if not d or d.get("status") != "success":
        return None
    return d["data"]["result"]


# ---------------------------------------------------------------- 1. Docker

def proveri_docker() -> None:
    print("\n1. Docker")
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=25, check=True)
        red(OK, "Docker Desktop radi")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        red(PAD, "Docker Desktop", "nije pokrenut -- pokreni ga pa probaj ponovo")
        return

    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "--format", "{{.Name}}\t{{.State}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=30, cwd=ROOT).stdout
    except (subprocess.SubprocessError, OSError):
        red(PAD, "docker compose ps", "ne moze da se pokrene")
        return

    nadjeni = {}
    for linija in out.strip().splitlines():
        delovi = linija.split("\t")
        if len(delovi) >= 3:
            nadjeni[delovi[0]] = (delovi[1], delovi[2])

    for ime in ("kp-dog-service", "kp-dog-prometheus", "kp-dog-grafana"):
        if ime not in nadjeni:
            red(PAD, ime, "kontejner ne postoji -- `docker compose up -d`")
        elif nadjeni[ime][0] != "running":
            red(PAD, ime, nadjeni[ime][1])
        else:
            red(OK, ime, nadjeni[ime][1])


# ---------------------------------------------------------------- 2. servis

def proveri_servis() -> None:
    print("\n2. Servis")
    h = dohvati(f"{SERVIS}/health")
    if not h:
        red(PAD, "GET /health", f"{SERVIS} ne odgovara")
        return
    red(OK if h.get("status") == "ok" else PAD, "GET /health", h.get("status"))
    red(OK, "model", f"{h['model_version']} | {h['n_classes']} rasa")
    red(OK if h.get("dog_gate") else UPOZ, "kapija 'nije pas'",
        "ukljucena" if h.get("dog_gate") else "iskljucena (nema notdog.onnx)")
    t = h.get("threshold", {})
    red(OK, "prag sigurnosti", f"{t.get('signal')} >= {t.get('value')}")

    d = h.get("drift") or {}
    if not d.get("enabled"):
        red(UPOZ, "detekcija drifta", "iskljucena -- nema reports/drift_reference.json")
    else:
        st = d.get("status")
        red(OK if st in ("stabilno", "nedovoljno podataka", None) else UPOZ,
            "detekcija drifta",
            f"{st or 'jos nije racunat'} | PSI {d.get('worst_psi')} | n={d.get('n_samples')}")

    c = h.get("counts", {})
    red(OK, "dnevnik predikcija",
        f"{c.get('predictions', 0)} predikcija, {c.get('feedback', 0)} ispravki")

    m = dohvati(f"{SERVIS}/breeds")
    red(OK if m and len(m.get("breeds", [])) == 120 else PAD, "GET /breeds",
        f"{len(m.get('breeds', [])) if m else 0} rasa")


# ---------------------------------------------------------------- 3. Prometheus

def proveri_prometheus() -> None:
    print("\n3. Prometheus")
    d = dohvati(f"{PROM}/api/v1/targets?state=active")
    if not d:
        red(PAD, "Prometheus", f"{PROM} ne odgovara")
        return

    for t in d["data"]["activeTargets"]:
        job = t["labels"].get("job", "?")
        if t["health"] != "up":
            red(PAD, f"meta '{job}'", t.get("lastError", "down"))
            continue
        # Zastareo uzorak je tih kvar: meta je "up", a brojke su od pre pola sata.
        # Interval skupljanja je 10 s, pa je sve preko 60 s sumnjivo.
        star = _starost_s(t.get("lastScrape", ""))
        if star is None:
            red(OK, f"meta '{job}'", "up")
        else:
            red(UPOZ if star > 60 else OK, f"meta '{job}'",
                f"up, poslednje skupljanje pre {star:.0f} s")

    r = dohvati(f"{PROM}/api/v1/rules")
    if not r:
        red(PAD, "pravila alarma", "ne mogu da se procitaju")
    else:
        pravila = [x for g in r["data"]["groups"] for x in g["rules"]]
        lose = [x["name"] for x in pravila if x.get("health") != "ok"]
        pali = [x["name"] for x in pravila if x.get("state") not in (None, "inactive")]
        red(PAD if lose else OK, "pravila alarma",
            f"{len(pravila)} ucitano" + (f", GRESKA u {lose}" if lose else ", sve ispravno"))
        if pali:
            red(UPOZ, "upaljeni alarmi", ", ".join(pali))
        else:
            red(OK, "upaljeni alarmi", "nijedan")

    # Serije koje posle restarta nisu imale nulu su pravile Bug 8 -- proveravamo da postoje.
    for izraz, naziv in (('dog_requests_total{outcome="ok"}', "brojac zahteva"),
                         ("dog_low_confidence_total", "brojac upozorenja"),
                         ("dog_service_ready", "spremnost servisa")):
        res = promql(izraz)
        red(OK if res else PAD, f"serija: {naziv}",
            f"= {res[0]['value'][1]}" if res else "ne postoji u Prometheus-u")


# ---------------------------------------------------------------- 4. odnosi

def proveri_odnose() -> None:
    """Bug 8: odnos je umeo da predje 100% kad imenilac nije bio izmeren."""
    print("\n4. Zdravlje odnosa (cuva od Buga 8)")
    izraz = ('sum(increase(dog_low_confidence_total[1h]))'
             ' / (sum(increase(dog_requests_total{outcome="ok"}[1h])) > 0)')
    res = promql(izraz)
    if not res:
        red(OK, "udeo 'nisam siguran'", "prazno -- nema prometa u poslednjih 1 h (ispravno)")
        return
    v = float(res[0]["value"][1])
    if 0.0 <= v <= 1.0:
        red(OK, "udeo 'nisam siguran'", f"{v:.1%} (u opsegu 0-100%)")
    else:
        red(PAD, "udeo 'nisam siguran'", f"{v:.1%} -- IZVAN 0-100%, odnos je pokvaren")


# ---------------------------------------------------------------- 5. Grafana

def proveri_grafanu() -> None:
    print("\n5. Grafana")
    h = dohvati(f"{GRAFANA}/api/health")
    if not h:
        red(PAD, "Grafana", f"{GRAFANA} ne odgovara")
        return
    red(OK, "Grafana", f"v{h.get('version')} | baza {h.get('database')}")

    ds = dohvati(f"{GRAFANA}/api/datasources")
    if not ds:
        red(PAD, "izvor podataka", "nijedan nije registrovan")
    else:
        for x in ds:
            red(OK, "izvor podataka", f"{x['name']} -> {x['url']}")

    d = dohvati(f"{GRAFANA}/api/dashboards/uid/kp-dog-breeds")
    if not d:
        red(PAD, "panel", "nije ucitan -- proveri monitoring/grafana/provisioning/")
        return
    dash = d["dashboard"]
    paneli = [p for p in dash["panels"] if p["type"] != "row"]
    red(OK, "panel", f"'{dash['title']}' | {len(paneli)} panela")

    # Stari obrazac clamp_min je uzrok Buga 8; ne sme da se vrati kroz izmenu u UI-ju.
    stari = [p["title"] for p in paneli
             for t in p.get("targets", []) if "clamp_min" in t.get("expr", "")]
    red(PAD if stari else OK, "izrazi bez clamp_min",
        f"NASAO u: {stari}" if stari else "ok -- nijedan panel ne koristi clamp_min")

    print(f"\n     -> {GRAFANA}/d/kp-dog-breeds/")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promet", action="store_true",
                    help="prvo posalji primeri/ na servis, pa proveravaj")
    a = ap.parse_args()

    print("\n" + "=" * 64)
    print("  Provera monitoring steka")
    print("=" * 64)

    proveri_docker()
    if _greske:
        print("\n  Docker nije u redu -- ostale provere nemaju smisla.\n")
        return 1

    if a.promet:
        print("\n   saljem primeri/ na servis...")
        subprocess.run([sys.executable, str(ROOT / "scripts/smoke_service.py"),
                        "--url", SERVIS], capture_output=True, timeout=120)
        print("   cekam da Prometheus pokupi nove brojke (2 skupljanja)...")
        time.sleep(22)

    proveri_servis()
    proveri_prometheus()
    proveri_odnose()
    proveri_grafanu()

    print("\n" + "=" * 64)
    if _greske:
        print(f"  PROBLEMA: {len(_greske)}")
        for g in _greske:
            print(f"    - {g}")
    else:
        print("  SVE JE U REDU")
    print("=" * 64 + "\n")
    return 1 if _greske else 0


if __name__ == "__main__":
    sys.exit(main())
