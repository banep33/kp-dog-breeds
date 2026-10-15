"""Testovi detekcije drifta.

Ne proveravaju "da kod radi" nego **invarijante mere** -- isti pristup koji je u Danu 1
uhvatio Bug 1. Detektor drifta koji uvek vrati nulu prolazi svaki test tipa "ne puca";
ovi testovi padaju ako mera izgubi smisao.
"""
from __future__ import annotations

import numpy as np
import pytest

from dogbreeds.drift import (
    MIN_SAMPLES,
    PSI_DRIFT,
    PSI_SHIFT,
    build_reference,
    compare,
    ks_two_sample,
    psi,
    status_of,
)

CLASSES = [f"breed_{i:03d}" for i in range(12)]


def _reference(rng, n=3000, loc=0.95):
    numeric = {
        "top3_mass": list(np.clip(rng.normal(loc, 0.04, n), 0, 1)),
        "max_prob": list(np.clip(rng.normal(0.85, 0.10, n), 0, 1)),
        "entropy": list(np.abs(rng.normal(0.3, 0.2, n))),
        "dog_score": list(np.clip(rng.normal(0.90, 0.05, n), 0, 1)),
    }
    top1 = list(rng.choice(CLASSES, n))
    return build_reference(numeric, top1, CLASSES), numeric, top1


# ---------------------------------------------------------------- PSI

def test_psi_identicnih_raspodela_je_nula():
    p = np.array([0.2, 0.3, 0.5])
    assert psi(p, p) == pytest.approx(0.0, abs=1e-12)


def test_psi_je_nenegativan_i_simetrican():
    """PSI je zbir dve KL divergencije, pa mora biti >= 0 i isti u oba smera."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.dirichlet(np.ones(8))
        b = rng.dirichlet(np.ones(8))
        assert psi(a, b) >= 0
        assert psi(a, b) == pytest.approx(psi(b, a), rel=1e-9)


def test_psi_raste_sa_velicinom_pomaka():
    base = np.array([0.25, 0.25, 0.25, 0.25])
    mali = np.array([0.30, 0.24, 0.23, 0.23])
    veliki = np.array([0.70, 0.10, 0.10, 0.10])
    assert psi(base, mali) < psi(base, veliki)


def test_psi_ne_puca_na_praznom_binu():
    """Bin bez ijednog uzorka je normalna pojava u produkciji, ne greska."""
    assert np.isfinite(psi(np.array([0.5, 0.5]), np.array([1.0, 0.0])))


def test_granice_statusa():
    assert status_of(PSI_SHIFT - 0.001) == "stabilno"
    assert status_of(PSI_SHIFT) == "pomak"
    assert status_of(PSI_DRIFT) == "drift"


# ---------------------------------------------------------------- KS

def test_ks_istog_uzorka_je_nula():
    x = np.linspace(0, 1, 500)
    d, p = ks_two_sample(x, x)
    assert d == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_ks_razdvaja_pomerene_raspodele():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(1.5, 1, 2000)
    d, p = ks_two_sample(a, b)
    assert d > 0.4
    assert p < 1e-10


def test_ks_ne_alarmira_na_istoj_raspodeli():
    """Dva uzorka iz ISTE raspodele ne smeju da budu znacajni -- inace je svaki dan alarm."""
    rng = np.random.default_rng(2)
    _, p = ks_two_sample(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    assert p > 0.01


# ---------------------------------------------------------------- ceo put

def test_referenca_sadrzi_sve_signale():
    rng = np.random.default_rng(3)
    ref, _, _ = _reference(rng)
    assert set(ref["numeric"]) == {"top3_mass", "max_prob", "entropy", "dog_score"}
    for d in ref["numeric"].values():
        assert len(d["props"]) == len(d["edges"]) - 1
        assert sum(d["props"]) == pytest.approx(1.0, abs=1e-9)
    assert sum(ref["categorical"]["top1_breed"]["props"]) == pytest.approx(1.0, abs=1e-9)


def test_isti_izvor_ne_pravi_drift():
    """Najvazniji test: produkcija iz iste raspodele mora biti 'stabilno'.

    Da ovo padne, detektor bi palio alarm bez razloga i bio bi iskljucen prve nedelje."""
    rng = np.random.default_rng(4)
    ref, _, _ = _reference(rng)
    _, numeric, top1 = _reference(rng)          # nov uzorak, ista raspodela
    rep = compare(ref, numeric, top1)
    assert rep["status"] == "stabilno"
    assert rep["worst_psi"] < PSI_SHIFT


def test_pomerena_raspodela_daje_drift():
    rng = np.random.default_rng(5)
    ref, _, _ = _reference(rng, loc=0.95)
    _, numeric, top1 = _reference(rng, loc=0.70)   # poverenje palo
    rep = compare(ref, numeric, top1)
    assert rep["features"]["top3_mass"]["psi"] > PSI_DRIFT
    assert rep["status"] == "drift"
    assert rep["features"]["top3_mass"]["ks_significant"] is True


def test_promenjena_raspodela_rasa_se_vidi():
    """Korisnici salju samo dve rase umesto svih 12.

    Izvestaj mora da imenuje OBA smera. Doprinos PSI-ju je veci za rasu koja je nestala
    nego za onu koja je porasla, pa bi jedna lista sortirana po doprinosu pokazala samo
    nestale i sakrila pravi uzrok."""
    rng = np.random.default_rng(6)
    ref, numeric, _ = _reference(rng)
    skew = list(rng.choice(CLASSES[:2], len(numeric["top3_mass"])))
    rep = compare(ref, numeric, skew)
    f = rep["features"]["top1_breed"]
    assert f["psi"] > PSI_DRIFT

    cesce = {d["breed"] for d in f["cesce_nego_u_referenci"]}
    redje = {d["breed"] for d in f["redje_nego_u_referenci"]}
    assert cesce == set(CLASSES[:2]), "rase koje korisnici salju moraju biti imenovane"
    assert redje <= set(CLASSES[2:]) and redje, "nestale rase takodje moraju biti imenovane"
    for d in f["cesce_nego_u_referenci"]:
        assert d["prod"] > d["ref"]


def test_mali_uzorak_ne_alarmira():
    """Sa 20 zahteva PSI meri sum uzorkovanja. Detektor mora da cuti, ne da pogadja."""
    rng = np.random.default_rng(7)
    ref, _, _ = _reference(rng)
    _, numeric, top1 = _reference(rng, n=MIN_SAMPLES - 1)
    rep = compare(ref, numeric, top1)
    assert rep["status"] == "nedovoljno podataka"
    assert "features" in rep and not rep["features"]
