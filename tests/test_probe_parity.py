"""Paritet izmedju linear probe-a i modela koji ide u produkciju.

OVAJ TEST POSTOJI ZBOG PRAVOG BUG-A:
dogbreeds.embed je u prvoj verziji L2-normalizovao embeddinge (trebalo je dedup-u
za kosinusnu slicnost). Glava u dogbreeds.probe je time ucila nad e/||e||, ali pun
timm model svom `classifier` sloju prosledjuje sirovo e (||e|| ~ 100). Izvezeni ONNX
je zato bio ekstremno preterano siguran (p = 1.000) i sva kalibracija besmislena.

Invarijante koje cuvamo:
  1. timm model sa num_classes=0 vraca TACNO vektor koji `classifier` sloj konzumira
  2. dogbreeds.embed podrazumevano NE normalizuje
"""
import numpy as np
import timm
import torch

from dogbreeds.dedup import l2_normalize
from dogbreeds.model import head_prefix

MODELS = ["efficientnet_b0", "mobilenetv4_conv_medium"]


def _split_backbone_and_head(name: str, num_classes: int = 7):
    """Isti nasumicni backbone u dve forme: sa glavom i bez nje."""
    torch.manual_seed(0)
    full = timm.create_model(name, pretrained=False, num_classes=num_classes).eval()
    backbone = timm.create_model(name, pretrained=False, num_classes=0).eval()
    prefix = head_prefix(full)
    missing = backbone.load_state_dict(
        {k: v for k, v in full.state_dict().items() if not k.startswith(prefix)},
        strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return full, backbone, prefix


def test_num_classes_0_output_is_what_classifier_consumes():
    """Temelj na kome stoji dogbreeds.probe: ako ovo padne, glava trenirana nad
    embeddinzima ne sme da se upise nazad u pun model."""
    x = torch.randn(3, 3, 224, 224)
    for name in MODELS:
        full, backbone, prefix = _split_backbone_and_head(name)
        sd = full.state_dict()
        w, b = sd[f"{prefix}.weight"], sd[f"{prefix}.bias"]
        with torch.no_grad():
            feats = backbone(x)
            logits_full = full(x)
            logits_from_feats = torch.nn.functional.linear(feats, w, b)
        diff = (logits_full - logits_from_feats).abs().max().item()
        assert diff < 1e-4, f"{name}: razlika {diff:.2e} - pooling/head se ne poklapaju"


def test_features_are_not_unit_norm():
    """Sirovi feature-i NISU jedinicne duzine -- zato normalizacija u ekstraktoru
    menja ono sto glava vidi. (Kod istreniranog efficientnet_b0 norma je ~100;
    ovde su tezine nasumicne pa je mala, ali kljucno je da NIJE 1.0.)"""
    x = torch.randn(4, 3, 224, 224)
    _, backbone, _ = _split_backbone_and_head("efficientnet_b0")
    with torch.no_grad():
        norms = backbone(x).norm(dim=1).numpy()
    assert not np.allclose(norms, 1.0, rtol=0.05), f"feature-i su vec normalizovani: {norms}"


def test_extract_does_not_normalize_by_default(tmp_path):
    from PIL import Image

    from dogbreeds.embed import extract

    rng = np.random.default_rng(0)
    ids = []
    for k in range(4):
        Image.fromarray(rng.integers(0, 255, (80, 90, 3), dtype=np.uint8)).save(tmp_path / f"im{k}.jpg")
        ids.append(f"im{k}")

    raw, meta = extract(ids, tmp_path, "efficientnet_b0", batch_size=2, pretrained=False,
                        log_every=0)
    assert meta["normalized"] is False
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0), "extract je normalizovao!"

    norm, meta_n = extract(ids, tmp_path, "efficientnet_b0", batch_size=2, pretrained=False,
                           log_every=0, normalize=True)
    assert meta_n["normalized"] is True
    np.testing.assert_allclose(np.linalg.norm(norm, axis=1), 1.0, rtol=1e-5)


def test_l2_normalize_is_idempotent():
    rng = np.random.default_rng(1)
    e = rng.normal(size=(6, 12)).astype(np.float32) * 40
    once = l2_normalize(e)
    np.testing.assert_allclose(once, l2_normalize(once), rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(once, axis=1), 1.0, rtol=1e-6)


def test_fold_standardization_is_exact():
    """Standardizacija uracunata u tezine mora dati ISTE logite kao eksplicitno
    standardizovanje ulaza. Na ovome stoji ceo linear-probe deployment put."""
    import torch.nn as nn

    from dogbreeds.probe import fold_standardization

    rng = np.random.default_rng(0)
    D, C, N = 32, 5, 11
    X = (rng.normal(size=(N, D)) * 37 + 12).astype(np.float32)   # proizvoljna skala
    mu, sd = X.mean(0).astype(np.float32), np.maximum(X.std(0), 1e-6).astype(np.float32)

    torch.manual_seed(0)
    head = nn.Linear(D, C).eval()
    with torch.no_grad():
        ref = head(torch.from_numpy((X - mu) / sd))          # standardizovan ulaz
    W, b = fold_standardization(head, mu, sd)
    got = torch.nn.functional.linear(torch.from_numpy(X), W, b)  # sirov ulaz
    assert (ref - got).abs().max().item() < 1e-3


def test_train_head_exports_weights_for_raw_features():
    """Posle train_head(standardize=True), apply_head nad SIROVIM feature-ima mora
    reprodukovati tacnost izmerenu tokom treninga."""
    from dogbreeds.probe import apply_head, train_head

    rng = np.random.default_rng(3)
    D, C = 24, 4
    centers = rng.normal(size=(C, D)) * 3
    y = np.repeat(np.arange(C), 40).astype(np.int64)
    X = (centers[y] + rng.normal(size=(len(y), D))).astype(np.float32) * 25 + 7
    yv = y.copy()
    Xv = (centers[yv] + rng.normal(size=(len(yv), D))).astype(np.float32) * 25 + 7

    head, best, _ = train_head(X, y, Xv, yv, num_classes=C, epochs=25, verbose=False)
    acc = (apply_head(head, Xv).argmax(1).numpy() == yv).mean()
    assert abs(acc - best["top1"]) < 1e-6, f"{acc} != {best['top1']}"
    assert acc > 0.9
