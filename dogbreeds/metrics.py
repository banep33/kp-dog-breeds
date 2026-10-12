"""Metrike. Proizvod prikazuje TRI odgovora, zato je top-3 accuracy glavna poslovna metrika,
a top-1 i log-loss su tehničke. ECE meri da li su prikazane verovatnoće "poštene"."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3)) -> dict[str, float]:
    ks = [k for k in ks if k <= logits.size(1)]
    _, pred = logits.topk(max(ks), dim=1)
    correct = pred.eq(targets.view(-1, 1))
    return {f"top{k}": correct[:, :k].any(dim=1).float().mean().item() for k in ks}


def log_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return F.cross_entropy(logits, targets).item()


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """ECE: prosečna razlika između 'koliko je model siguran' i 'koliko je zaista tačan'.
    Ako model kaže 90% za 100 slika, treba da pogodi ~90 njih."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == targets).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017): jedan parametar T, fitovan na VALIDACIONOM skupu.
    Ne menja top-1/top-3 (redosled klasa ostaje isti), samo kalibriše verovatnoće."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
