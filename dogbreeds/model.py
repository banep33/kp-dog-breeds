"""Kreiranje modela (timm) i pomoćne funkcije za zamrzavanje i param grupe."""
from __future__ import annotations

import timm
import torch
from timm.data import resolve_data_config


def create_model(model_cfg: dict, num_classes: int) -> torch.nn.Module:
    # timm sam zamenjuje poslednji sloj (1000 ImageNet klasa -> num_classes).
    return timm.create_model(
        model_cfg["name"],
        pretrained=model_cfg.get("pretrained", True),
        num_classes=num_classes,
        drop_rate=model_cfg.get("drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )


def get_data_config(model: torch.nn.Module, img_size: int | None = None) -> dict:
    """Preprocessing koji model očekuje (input_size, mean, std, interpolation, crop_pct)."""
    dc = resolve_data_config({}, model=model)
    if img_size:
        dc["input_size"] = (3, img_size, img_size)
    return dict(dc)


def head_prefix(model: torch.nn.Module) -> str:
    # npr. 'classifier' (EfficientNet/MobileNet) ili 'head.fc' (ConvNeXt)
    return model.pretrained_cfg.get("classifier", "classifier")


def split_params(model: torch.nn.Module):
    prefix = head_prefix(model)
    head, backbone = [], []
    for name, p in model.named_parameters():
        (head if name.startswith(prefix) else backbone).append(p)
    return backbone, head


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    backbone, _ = split_params(model)
    for p in backbone:
        p.requires_grad = trainable


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
