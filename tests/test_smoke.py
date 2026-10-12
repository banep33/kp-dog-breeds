"""Smoke test celog pipeline-a na lažnim podacima (CPU, ~1 min, bez interneta).

    python -m pytest tests -q
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sh(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def test_pipeline(tmp_path):
    data = tmp_path / "dummy"
    sh("scripts/make_dummy_data.py", "--out", str(data), "--classes", "4", "--per-class", "15")
    common = [
        "--config", "configs/baseline.yaml", "--set",
        f"data.labels_csv={data / 'labels.csv'}", f"data.images_dir={data / 'train'}",
        f"data.splits_dir={tmp_path / 'splits'}", "data.num_workers=0", "data.batch_size=8",
        "data.val_size=0.2", "data.test_size=0.2",
    ]
    sh("-m", "dogbreeds.data", *common)
    sh("-m", "dogbreeds.train", *common, "model.name=efficientnet_b0", "model.pretrained=false",
       "model.img_size=96", "train.epochs=2", "train.amp=false", "train.ema_decay=null",
       f"output.dir={tmp_path / 'runs'}", "output.run_name=smoke")
    ckpt = tmp_path / "runs" / "smoke" / "best.pt"
    assert ckpt.exists()
    sh("-m", "dogbreeds.evaluate", "--checkpoint", str(ckpt), *common)
    assert (ckpt.parent / "eval" / "metrics.json").exists()
    sh("-m", "dogbreeds.export", "--checkpoint", str(ckpt), "--out", str(tmp_path / "artifacts"),
       "--int8", "--calib", "10", *common)
    assert (tmp_path / "artifacts" / "model.int8.onnx").exists()

    from dogbreeds.inference import Predictor
    p = Predictor(tmp_path / "artifacts" / "model.onnx")
    img = next((data / "train").glob("*.jpg"))
    r = p.predict(Image.open(img), k=3)
    assert len(r["predictions"]) == 3
    assert sum(x["probability"] for x in r["predictions"]) <= 1 + 1e-6


def test_preprocess_matches_timm():
    """NumPy preprocessing u produkciji mora da odgovara timm transformaciji iz treninga."""
    import timm
    from timm.data import create_transform, resolve_data_config

    from dogbreeds.inference import preprocess

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=3)
    dc = dict(resolve_data_config({}, model=model))
    tf = create_transform(**dc, is_training=False)
    rng = np.random.default_rng(0)
    for size in [(320, 240), (240, 500), (224, 224)]:
        img = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        ref = tf(img).numpy()
        got = preprocess(img, dc)[0]
        assert ref.shape == got.shape
        assert np.abs(ref - got).max() < 1e-4, np.abs(ref - got).max()
