"""YOLO detection wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO

from biomae.paths import checkpoint_path


def load_detector(weights: Optional[Union[str, Path]] = None) -> YOLO:
    """Load a YOLO detector from weights path or default checkpoint."""
    if weights is None:
        weights = checkpoint_path("yolov11n_best.pt")
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. "
            "See checkpoints/README.md for setup instructions."
        )
    return YOLO(str(weights))


def predict(
    model: YOLO,
    img_path: Union[str, Path],
    *,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.60,
    device: Optional[Union[int, str]] = None,
):
    """Run YOLO inference on a single image path."""
    if device is None:
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"

    return model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        save=False,
    )
