"""MobileNetV3 classification for gammarus crops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from biomae.paths import checkpoint_path


@dataclass
class ClassifierConfig:
    class_names: List[str]
    name_to_idx: Dict[str, int]
    crop_size: int = 512
    use_argmax: bool = True
    thresholds: Optional[Dict[str, float]] = None


def build_mobilenetv3(num_classes: int) -> nn.Module:
    """Build MobileNetV3-large with custom classifier head (matches training notebooks)."""
    m = models.mobilenet_v3_large(weights=None)
    in_feats = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_feats, 512),
        nn.LayerNorm(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return m


def _load_state_dict(weights_path: Path, device: torch.device) -> dict:
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_classifier(
    weights: Optional[Union[str, Path]] = None,
    meta_json: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, ClassifierConfig, transforms.Compose]:
    """
    Load MobileNetV3 classifier and preprocessing pipeline.

    Returns model, config, and preprocess transform.
    """
    if weights is None:
        weights = checkpoint_path("mobilenet_v3_v4.pth")
    if meta_json is None:
        meta_json = checkpoint_path("model_meta.json")

    weights = Path(weights)
    meta_json = Path(meta_json)

    if not weights.exists():
        raise FileNotFoundError(f"Classifier weights not found: {weights}")
    if not meta_json.exists():
        raise FileNotFoundError(f"Classifier metadata not found: {meta_json}")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    class_names = meta["labels_map"]["names"]
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    crop_size = meta.get("input_config", {}).get("input_shape", [1, 512, 512, 3])[1]

    thr = meta.get("thresholds_used_in_training_val", {})
    thresholds = {
        "couple": thr.get("couple", 0.50),
        "femelle": thr.get("female", 0.40),
        "male": 0.40,
        "indeterminee": thr.get("indet", 0.50),
    }

    config = ClassifierConfig(
        class_names=class_names,
        name_to_idx=name_to_idx,
        crop_size=int(crop_size),
        use_argmax=True,
        thresholds=thresholds,
    )

    model = build_mobilenetv3(len(class_names)).to(device)
    model.load_state_dict(_load_state_dict(weights, device), strict=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((config.crop_size, config.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, config, preprocess


def decide_from_probs(probs: torch.Tensor, config: ClassifierConfig) -> int:
    """
    Map class probabilities to a predicted index.

    Uses argmax by default (inference notebook setting). Set config.use_argmax=False
    to apply the hierarchical threshold logic used during training validation.
    """
    if config.use_argmax:
        return int(torch.argmax(probs).item())

    thr = config.thresholds or {}
    idx_c = config.name_to_idx.get("couple")
    idx_f = config.name_to_idx.get("femelle")
    idx_m = config.name_to_idx.get("male")
    idx_i = config.name_to_idx.get("indeterminee")

    pc = float(probs[idx_c]) if idx_c is not None else 0.0
    pf = float(probs[idx_f]) if idx_f is not None else 0.0
    pm = float(probs[idx_m]) if idx_m is not None else 0.0
    pi = float(probs[idx_i]) if idx_i is not None else 0.0

    thr_i = thr.get("indeterminee", 0.50)
    if pi >= thr_i:
        return idx_i

    thr_c = thr.get("couple", 0.50)
    if idx_c is not None and pc >= thr_c and pc >= pf and pc >= pm:
        return idx_c

    thr_f = thr.get("femelle", 0.40)
    if idx_f is not None and pf >= thr_f and pf >= pc and pf >= pm:
        return idx_f

    thr_m = thr.get("male", 0.40)
    if idx_m is not None and pm >= thr_m and pm >= pc and pm >= pf:
        return idx_m

    return int(torch.argmax(probs).item())


@torch.no_grad()
def predict_pil(
    pil_img: Image.Image,
    model: nn.Module,
    config: ClassifierConfig,
    preprocess: transforms.Compose,
    device: torch.device,
) -> Tuple[torch.Tensor, int, str]:
    """Classify a single PIL image crop."""
    pil_img = pil_img.convert("RGB")
    x = preprocess(pil_img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    pred_idx = decide_from_probs(probs, config)
    pred_name = config.class_names[pred_idx]
    return probs, pred_idx, pred_name
