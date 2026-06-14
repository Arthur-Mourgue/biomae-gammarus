"""Legacy MLP fusion of dual-view YOLO embeddings (from Train_MLP.ipynb)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


def calculate_iou(box_a, box_b) -> float:
    """IoU between two YOLO xywh normalized boxes."""

    def to_coords(b):
        x, y, w, h = b
        x1, x2 = x - w / 2, x + w / 2
        y1, y2 = y - h / 2, y + h / 2
        return [x1, y1, x2, y2]

    b1, b2 = to_coords(box_a), to_coords(box_b)
    x_a = max(b1[0], b2[0])
    y_a = max(b1[1], b2[1])
    x_b = min(b1[2], b2[2])
    y_b = min(b1[3], b2[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    box_b_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter_area / float(box_a_area + box_b_area - inter_area + 1e-6)


class FeatureExtractorMulti:
    """
    Extract per-detection YOLO backbone embeddings via crop-and-re-pass.

    Detections are sorted left-to-right so view A and view B can be paired by index.
    """

    def __init__(self, model_path: Union[str, Path]):
        self.yolo = YOLO(str(model_path))
        self.model = self.yolo.model
        self.target_layer = self.model.model[9]
        self.features = None
        self.target_layer.register_forward_hook(self.hook_fn)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def hook_fn(self, module, input, output):
        self.features = output

    def extract_all(self, img_path: Union[str, Path]) -> List[dict]:
        """
        Return all detections sorted left-to-right.

        Each item: {cls, emb, box, center_x}
        """
        results = self.yolo.predict(str(img_path), conf=0.25, verbose=False)

        if len(results[0].boxes) == 0:
            return []

        detections = []
        img_full = results[0].orig_img
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls)
            xywh = box.xywhn[0].cpu().numpy()

            h, w, _ = img_full.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            crop = img_full[y1:y2, x1:x2]
            _ = self.yolo.predict(crop, verbose=False)

            pooled = self.avg_pool(self.features)
            emb_vec = pooled.flatten().cpu().detach().numpy()

            detections.append({
                "cls": cls,
                "emb": emb_vec,
                "box": xywh,
                "center_x": xywh[0],
            })

        detections.sort(key=lambda x: x["center_x"])
        return detections


class FusionMLP(nn.Module):
    """Concatenate view-A and view-B embeddings, then classify."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.concat_dim = input_dim * 2
        self.layer1 = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat((emb_a, emb_b), dim=1)
        x = self.layer1(x)
        return self.classifier(x)


def load_fusion_mlp(
    weights: Union[str, Path],
    input_dim: int,
    hidden_dim: int = 256,
    num_classes: int = 3,
    dropout_rate: float = 0.5,
    device: Optional[torch.device] = None,
) -> FusionMLP:
    """Load a trained FusionMLP checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionMLP(input_dim, hidden_dim, num_classes, dropout_rate).to(device)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def fuse_embeddings(
    model: FusionMLP,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    device: torch.device,
) -> int:
    """Predict fused class index from two embedding vectors."""
    t_a = torch.tensor(emb_a, dtype=torch.float32).unsqueeze(0).to(device)
    t_b = torch.tensor(emb_b, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(t_a, t_b)
    return int(torch.argmax(output, dim=1).item())
