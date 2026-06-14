"""End-to-end inference pipeline: detect → crop → classify."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from biomae.classification import (
    ClassifierConfig,
    load_classifier,
    predict_pil,
)
from biomae.cropping import boxes_to_crops_512, yolo_result_to_numpy
from biomae.detection import load_detector, predict as yolo_predict


def aggregate_probs(per_crop_probs: List[np.ndarray], mode: str = "mean") -> np.ndarray:
    """Aggregate per-crop probability vectors into one vector."""
    stack = np.stack(per_crop_probs, axis=0)
    if mode == "mean":
        return stack.mean(axis=0)
    if mode == "max":
        return stack.max(axis=0)
    raise ValueError(f"Unknown aggregation mode: {mode}")


class GammarusPipeline:
    """
    YOLO detection + 512 crop + MobileNetV3 classification.

    Processes each view independently. For legacy dual-view MLP fusion, see biomae.fusion.
    """

    def __init__(
        self,
        yolo_weights: Optional[Union[str, Path]] = None,
        clf_weights: Optional[Union[str, Path]] = None,
        clf_meta: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.det_model = load_detector(yolo_weights)
        self.clf_model, self.clf_config, self.preprocess = load_classifier(
            clf_weights, clf_meta, device
        )

    def process_image(
        self,
        img_path: Union[str, Path],
        *,
        yolo_imgsz: int = 1024,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.60,
        crop_conf_filter: float = 0.25,
        group_iou_thr: float = 0.25,
        agg_mode: str = "mean",
        crop_size: int = 512,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on one image.

        Returns dict with pred_name, score, n_crops, per_crop details, and agg_probs.
        """
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")

        results = yolo_predict(
            self.det_model,
            img_path,
            imgsz=yolo_imgsz,
            conf=yolo_conf,
            iou=yolo_iou,
            device=0 if self.device.type == "cuda" else "cpu",
        )

        boxes, scores, _cls_ids = yolo_result_to_numpy(results[0])

        crops_bgr, metas = boxes_to_crops_512(
            img_bgr=img_bgr,
            boxes_xyxy=boxes,
            scores=scores,
            conf_filter=crop_conf_filter,
            group_mode="iou",
            group_iou_thr=group_iou_thr,
            canvas_size=crop_size,
            downscale_only=True,
        )

        if len(crops_bgr) == 0:
            return {
                "img": str(img_path),
                "pred_name": "none",
                "score": 0.0,
                "n_crops": 0,
                "per_crop": [],
            }

        per_crop = []
        per_crop_probs = []

        for crop_bgr, m in zip(crops_bgr, metas):
            pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            probs_t, pred_idx, pred_name = predict_pil(
                pil_crop,
                self.clf_model,
                self.clf_config,
                self.preprocess,
                self.device,
            )
            probs = probs_t.numpy().astype(float)

            per_crop.append({
                "meta": m,
                "pred_idx": int(pred_idx),
                "pred_name": pred_name,
                "probs": probs,
                "score": float(probs[pred_idx]),
            })
            per_crop_probs.append(probs)

        agg_probs = aggregate_probs(per_crop_probs, mode=agg_mode)
        final_idx = int(np.argmax(agg_probs))
        final_name = self.clf_config.class_names[final_idx]
        final_score = float(agg_probs[final_idx])

        return {
            "img": str(img_path),
            "pred_name": final_name,
            "score": final_score,
            "n_crops": len(crops_bgr),
            "agg_probs": agg_probs,
            "per_crop": per_crop,
        }

    def process_dual_view(
        self,
        img_a: Union[str, Path],
        img_b: Union[str, Path],
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process paired orthogonal views independently.

        Does not fuse predictions — returns separate results for view A and B.
        """
        return {
            "view_a": self.process_image(img_a, **kwargs),
            "view_b": self.process_image(img_b, **kwargs),
        }


def process_image(
    img_path: Union[str, Path],
    pipeline: Optional[GammarusPipeline] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function matching the original notebook API."""
    if pipeline is None:
        pipeline = GammarusPipeline()
    return pipeline.process_image(img_path, **kwargs)
