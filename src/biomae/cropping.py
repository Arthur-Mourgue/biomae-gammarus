"""512×512 crop utilities for YOLO detections (from notebook/lib/crop512.py)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch


def compute_iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / (denom + 1e-6))


def boxes_intersect_xyxy(a, b) -> bool:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return (x2 > x1) and (y2 > y1)


def merge_xyxy(boxes: Sequence[Sequence[float]]) -> List[int]:
    xs1 = [float(b[0]) for b in boxes]
    ys1 = [float(b[1]) for b in boxes]
    xs2 = [float(b[2]) for b in boxes]
    ys2 = [float(b[3]) for b in boxes]
    return [int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))]


def clip_xyxy(box_xyxy: Sequence[float], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = map(float, box_xyxy)

    x1 = max(0.0, min(x1, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    x2 = max(0.0, min(x2, float(w)))
    y2 = max(0.0, min(y2, float(h)))

    if x2 <= x1:
        x2 = min(float(w), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(h), y1 + 1.0)

    return [int(x1), int(y1), int(x2), int(y2)]


def expand_xyxy(
    box_xyxy: Sequence[float],
    margin_px: int = 0,
    margin_ratio: float = 0.0,
) -> List[float]:
    """Expand a box (xyxy) by fixed pixels and/or relative margin."""
    x1, y1, x2, y2 = map(float, box_xyxy)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    mx = float(margin_px) + 0.5 * float(margin_ratio) * w
    my = float(margin_px) + 0.5 * float(margin_ratio) * h

    return [x1 - mx, y1 - my, x2 + mx, y2 + my]


def group_overlapping_boxes(
    boxes_xyxy,
    mode: str = "intersect",
    iou_thr: float = 0.05,
):
    """
    Group boxes into connected components.

    mode:
      - "intersect": strict overlap (recommended for amplexus / couples)
      - "iou": IoU >= iou_thr
    """
    boxes = np.asarray(boxes_xyxy, dtype=float)
    n = len(boxes)
    if n == 0:
        return []

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if mode == "intersect":
                ok = boxes_intersect_xyxy(boxes[i], boxes[j])
            elif mode == "iou":
                ok = compute_iou_xyxy(boxes[i], boxes[j]) >= float(iou_thr)
            else:
                raise ValueError("mode must be 'intersect' or 'iou'")

            if ok:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        groups.append(comp)

    return groups


def crop_pad_512(
    img_bgr: np.ndarray,
    box_xyxy: Sequence[float],
    canvas_size: int = 512,
    downscale_only: bool = True,
    pad_value: Union[int, Tuple[int, int, int]] = 0,
    return_info: bool = False,
):
    """
    Return a BGR image (canvas_size, canvas_size, 3) padded to square.

    downscale_only=True never upscales (preserves real-world scale).
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = clip_xyxy(box_xyxy, w=w, h=h)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return (None, None) if return_info else None

    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return (None, None) if return_info else None

    scale = 1.0
    if downscale_only:
        if ch > canvas_size or cw > canvas_size:
            scale = canvas_size / float(max(ch, cw))
            new_w = max(1, int(round(cw * scale)))
            new_h = max(1, int(round(ch * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.resize(crop, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
        if return_info:
            info = {
                "src_box_xyxy": [x1, y1, x2, y2],
                "scale": float(canvas_size / max(1, cw)),
                "pad_xy": (0, 0),
                "dst_size": (canvas_size, canvas_size),
                "crop_size": (canvas_size, canvas_size),
            }
            return crop, info
        return crop

    fh, fw = crop.shape[:2]

    if isinstance(pad_value, tuple):
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        canvas[:] = pad_value
    else:
        canvas = np.full((canvas_size, canvas_size, 3), int(pad_value), dtype=np.uint8)

    pad_x = (canvas_size - fw) // 2
    pad_y = (canvas_size - fh) // 2
    canvas[pad_y : pad_y + fh, pad_x : pad_x + fw] = crop

    if return_info:
        info = {
            "src_box_xyxy": [x1, y1, x2, y2],
            "scale": float(scale),
            "pad_xy": (int(pad_x), int(pad_y)),
            "dst_size": (int(canvas_size), int(canvas_size)),
            "crop_size": (int(fw), int(fh)),
        }
        return canvas, info

    return canvas


def yolo_result_to_numpy(result0):
    """
    Convert Ultralytics result[0] to numpy arrays.

    Returns: boxes (N,4), scores (N,), cls_ids (N,)
    """
    boxes = result0.boxes.xyxy
    scores = result0.boxes.conf
    cls_ids = result0.boxes.cls

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes)

    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)

    if torch.is_tensor(cls_ids):
        cls_ids = cls_ids.detach().cpu().numpy()
    else:
        cls_ids = np.asarray(cls_ids)

    return boxes, scores, cls_ids


def boxes_to_crops_512(
    img_bgr: np.ndarray,
    boxes_xyxy,
    scores=None,
    conf_filter: float = 0.25,
    group_mode: str = "intersect",
    group_iou_thr: float = 0.05,
    canvas_size: int = 512,
    downscale_only: bool = True,
    margin_px: int = 0,
    margin_ratio: float = 0.0,
    pad_value: Union[int, Tuple[int, int, int]] = 0,
    include_crop_info: bool = True,
):
    """
    Convert detection boxes to padded 512×512 crops with metadata.

    Returns:
      crops_bgr: list of BGR arrays
      meta: list of dicts (merged_box, is_couple, group_scores, crop_info, ...)
    """
    boxes_in = np.asarray(boxes_xyxy, dtype=float)
    n0 = len(boxes_in)
    if n0 == 0:
        return [], []

    orig_idx = np.arange(n0, dtype=int)

    if scores is not None:
        scores_in = np.asarray(scores, dtype=float)
        keep = scores_in >= float(conf_filter)
        boxes = boxes_in[keep]
        scores_f = scores_in[keep]
        orig_idx_f = orig_idx[keep]
    else:
        boxes = boxes_in
        scores_f = None
        orig_idx_f = orig_idx

    if len(boxes) == 0:
        return [], []

    groups = group_overlapping_boxes(boxes, mode=group_mode, iou_thr=group_iou_thr)

    h, w = img_bgr.shape[:2]
    crops: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    for g in groups:
        g_boxes = [boxes[i].tolist() for i in g]
        merged_box = merge_xyxy(g_boxes)

        merged_box = expand_xyxy(merged_box, margin_px=margin_px, margin_ratio=margin_ratio)
        merged_box = clip_xyxy(merged_box, w=w, h=h)

        is_couple = len(g) > 1

        if include_crop_info:
            crop512, crop_info = crop_pad_512(
                img_bgr,
                merged_box,
                canvas_size=canvas_size,
                downscale_only=downscale_only,
                pad_value=pad_value,
                return_info=True,
            )
            if crop512 is None:
                continue
        else:
            crop512 = crop_pad_512(
                img_bgr,
                merged_box,
                canvas_size=canvas_size,
                downscale_only=downscale_only,
                pad_value=pad_value,
                return_info=False,
            )
            crop_info = None
            if crop512 is None:
                continue

        crops.append(crop512)
        meta.append({
            "merged_box": merged_box,
            "group_indices": g,
            "orig_group_indices": [int(orig_idx_f[i]) for i in g],
            "is_couple": is_couple,
            "group_scores": None if scores_f is None else [float(scores_f[i]) for i in g],
            "crop_info": crop_info if include_crop_info else None,
        })

    return crops, meta
