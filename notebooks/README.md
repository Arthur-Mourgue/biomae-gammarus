# Notebooks

Curated notebooks for training and inference. Run from the project root or this directory.

| Notebook | Purpose |
|----------|---------|
| `01_train_yolo.ipynb` | Train YOLOv11n detector |
| `02_train_mobilenet.ipynb` | Train MobileNetV3 4-class classifier (v4) |
| `03_train_mlp_fusion.ipynb` | **Legacy** — YOLO embedding MLP fusion across A/B views |
| `04_inference_pipeline.ipynb` | End-to-end inference using `src/biomae/` |

Experimental and utility notebooks are in `experiments/` (CLIP, SigLIP, ConvNeXt, dataset prep).

## Setup

```bash
pip install -e .
jupyter lab
```

Place checkpoints in `checkpoints/` before running inference (see `checkpoints/README.md`).
