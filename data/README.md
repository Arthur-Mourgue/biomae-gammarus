# Dataset layout

This pipeline expects a **dual-view** dataset: synchronized orthogonal camera frames for each specimen.

## Raw detection dataset (YOLO format)

```
data/01_raw/mfi_dataset/
├── train/
│   ├── images/
│   │   ├── img_A_00001.png
│   │   ├── img_B_00001.png
│   │   └── ...
│   └── labels/
│       ├── img_A_00001.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

- `img_A_*` and `img_B_*` share the same numeric ID and depict the same specimen from two orthogonal views.
- Labels are YOLO format: `class_id x_center y_center width height` (normalized).

## Classification crops (MobileNetV3)

```
data/02_processed/mobilenet_mfic_dataset/
├── train/
│   ├── male/
│   ├── femelle/
│   ├── indeterminee/
│   └── couple/
└── val/
    └── ...
```

Crops are 512×512 PNGs produced by the detection + cropping stage (`src/biomae/cropping.py`).

## Notes

- Raw imagery from the Biomae laboratory is **not** included in this repository.
- You must obtain or recreate a compatible dataset to train or run inference.
- Place trained model weights in `checkpoints/` (see `checkpoints/README.md`).
