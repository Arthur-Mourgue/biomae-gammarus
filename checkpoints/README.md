# Checkpoints

Model weights are **not** committed to git. Copy or symlink your trained checkpoints here:

| File | Source (local training output) | Used by |
|------|-------------------------------|---------|
| `yolov11n_best.pt` | `outputs/yolov11n_i1024b2_e100/det/weights/best.pt` | Detection |
| `mobilenet_v3_v4.pth` | `outputs/mobilenet_v3_mfic_4classes_v4/best_model.pth` | Classification (current) |
| `model_meta.json` | `outputs/mobilenet_v3_mfic_4classes_v4/model_meta.json` | Classification metadata |
| `fusion_mlp.pth` | `best_fusion_mlp.pth` (legacy MLP fusion) | `notebooks/03_train_mlp_fusion.ipynb` |

## Quick setup (if you have local training outputs)

```bash
# From the legacy workspace (sibling folders under BIOMAE/):
cp ../Gammare-detection---classification/ws_det-clas-mal-fem/outputs/yolov11n_i1024b2_e100/det/weights/best.pt checkpoints/yolov11n_best.pt
cp ../Gammare-detection---classification/ws_det-clas-mal-fem/outputs/mobilenet_v3_mfic_4classes_v4/best_model.pth checkpoints/mobilenet_v3_v4.pth
cp ../Gammare-detection---classification/ws_det-clas-mal-fem/outputs/mobilenet_v3_mfic_4classes_v4/model_meta.json checkpoints/model_meta.json
cp ../Gammare-detection---classification/best_fusion_mlp.pth checkpoints/fusion_mlp.pth
```
