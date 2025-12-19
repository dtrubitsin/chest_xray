# ChestXray NIHYOLO Baseline

Baseline multiclass object detector for NIH ChestXray bounding boxes using
Ultralytics YOLO, tested with Apple MPS acceleration.

## Setup

- Python 3.10+ recommended.
- Install deps:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Data

1) Download images and bbox CSV (defaults to `data/raw`) and extract:
   ```bash
   python download_dataset.py --extract
   ```
   Use `--start/--end` to grab fewer archives for a quick run.
2) Prepare YOLO labels and splits (writes to `data/labels`, `data/splits` and
   updates `config/dataset.yaml`). Use provided NIH lists for reproducible
   splits:
   ```bash
   python scripts/prepare_labels.py --test-list test_list.txt
   ```
   **NOTE:** only test list contain images with bounding boxes.

Data layout after prep:

- `data/images/*.png` extracted images.
- `data/labels/*.txt` YOLO bboxes.
- `data/splits/train.txt`, `data/splits/val.txt`.
- `config/dataset.yaml` referencing splits and class names.

## Train

Trains YOLOv11 nano with single class option with defaults from `config/model.yaml`:

```bash
python3 train.py --workers 8 --single-cls --data config/dataset.yaml --model yolo11n.pt --device mps
```

## Evaluate & Visualize

```bash
python eval.py --weights runs/train/chestxray-yolo11n/weights/best.pt \
  --data config/dataset.yaml --max-viz 16 --device mps
```

Creates metrics under `runs/eval/chestxray-eval` and optional predictions under
`runs/eval/chestxray-eval-viz`.

## Baseline results

1. Dataset contains very few images with bbox annotations (880);
2. Result metrics on single class training:
    - precision: 0.4
    - recall: 0.23
    - mAP50: 0.22
    - mAP50-95: 0.1
3. This tells us about that pipeline is working, but need strong improvements to perform well;
4. [Train curves](runs/train/chestxray-yolo11n/results.png)
5. Validation curves available
   here: [F1 curve](runs/eval/chestxray-eval/BoxF1_curve.png), [PR_curve](runs/eval/chestxray-eval/BoxPR_curve.png)

## Suggestions / Next Steps

- **Train the actual 8-class detector (not `--single-cls`)**:
  - Remove `--single-cls` and train with `config/dataset.yaml` `names` as-is to get per-class metrics.
  - Inspect class distribution (# boxes per class) and consider merging very rare classes for a stronger initial baseline.

- **Make the most of the small labeled set (880 images)**:
  - Use k-fold cross validation or multiple random seeds and report mean/std for mAP50 and mAP50-95.
  - Verify the split has enough bbox-annotated images in both train/val (and no patient-level leakage).

- **Address class imbalance explicitly**:
  - Oversample images containing rare classes (or use a weighted sampler).
  - Consider loss tweaks that help imbalance (e.g., focal-style classification loss) and compare per-class AP.

- **Tune YOLO training hyperparameters**:
  - Increase `imgsz`, train longer, and run Ultralytics hyperparameter tuning.
  - Try different learning rates / schedules, stronger regularization, and adjust augmentation strength (too much aug can hurt small datasets).

- **Try stronger backbones and higher capacity**:
  - Compare `yolo11n.pt` vs `yolo11s.pt`/`yolo11m.pt` (and potentially freeze the backbone for a few epochs before full fine-tuning).

- **Use domain-appropriate augmentations & preprocessing**:
  - Keep geometry changes realistic (avoid extreme perspective/rotation for frontal CXR).
  - Add intensity-only transforms (contrast/brightness/gamma, CLAHE) via Albumentations; keep bbox-safe transforms.

- **Improve label quality & bbox conventions**:
  - Clip bboxes to image bounds, remove degenerate boxes, and audit a sample of labels visually.
  - Standardize how you handle images without boxes (true negatives) vs missing annotations.

- **Leverage external data / pretraining**:
  - Initialize from a detector pretrained on a similar CXR bbox dataset or pseudo-label additional NIH images and fine-tune.

- **Evaluation improvements**:
  - Report per-class AP and confusion between visually similar classes.
  - Tune confidence/NMS thresholds on validation to optimize F1 or mAP.

## References

- NIH ChestXray dataset and bbox annotations: [ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
