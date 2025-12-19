#!/usr/bin/env python3
"""
Evaluate a trained Ultralytics YOLO model on the NIH ChestXray dataset.

Runs validation to compute metrics (mAP, precision/recall) and optionally
produces sample predictions for qualitative inspection.

:param weights: Path to trained weights (.pt).
:param data: Path to dataset YAML.
:param imgsz: Input image size for evaluation.
:param conf: Confidence threshold for prediction visualization.
:param split: Dataset split to evaluate (default: val).
:param device: Device string; auto-selects mps when available.
:param max-viz: Number of validation images to visualize.
:param project: Output project directory for evaluation artifacts.
:param name: Run name within the project directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for evaluation.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on NIH ChestXray.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights (.pt).")
    parser.add_argument("--data", type=Path, default=Path("config/dataset.yaml"), help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for visualization.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate.")
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g., mps, cpu).")
    parser.add_argument("--max-viz", dest="max_viz", type=int, default=16, help="Number of images to visualize.")
    parser.add_argument("--project", type=Path, default=Path("runs/eval"), help="Output project directory.")
    parser.add_argument("--name", type=str, default="chestxray-eval", help="Run name.")
    return parser.parse_args()


def select_device(preferred: str | None) -> str:
    """
    Pick the best available device.

    :param preferred: Preferred device string.
    :return: Device string.
    """
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dataset_yaml(path: Path) -> Dict[str, Any]:
    """
    Load dataset YAML.

    :param path: Path to dataset YAML.
    :return: Parsed dictionary.
    """
    with path.open("r") as f:
        return yaml.safe_load(f)


def _load_split_samples(data_cfg: Dict[str, Any], split: str, max_viz: int) -> List[str]:
    """
    Load a limited list of image paths from a split file.

    :param data_cfg: Dataset configuration dictionary.
    :param split: Split key ('train' or 'val').
    :param max_viz: Maximum number of samples to load.
    :return: List of image paths.
    """
    split_path = Path(data_cfg.get(split, ""))
    if not split_path.exists():
        return []
    lines = split_path.read_text().strip().splitlines()
    return lines[:max_viz]


def main() -> None:
    """
    Run validation and optional visualization.
    """
    args = parse_args()
    device = select_device(args.device)
    model = YOLO(str(args.weights))

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        device=device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    print("[val] metrics:", metrics.results_dict)

    data_cfg = _load_dataset_yaml(args.data)
    samples = _load_split_samples(data_cfg, args.split, args.max_viz)
    if samples:
        viz_name = f"{args.name}-viz"
        print(f"[predict] visualizing {len(samples)} samples")
        model.predict(
            source=samples,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            project=str(args.project),
            name=viz_name,
            save=True,
            exist_ok=True,
        )


if __name__ == "__main__":
    main()

