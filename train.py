#!/usr/bin/env python3
"""
Train an Ultralytics YOLO model on the NIH ChestXray dataset.

Loads dataset/model configs, selects the best available device (preferring MPS
on Apple Silicon), and runs training with sensible defaults. CLI flags override
values found in `config/model.yaml`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for training.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO on NIH ChestXray.")
    parser.add_argument("--data", type=Path, default=Path("config/dataset.yaml"), help="Dataset YAML path.")
    parser.add_argument("--model", type=str, default=None, help="Model weights or YAML (default from model_config).")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model.yaml"),
        help="YAML with default hyperparameters.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g., mps, cpu).")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--project", type=Path, default=Path("runs/train"), help="Output project directory.")
    parser.add_argument("--name", type=str, default="chestxray-yolo11n", help="Run name.")
    parser.add_argument("--resume", action="store_true", help="Resume last run from project/name.")
    parser.add_argument("--single-cls", action="store_true", help="Train as single-class detector.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def load_model_config(path: Path) -> Dict[str, Any]:
    """
    Load optional model configuration YAML.

    :param path: Path to the YAML file.
    :return: Parsed dictionary or empty dict if missing.
    """
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def select_device(preferred: str | None) -> str:
    """
    Select the best available device, preferring MPS on Apple Silicon.

    :param preferred: User-requested device string.
    :return: Device string compatible with Ultralytics.
    """
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """
    Run YOLO training with merged CLI and config defaults.
    """
    args = parse_args()
    cfg = load_model_config(args.model_config)

    model_path = args.model or cfg.get("model")
    device = select_device(args.device or cfg.get("device"))

    train_kwargs: Dict[str, Any] = {
        "data": str(args.data),
        "epochs": args.epochs or cfg.get("epochs", 50),
        "imgsz": args.imgsz or cfg.get("imgsz", 640),
        "batch": args.batch or cfg.get("batch", 8),
        "patience": args.patience or cfg.get("patience", 20),
        "lr0": args.lr0 or cfg.get("lr0", 0.01),
        "device": device,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "resume": args.resume,
        "single_cls": args.single_cls,
        "seed": args.seed or cfg.get("seed", 42),
        "optimizer": cfg["optimizer"],
        "save_period": cfg["save_period"]
    }

    model = YOLO(model_path)
    results = model.train(**train_kwargs)


if __name__ == "__main__":
    main()
