#!/usr/bin/env python3
"""
Convert NIH ChestXray bounding boxes to YOLO format and create splits.

The script expects the NIH bounding-box CSV (BBox_List_2017.csv). It filters to
images available in the provided images directory, writes YOLO label TXT files,
and creates train/val splits for Ultralytics.

Only images with annotations in the bbox CSV are included in splits. If
train_val_list or test_list are provided, they are filtered to only include
images that have annotations in the CSV.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml

BBOX_DEFAULT = Path("data/raw/BBox_List_2017.csv")
DATASET_YAML_PATH = Path("config/dataset.yaml")
DEFAULT_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltrate",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]
DEFAULT_IMG_SIZE = 1024


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert NIH ChestXray bbox CSV to YOLO labels and splits.")
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"), help="Directory with extracted images.")
    parser.add_argument("--bbox-csv", type=Path, default=BBOX_DEFAULT, help="Path to BBox_List_2017.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output root for labels/splits.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    parser.add_argument("--test-list", type=Path, default='test_list.txt',
                        help="File with image names reserved for test split (one per line).")

    return parser.parse_args()


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure expected columns exist and coerce numeric fields.

    :param df: Raw dataframe from CSV.
    :return: Cleaned dataframe with numeric bbox fields.
    :raises ValueError: When required columns are missing.
    """
    required = {
        "Image Index",
        "Finding Label",
        "Bbox [x",
        "y",
        "w",
        "h]"
    }
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns in bbox CSV: {missing}")

    numeric_cols = ["Bbox [x", "y", "w", "h]"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols)
    return df


def _build_class_map(df: pd.DataFrame) -> List[str]:
    """
    Build ordered class list.

    :param df: Annotation dataframe.
    :return: Ordered list of class names.
    """
    labels = sorted({label.strip() for label in df["Finding Label"].unique()})
    return labels or DEFAULT_CLASSES


def _yolo_bbox(row: pd.Series) -> Tuple[float, float, float, float]:
    """
    Convert bbox row to YOLO normalized format.

    :param row: Single annotation row.
    :return: (x_center, y_center, width, height) normalized to [0,1].
    """
    x_center = (row["Bbox [x"] + row["w"] / 2.0) / DEFAULT_IMG_SIZE
    y_center = (row["y"] + row["h]"] / 2.0) / DEFAULT_IMG_SIZE
    w_norm = row["w"] / DEFAULT_IMG_SIZE
    h_norm = row["h]"] / DEFAULT_IMG_SIZE
    return x_center, y_center, w_norm, h_norm


def _write_label_file(label_path: Path, rows: Iterable[pd.Series], class_to_idx: Dict[str, int]) -> None:
    """
    Write a YOLO label file for a single image.

    :param label_path: Destination TXT path.
    :param rows: Iterable of annotation rows for the image.
    :param class_to_idx: Mapping from class name to class index.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for row in rows:
        cls = class_to_idx[row["Finding Label"]] if class_to_idx else 0
        x_center, y_center, w_norm, h_norm = _yolo_bbox(row)
        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    label_path.write_text("\n".join(lines))


def _gather_images(
        df: pd.DataFrame,
        images_dir: Path,
        allowed_ids: set[str] | None,
) -> List[Tuple[str, Path]]:
    """
    Match annotation rows to existing image files.

    :param df: Annotation dataframe.
    :param images_dir: Directory containing images.
    :param allowed_ids: Optional whitelist of image ids to keep.
    :return: List of (image_index, image_path) pairs present on disk.
    """
    images_dir = images_dir.resolve()
    grouped = df.groupby("Image Index")
    pairs: List[Tuple[str, Path]] = []
    for image_id, _ in grouped:
        if allowed_ids is not None and image_id not in allowed_ids:
            continue
        img_path = images_dir / image_id
        if img_path.exists():
            pairs.append((image_id, img_path))
    return pairs


def _train_val_split(items: List[Tuple[str, Path]], train_ratio: float, seed: int) -> Tuple[List, List]:
    """
    Split image list into train and val subsets.

    :param items: List of (image_id, image_path) pairs.
    :param train_ratio: Fraction for training.
    :param seed: Random seed.
    :return: (train_items, val_items).
    """
    rng = random.Random(seed)
    rng.shuffle(items)
    split_idx = int(len(items) * train_ratio)
    return items[:split_idx], items[split_idx:]


def _write_split_lists(
        train_items: List[Tuple[str, Path]],
        val_items: List[Tuple[str, Path]],
        split_dir: Path,
) -> None:
    """
    Write train/val/test split text files expected by Ultralytics.

    :param train_items: Training image pairs.
    :param val_items: Validation image pairs.
    :param split_dir: Directory to store split lists.
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    train_txt.write_text("\n".join(str(path) for _, path in train_items))
    val_txt.write_text("\n".join(str(path) for _, path in val_items))


def _write_dataset_yaml(names: List[str], split_dir: Path, dataset_yaml: Path, has_test: bool) -> None:
    """
    Create or update the Ultralytics dataset YAML file.

    :param names: Ordered class list.
    :param split_dir: Directory containing train/val split files.
    :param dataset_yaml: Destination dataset YAML path.
    :param has_test: Whether a test split file is available.
    """
    data = {
        "path": ".",
        "train": str((split_dir / "train.txt").as_posix()),
        "val": str((split_dir / "val.txt").as_posix()),
        "names": {idx: name for idx, name in enumerate(names)},
    }
    test_path = split_dir / "test.txt"
    if has_test and test_path.exists():
        data["test"] = str(test_path.as_posix())
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"[dataset.yaml] wrote {dataset_yaml}")


def _load_list(path: Path) -> set[str]:
    """
    Load image names from a split list file.

    :param path: Path to a text file with one image name per line.
    :return: Set of image identifiers.
    """
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    """
    Entry point for label conversion and split generation.
    
    Only images with annotations in the bbox CSV are included in splits,
    even if train_val_list or test_list are provided.
    """
    args = parse_args()
    df = pd.read_csv(args.bbox_csv)
    df = _validate_columns(df)

    # Get all unique image IDs that have annotations in the CSV
    annotated_image_ids = set(df["Image Index"].unique())
    print(f"[info] Found {len(annotated_image_ids)} unique images with annotations in CSV")

    class_names = _build_class_map(df)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)} if not args.single_class else {}

    # Load split lists if provided
    train_val_ids = _load_list(args.test_list) if args.test_list else set()

    # Filter split lists to only include images that have annotations
    if train_val_ids is not None:
        train_val_ids = train_val_ids & annotated_image_ids
        print(f"[info] Filtered train_val_list: {len(train_val_ids)} images with annotations")
    else:
        raise ValueError(f"No image is inside given list.")

    available_images = _gather_images(df, args.images_dir, train_val_ids)
    if not available_images:
        raise FileNotFoundError(
            f"No images found in {args.images_dir.resolve()} matching annotations and provided lists."
        )

    train_items, val_items = _train_val_split(available_images, args.train_ratio, args.seed)

    labels_dir = args.output_dir / "labels"
    grouped = df.groupby("Image Index")
    for image_id, img_path in train_items + val_items:
        if image_id not in grouped.groups:
            continue
        rows = grouped.get_group(image_id)
        label_path = labels_dir / f"{Path(image_id).stem}.txt"
        _write_label_file(label_path, (row for _, row in rows.iterrows()), class_to_idx)

    split_dir = args.output_dir / "splits"
    _write_split_lists(train_items, val_items, split_dir)

    names_path = args.output_dir / "classes.txt"
    names_path.write_text("\n".join(class_names))
    print(f"[classes] wrote {names_path}")

    print(f"[done] train images: {len(train_items)} | val images: {len(val_items)}")


if __name__ == "__main__":
    main()
