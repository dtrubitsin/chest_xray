#!/usr/bin/env python3
"""
Visualize bounding boxes from CSV and dataset to check for mismatches.

Loads bounding boxes from both the original CSV file and the YOLO label files,
then displays them side-by-side or overlaid on the same image to identify
discrepancies that could cause training issues.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_IMG_SIZE = 1024
BBOX_DEFAULT = Path("data/raw/BBox_List_2017.csv")
IMAGES_DIR = Path("data/images")
LABELS_DIR = Path("data/labels")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize bboxes from CSV and dataset.")
    parser.add_argument("--bbox-csv", type=Path, default=BBOX_DEFAULT, help="Path to BBox_List_2017.csv.")
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR, help="Directory with images.")
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR, help="Directory with YOLO label files.")
    parser.add_argument("--image-name", type=str, default=None, help="Specific image to visualize (e.g., '00013118_008.png').")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of random images to visualize (if --image-name not provided).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--overlay", action="store_true", help="Overlay both bboxes on same image instead of side-by-side.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Save visualizations to directory instead of displaying.")
    return parser.parse_args()


def load_csv_bboxes(csv_path: Path) -> dict[str, List[Tuple[int, int, int, int, str]]]:
    """
    Load bounding boxes from CSV file.

    :param csv_path: Path to BBox_List_2017.csv.
    :return: Dictionary mapping image names to list of (x, y, w, h, class_name) tuples.
    """
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required = {"Image Index", "Finding Label", "Bbox [x", "y", "w", "h]"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns in bbox CSV: {missing}")
    
    # Convert to numeric
    numeric_cols = ["Bbox [x", "y", "w", "h]"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols)
    
    # Group by image
    bboxes = {}
    for idx, row in df.iterrows():
        image_name = row["Image Index"]
        x = row["Bbox [x"]
        y = row["y"]
        w = row["w"]
        h = row["h]"]
        class_name = row["Finding Label"].strip()
        
        # Check for invalid values
        if any(not np.isfinite(val) for val in [x, y, w, h]):
            print(f"  [WARNING] Invalid values in CSV for {image_name} row {idx}: NaN or Inf detected")
            continue
        
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Validate coordinates
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            print(f"  [WARNING] Invalid bbox in CSV for {image_name} row {idx}: x={x}, y={y}, w={w}, h={h}")
        
        if image_name not in bboxes:
            bboxes[image_name] = []
        bboxes[image_name].append((x, y, w, h, class_name))
    
    return bboxes


def load_yolo_bboxes(label_path: Path, img_size: int = DEFAULT_IMG_SIZE) -> List[Tuple[int, int, int, int, int]]:
    """
    Load bounding boxes from YOLO label file.

    :param label_path: Path to YOLO label .txt file.
    :param img_size: Image size for denormalization. Default: 1024.
    :return: List of (x, y, w, h, class_id) tuples in pixel coordinates.
    """
    if not label_path.exists():
        return []
    
    bboxes = []
    for line_num, line in enumerate(label_path.read_text().strip().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        try:
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
            
            # Check for invalid values
            if any(not np.isfinite(val) for val in [x_center_norm, y_center_norm, w_norm, h_norm]):
                print(f"  [WARNING] Invalid values in {label_path.name} line {line_num}: NaN or Inf detected")
                continue
            
            if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1):
                print(f"  [WARNING] Invalid center coordinates in {label_path.name} line {line_num}: "
                      f"x_center={x_center_norm}, y_center={y_center_norm}")
            
            if not (0 < w_norm <= 1 and 0 < h_norm <= 1):
                print(f"  [WARNING] Invalid dimensions in {label_path.name} line {line_num}: "
                      f"w={w_norm}, h={h_norm}")
            
            # Convert from YOLO format (normalized center, width, height) to pixel coordinates (x, y, w, h)
            x_center = x_center_norm * img_size
            y_center = y_center_norm * img_size
            w = w_norm * img_size
            h = h_norm * img_size
            
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)
            w = int(w)
            h = int(h)
            
            # Validate converted coordinates
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                print(f"  [WARNING] Invalid bbox in {label_path.name} line {line_num}: "
                      f"x={x}, y={y}, w={w}, h={h}")
            
            if x + w > img_size or y + h > img_size:
                print(f"  [WARNING] Bbox out of bounds in {label_path.name} line {line_num}: "
                      f"x={x}, y={y}, w={w}, h={h}, img_size={img_size}")
            
            bboxes.append((x, y, w, h, class_id))
        except (ValueError, IndexError) as e:
            print(f"  [ERROR] Failed to parse line {line_num} in {label_path.name}: {e}")
            continue
    
    return bboxes


def load_class_names(classes_path: Path | None = None) -> List[str]:
    """
    Load class names from classes.txt or use defaults.

    :param classes_path: Path to classes.txt file.
    :return: List of class names.
    """
    if classes_path and classes_path.exists():
        return [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    
    # Default classes
    return [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltrate",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
    ]


def draw_bboxes_on_image(
    img: np.ndarray,
    bboxes: List[Tuple[int, int, int, int, str | int]],
    color: Tuple[int, int, int],
    label_prefix: str = "",
    class_names: List[str] | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    :param img: Image array (H, W, 3).
    :param bboxes: List of (x, y, w, h, class_id_or_name) tuples.
    :param color: RGB color tuple for bbox.
    :param label_prefix: Prefix for labels.
    :param class_names: Optional list of class names for class_id lookup.
    :return: Image with drawn bboxes.
    """
    img_copy = img.copy()
    
    for x, y, w, h, class_info in bboxes:
        # Get class name
        if isinstance(class_info, int):
            class_name = class_names[class_info] if class_names and class_info < len(class_names) else f"Class{class_info}"
        else:
            class_name = class_info
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{label_prefix}{class_name}" if label_prefix else class_name
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y, label_size[1] + 5)
        cv2.rectangle(img_copy, (x, label_y - label_size[1] - 5), (x + label_size[0], label_y), color, -1)
        cv2.putText(img_copy, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy


def compare_bboxes(
    csv_bboxes: List[Tuple[int, int, int, int, str]],
    yolo_bboxes: List[Tuple[int, int, int, int, int]],
    class_names: List[str],
    tolerance: int = 2,
) -> None:
    """
    Compare CSV and YOLO bboxes and report mismatches.

    :param csv_bboxes: Bboxes from CSV in (x, y, w, h, class_name) format.
    :param yolo_bboxes: Bboxes from YOLO labels in (x, y, w, h, class_id) format.
    :param class_names: List of class names.
    :param tolerance: Pixel tolerance for coordinate comparison.
    """
    if len(csv_bboxes) != len(yolo_bboxes):
        print(f"  [MISMATCH] Different number of bboxes: CSV={len(csv_bboxes)}, YOLO={len(yolo_bboxes)}")
        return
    
    # Try to match bboxes by position and class
    csv_matched = [False] * len(csv_bboxes)
    yolo_matched = [False] * len(yolo_bboxes)
    
    for i, (cx, cy, cw, ch, cclass) in enumerate(csv_bboxes):
        for j, (yx, yy, yw, yh, yclass_id) in enumerate(yolo_bboxes):
            if yolo_matched[j]:
                continue
            
            yclass = class_names[yclass_id] if yclass_id < len(class_names) else f"Class{yclass_id}"
            
            # Check if classes match
            if cclass != yclass:
                continue
            
            # Check if positions match (within tolerance)
            if (abs(cx - yx) <= tolerance and abs(cy - yy) <= tolerance and
                abs(cw - yw) <= tolerance and abs(ch - yh) <= tolerance):
                csv_matched[i] = True
                yolo_matched[j] = True
                break
    
    # Report unmatched bboxes
    for i, matched in enumerate(csv_matched):
        if not matched:
            cx, cy, cw, ch, cclass = csv_bboxes[i]
            print(f"  [MISMATCH] CSV bbox {i} not matched: {cclass} at ({cx}, {cy}, {cw}, {ch})")
    
    for j, matched in enumerate(yolo_matched):
        if not matched:
            yx, yy, yw, yh, yclass_id = yolo_bboxes[j]
            yclass = class_names[yclass_id] if yclass_id < len(class_names) else f"Class{yclass_id}"
            print(f"  [MISMATCH] YOLO bbox {j} not matched: {yclass} at ({yx}, {yy}, {yw}, {yh})")


def visualize_image(
    image_path: Path,
    csv_bboxes: List[Tuple[int, int, int, int, str]],
    yolo_bboxes: List[Tuple[int, int, int, int, int]],
    class_names: List[str],
    overlay: bool = False,
) -> np.ndarray:
    """
    Visualize bounding boxes for a single image.

    :param image_path: Path to image file.
    :param csv_bboxes: Bboxes from CSV in (x, y, w, h, class_name) format.
    :param yolo_bboxes: Bboxes from YOLO labels in (x, y, w, h, class_id) format.
    :param class_names: List of class names.
    :param overlay: Whether to overlay both sets on same image.
    :return: Visualization image array.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if overlay:
        # Overlay both sets on same image
        img = draw_bboxes_on_image(img, csv_bboxes, (255, 0, 0), "CSV: ", class_names)
        img = draw_bboxes_on_image(img, yolo_bboxes, (0, 255, 0), "YOLO: ", class_names)
    else:
        # Side-by-side comparison
        img_csv = draw_bboxes_on_image(img.copy(), csv_bboxes, (255, 0, 0), "", class_names)
        img_yolo = draw_bboxes_on_image(img.copy(), yolo_bboxes, (0, 255, 0), "", class_names)
        
        # Concatenate horizontally
        img = np.hstack([img_csv, img_yolo])
    
    return img


def main() -> None:
    """
    Main entry point for bbox visualization.
    """
    args = parse_args()
    
    # Load data
    print(f"[info] Loading CSV bboxes from {args.bbox_csv}...")
    csv_bboxes_dict = load_csv_bboxes(args.bbox_csv)
    print(f"[info] Found {len(csv_bboxes_dict)} images with annotations in CSV")
    
    class_names = load_class_names(Path("data/classes.txt"))
    print(f"[info] Loaded {len(class_names)} classes")
    
    # Determine which images to visualize
    if args.image_name:
        image_names = [args.image_name]
    else:
        # Get images that exist in both CSV and have label files
        available_images = []
        for img_name in csv_bboxes_dict.keys():
            img_path = args.images_dir / img_name
            label_path = args.labels_dir / f"{Path(img_name).stem}.txt"
            if img_path.exists() and label_path.exists():
                available_images.append(img_name)
        
        if not available_images:
            raise ValueError("No images found with both CSV annotations and label files")
        
        print(f"[info] Found {len(available_images)} images with both CSV and label files")
        
        # Sample random images
        rng = random.Random(args.seed)
        image_names = rng.sample(available_images, min(args.num_samples, len(available_images)))
        print(f"[info] Sampling {len(image_names)} images for visualization")
    
    # Visualize each image
    for img_name in image_names:
        print(f"[info] Processing {img_name}...")
        
        img_path = args.images_dir / img_name
        if not img_path.exists():
            print(f"[warn] Image not found: {img_path}, skipping")
            continue
        
        label_path = args.labels_dir / f"{Path(img_name).stem}.txt"
        
        # Get bboxes
        csv_bboxes = csv_bboxes_dict.get(img_name, [])
        yolo_bboxes = load_yolo_bboxes(label_path) if label_path.exists() else []
        
        print(f"  CSV bboxes: {len(csv_bboxes)}")
        print(f"  YOLO bboxes: {len(yolo_bboxes)}")
        
        # Check for mismatches
        if len(csv_bboxes) != len(yolo_bboxes):
            print(f"  [WARNING] Mismatch: CSV has {len(csv_bboxes)} bboxes, YOLO has {len(yolo_bboxes)}")
        
        # Compare bboxes
        compare_bboxes(csv_bboxes, yolo_bboxes, class_names)
        
        # Visualize
        try:
            vis_img = visualize_image(img_path, csv_bboxes, yolo_bboxes, class_names, overlay=args.overlay)
            
            if args.output_dir:
                # Save to file
                args.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.output_dir / f"{Path(img_name).stem}_bboxes.png"
                plt.figure(figsize=(16, 8) if not args.overlay else (8, 8))
                plt.imshow(vis_img)
                plt.axis("off")
                if not args.overlay:
                    plt.text(vis_img.shape[1] // 4, 20, "CSV Bboxes (Red)", color="red", fontsize=12, weight="bold")
                    plt.text(3 * vis_img.shape[1] // 4, 20, "YOLO Bboxes (Green)", color="green", fontsize=12, weight="bold")
                else:
                    plt.text(10, 20, "CSV (Red) | YOLO (Green)", color="white", fontsize=12, weight="bold", 
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved to {output_path}")
            else:
                # Display
                plt.figure(figsize=(16, 8) if not args.overlay else (8, 8))
                plt.imshow(vis_img)
                plt.axis("off")
                if not args.overlay:
                    plt.text(vis_img.shape[1] // 4, 20, "CSV Bboxes (Red)", color="red", fontsize=12, weight="bold")
                    plt.text(3 * vis_img.shape[1] // 4, 20, "YOLO Bboxes (Green)", color="green", fontsize=12, weight="bold")
                else:
                    plt.text(10, 20, "CSV (Red) | YOLO (Green)", color="white", fontsize=12, weight="bold",
                            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
                plt.title(f"{img_name}", fontsize=14, weight="bold")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"  [ERROR] Failed to visualize {img_name}: {e}")
            continue
    
    print("[done] Visualization complete")


if __name__ == "__main__":
    main()
