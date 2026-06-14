#!/usr/bin/env python3
"""
Organize blurred crop datasets into MobileNetV3 class folders.

Expects a local layout with datasets_blur/, Nouvelles/labels/, etc.
See the original script context in the legacy 03_inprocess/dataset folder.
"""

import shutil
from collections import defaultdict
from pathlib import Path


CLASS_MAPPING = {
    0: "male",
    1: "femelle",
}


def find_label_file(
    image_name: str,
    bdd_nouvelles_labels_dir: Path,
    nouvelles_labels_dir: Path,
) -> Path | None:
    """Find the YOLO label file for a cropped image name."""
    base_name = image_name.replace(".png", "")
    if "_crop" in base_name:
        base_without_crop = base_name.rsplit("_crop", 1)[0]
    else:
        base_without_crop = base_name

    if bdd_nouvelles_labels_dir.exists():
        label_path = bdd_nouvelles_labels_dir / f"{base_without_crop}.txt"
        if label_path.exists():
            return label_path

    label_path = nouvelles_labels_dir / f"{base_without_crop}.txt"
    if label_path.exists():
        return label_path

    return None


def get_class_from_label(label_path: Path | None) -> str | None:
    """Read the first object class from a YOLO label file."""
    if not label_path or not label_path.exists():
        return None

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return None
            first_line = lines[0].strip()
            if not first_line:
                return None
            parts = first_line.split()
            if len(parts) < 1:
                return None
            class_id = int(parts[0])
            return CLASS_MAPPING.get(class_id)
    except OSError as e:
        print(f"Error reading {label_path}: {e}")
        return None


def organize_dataset(base_dir: Path) -> dict:
    """
    Copy images from datasets_blur into train/val class folders.

    base_dir should contain datasets_blur/, Nouvelles/labels/, etc.
    """
    datasets_blur_dir = base_dir / "datasets_blur"
    bdd_nouvelles_labels_dir = base_dir / "BDD(1)" / "Nouvelles" / "labels"
    nouvelles_labels_dir = base_dir / "Nouvelles" / "labels"
    output_dir = base_dir / "dataset"

    stats: dict = defaultdict(int)

    for split_name, output_split in [("train", "train"), ("test", "val")]:
        split_dir = datasets_blur_dir / split_name

        for blur_type in ["blur", "not_blur"]:
            images_dir = split_dir / blur_type
            if not images_dir.exists():
                print(f"Skipping missing directory: {images_dir}")
                continue

            print(f"\nProcessing {split_name}/{blur_type}...")

            for image_file in images_dir.glob("*.png"):
                label_path = find_label_file(
                    image_file.name,
                    bdd_nouvelles_labels_dir,
                    nouvelles_labels_dir,
                )
                if not label_path:
                    stats["sans_label"] += 1
                    continue

                class_name = get_class_from_label(label_path)
                if not class_name:
                    stats["sans_classe"] += 1
                    continue

                dest_dir = output_dir / output_split / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_file, dest_dir / image_file.name)
                stats[f"{output_split}_{class_name}"] += 1

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Organize blur dataset for MobileNetV3")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing datasets_blur/ and label folders",
    )
    args = parser.parse_args()

    print(f"Base directory: {args.base_dir}")
    stats = organize_dataset(args.base_dir)

    print("\n" + "=" * 50)
    print("FINAL STATS")
    print("=" * 50)
    for key in ["train_male", "train_femelle", "val_male", "val_femelle"]:
        print(f"{key}: {stats[key]}")
    print(f"Missing labels: {stats['sans_label']}")
    print(f"Missing class: {stats['sans_classe']}")


if __name__ == "__main__":
    main()
