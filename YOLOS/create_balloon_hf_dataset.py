#!/usr/bin/env python
"""Convert the Matterport Balloon dataset to a Hub-compatible imagefolder dataset.

The script downloads the original Balloon dataset, converts the VIA polygon
annotations to COCO bounding boxes, creates a Hugging Face `imagefolder`
dataset with `metadata.jsonl` files, and can optionally upload the result to a
dataset repository such as `nielsr/balloon-dataset`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image

BALLOON_ZIP_URL = (
    "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/"
    "balloon_dataset.zip"
)
CATEGORIES = [{"id": 0, "name": "balloon", "supercategory": "object"}]
SPLIT_MAP = {"train": "train", "val": "validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the imagefolder dataset will be written.",
    )
    parser.add_argument(
        "--repo-id",
        default="nielsr/balloon-dataset",
        help="Dataset repo to create/update on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the generated dataset folder to the Hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private when pushing to the Hub.",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep the downloaded zip and extracted raw files under output-dir/raw.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with urllib.request.urlopen(url) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def unzip_file(zip_path: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def polygon_area(points_x: list[float], points_y: list[float]) -> float:
    if len(points_x) < 3 or len(points_y) < 3:
        return 0.0
    area = 0.0
    for index in range(len(points_x)):
        next_index = (index + 1) % len(points_x)
        area += points_x[index] * points_y[next_index]
        area -= points_y[index] * points_x[next_index]
    return abs(area) / 2.0


def normalize_regions(regions: Any) -> list[dict[str, Any]]:
    if isinstance(regions, list):
        return regions
    if isinstance(regions, dict):
        return list(regions.values())
    return []


def build_objects(regions: Any, annotation_id_start: int) -> tuple[dict[str, list[Any]], list[dict[str, Any]], int]:
    objects = {
        "id": [],
        "bbox": [],
        "category": [],
        "area": [],
    }
    coco_annotations: list[dict[str, Any]] = []
    annotation_id = annotation_id_start

    for region in normalize_regions(regions):
        shape = region.get("shape_attributes", {})
        all_points_x = shape.get("all_points_x", [])
        all_points_y = shape.get("all_points_y", [])
        if not all_points_x or not all_points_y:
            continue

        x_min = float(min(all_points_x))
        y_min = float(min(all_points_y))
        width = float(max(all_points_x) - min(all_points_x))
        height = float(max(all_points_y) - min(all_points_y))
        area = polygon_area(all_points_x, all_points_y)
        segmentation = [
            [coord for point in zip(all_points_x, all_points_y) for coord in point]
        ]

        objects["id"].append(annotation_id)
        objects["bbox"].append([x_min, y_min, width, height])
        objects["category"].append("balloon")
        objects["area"].append(area)
        coco_annotations.append(
            {
                "id": annotation_id,
                "bbox": [x_min, y_min, width, height],
                "area": area,
                "category_id": 0,
                "iscrowd": 0,
                "segmentation": segmentation,
            }
        )
        annotation_id += 1

    return objects, coco_annotations, annotation_id


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_dataset_card(path: Path) -> None:
    content = """---
task_categories:
- object-detection
pretty_name: Balloon Dataset
---

# Balloon Dataset

This dataset is derived from the original Balloon dataset released in the
[Matterport Mask R-CNN repository](https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip).

It is stored in the Hugging Face `imagefolder` object-detection layout, with one
`metadata.jsonl` file per split, so it can be loaded directly with:

```python
from datasets import load_dataset

dataset = load_dataset("nielsr/balloon-dataset")
```
"""
    path.write_text(content, encoding="utf-8")


def convert_split(
    raw_split_dir: Path,
    output_split_dir: Path,
    coco_path: Path,
    image_id_start: int,
    annotation_id_start: int,
) -> tuple[int, int]:
    with (raw_split_dir / "via_region_data.json").open(encoding="utf-8") as file:
        via_annotations = json.load(file)

    output_split_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_split_dir / "metadata.jsonl"
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    metadata_lines: list[str] = []

    image_id = image_id_start
    annotation_id = annotation_id_start

    for item in via_annotations.values():
        file_name = item["filename"]
        source_image_path = raw_split_dir / file_name
        destination_image_path = output_split_dir / file_name
        shutil.copy2(source_image_path, destination_image_path)

        with Image.open(source_image_path) as image:
            width, height = image.size

        objects, image_annotations, annotation_id = build_objects(
            item.get("regions", []),
            annotation_id,
        )
        for annotation in image_annotations:
            annotation["image_id"] = image_id

        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )
        coco_annotations.extend(image_annotations)

        metadata_lines.append(
            json.dumps(
                {
                    "file_name": file_name,
                    "image_id": image_id,
                    "width": width,
                    "height": height,
                    "objects": {
                        "id": objects["id"],
                        "bbox": objects["bbox"],
                        "categories": [0] * len(objects["bbox"]),
                        "category": objects["category"],
                        "area": objects["area"],
                    },
                }
            )
        )
        image_id += 1

    metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    write_json(
        coco_path,
        {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": CATEGORIES,
        },
    )
    return image_id, annotation_id


def create_dataset(raw_root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id = 0
    annotation_id = 0

    for raw_split, hf_split in SPLIT_MAP.items():
        image_id, annotation_id = convert_split(
            raw_split_dir=raw_root / "balloon" / raw_split,
            output_split_dir=output_dir / hf_split,
            coco_path=output_dir / "annotations" / f"{hf_split}.json",
            image_id_start=image_id,
            annotation_id_start=annotation_id,
        )

    write_json(
        output_dir / "categories.json",
        {"id2label": {"0": "balloon"}, "label2id": {"balloon": 0}},
    )
    write_dataset_card(output_dir / "README.md")


def validate_dataset(output_dir: Path):
    dataset = load_dataset("imagefolder", data_dir=str(output_dir))
    for split in ["train", "validation"]:
        sample = dataset[split][0]
        bbox_count = len(sample["objects"]["bbox"])
        print(f"{split}: {len(dataset[split])} examples")
        print(f"  first example: image_id={sample['image_id']} boxes={bbox_count}")
    return dataset


def push_to_hub(dataset, repo_id: str, private: bool) -> None:
    dataset.push_to_hub(repo_id, private=private)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        if args.keep_downloads:
            raw_parent = output_dir / "raw"
        else:
            raw_parent = temp_dir / "raw"

        zip_path = raw_parent / "balloon_dataset.zip"
        raw_extract_dir = raw_parent / "extracted"

        print(f"Downloading {BALLOON_ZIP_URL}")
        download_file(BALLOON_ZIP_URL, zip_path)

        print(f"Extracting to {raw_extract_dir}")
        unzip_file(zip_path, raw_extract_dir)

        print(f"Creating dataset under {output_dir}")
        create_dataset(raw_extract_dir, output_dir)

        print("Validating imagefolder dataset")
        dataset = validate_dataset(output_dir)

        if args.push_to_hub:
            print(f"Pushing dataset to {args.repo_id}")
            push_to_hub(dataset=dataset, repo_id=args.repo_id, private=args.private)
            print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
