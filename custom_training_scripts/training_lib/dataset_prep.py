"""Generic dataset validate/split/COCO-convert used by run_training.py.

Understands exactly two on-disk shapes under ``<data_root>/<dataset_name>/``:

1. Already split: ``images/{train,val[,test]}`` + a matching
   ``annotations/instances_<split>2017.json`` for each. Nothing to do.
2. Flat: ``images/`` + ``labels/`` (YOLO ``.txt`` labels), no split
   subdirectories. Every image must have a same-stem label file; the dataset
   is then shuffled and split by ``split_ratio`` into
   ``images/{train,val,test}`` + ``labels/{train,val,test}``.

Either shape is missing YOLO labels for a split that has no COCO annotations,
they're generated from ``classes.txt`` + the ``labels/<split>`` files.

Anything else (nested raw exports, CVAT XML, etc.) is out of scope and raises
a clear ``DatasetError`` instead of guessing.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SPLITS = ("train", "val", "test")
REQUIRED_SPLITS = ("train", "val")


class DatasetError(RuntimeError):
    pass


@dataclass
class DatasetInfo:
    name: str
    host_path: Path
    container_path: str
    class_names: list[str]
    num_classes: int
    input_size: tuple[int, int]  # (height, width), both multiples of 32


def _find_images(dir_path: Path) -> dict[str, Path]:
    if not dir_path.is_dir():
        return {}
    return {
        p.stem: p
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


def _find_labels(dir_path: Path) -> dict[str, Path]:
    if not dir_path.is_dir():
        return {}
    return {p.stem: p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}


def _load_class_names(dataset_root: Path) -> list[str]:
    classes_file = dataset_root / "classes.txt"
    if not classes_file.is_file():
        raise DatasetError(f"{dataset_root}: no classes.txt — can't determine class names/count.")
    names = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    if not names:
        raise DatasetError(f"{classes_file}: empty.")
    return names


def _match_images_labels(images_dir: Path, labels_dir: Path) -> dict[str, Path]:
    images = _find_images(images_dir)
    labels = _find_labels(labels_dir)
    missing = sorted(stem for stem in images if stem not in labels)
    if missing:
        raise DatasetError(
            f"{images_dir}: {len(missing)} image(s) have no matching label in {labels_dir}, "
            f"e.g. {missing[:5]} — fix labeling before this dataset can be used."
        )
    return images


def _split_is_annotated(dataset_root: Path, split: str) -> bool:
    ann_path = dataset_root / "annotations" / f"instances_{split}2017.json"
    images_dir = dataset_root / "images" / split
    if not ann_path.is_file() or not images_dir.is_dir():
        return False
    try:
        data = json.loads(ann_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not data.get("images"):
        return False
    json_names = {Path(im["file_name"]).name for im in data["images"]}
    disk_names = {p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    return json_names == disk_names


def _yolo_to_coco(dataset_root: Path, split: str, class_names: list[str]) -> None:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    images = _find_images(images_dir)

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    coco_images = []
    coco_annotations = []
    ann_id = 1
    for image_id, (stem, img_path) in enumerate(sorted(images.items()), start=1):
        with Image.open(img_path) as im:
            width, height = im.size
        coco_images.append(
            {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
        )
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.is_file():
            continue
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_idx = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:5])
            box_w, box_h = w * width, h * height
            x, y = xc * width - box_w / 2, yc * height - box_h / 2
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_idx + 1,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    ann_dir = dataset_root / "annotations"
    ann_dir.mkdir(exist_ok=True)
    out_path = ann_dir / f"instances_{split}2017.json"
    out_path.write_text(
        json.dumps({"images": coco_images, "annotations": coco_annotations, "categories": categories})
    )
    print(f"[dataset] wrote {out_path} ({len(coco_images)} images, {len(coco_annotations)} boxes)")


def _ensure_split_annotated(
    dataset_root: Path, split: str, class_names: list[str], force_reformat: bool
) -> bool:
    """Returns True if images/<split> exists (annotated or not). Raises on bad state."""
    images_dir = dataset_root / "images" / split
    if not images_dir.is_dir():
        return False
    if not force_reformat and _split_is_annotated(dataset_root, split):
        return True
    labels_dir = dataset_root / "labels" / split
    if not labels_dir.is_dir():
        raise DatasetError(
            f"{dataset_root}: '{split}' has no valid annotations/instances_{split}2017.json "
            f"and no labels/{split} to regenerate it from."
        )
    _match_images_labels(images_dir, labels_dir)
    print(f"[dataset] {dataset_root.name}: (re)generating COCO annotations for '{split}'")
    _yolo_to_coco(dataset_root, split, class_names)
    return True


def _do_split(dataset_root: Path, ratios: tuple[float, float, float], seed: int) -> None:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    images = _match_images_labels(images_dir, labels_dir)
    labels = _find_labels(labels_dir)

    stems = sorted(images.keys())
    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    split_of = {}
    for stem in stems[:n_train]:
        split_of[stem] = "train"
    for stem in stems[n_train : n_train + n_val]:
        split_of[stem] = "val"
    for stem in stems[n_train + n_val :]:
        split_of[stem] = "test"

    for split in SPLITS:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    for stem, split in split_of.items():
        images[stem].rename(images_dir / split / images[stem].name)
        labels[stem].rename(labels_dir / split / labels[stem].name)
        counts[split] += 1

    print(f"[dataset] split {n} images -> {counts}")


def _detect_input_size(dataset_root: Path, sample_size: int = 50) -> tuple[int, int]:
    images_dir = dataset_root / "images" / "train"
    images = list(_find_images(images_dir).values())
    if not images:
        raise DatasetError(f"{images_dir}: no images found to auto-detect resolution from.")
    sample = images[:]
    random.Random(42).shuffle(sample)
    sample = sample[:sample_size]

    widths, heights = [], []
    for p in sample:
        with Image.open(p) as im:
            w, h = im.size
        widths.append(w)
        heights.append(h)

    def pct95(values: list[int]) -> int:
        values = sorted(values)
        return values[min(len(values) - 1, int(len(values) * 0.95))]

    def round_up_32(v: int) -> int:
        return ((v + 31) // 32) * 32

    return round_up_32(pct95(heights)), round_up_32(pct95(widths))


def ensure_dataset(
    data_root: Path,
    dataset_name: str,
    split_ratio: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    force_reformat: bool = False,
) -> DatasetInfo:
    dataset_root = data_root / dataset_name
    if not dataset_root.is_dir():
        raise DatasetError(f"{dataset_root}: dataset directory does not exist.")

    class_names = _load_class_names(dataset_root)

    already_split = (dataset_root / "images" / "train").is_dir()
    if not already_split:
        flat_images = dataset_root / "images"
        flat_labels = dataset_root / "labels"
        if not flat_images.is_dir() or not flat_labels.is_dir():
            raise DatasetError(
                f"{dataset_root}: expected either images/{{train,val}} (already split) or a "
                f"flat images/+labels/ pair to split — found neither."
            )
        if any(p.is_dir() for p in flat_images.iterdir()):
            raise DatasetError(
                f"{flat_images}: has subdirectories but none named 'train' — unrecognized "
                f"layout, not touching it."
            )
        print(f"[dataset] {dataset_name}: flat images/+labels/ found, splitting by {split_ratio}")
        _do_split(dataset_root, split_ratio, seed)

    for split in REQUIRED_SPLITS:
        if not _ensure_split_annotated(dataset_root, split, class_names, force_reformat):
            raise DatasetError(f"{dataset_root}: images/{split} missing after the split step.")
    _ensure_split_annotated(dataset_root, "test", class_names, force_reformat)  # optional

    height, width = _detect_input_size(dataset_root)

    return DatasetInfo(
        name=dataset_name,
        host_path=dataset_root,
        container_path=f"/data/{dataset_name}",
        class_names=class_names,
        num_classes=len(class_names),
        input_size=(height, width),
    )
