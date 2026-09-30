from collections import defaultdict
from configparser import ConfigParser
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import random
import shutil

import cv2
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MOT17_TRAIN_SEQUENCES = ("02", "04", "05", "10", "11")
MOT17_VALIDATION_SEQUENCES = ("09", "13")


@dataclass
class DatasetReport:
    dataset: str
    image_count: int = 0
    labeled_image_count: int = 0
    empty_image_count: int = 0
    box_count: int = 0
    ignored_box_count: int = 0
    rejected_box_count: int = 0
    clipped_box_count: int = 0
    missing_image_count: int = 0
    hardlink_count: int = 0
    copy_count: int = 0
    missing_images: list[str] = field(default_factory=list)
    validation: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def convert_mot17(
    source_root: Path,
    output_root: Path,
    link_mode: str = "hardlink",
    train_sequences: tuple[str, ...] = MOT17_TRAIN_SEQUENCES,
    validation_sequences: tuple[str, ...] = MOT17_VALIDATION_SEQUENCES,
    preview_count: int = 12,
) -> DatasetReport:
    _validate_link_mode(link_mode)
    report = DatasetReport(dataset="MOT17")
    split_sequences = {
        "train": train_sequences,
        "val": validation_sequences,
    }

    for split, sequence_numbers in split_sequences.items():
        for sequence_number in sequence_numbers:
            sequence_path = (
                source_root / "train" / f"MOT17-{sequence_number}-FRCNN"
            )
            info = _load_mot_sequence_info(sequence_path / "seqinfo.ini")
            annotations = _load_mot_person_boxes(sequence_path / "gt" / "gt.txt")
            for frame_id in range(1, info["length"] + 1):
                source_image = (
                    sequence_path
                    / info["image_dir"]
                    / f"{frame_id:06d}{info['extension']}"
                )
                output_stem = f"MOT17-{sequence_number}_{frame_id:06d}"
                output_image = (
                    output_root / "images" / split / f"{output_stem}{info['extension']}"
                )
                output_label = output_root / "labels" / split / f"{output_stem}.txt"
                if not source_image.exists():
                    _record_missing_image(report, source_image)
                    continue

                materialization = _materialize_image(
                    source_image,
                    output_image,
                    link_mode,
                )
                _record_materialization(report, materialization)
                label_rows = []
                for box in annotations.get(frame_id, []):
                    converted, clipped = _convert_box_to_yolo(
                        box,
                        width=info["width"],
                        height=info["height"],
                    )
                    if converted is None:
                        report.rejected_box_count += 1
                        continue
                    if clipped:
                        report.clipped_box_count += 1
                    label_rows.append(_format_yolo_row(converted))

                _write_labels(output_label, label_rows)
                report.image_count += 1
                report.box_count += len(label_rows)
                if label_rows:
                    report.labeled_image_count += 1
                else:
                    report.empty_image_count += 1

    _write_dataset_yaml(output_root, "MOT17 pedestrians")
    report.validation = validate_yolo_dataset(output_root)
    _write_previews(output_root, preview_count)
    _write_report(output_root / "report.json", report.to_dict())
    return report


def convert_crowdhuman(
    source_root: Path,
    output_root: Path,
    link_mode: str = "hardlink",
    preview_count: int = 12,
) -> DatasetReport:
    _validate_link_mode(link_mode)
    report = DatasetReport(dataset="CrowdHuman")
    split_config = {
        "train": (
            source_root / "annotation_train.odgt",
            (
                source_root / "CrowdHuman_train01" / "Images",
                source_root / "CrowdHuman_train02" / "Images",
                source_root / "CrowdHuman_train03" / "Images",
            ),
        ),
        "val": (
            source_root / "annotation_val.odgt",
            (source_root / "CrowdHuman_val" / "Images",),
        ),
    }

    for split, (annotation_path, image_directories) in split_config.items():
        image_lookup = _build_image_lookup(image_directories)
        with annotation_path.open(encoding="utf-8") as annotation_file:
            for line in annotation_file:
                annotation = json.loads(line)
                image_id = annotation["ID"]
                source_image = image_lookup.get(image_id)
                if source_image is None:
                    _record_missing_image(report, Path(f"{image_id}.jpg"))
                    continue

                image = cv2.imread(str(source_image))
                if image is None:
                    _record_missing_image(report, source_image)
                    continue
                height, width = image.shape[:2]
                output_image = output_root / "images" / split / source_image.name
                output_label = output_root / "labels" / split / f"{image_id}.txt"
                materialization = _materialize_image(
                    source_image,
                    output_image,
                    link_mode,
                )
                _record_materialization(report, materialization)

                label_rows = []
                for ground_truth in annotation.get("gtboxes", []):
                    if (
                        ground_truth.get("tag") != "person"
                        or ground_truth.get("extra", {}).get("ignore", 0) == 1
                    ):
                        report.ignored_box_count += 1
                        continue
                    full_box = ground_truth.get("fbox")
                    if not full_box or len(full_box) != 4:
                        report.rejected_box_count += 1
                        continue
                    converted, clipped = _convert_box_to_yolo(
                        tuple(float(value) for value in full_box),
                        width=width,
                        height=height,
                    )
                    if converted is None:
                        report.rejected_box_count += 1
                        continue
                    if clipped:
                        report.clipped_box_count += 1
                    label_rows.append(_format_yolo_row(converted))

                _write_labels(output_label, label_rows)
                report.image_count += 1
                report.box_count += len(label_rows)
                if label_rows:
                    report.labeled_image_count += 1
                else:
                    report.empty_image_count += 1

    _write_dataset_yaml(output_root, "CrowdHuman pedestrians")
    report.validation = validate_yolo_dataset(output_root)
    _write_previews(output_root, preview_count)
    _write_report(output_root / "report.json", report.to_dict())
    return report


def create_combined_dataset_yaml(
    mot17_root: Path,
    crowdhuman_root: Path,
    output_path: Path,
) -> None:
    data = {
        "train": [
            str((mot17_root / "images" / "train").resolve()),
            str((crowdhuman_root / "images" / "train").resolve()),
        ],
        "val": [
            str((mot17_root / "images" / "val").resolve()),
            str((crowdhuman_root / "images" / "val").resolve()),
        ],
        "names": {0: "person"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def validate_yolo_dataset(dataset_root: Path) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    duplicate_label_count = 0
    image_count = 0
    label_count = 0
    box_count = 0

    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        } if image_dir.exists() else {}
        labels = {
            path.stem: path
            for path in label_dir.glob("*.txt")
        } if label_dir.exists() else {}
        image_count += len(images)
        label_count += len(labels)

        for missing_label in sorted(set(images) - set(labels)):
            errors.append(f"{split}: missing label for image {missing_label}")
        for missing_image in sorted(set(labels) - set(images)):
            errors.append(f"{split}: missing image for label {missing_image}")

        for stem, label_path in labels.items():
            seen_rows: set[str] = set()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped in seen_rows:
                    duplicate_label_count += 1
                    warnings.append(f"{split}/{stem}:{line_number} duplicate row")
                seen_rows.add(stripped)
                columns = stripped.split()
                if len(columns) != 5:
                    errors.append(f"{split}/{stem}:{line_number} invalid column count")
                    continue
                try:
                    class_id = int(columns[0])
                    values = [float(value) for value in columns[1:]]
                except ValueError:
                    errors.append(f"{split}/{stem}:{line_number} invalid numeric value")
                    continue
                if class_id != 0:
                    errors.append(f"{split}/{stem}:{line_number} invalid class {class_id}")
                if not all(math.isfinite(value) for value in values):
                    errors.append(f"{split}/{stem}:{line_number} non-finite value")
                    continue
                center_x, center_y, width, height = values
                if width <= 0 or height <= 0:
                    errors.append(f"{split}/{stem}:{line_number} non-positive box")
                if not all(0.0 <= value <= 1.0 for value in values):
                    errors.append(f"{split}/{stem}:{line_number} value outside [0, 1]")
                if (
                    center_x - width / 2 < -1e-6
                    or center_x + width / 2 > 1.0 + 1e-6
                    or center_y - height / 2 < -1e-6
                    or center_y + height / 2 > 1.0 + 1e-6
                ):
                    errors.append(f"{split}/{stem}:{line_number} box outside image")
                box_count += 1

    return {
        "valid": not errors,
        "image_count": image_count,
        "label_count": label_count,
        "box_count": box_count,
        "duplicate_label_count": duplicate_label_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
    }


def _load_mot_sequence_info(path: Path) -> dict[str, int | str]:
    parser = ConfigParser()
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"MOT17 sequence metadata not found: {path}")
    sequence = parser["Sequence"]
    return {
        "image_dir": sequence.get("imDir", "img1"),
        "length": sequence.getint("seqLength"),
        "width": sequence.getint("imWidth"),
        "height": sequence.getint("imHeight"),
        "extension": sequence.get("imExt", ".jpg"),
    }


def _load_mot_person_boxes(path: Path) -> dict[int, list[tuple[float, ...]]]:
    boxes: dict[int, list[tuple[float, ...]]] = defaultdict(list)
    with path.open(encoding="utf-8") as ground_truth_file:
        for line in ground_truth_file:
            columns = line.strip().split(",")
            if len(columns) < 9:
                continue
            frame_id = int(float(columns[0]))
            mark = float(columns[6])
            class_id = int(float(columns[7]))
            if mark <= 0 or class_id != 1:
                continue
            boxes[frame_id].append(tuple(float(value) for value in columns[2:6]))
    return dict(boxes)


def _build_image_lookup(image_directories: tuple[Path, ...]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for directory in image_directories:
        if not directory.exists():
            raise FileNotFoundError(f"CrowdHuman image directory not found: {directory}")
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                lookup[path.stem] = path
    return lookup


def _convert_box_to_yolo(
    box: tuple[float, ...],
    width: int,
    height: int,
) -> tuple[tuple[float, float, float, float] | None, bool]:
    x, y, box_width, box_height = box
    original = (x, y, x + box_width, y + box_height)
    x1 = min(max(0.0, original[0]), float(width))
    y1 = min(max(0.0, original[1]), float(height))
    x2 = min(max(0.0, original[2]), float(width))
    y2 = min(max(0.0, original[3]), float(height))
    if x2 <= x1 or y2 <= y1:
        return None, original != (x1, y1, x2, y2)
    clipped = original != (x1, y1, x2, y2)
    converted = (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
    return converted, clipped


def _format_yolo_row(box: tuple[float, float, float, float]) -> str:
    return "0 " + " ".join(f"{value:.8f}" for value in box)


def _write_labels(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(rows)
    if contents:
        contents += "\n"
    path.write_text(contents, encoding="utf-8")


def _materialize_image(source: Path, destination: Path, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    if link_mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _record_materialization(report: DatasetReport, materialization: str) -> None:
    if materialization == "hardlink":
        report.hardlink_count += 1
    elif materialization == "copy":
        report.copy_count += 1


def _record_missing_image(report: DatasetReport, path: Path) -> None:
    report.missing_image_count += 1
    if len(report.missing_images) < 100:
        report.missing_images.append(str(path))


def _write_dataset_yaml(output_root: Path, description: str) -> None:
    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
        "description": description,
    }
    (output_root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def _write_previews(dataset_root: Path, preview_count: int) -> None:
    if preview_count <= 0:
        return
    random_generator = random.Random(17)
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue
        selected = random_generator.sample(images, min(preview_count, len(images)))
        preview_dir = dataset_root / "previews" / split
        preview_dir.mkdir(parents=True, exist_ok=True)
        for image_path in selected:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
            for line in label_path.read_text(encoding="utf-8").splitlines():
                columns = line.split()
                if len(columns) != 5:
                    continue
                center_x, center_y, box_width, box_height = (
                    float(value) for value in columns[1:]
                )
                x1 = round((center_x - box_width / 2) * width)
                y1 = round((center_y - box_height / 2) * height)
                x2 = round((center_x + box_width / 2) * width)
                y2 = round((center_y + box_height / 2) * height)
                cv2.rectangle(image, (x1, y1), (x2, y2), (30, 220, 80), 2)
            cv2.imwrite(str(preview_dir / image_path.name), image)


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _validate_link_mode(link_mode: str) -> None:
    if link_mode not in {"hardlink", "copy"}:
        raise ValueError("link_mode must be 'hardlink' or 'copy'.")
