import json
from pathlib import Path

import cv2
import numpy as np

from src.dataset_tools import (
    convert_crowdhuman,
    convert_mot17,
    validate_yolo_dataset,
)


def write_image(path: Path, width: int = 100, height: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_converts_mot17_sequence_to_yolo(tmp_path: Path):
    source = tmp_path / "MOT17"
    sequence = source / "train" / "MOT17-02-FRCNN"
    (sequence / "gt").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        """[Sequence]
name=MOT17-02-FRCNN
imDir=img1
frameRate=30
seqLength=2
imWidth=100
imHeight=80
imExt=.jpg
""",
        encoding="utf-8",
    )
    write_image(sequence / "img1" / "000001.jpg")
    write_image(sequence / "img1" / "000002.jpg")
    (sequence / "gt" / "gt.txt").write_text(
        "1,1,10,20,30,40,1,1,1\n"
        "1,2,50,20,20,40,0,7,1\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    report = convert_mot17(
        source,
        output,
        train_sequences=("02",),
        validation_sequences=(),
        preview_count=0,
    )

    label = (output / "labels" / "train" / "MOT17-02_000001.txt").read_text()
    assert label == "0 0.25000000 0.50000000 0.30000000 0.50000000\n"
    assert (output / "labels" / "train" / "MOT17-02_000002.txt").read_text() == ""
    assert report.image_count == 2
    assert report.box_count == 1
    assert report.validation["valid"] is True


def test_converts_crowdhuman_and_reports_missing_images(tmp_path: Path):
    source = tmp_path / "crowdhuman"
    train_images = source / "CrowdHuman_train01" / "Images"
    val_images = source / "CrowdHuman_val" / "Images"
    write_image(train_images / "train-image.jpg")
    write_image(val_images / "val-image.jpg")
    (source / "CrowdHuman_train02" / "Images").mkdir(parents=True)
    (source / "CrowdHuman_train03" / "Images").mkdir(parents=True)
    train_annotations = [
        {
            "ID": "train-image",
            "gtboxes": [
                {"tag": "person", "fbox": [10, 10, 30, 40], "extra": {}},
                {"tag": "mask", "fbox": [0, 0, 5, 5], "extra": {"ignore": 1}},
                {
                    "tag": "person",
                    "fbox": [50, 10, 20, 20],
                    "extra": {"ignore": 1},
                },
            ],
        },
        {"ID": "missing-image", "gtboxes": []},
    ]
    (source / "annotation_train.odgt").write_text(
        "\n".join(json.dumps(item) for item in train_annotations) + "\n",
        encoding="utf-8",
    )
    (source / "annotation_val.odgt").write_text(
        json.dumps(
            {
                "ID": "val-image",
                "gtboxes": [
                    {"tag": "person", "fbox": [20, 20, 20, 20], "extra": {}}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    report = convert_crowdhuman(source, output, preview_count=0)

    assert report.image_count == 2
    assert report.box_count == 2
    assert report.ignored_box_count == 2
    assert report.missing_image_count == 1
    assert report.validation["valid"] is True


def test_validator_rejects_out_of_bounds_label(tmp_path: Path):
    dataset = tmp_path / "dataset"
    write_image(dataset / "images" / "train" / "image.jpg")
    label_path = dataset / "labels" / "train" / "image.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("0 0.9 0.5 0.4 0.5\n", encoding="utf-8")

    validation = validate_yolo_dataset(dataset)

    assert validation["valid"] is False
    assert validation["error_count"] == 1
    assert "box outside image" in validation["errors"][0]
