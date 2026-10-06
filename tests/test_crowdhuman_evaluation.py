import json

import cv2
import numpy as np
import pytest

from scripts.evaluate_crowdhuman import (
    build_detection_records,
    group_predictions,
    write_odgt,
)


def test_builds_complete_crowdhuman_detection_records(tmp_path):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "present.jpg"), image)
    ground_truth = [{"ID": "present"}, {"ID": "missing"}]
    predictions = group_predictions(
        [
            {
                "image_id": "present",
                "bbox": [1, 2, 3, 4],
                "score": 0.75,
            }
        ]
    )

    records, missing = build_detection_records(
        ground_truth,
        predictions,
        tmp_path,
    )

    assert records == [
        {
            "ID": "present",
            "width": 30,
            "height": 20,
            "dtboxes": [{"box": [1.0, 2.0, 3.0, 4.0], "score": 0.75}],
        },
        {
            "ID": "missing",
            "width": 1,
            "height": 1,
            "dtboxes": [],
        },
    ]
    assert missing == ["missing"]

    output_path = tmp_path / "detections.odgt"
    write_odgt(output_path, records)
    assert [json.loads(line) for line in output_path.read_text().splitlines()] == records


def test_rejects_predictions_for_unknown_image(tmp_path):
    with pytest.raises(ValueError, match="unknown image IDs"):
        build_detection_records(
            [{"ID": "known"}],
            {"unknown": [{"box": [1, 2, 3, 4], "score": 0.5}]},
            tmp_path,
        )


def test_rejects_non_finite_prediction():
    with pytest.raises(ValueError, match="Invalid prediction"):
        group_predictions(
            [{"image_id": "image", "bbox": [1, 2, 3, 4], "score": float("nan")}]
        )
