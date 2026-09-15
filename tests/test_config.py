from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from configs.config import load_config


@pytest.fixture
def valid_config_data() -> dict:
    return {
        "detection": {
            "model_path": "yolo26n.pt",
            "confidence": 0.4,
            "target_classes": ["person"],
        },
        "tracking": {
            "history_length": 30,
            "max_missing_frames": 30,
        },
        "events": {
            "line_crossing": {
                "enabled": True,
                "start": [50, 300],
                "end": [600, 300],
                "dead_zone_px": 5.0,
                "confirmation_frames": 3,
                "negative_to_positive": "IN",
                "positive_to_negative": "OUT",
            },
            "zones": {
                "restricted-zone-1": {
                    "polygon": [
                        [100, 100],
                        [200, 100],
                        [200, 200],
                        [100, 200],
                    ]
                }
            },
            "intrusion": {
                "enabled": True,
                "zone_id": "restricted-zone-1",
            },
            "loitering": {
                "enabled": True,
                "zone_id": "restricted-zone-1",
                "dwell_threshold_seconds": 5.0,
            },
        },
        "output": {
            "display": True,
            "codec": "mp4v",
        },
        "logging": {
            "level": "INFO",
            "event_jsonl_path": "data/outputs/events.jsonl",
        },
    }


def write_config(tmp_path: Path, data: object) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def test_loads_valid_config(tmp_path: Path, valid_config_data: dict):
    config = load_config(write_config(tmp_path, valid_config_data))

    assert config.detection.confidence == 0.4
    assert config.detection.target_classes == {"person"}
    assert config.events.loitering.dwell_threshold_seconds == 5.0
    assert config.events.zones["restricted-zone-1"].polygon[0] == (100, 100)


def test_default_config_loads():
    project_root = Path(__file__).resolve().parents[1]

    config = load_config(project_root / "configs" / "default.yaml")

    assert config.events.intrusion.zone_id == "restricted-zone-1"


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_rejects_confidence_outside_valid_range(
    tmp_path: Path,
    valid_config_data: dict,
    confidence: float,
):
    data = deepcopy(valid_config_data)
    data["detection"]["confidence"] = confidence

    with pytest.raises(ValidationError, match="confidence"):
        load_config(write_config(tmp_path, data))


def test_rejects_identical_line_endpoints(tmp_path: Path, valid_config_data: dict):
    data = deepcopy(valid_config_data)
    data["events"]["line_crossing"]["end"] = [50, 300]

    with pytest.raises(ValidationError, match="cannot be the same"):
        load_config(write_config(tmp_path, data))


def test_rejects_zero_confirmation_frames(tmp_path: Path, valid_config_data: dict):
    data = deepcopy(valid_config_data)
    data["events"]["line_crossing"]["confirmation_frames"] = 0

    with pytest.raises(ValidationError, match="confirmation_frames"):
        load_config(write_config(tmp_path, data))


def test_rejects_non_positive_loitering_threshold(tmp_path: Path, valid_config_data: dict):
    data = deepcopy(valid_config_data)
    data["events"]["loitering"]["dwell_threshold_seconds"] = 0

    with pytest.raises(ValidationError, match="dwell_threshold_seconds"):
        load_config(write_config(tmp_path, data))


def test_rejects_polygon_with_too_few_vertices(tmp_path: Path, valid_config_data: dict):
    data = deepcopy(valid_config_data)
    data["events"]["zones"]["restricted-zone-1"]["polygon"] = [
        [100, 100],
        [200, 100],
    ]

    with pytest.raises(ValidationError, match="at least three"):
        load_config(write_config(tmp_path, data))


def test_rejects_self_intersecting_polygon(tmp_path: Path, valid_config_data: dict):
    data = deepcopy(valid_config_data)
    data["events"]["zones"]["restricted-zone-1"]["polygon"] = [
        [0, 0],
        [10, 10],
        [0, 10],
        [10, 0],
    ]

    with pytest.raises(ValidationError, match="intersecting edges"):
        load_config(write_config(tmp_path, data))


def test_rejects_unknown_zone_reference(tmp_path: Path, valid_config_data: dict):
    data = deepcopy(valid_config_data)
    data["events"]["intrusion"]["zone_id"] = "missing-zone"

    with pytest.raises(ValidationError, match="Unknown zone_id"):
        load_config(write_config(tmp_path, data))


def test_rejects_unknown_yaml_field(tmp_path: Path, valid_config_data: dict):
    data = deepcopy(valid_config_data)
    data["detection"]["confidense"] = 0.4

    with pytest.raises(ValidationError, match="confidense"):
        load_config(write_config(tmp_path, data))


def test_rejects_empty_config(tmp_path: Path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_config(config_path)


def test_rejects_missing_config_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_rejects_non_mapping_config_root(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top level"):
        load_config(config_path)
