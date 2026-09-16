import json
from pathlib import Path

import numpy as np

from src.event_jsonl import EventJsonlWriter
from src.events import Event

from src.pipeline import PipelineResult

def test_writes_event_as_one_json_line(tmp_path):
    writer = EventJsonlWriter(
        event_path=tmp_path / "events.jsonl",
        metadata_path=tmp_path / "metadata.json",
    )

    event = Event(
        event_type="intrusion",
        track_id=7,
        timestamp=5.0,
        position=(120, 240),
        zone_id="restricted-zone-1",
    )

    try:
        writer.write_event(event, frame_id=150)
    finally:
        writer.close()

    lines = (tmp_path / "events.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["frame_id"] == 150
    assert record["event_type"] == "intrusion"
    assert record["track_id"] == 7
    assert record["video_timestamp"] == 5.0
    assert record["position"] == {"x": 120, "y": 240}
    assert record["zone_id"] == "restricted-zone-1"
    assert record["direction"] is None
    assert record["duration_seconds"] is None
    assert record["run_id"]


def test_writes_numpy_event_coordinates_as_json_numbers(tmp_path):
    writer = EventJsonlWriter(
        event_path=tmp_path / "events.jsonl",
        metadata_path=tmp_path / "metadata.json",
    )

    event = Event(
        event_type="intrusion",
        track_id=7,
        timestamp=5.0,
        position=(np.int64(120), np.int64(240)),
        zone_id="restricted-zone-1",
    )

    try:
        writer.write_event(event, frame_id=150)
    finally:
        writer.close()

    record = json.loads(
        (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    )

    assert record["position"] == {"x": 120, "y": 240}

def test_writes_run_metadata(tmp_path):
    writer = EventJsonlWriter(
        event_path=tmp_path / "events.jsonl",
        metadata_path=tmp_path / "metadata.json",
    )

    result = PipelineResult(
        source_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        processed_frames=100,
        source_fps=30.0,
        effective_fps=30.0,
        core_processing_fps=25.0,
        end_to_end_fps=20.0,
        stopped_early=False,
        elapsed_seconds=5.0,
        in_count=2,
        out_count=1,
        intrusion_count=3,
        loitering_count=1,
    )

    try:
        writer.write_metadata(
            config_data={"detection": {"confidence": 0.4}},
            result=result,
        )
    finally:
        writer.close()

    metadata = json.loads(
        (tmp_path / "metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["status"] == "completed"
    assert metadata["processed_frames"] == 100
    assert metadata["core_processing_fps"] == 25.0
    assert metadata["end_to_end_fps"] == 20.0
    assert not metadata["stopped_early"]
    assert metadata["counts"]["intrusion"] == 3
