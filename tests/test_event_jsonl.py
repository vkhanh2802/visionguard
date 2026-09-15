import json

from src.event_jsonl import EventJsonlWriter
from src.events import Event


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