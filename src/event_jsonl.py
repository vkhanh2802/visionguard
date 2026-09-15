import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.events import Event
from src.pipeline.types import PipelineResult


class EventJsonlWriter:
    def __init__(self, event_path: Path, metadata_path: Path):
        self.event_path = Path(event_path)
        self.metadata_path = Path(metadata_path)
        self.run_id = str(uuid4())

        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = self.event_path.open(
            mode="w",
            encoding="utf-8",
        )

    def write_event(self, event: Event, frame_id: int) -> None:
        record = {
            "run_id": self.run_id,
            "frame_id": frame_id,
            "event_type": event.event_type,
            "track_id": event.track_id,
            "video_timestamp": event.timestamp,
            "position": {
                "x": event.position[0],
                "y": event.position[1],
            },
            "direction": event.direction,
            "zone_id": event.zone_id,
            "duration_seconds": event.duration_seconds,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }

        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def write_metadata(
        self,
        config_data: dict,
        result: PipelineResult,
    ) -> None:
        metadata = {
            "run_id": self.run_id,
            "status": "completed",
            "source_path": str(result.source_path),
            "output_path": str(result.output_path),
            "processed_frames": result.processed_frames,
            "source_fps": result.source_fps,
            "processing_fps": result.processing_fps,
            "elapsed_seconds": result.elapsed_seconds,
            "counts": {
                "in": result.in_count,
                "out": result.out_count,
                "intrusion": result.intrusion_count,
                "loitering": result.loitering_count,
            },
            "config": config_data,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        self.metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.file.close()