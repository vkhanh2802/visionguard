from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PipelineResult:
    source_path: Path
    output_path: Path
    processed_frames: int
    source_fps: float
    effective_fps: float
    core_processing_fps: float
    end_to_end_fps: float
    stopped_early: bool
    elapsed_seconds: float
    in_count: int
    out_count: int
    intrusion_count: int
    loitering_count: int

