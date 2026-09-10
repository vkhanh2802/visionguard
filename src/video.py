from pathlib import Path

import cv2


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    return writer
from pathlib import Path

import cv2


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    safe_fps = fps if fps and fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    return writer
