from pathlib import Path

import cv2

def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    if len(codec) != 4:
        raise ValueError("Video codec must contain exactly four characters.")

    if fps <= 0:
        raise ValueError("Output video FPS must be positive.")

    if width <= 0 or height <= 0:
        raise ValueError("Output video dimensions must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    return writer