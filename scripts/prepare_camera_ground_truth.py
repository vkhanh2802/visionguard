import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and normalize a CVAT MOT export for camera evaluation",
    )
    parser.add_argument("--video", required=True, help="Source camera video")
    parser.add_argument("--source-gt", required=True, help="CVAT MOT gt.txt path")
    parser.add_argument("--output-dir", required=True, help="Normalized MOT sequence directory")
    parser.add_argument("--name", required=True, help="MOT sequence name")
    return parser.parse_args()


def _integer(value: str, field: str, line_number: int) -> int:
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"Invalid {field} at line {line_number}: {value}")
    return int(parsed)


def normalize_mot_rows(
    rows: Sequence[Sequence[str]],
    frame_count: int,
    width: int,
    height: int,
) -> tuple[list[str], int]:
    normalized = []
    seen_frame_tracks: set[tuple[int, int]] = set()
    clipped_boxes = 0

    for line_number, row in enumerate(rows, start=1):
        if len(row) < 9:
            raise ValueError(f"Expected at least 9 MOT columns at line {line_number}")

        frame_id = _integer(row[0], "frame ID", line_number)
        track_id = _integer(row[1], "track ID", line_number)
        if not 1 <= frame_id <= frame_count:
            raise ValueError(
                f"Frame ID outside video range at line {line_number}: {frame_id}"
            )
        key = (frame_id, track_id)
        if key in seen_frame_tracks:
            raise ValueError(f"Duplicate frame/track pair at line {line_number}: {key}")
        seen_frame_tracks.add(key)

        x, y, box_width, box_height = (float(value) for value in row[2:6])
        if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
            raise ValueError(f"Non-finite box at line {line_number}")
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Non-positive box at line {line_number}")

        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(width), x + box_width)
        y2 = min(float(height), y + box_height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Box is outside the image at line {line_number}")
        if (x1, y1, x2 - x1, y2 - y1) != (x, y, box_width, box_height):
            clipped_boxes += 1

        confidence = float(row[6])
        class_id = _integer(row[7], "class ID", line_number)
        visibility = float(row[8])
        if not all(math.isfinite(value) for value in (confidence, visibility)):
            raise ValueError(f"Non-finite MOT metadata at line {line_number}")
        if confidence <= 0:
            raise ValueError(f"Ignored ground truth is not supported at line {line_number}")
        if class_id != 1:
            raise ValueError(f"Expected person class ID 1 at line {line_number}")

        normalized.append(
            f"{frame_id},{track_id},{x1:.2f},{y1:.2f},"
            f"{x2 - x1:.2f},{y2 - y1:.2f},{confidence:g},{class_id},"
            f"{visibility:g}\n"
        )

    if not normalized:
        raise ValueError("Ground-truth file is empty")
    return normalized, clipped_boxes


def prepare_ground_truth(
    video_path: Path,
    source_gt_path: Path,
    output_dir: Path,
    sequence_name: str,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0 or frame_rate <= 0:
        raise ValueError(f"Invalid video metadata: {video_path}")

    with source_gt_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.reader(handle))
    normalized_rows, clipped_boxes = normalize_mot_rows(
        source_rows,
        frame_count=frame_count,
        width=width,
        height=height,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir = output_dir / "gt"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    (ground_truth_dir / "gt.txt").write_text("".join(normalized_rows), encoding="utf-8")
    (ground_truth_dir / "labels.txt").write_text("person\n", encoding="utf-8")
    (output_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={sequence_name}\n"
        "imDir=img1\n"
        f"frameRate={frame_rate:g}\n"
        f"seqLength={frame_count}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )

    frame_ids = {int(float(row[0])) for row in source_rows}
    track_ids = {int(float(row[1])) for row in source_rows}
    report: dict[str, object] = {
        "name": sequence_name,
        "video": str(video_path),
        "source_ground_truth": str(source_gt_path),
        "normalized_ground_truth": str(ground_truth_dir / "gt.txt"),
        "frames": frame_count,
        "annotated_frames": len(frame_ids),
        "frame_id_min": min(frame_ids),
        "frame_id_max": max(frame_ids),
        "width": width,
        "height": height,
        "frame_rate": frame_rate,
        "detections": len(normalized_rows),
        "tracks": len(track_ids),
        "clipped_boxes": clipped_boxes,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    report = prepare_ground_truth(
        video_path=Path(args.video),
        source_gt_path=Path(args.source_gt),
        output_dir=Path(args.output_dir),
        sequence_name=args.name,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
