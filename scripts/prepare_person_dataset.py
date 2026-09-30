import argparse
import json
from pathlib import Path

import cv2

from src.dataset_tools import (
    convert_crowdhuman,
    convert_mot17,
    create_combined_dataset_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MOT17 and CrowdHuman person annotations to YOLO format",
    )
    parser.add_argument("--mot17-root", default="datasets/MOT17")
    parser.add_argument(
        "--crowdhuman-root",
        default="datasets/person_tracking/crownhuman",
    )
    parser.add_argument("--camera-root", default="datasets/camera")
    parser.add_argument("--output-root", default="datasets/yolo")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("mot17", "crowdhuman"),
        default=("mot17", "crowdhuman"),
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
    )
    parser.add_argument("--preview-count", type=int, default=12)
    return parser.parse_args()


def inventory_camera_videos(camera_root: Path) -> dict[str, object]:
    videos = []
    for video_path in sorted(camera_root.glob("*.mp4")):
        capture = cv2.VideoCapture(str(video_path))
        videos.append(
            {
                "path": str(video_path),
                "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": capture.get(cv2.CAP_PROP_FPS),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "annotated": False,
            }
        )
        capture.release()
    return {
        "video_count": len(videos),
        "training_ready": False,
        "reason": "Bounding-box and event annotations are not present.",
        "videos": videos,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    reports: dict[str, object] = {}

    if "mot17" in args.sources:
        print("Converting MOT17...")
        mot17_report = convert_mot17(
            source_root=Path(args.mot17_root),
            output_root=output_root / "mot17",
            link_mode=args.link_mode,
            preview_count=args.preview_count,
        )
        reports["mot17"] = mot17_report.to_dict()

    if "crowdhuman" in args.sources:
        print("Converting CrowdHuman...")
        crowdhuman_report = convert_crowdhuman(
            source_root=Path(args.crowdhuman_root),
            output_root=output_root / "crowdhuman",
            link_mode=args.link_mode,
            preview_count=args.preview_count,
        )
        reports["crowdhuman"] = crowdhuman_report.to_dict()

    if {"mot17", "crowdhuman"}.issubset(args.sources):
        create_combined_dataset_yaml(
            mot17_root=output_root / "mot17",
            crowdhuman_root=output_root / "crowdhuman",
            output_path=output_root / "mot17_crowdhuman.yaml",
        )

    reports["camera"] = inventory_camera_videos(Path(args.camera_root))
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "preparation_report.json"
    summary_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    print()
    print("Dataset preparation completed")
    for name in ("mot17", "crowdhuman"):
        report = reports.get(name)
        if report is None:
            continue
        print(
            f"{name}: images={report['image_count']} boxes={report['box_count']} "
            f"missing={report['missing_image_count']} "
            f"valid={report['validation']['valid']}"
        )
    print(f"Report: {summary_path}")


if __name__ == "__main__":
    main()
