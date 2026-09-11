import argparse
from pathlib import Path

import cv2

from src.tracking import YOLOByteTracker
from src.fps_meter import FPSMeter
from src.video import create_video_writer
from src.visualization import draw_tracks, draw_fps, draw_trajectories
from src.tracking.history import TrackHistory

TARGET_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionGuard object detection pipeline")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--output", default="data/outputs/output.mp4", help="Path to output video")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(f"Video does not exist: {source}")

    tracker = YOLOByteTracker(model_path=args.model, confidence=args.conf, target_classes=TARGET_CLASSES)
    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = create_video_writer(output, source_fps, width, height)
    fps_meter = FPSMeter()
    track_history = TrackHistory(max_length=30)  # Store the last 30 positions for each track
    print(f"Input: {source}")
    print(f"Resolution: {width}x{height}")
    print(f"Source FPS: {source_fps:.2f}")
    print(f"Frames: {frame_count}")

    processed_frames = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            fps_meter.start()
            tracks = tracker.track(frame)
            track_history.update(tracks)
            processing_fps = fps_meter.stop()

            draw_tracks(frame, tracks)
            draw_fps(frame, processing_fps)
            draw_trajectories(frame, tracks, track_history)

            writer.write(frame)
            processed_frames += 1

            if not args.no_display:
                cv2.imshow("VisionGuard", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Processed frames: {processed_frames}")
    print(f"Processing FPS: {fps_meter.fps:.2f}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()