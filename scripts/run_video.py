import argparse
from pathlib import Path

import cv2

from src.tracking import YOLOByteTracker
from src.fps_meter import FPSMeter
from src.video import create_video_writer
from src.visualization import draw_tracks, draw_fps, draw_trajectories
from src.tracking.history import TrackHistory

from src.events import LineCrossingEngine
from src.visualization import draw_counts, draw_fps, draw_line_crossing, draw_line_directions, draw_tracks, draw_trajectories

LINE_START =  (50, 400)
LINE_END = (1850, 400)
DEAD_ZONE_PX = 5
CONFIRMATION_FRAMES = 5
TARGET_CLASSES = {"person"}


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
    line_crossing_engine = LineCrossingEngine(
        line_start = LINE_START, 
        line_end = LINE_END, 
        dead_zone_px = DEAD_ZONE_PX,
        max_missing_frames = 30,
        confirmation_frames = CONFIRMATION_FRAMES,
    )
    print(f"Input: {source}")
    print(f"Resolution: {width}x{height}")
    print(f"Source FPS: {source_fps:.2f}")
    print(f"Frames: {frame_count}")

    frame_id = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            fps_meter.start()
            tracks = tracker.track(frame)
            track_history.update(tracks)

            timestamp = frame_id / source_fps if source_fps > 0 else 0.0
            events = line_crossing_engine.process(tracks, frame_id=frame_id, timestamp=timestamp)

            tracking_fps = fps_meter.stop()

            for event in events:
                print(f"[{event.timestamp:7.2f}s] Track #{event.track_id} {event.direction} at {event.position}")

            


            draw_tracks(frame, tracks)
            draw_fps(frame, tracking_fps)
            draw_trajectories(frame, tracks, track_history)
            draw_line_crossing(frame, line_crossing_engine)
            draw_line_directions(frame, line_crossing_engine)
            draw_counts(frame, line_crossing_engine)

            writer.write(frame)
            frame_id += 1

            if not args.no_display:
                cv2.imshow("VisionGuard", frame) 

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print()
    print("Processing completed")
    print(f"Frames: {frame_id}")
    print(f"Tracking FPS: {fps_meter.fps:.2f}")
    print(f"IN: {line_crossing_engine.in_count}")
    print(f"OUT: {line_crossing_engine.out_count}")
    print(f"NET: {line_crossing_engine.net_count}")
    print(f"Total crossings: {line_crossing_engine.total_count}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()