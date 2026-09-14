import argparse
from pathlib import Path

import cv2

from src.tracking import YOLOByteTracker
from src.fps_meter import FPSMeter
from src.video import create_video_writer
from src.visualization import draw_tracks, draw_fps, draw_trajectories
from src.tracking.history import TrackHistory

from src.events import LineCrossingEngine, IntrusionEngine, LoiteringEngine

from src.visualization import draw_fps, draw_line_crossing, draw_line_directions, draw_tracks, draw_trajectories, draw_event_counts, draw_polygon_roi 
from src.events import Event
LINE_START =  (50, 300)
LINE_END = (750, 300)
RESTRICTED_ZONE = (
    (100, 450),
    (300, 450),
    (300, 350),
    (100, 350),
)

DEAD_ZONE_PX = 5.0
CONFIRMATION_FRAMES = 3
MAX_MISSING_FRAMES = 30
LOITERING_THRESHOLD_SECONDS = 10.0
ZONE_ID = "restricted-zone-1"
TARGET_CLASSES = {"person"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionGuard object detection pipeline")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--output", default="data/outputs/output.mp4", help="Path to output video")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    return parser.parse_args()

def log_event(event: Event) -> None:
    message = (
        f"[{event.timestamp:7.2f}s] "
        f"{event.event_type} "
        f"track=#{event.track_id}"
    )

    if event.direction is not None:
        message += f" direction={event.direction}"

    if event.zone_id is not None:
        message += f" zone={event.zone_id}"

    if event.duration_seconds is not None:
        message += f" duration={event.duration_seconds:.1f}s"

    print(message)

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
    intrusion_engine = IntrusionEngine(
        polygon = RESTRICTED_ZONE,
        zone_id = ZONE_ID,
        max_missing_frames = MAX_MISSING_FRAMES,
    )
    loitering_engine = LoiteringEngine(
        polygon = RESTRICTED_ZONE,
        dwell_threshold_seconds = LOITERING_THRESHOLD_SECONDS,
        zone_id = ZONE_ID,
        max_missing_frames = MAX_MISSING_FRAMES,
    )

    print(f"Input: {source}")
    print(f"Resolution: {width}x{height}")
    print(f"Source FPS: {source_fps:.2f}")
    print(f"Frames: {frame_count}")

    frame_id = 0
    video_fps = source_fps if source_fps > 0 else 30.0  # Default to 30 FPS if source FPS is not available
    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            fps_meter.start()
            tracks = tracker.track(frame)
            track_history.update(tracks)

            timestamp = frame_id / video_fps

            line_events = line_crossing_engine.process(tracks, frame_id=frame_id, timestamp=timestamp)
            intrusion_events = intrusion_engine.process(tracks, frame_id=frame_id, timestamp=timestamp)
            loitering_events = loitering_engine.process(tracks, frame_id=frame_id, timestamp=timestamp)
            events = [*line_events, *intrusion_events, *loitering_events]
            tracking_fps = fps_meter.stop()

            for event in events:
                log_event(event)

            draw_tracks(frame, tracks)
            draw_fps(frame, tracking_fps)
            draw_trajectories(frame, tracks, track_history)
            draw_line_crossing(frame, line_crossing_engine)
            draw_line_directions(frame, line_crossing_engine)
            draw_polygon_roi(frame, RESTRICTED_ZONE, label = "restricted_zone")
            draw_event_counts(frame, line_crossing_engine, intrusion_engine, loitering_engine)

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