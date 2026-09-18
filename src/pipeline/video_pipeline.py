from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.config import AppConfig
from src.events import Event, IntrusionEngine, LineCrossingEngine, LoiteringEngine
from src.fps_meter import FPSMeter
from src.tracking import YOLOByteTracker
from src.tracking.history import TrackHistory
from src.tracking.types import Track
from src.video import create_video_writer
from src.visualization import (
    draw_event_counts,
    draw_fps,
    draw_line_crossing,
    draw_line_directions,
    draw_polygon_roi,
    draw_tracks,
    draw_trajectories,
)

from .types import PipelineResult


EventHandler = Callable[[Event, int], None]


class VideoPipeline:
    def __init__(
        self,
        config: AppConfig,
        event_handler: EventHandler | None = None,
    ):
        self.config = config
        self.event_handler = event_handler
    def _resolve_effective_fps(self, source_fps: float) -> float:
        fps_override = self.config.video.fps_override

        if fps_override is not None:
            return fps_override

        if source_fps <= 0:
            raise RuntimeError(
                "Input video does not provide a valid FPS. "
                "Set video.fps_override in the config."
            )

        return source_fps

    def run(
        self,
        source_path: Path,
        output_path: Path,
    ) -> PipelineResult:
        source_path = Path(source_path)
        output_path = Path(output_path)

        if not source_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {source_path}")

        if source_path.resolve() == output_path.resolve():
            raise ValueError("Input and output video paths must be different.")

        capture = cv2.VideoCapture(str(source_path))
        writer = None
        start_time = perf_counter()

        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {source_path}")

            source_fps = capture.get(cv2.CAP_PROP_FPS)
            effective_fps = self._resolve_effective_fps(source_fps)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if source_fps <= 0:
                raise RuntimeError("Input video does not provide a valid FPS.")
            if width <= 0 or height <= 0:
                raise RuntimeError(
                    f"Input video has invalid dimensions: {width}x{height}")
            
            writer = create_video_writer(
                output_path=output_path,
                fps=effective_fps,
                width=width,
                height=height,
                codec=self.config.output.codec,
            )

            tracker = YOLOByteTracker(
                model_path=self.config.detection.model_path,
                confidence=self.config.detection.confidence,
                target_classes=self.config.detection.target_classes,
            )

            track_history = TrackHistory(
                max_length=self.config.tracking.history_length,
                max_missing_frames=self.config.tracking.max_missing_frames,
            )

            line_engine, intrusion_engine, loitering_engine = self._create_event_engines()

            fps_meter = FPSMeter()
            frame_id = 0
            frame_loop_start = perf_counter()
            stopped_early = False
            while True:
                success, frame = capture.read()

                if not success:
                    break

                fps_meter.start()

                tracks = tracker.track(frame)
                track_history.update(tracks)

                timestamp = frame_id / effective_fps

                events = self._process_events(
                    tracks=tracks,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    line_engine=line_engine,
                    intrusion_engine=intrusion_engine,
                    loitering_engine=loitering_engine,
                )

                processing_fps = fps_meter.stop()

                for event in events:
                    if self.event_handler is not None:
                        self.event_handler(event, frame_id)

                self._draw_frame(
                    frame=frame,
                    tracks=tracks,
                    track_history=track_history,
                    processing_fps=processing_fps,
                    line_engine=line_engine,
                    intrusion_engine=intrusion_engine,
                    loitering_engine=loitering_engine,
                )

                writer.write(frame)
                frame_id += 1

                if self.config.output.display:
                    cv2.imshow("VisionGuard", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stopped_early = True
                        break
            frame_loop_elapsed = perf_counter() - frame_loop_start
            end_to_end_fps = (
                frame_id / frame_loop_elapsed
                if frame_loop_elapsed > 0
                else 0.0
            )
            if frame_id == 0:
                raise RuntimeError(f"Input video contains no readable frames: {source_path}")

            return PipelineResult(
            source_path=source_path,
            output_path=output_path,
            processed_frames=frame_id,
            source_fps=source_fps,
            effective_fps=effective_fps,
            core_processing_fps=fps_meter.fps,
            end_to_end_fps=end_to_end_fps,
            elapsed_seconds=perf_counter() - start_time,
            stopped_early=stopped_early,
            in_count=line_engine.in_count if line_engine else 0,
            out_count=line_engine.out_count if line_engine else 0,
            intrusion_count=(
                intrusion_engine.intrusion_count
                if intrusion_engine
                else 0
            ),
            loitering_count=(
                loitering_engine.loitering_count
                if loitering_engine
                else 0
            ),
        )
        finally:
            capture.release()

            if writer is not None:
                writer.release()

            if self.config.output.display:
                cv2.destroyAllWindows()

    def _create_event_engines(
        self,
    ) -> tuple[
        LineCrossingEngine | None,
        IntrusionEngine | None,
        LoiteringEngine | None,
    ]:
        events_config = self.config.events

        line_engine = None
        if events_config.line_crossing.enabled:
            line_config = events_config.line_crossing

            line_engine = LineCrossingEngine(
                line_start=line_config.start,
                line_end=line_config.end,
                dead_zone_px=line_config.dead_zone_px,
                confirmation_frames=line_config.confirmation_frames,
                max_missing_frames=self.config.tracking.max_missing_frames,
                negative_to_positive=line_config.negative_to_positive,
                positive_to_negative=line_config.positive_to_negative,
            )

        intrusion_engine = None
        if events_config.intrusion.enabled:
            intrusion_config = events_config.intrusion
            zone = events_config.zones[intrusion_config.zone_id]

            intrusion_engine = IntrusionEngine(
                polygon=tuple(zone.polygon),
                zone_id=intrusion_config.zone_id,
                max_missing_frames=self.config.tracking.max_missing_frames,
                entry_confirmation_frames=intrusion_config.entry_confirmation_frames,
                exit_confirmation_frames=intrusion_config.exit_confirmation_frames,
            )

        loitering_engine = None
        if events_config.loitering.enabled:
            loitering_config = events_config.loitering
            zone = events_config.zones[loitering_config.zone_id]

            loitering_engine = LoiteringEngine(
                polygon=tuple(zone.polygon),
                zone_id=loitering_config.zone_id,
                dwell_threshold_seconds=loitering_config.dwell_threshold_seconds,
                max_missing_frames=self.config.tracking.max_missing_frames,
            )

        return line_engine, intrusion_engine, loitering_engine

    def _process_events(
        self,
        tracks: list[Track],
        frame_id: int,
        timestamp: float,
        line_engine: LineCrossingEngine | None,
        intrusion_engine: IntrusionEngine | None,
        loitering_engine: LoiteringEngine | None,
    ) -> list[Event]:
        events = []

        if line_engine is not None:
            events.extend(
                line_engine.process(
                    tracks,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )

        if intrusion_engine is not None:
            events.extend(
                intrusion_engine.process(
                    tracks,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )

        if loitering_engine is not None:
            events.extend(
                loitering_engine.process(
                    tracks,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )

        return events

    def _draw_frame(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        track_history: TrackHistory,
        processing_fps: float,
        line_engine: LineCrossingEngine | None,
        intrusion_engine: IntrusionEngine | None,
        loitering_engine: LoiteringEngine | None,
    ) -> None:
        draw_tracks(frame, tracks)
        draw_trajectories(frame, tracks, track_history)

        if line_engine is not None:
            draw_line_crossing(frame, line_engine)
            draw_line_directions(frame, line_engine)

        zone_engine = intrusion_engine or loitering_engine
        if zone_engine is not None:
            draw_polygon_roi(
                frame,
                zone_engine.polygon,
                label=zone_engine.zone_id,
            )

        draw_event_counts(
            frame,
            in_count=line_engine.in_count if line_engine is not None else 0,
            out_count=line_engine.out_count if line_engine is not None else 0,
            intrusion_count=(
                intrusion_engine.intrusion_count
                if intrusion_engine is not None
                else 0
            ),
            loitering_count=(
                loitering_engine.loitering_count
                if loitering_engine is not None
                else 0
            ),
        )

        draw_fps(frame, processing_fps)

    
