import subprocess
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Protocol

import cv2
import numpy as np


class VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


class AsyncVideoWriter:
    def __init__(self, writer: VideoWriter, queue_size: int):
        self._writer = writer
        self._sentinel = object()
        self._queue: Queue[np.ndarray | object] = Queue(maxsize=queue_size)
        self._error: Exception | None = None
        self._released = False
        self.encoding_seconds = 0.0
        self._thread = Thread(target=self._run, name="video-writer", daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise RuntimeError("Cannot write after video writer release.")
        self._raise_if_failed()
        self._queue.put(frame)
        self._raise_if_failed()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._queue.put(self._sentinel)
        self._thread.join()
        self._writer.release()
        self._raise_if_failed()

    def _run(self) -> None:
        from time import perf_counter

        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                if self._error is None:
                    started_at = perf_counter()
                    try:
                        self._writer.write(item)
                    finally:
                        self.encoding_seconds += perf_counter() - started_at
            except Exception as error:
                self._error = error
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Asynchronous video encoding failed.") from self._error


class FFmpegNVENCVideoWriter:
    def __init__(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
        ffmpeg_path: str,
        quality: int,
    ):
        command = _build_ffmpeg_nvenc_command(
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            ffmpeg_path=ffmpeg_path,
            quality=quality,
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
            )
        except FileNotFoundError as error:
            raise RuntimeError(f"FFmpeg executable not found: {ffmpeg_path}") from error
        if self._process.stdin is None:
            self._process.kill()
            raise RuntimeError("Could not open FFmpeg input pipe.")
        self._stdin = self._process.stdin
        self._released = False

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise RuntimeError("Cannot write after video writer release.")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("FFmpeg NVENC expects an 8-bit BGR frame.")
        contiguous = np.ascontiguousarray(frame)
        try:
            self._stdin.write(memoryview(contiguous).cast("B"))
        except BrokenPipeError as error:
            raise RuntimeError("FFmpeg NVENC stopped while encoding.") from error

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._stdin.close()
        except BrokenPipeError:
            pass
        stderr = (
            self._process.stderr.read().decode(errors="replace")
            if self._process.stderr is not None
            else ""
        )
        return_code = self._process.wait()
        if self._process.stderr is not None:
            self._process.stderr.close()
        if return_code != 0:
            detail = stderr.strip() or f"exit code {return_code}"
            raise RuntimeError(f"FFmpeg NVENC encoding failed: {detail}")


def _build_ffmpeg_nvenc_command(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    ffmpeg_path: str,
    quality: int,
) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p1",
        "-tune",
        "ll",
        "-rc",
        "constqp",
        "-qp",
        str(quality),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    codec: str = "mp4v",
    encoder: str = "opencv",
    async_write: bool = False,
    queue_size: int = 4,
    ffmpeg_path: str = "ffmpeg",
    nvenc_quality: int = 23,
) -> VideoWriter:
    if len(codec) != 4:
        raise ValueError("Video codec must contain exactly four characters.")

    if fps <= 0:
        raise ValueError("Output video FPS must be positive.")

    if width <= 0 or height <= 0:
        raise ValueError("Output video dimensions must be positive.")
    if queue_size <= 0:
        raise ValueError("Video writer queue size must be positive.")
    if not 0 <= nvenc_quality <= 51:
        raise ValueError("NVENC quality must be between 0 and 51.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if encoder == "ffmpeg_nvenc":
        writer: VideoWriter = FFmpegNVENCVideoWriter(
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            ffmpeg_path=ffmpeg_path,
            quality=nvenc_quality,
        )
    elif encoder == "opencv":
        opencv_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not opencv_writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")
        writer = opencv_writer
    else:
        raise ValueError(f"Unknown video encoder: {encoder}")

    if async_write:
        return AsyncVideoWriter(writer, queue_size=queue_size)
    return writer
