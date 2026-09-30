import subprocess
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Protocol

import cv2
import numpy as np


class VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def release(self) -> None: ...


class AsyncVideoWriter:
    def __init__(
        self,
        writer: VideoWriter,
        queue_size: int,
        timeout_seconds: float = 30.0,
    ):
        self._writer = writer
        self._sentinel = object()
        self._queue: Queue[np.ndarray | object] = Queue(maxsize=queue_size)
        self._error: Exception | None = None
        self._released = False
        self._writer_released = False
        self._stop = Event()
        self._timeout_seconds = timeout_seconds
        self.encoding_seconds = 0.0
        self._thread = Thread(target=self._run, name="video-writer", daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise RuntimeError("Cannot write after video writer release.")
        self._raise_if_failed()
        try:
            self._queue.put(frame, timeout=self._timeout_seconds)
        except Full as error:
            self._error = TimeoutError("Video writer queue did not drain in time.")
            raise RuntimeError("Asynchronous video encoding timed out.") from error
        self._raise_if_failed()

    def release(self) -> None:
        if self._writer_released:
            return
        self._released = True
        shutdown_error: Exception | None = None
        if self._thread.is_alive():
            try:
                self._queue.put(self._sentinel, timeout=self._timeout_seconds)
                self._thread.join(timeout=self._timeout_seconds)
            except Full:
                shutdown_error = TimeoutError("Video writer queue did not drain in time.")

        if self._thread.is_alive():
            shutdown_error = shutdown_error or TimeoutError(
                "Video writer thread did not finish in time."
            )
            self._stop.set()
            abort = getattr(self._writer, "abort", None)
            if callable(abort):
                try:
                    abort()
                except Exception as error:
                    shutdown_error = shutdown_error or error
            self._thread.join(timeout=self._timeout_seconds)
            if self._thread.is_alive():
                shutdown_error = shutdown_error or TimeoutError(
                    "Video writer thread did not stop in time."
                )

        if not self._thread.is_alive():
            try:
                self._writer.release()
            except Exception as error:
                shutdown_error = shutdown_error or error
            else:
                self._writer_released = True

        self._raise_if_failed()
        if shutdown_error is not None:
            raise RuntimeError("Asynchronous video writer shutdown failed.") from shutdown_error

    def _run(self) -> None:
        from time import perf_counter

        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                if item is self._sentinel:
                    return
                if self._error is None and not self._stop.is_set():
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
        shutdown_timeout_seconds: float = 30.0,
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
                bufsize=0,
                creationflags=creation_flags,
            )
        except FileNotFoundError as error:
            raise RuntimeError(f"FFmpeg executable not found: {ffmpeg_path}") from error
        if self._process.stdin is None:
            self._process.kill()
            raise RuntimeError("Could not open FFmpeg input pipe.")
        self._stdin = self._process.stdin
        self._released = False
        self._cleanup_complete = False
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._stderr_chunks: deque[bytes] = deque(maxlen=64)
        self._stderr_lock = Lock()
        self._stderr_thread: Thread | None = None
        if self._process.stderr is not None:
            self._stderr_thread = Thread(
                target=self._drain_stderr,
                name="ffmpeg-stderr",
                daemon=True,
            )
            self._stderr_thread.start()

    def write(self, frame: np.ndarray) -> None:
        if self._released:
            raise RuntimeError("Cannot write after video writer release.")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("FFmpeg NVENC expects an 8-bit BGR frame.")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"FFmpeg NVENC stopped while encoding: {self._stderr_detail()}"
            )
        contiguous = np.ascontiguousarray(frame)
        try:
            remaining = memoryview(contiguous).cast("B")
            while remaining:
                written = self._stdin.write(remaining)
                if written is None or written <= 0:
                    raise BrokenPipeError("FFmpeg input pipe accepted no frame data.")
                remaining = remaining[written:]
        except (BrokenPipeError, OSError) as error:
            detail = self._stderr_detail()
            raise RuntimeError(
                f"FFmpeg NVENC stopped while encoding: {detail}"
            ) from error

    def release(self) -> None:
        if self._cleanup_complete:
            return
        self._released = True
        close_error: Exception | None = None
        try:
            self._stdin.close()
        except (BrokenPipeError, OSError, ValueError) as error:
            close_error = error
        timed_out = False
        wait_error: Exception | None = None
        return_code: int | None = None
        try:
            return_code = self._process.wait(timeout=self._shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                self._process.kill()
                return_code = self._process.wait(timeout=self._shutdown_timeout_seconds)
            except Exception as error:
                wait_error = error
        finally:
            process_reaped = return_code is not None
            if process_reaped and self._stderr_thread is not None:
                self._stderr_thread.join(timeout=self._shutdown_timeout_seconds)
            stderr = self._stderr_detail()
            stderr_thread_finished = (
                self._stderr_thread is None or not self._stderr_thread.is_alive()
            )
            if process_reaped and stderr_thread_finished and self._process.stderr is not None:
                try:
                    self._process.stderr.close()
                except (OSError, ValueError):
                    pass
            if process_reaped:
                self._cleanup_complete = True

        if wait_error is not None:
            raise RuntimeError("FFmpeg NVENC did not stop after being killed.") from wait_error
        if timed_out:
            detail = stderr or "no FFmpeg error output"
            raise RuntimeError(f"FFmpeg NVENC shutdown timed out: {detail}")
        if return_code != 0:
            detail = stderr or f"exit code {return_code}"
            raise RuntimeError(f"FFmpeg NVENC encoding failed: {detail}")
        if close_error is not None:
            raise RuntimeError("Could not close FFmpeg NVENC input pipe.") from close_error

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.kill()

    def _drain_stderr(self) -> None:
        if self._process.stderr is None:
            return
        try:
            while chunk := self._process.stderr.read(4096):
                with self._stderr_lock:
                    self._stderr_chunks.append(chunk)
        except (OSError, ValueError):
            return

    def _stderr_detail(self) -> str:
        with self._stderr_lock:
            stderr = b"".join(self._stderr_chunks).decode(errors="replace").strip()
        return stderr or "no FFmpeg error output"


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
