from pathlib import Path
import subprocess
from io import BytesIO
from threading import Event

import cv2
import numpy as np
import pytest

from src.video import (
    AsyncVideoWriter,
    FFmpegNVENCVideoWriter,
    _build_ffmpeg_nvenc_command,
    create_video_writer,
)


def test_async_video_writer_writes_every_frame(tmp_path: Path):
    output_path = tmp_path / "output.avi"
    writer = create_video_writer(
        output_path=output_path,
        fps=10,
        width=64,
        height=48,
        codec="MJPG",
        async_write=True,
        queue_size=2,
    )

    assert isinstance(writer, AsyncVideoWriter)
    for value in range(10):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    capture = cv2.VideoCapture(str(output_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    capture.release()


def test_video_writer_rejects_invalid_queue_size(tmp_path: Path):
    with pytest.raises(ValueError, match="queue size"):
        create_video_writer(
            output_path=tmp_path / "output.avi",
            fps=10,
            width=64,
            height=48,
            codec="MJPG",
            async_write=True,
            queue_size=0,
        )


def test_builds_ffmpeg_nvenc_command(tmp_path: Path):
    output_path = tmp_path / "output.mp4"

    command = _build_ffmpeg_nvenc_command(
        output_path=output_path,
        fps=29.97,
        width=1920,
        height=1080,
        ffmpeg_path="custom-ffmpeg",
        quality=21,
    )

    assert command[0] == "custom-ffmpeg"
    assert command[command.index("-video_size") + 1] == "1920x1080"
    assert command[command.index("-framerate") + 1] == "29.97"
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-qp") + 1] == "21"
    assert command[-1] == str(output_path)


def test_async_video_writer_propagates_encoding_failure():
    class FailingWriter:
        def write(self, frame: np.ndarray) -> None:
            raise RuntimeError("encoder failed")

        def release(self) -> None:
            pass

    writer = AsyncVideoWriter(FailingWriter(), queue_size=1, timeout_seconds=0.5)
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="Asynchronous video encoding failed"):
        writer.release()


def test_async_video_writer_times_out_and_unblocks_stuck_writer():
    class BlockingWriter:
        def __init__(self):
            self.started = Event()
            self.write_finished = Event()
            self.released = Event()
            self.released_during_write = False

        def write(self, frame: np.ndarray) -> None:
            self.started.set()
            self.released.wait()
            self.write_finished.set()

        def abort(self) -> None:
            self.released.set()

        def release(self) -> None:
            self.released_during_write = not self.write_finished.is_set()

    blocking_writer = BlockingWriter()
    writer = AsyncVideoWriter(
        blocking_writer,
        queue_size=1,
        timeout_seconds=0.1,
    )
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    assert blocking_writer.started.wait(timeout=1)
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="timed out"):
        writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="Asynchronous video encoding failed"):
        writer.release()

    assert blocking_writer.released.is_set()
    assert not blocking_writer.released_during_write


def test_async_video_writer_reports_forced_shutdown_without_queue_timeout():
    class BlockingWriter:
        def __init__(self):
            self.started = Event()
            self.unblocked = Event()

        def write(self, frame: np.ndarray) -> None:
            self.started.set()
            self.unblocked.wait()

        def abort(self) -> None:
            self.unblocked.set()

        def release(self) -> None:
            pass

    blocking_writer = BlockingWriter()
    writer = AsyncVideoWriter(blocking_writer, queue_size=1, timeout_seconds=0.1)
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    assert blocking_writer.started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        writer.release()

    assert blocking_writer.unblocked.is_set()


def test_async_video_writer_bounds_shutdown_for_writer_without_abort():
    class BlockingWriter:
        def __init__(self):
            self.started = Event()
            self.unblocked = Event()
            self.release_calls = 0

        def write(self, frame: np.ndarray) -> None:
            self.started.set()
            self.unblocked.wait()

        def release(self) -> None:
            self.release_calls += 1

    blocking_writer = BlockingWriter()
    writer = AsyncVideoWriter(blocking_writer, queue_size=1, timeout_seconds=0.05)
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    assert blocking_writer.started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        writer.release()

    assert blocking_writer.release_calls == 0
    blocking_writer.unblocked.set()
    writer._thread.join(timeout=1)
    writer.release()
    assert blocking_writer.release_calls == 1


def test_ffmpeg_writer_rejects_missing_executable(tmp_path: Path):
    with pytest.raises(RuntimeError, match="FFmpeg executable not found"):
        FFmpegNVENCVideoWriter(
            output_path=tmp_path / "output.mp4",
            fps=30,
            width=64,
            height=48,
            ffmpeg_path="definitely-missing-visionguard-ffmpeg",
            quality=23,
        )


def test_ffmpeg_writer_reports_stderr_without_waiting_for_release(
    tmp_path: Path,
    monkeypatch,
):
    class FakeStdin(BytesIO):
        pass

    class FailedProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stderr = BytesIO(b"NVENC device unavailable")

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FailedProcess())
    writer = FFmpegNVENCVideoWriter(
        output_path=tmp_path / "output.mp4",
        fps=30,
        width=64,
        height=48,
        ffmpeg_path="ffmpeg",
        quality=23,
    )
    writer._stderr_thread.join(timeout=1)

    with pytest.raises(RuntimeError, match="NVENC device unavailable"):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))


def test_ffmpeg_writer_retries_partial_pipe_writes(tmp_path: Path, monkeypatch):
    class ShortWriteStdin:
        def __init__(self):
            self.data = bytearray()
            self.closed = False

        def write(self, data):
            chunk = bytes(data[:2])
            self.data.extend(chunk)
            return len(chunk)

        def close(self):
            self.closed = True

    class CompletedProcess:
        def __init__(self):
            self.stdin = ShortWriteStdin()
            self.stderr = BytesIO()

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    process = CompletedProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    writer = FFmpegNVENCVideoWriter(
        output_path=tmp_path / "output.mp4",
        fps=30,
        width=2,
        height=2,
        ffmpeg_path="ffmpeg",
        quality=23,
    )

    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    writer.release()

    assert len(process.stdin.data) == 12


def test_ffmpeg_writer_kills_process_after_shutdown_timeout(
    tmp_path: Path,
    monkeypatch,
):
    class HungProcess:
        def __init__(self):
            self.stdin = BytesIO()
            self.stderr = BytesIO()
            self.killed = False

        def poll(self):
            return None

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return -9

        def kill(self):
            self.killed = True

    process = HungProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    writer = FFmpegNVENCVideoWriter(
        output_path=tmp_path / "output.mp4",
        fps=30,
        width=64,
        height=48,
        ffmpeg_path="ffmpeg",
        quality=23,
        shutdown_timeout_seconds=0.1,
    )

    with pytest.raises(RuntimeError, match="shutdown timed out"):
        writer.release()

    assert process.killed


def test_ffmpeg_writer_cleans_up_after_input_close_error(
    tmp_path: Path,
    monkeypatch,
):
    class FailingCloseStdin(BytesIO):
        def close(self):
            raise OSError("pipe close failed")

    class CompletedProcess:
        def __init__(self):
            self.stdin = FailingCloseStdin()
            self.stderr = BytesIO()
            self.waited = False

        def poll(self):
            return 0

        def wait(self, timeout=None):
            self.waited = True
            return 0

        def kill(self):
            pass

    process = CompletedProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    writer = FFmpegNVENCVideoWriter(
        output_path=tmp_path / "output.mp4",
        fps=30,
        width=64,
        height=48,
        ffmpeg_path="ffmpeg",
        quality=23,
    )

    with pytest.raises(RuntimeError, match="close FFmpeg NVENC input pipe"):
        writer.release()

    assert process.waited
    assert process.stderr.closed


def test_ffmpeg_writer_retries_cleanup_when_kill_fails(
    tmp_path: Path,
    monkeypatch,
):
    class RetryProcess:
        def __init__(self):
            self.stdin = BytesIO()
            self.stderr = BytesIO()
            self.kill_calls = 0
            self.killed = False

        def poll(self):
            return None if not self.killed else -9

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return -9

        def kill(self):
            self.kill_calls += 1
            if self.kill_calls == 1:
                raise OSError("temporary kill failure")
            self.killed = True

    process = RetryProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    writer = FFmpegNVENCVideoWriter(
        output_path=tmp_path / "output.mp4",
        fps=30,
        width=64,
        height=48,
        ffmpeg_path="ffmpeg",
        quality=23,
        shutdown_timeout_seconds=0.1,
    )

    with pytest.raises(RuntimeError, match="did not stop"):
        writer.release()
    with pytest.raises(RuntimeError, match="shutdown timed out"):
        writer.release()
    writer.release()

    assert process.kill_calls == 2
    assert process.killed
