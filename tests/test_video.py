from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video import (
    AsyncVideoWriter,
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
