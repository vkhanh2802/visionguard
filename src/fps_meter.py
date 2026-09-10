import time
from collections import deque


class FPSMeter:
    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)
        self.start_time = None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("FPSMeter.start() must be called before stop().")

        elapsed = time.perf_counter() - self.start_time
        self.timestamps.append(elapsed)
        return self.fps

    @property
    def fps(self) -> float:
        if not self.timestamps:
            return 0.0

        average_time = sum(self.timestamps) / len(self.timestamps)
        return 1.0 / average_time if average_time > 0 else 0.0