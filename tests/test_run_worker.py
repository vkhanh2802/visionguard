from threading import Event

from rq.worker import WorkerStatus

from scripts.run_worker import HeartbeatSimpleWorker


def test_windows_simple_worker_heartbeats_during_long_job():
    worker = object.__new__(HeartbeatSimpleWorker)
    calls = []
    two_heartbeats = Event()
    worker.job_monitoring_interval = 0.01
    worker.prepare_execution = lambda job: calls.append("prepare")

    def maintain_heartbeats(job):
        calls.append("heartbeat")
        if calls.count("heartbeat") == 2:
            two_heartbeats.set()

    worker.perform_job = lambda job, queue: two_heartbeats.wait(timeout=1)
    worker.maintain_heartbeats = maintain_heartbeats
    worker.set_state = lambda state: calls.append(state)

    worker.execute_job(object(), object())

    assert calls[0] == "prepare"
    assert calls.count("heartbeat") >= 2
    assert calls[-1] == WorkerStatus.IDLE
