import os
from threading import Event, Thread

from redis import Redis
from redis.exceptions import RedisError
from rq import SimpleWorker, Worker
from rq.worker import WorkerStatus

from src.api.settings import ApiSettings


class HeartbeatSimpleWorker(SimpleWorker):
    """Keep Windows worker registration current while a job runs in-process."""

    def execute_job(self, job, queue):
        self.prepare_execution(job)
        stop_heartbeat = Event()
        heartbeat_thread = Thread(
            target=self._maintain_job_heartbeat,
            args=(job, stop_heartbeat),
            name="rq-job-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            self.perform_job(job, queue)
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join()
        self.set_state(WorkerStatus.IDLE)

    def _maintain_job_heartbeat(self, job, stop_heartbeat: Event) -> None:
        while not stop_heartbeat.wait(self.job_monitoring_interval):
            try:
                self.maintain_heartbeats(job)
            except RedisError as error:
                self.log.warning("Could not maintain worker heartbeat: %s", error)


def main() -> None:
    settings = ApiSettings.from_environment()
    connection = Redis.from_url(settings.redis_url)
    worker_class = HeartbeatSimpleWorker if os.name == "nt" else Worker
    worker = worker_class([settings.queue_name], connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
