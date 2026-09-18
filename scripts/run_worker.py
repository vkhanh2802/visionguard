import os

from redis import Redis
from rq import SimpleWorker, Worker

from src.api.settings import ApiSettings


def main() -> None:
    settings = ApiSettings.from_environment()
    connection = Redis.from_url(settings.redis_url)
    worker_class = SimpleWorker if os.name == "nt" else Worker
    worker = worker_class([settings.queue_name], connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
