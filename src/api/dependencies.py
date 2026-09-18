from fastapi import Request
from rq import Queue

from src.database import SQLiteRepository

from .settings import ApiSettings


def get_repository(request: Request) -> SQLiteRepository:
    return request.app.state.repository


def get_settings(request: Request) -> ApiSettings:
    return request.app.state.settings


def get_queue(request: Request) -> Queue:
    return request.app.state.queue
