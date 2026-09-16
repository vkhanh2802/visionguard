from fastapi import Request

from src.database import SQLiteRepository


def get_repository(request: Request) -> SQLiteRepository:
    return request.app.state.repository
