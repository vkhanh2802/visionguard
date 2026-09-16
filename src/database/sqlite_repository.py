import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.events import Event
from src.pipeline import PipelineResult


class SQLiteRepository:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_runs (
                        run_id TEXT PRIMARY KEY,
                        source_path TEXT NOT NULL,
                        output_path TEXT NOT NULL,
                        status TEXT NOT NULL
                            CHECK (status IN ('running', 'completed', 'failed')),
                        created_at TEXT NOT NULL,
                        completed_at TEXT,
                        error_message TEXT,
                        config_json TEXT NOT NULL,
                        processed_frames INTEGER,
                        source_fps REAL,
                        effective_fps REAL,
                        core_processing_fps REAL,
                        end_to_end_fps REAL,
                        elapsed_seconds REAL,
                        stopped_early INTEGER,
                        in_count INTEGER,
                        out_count INTEGER,
                        intrusion_count INTEGER,
                        loitering_count INTEGER
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        frame_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        track_id INTEGER NOT NULL,
                        video_timestamp REAL NOT NULL,
                        position_x INTEGER NOT NULL,
                        position_y INTEGER NOT NULL,
                        direction TEXT,
                        zone_id TEXT,
                        duration_seconds REAL,
                        logged_at TEXT NOT NULL,
                        FOREIGN KEY (run_id)
                            REFERENCES analysis_runs(run_id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_run_timestamp
                    ON events(run_id, video_timestamp);

                    CREATE INDEX IF NOT EXISTS idx_events_type
                    ON events(event_type);
                    """
                )

                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(analysis_runs)")
                }
                if "error_message" not in columns:
                    connection.execute(
                        "ALTER TABLE analysis_runs ADD COLUMN error_message TEXT"
                    )

    def create_run(
        self,
        run_id: str,
        source_path: str | Path,
        output_path: str | Path,
        config_data: dict,
    ) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO analysis_runs (
                        run_id,
                        source_path,
                        output_path,
                        status,
                        created_at,
                        config_json
                    ) VALUES (?, ?, ?, 'running', ?, ?)
                    """,
                    (
                        run_id,
                        str(source_path),
                        str(output_path),
                        self._utc_now(),
                        json.dumps(config_data, sort_keys=True, default=str),
                    ),
                )

    def record_event(
        self,
        run_id: str,
        event: Event,
        frame_id: int,
    ) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO events (
                        run_id,
                        frame_id,
                        event_type,
                        track_id,
                        video_timestamp,
                        position_x,
                        position_y,
                        direction,
                        zone_id,
                        duration_seconds,
                        logged_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(frame_id),
                        str(event.event_type),
                        int(event.track_id),
                        float(event.timestamp),
                        int(event.position[0]),
                        int(event.position[1]),
                        event.direction,
                        event.zone_id,
                        (
                            float(event.duration_seconds)
                            if event.duration_seconds is not None
                            else None
                        ),
                        self._utc_now(),
                    ),
                )

    def complete_run(
        self,
        run_id: str,
        result: PipelineResult,
    ) -> None:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE analysis_runs
                    SET status = 'completed',
                        completed_at = ?,
                        processed_frames = ?,
                        source_fps = ?,
                        effective_fps = ?,
                        core_processing_fps = ?,
                        end_to_end_fps = ?,
                        elapsed_seconds = ?,
                        stopped_early = ?,
                        in_count = ?,
                        out_count = ?,
                        intrusion_count = ?,
                        loitering_count = ?
                    WHERE run_id = ?
                    """,
                    (
                        self._utc_now(),
                        result.processed_frames,
                        result.source_fps,
                        result.effective_fps,
                        result.core_processing_fps,
                        result.end_to_end_fps,
                        result.elapsed_seconds,
                        int(result.stopped_early),
                        result.in_count,
                        result.out_count,
                        result.intrusion_count,
                        result.loitering_count,
                        run_id,
                    ),
                )

                if cursor.rowcount != 1:
                    raise KeyError(f"Unknown run_id: {run_id}")

    def fail_run(self, run_id: str, error_message: str) -> None:
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE analysis_runs
                    SET status = 'failed',
                        completed_at = ?,
                        error_message = ?
                    WHERE run_id = ?
                    """,
                    (self._utc_now(), error_message, run_id),
                )

                if cursor.rowcount != 1:
                    raise KeyError(f"Unknown run_id: {run_id}")

    def get_run(self, run_id: str) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM analysis_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        return dict(row) if row is not None else None

    def list_runs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        self._validate_pagination(limit, offset)

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM analysis_runs
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [dict(row) for row in rows]

    def list_events(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        self._validate_pagination(limit, offset)

        conditions = []
        parameters: list[object] = []

        if run_id is not None:
            conditions.append("run_id = ?")
            parameters.append(run_id)

        if event_type is not None:
            conditions.append("event_type = ?")
            parameters.append(event_type)

        query = "SELECT * FROM events"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY video_timestamp, event_id LIMIT ? OFFSET ?"
        parameters.extend([limit, offset])

        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [dict(row) for row in rows]

    def is_healthy(self) -> bool:
        try:
            with closing(self._connect()) as connection:
                connection.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            return False

        return True

    @staticmethod
    def _validate_pagination(limit: int, offset: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        if offset < 0:
            raise ValueError("offset must be non-negative")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
