import sqlite3
import threading
from pathlib import Path

import psycopg

from src.schemas import DecisionStatus, EscalationPacket, HumanReviewOutcome

OPEN = "('pending_human_review', 'escalated')"


class ReviewStore:
    def __init__(self, sqlite_path: str, database_url: str = ""):
        self._postgres = bool(database_url)
        self._lock = threading.Lock()
        if self._postgres:
            self._conn = psycopg.connect(database_url, autocommit=True, prepare_threshold=None)
            created = "TIMESTAMPTZ DEFAULT now()"
        else:
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(sqlite_path, check_same_thread=False, isolation_level=None)
            created = "TEXT DEFAULT CURRENT_TIMESTAMP"
        self._run(
            "CREATE TABLE IF NOT EXISTS reviews ("
            "submission_id TEXT PRIMARY KEY, packet TEXT NOT NULL, "
            f"status TEXT NOT NULL, outcome TEXT, created_at {created})"
        )

    def _run(self, query: str, params: tuple = ()):
        if self._postgres:
            query = query.replace("?", "%s")
        with self._lock:
            try:
                cursor = self._conn.execute(query, params)
                return cursor.fetchall() if cursor.description else cursor.rowcount
            except psycopg.OperationalError:
                self._conn = psycopg.connect(self._conn.info.dsn, autocommit=True, prepare_threshold=None)
                raise

    def add(self, packet: EscalationPacket) -> None:
        self._run(
            "INSERT INTO reviews (submission_id, packet, status) VALUES (?, ?, ?) "
            "ON CONFLICT (submission_id) DO UPDATE SET packet = excluded.packet, status = excluded.status",
            (packet.submission_id, packet.model_dump_json(), packet.status.value),
        )

    def list(self, pending_only: bool = True) -> list[EscalationPacket]:
        where = f" WHERE status IN {OPEN}" if pending_only else ""
        rows = self._run(f"SELECT packet, status FROM reviews{where} ORDER BY created_at DESC")
        packets = [EscalationPacket.model_validate_json(packet) for packet, _ in rows]
        for packet, (_, status) in zip(packets, rows):
            packet.status = DecisionStatus(status)
        return packets

    def decide(self, outcome: HumanReviewOutcome) -> bool:
        return self._run(
            f"UPDATE reviews SET status = ?, outcome = ? WHERE submission_id = ? AND status IN {OPEN}",
            (outcome.final_status.value, outcome.model_dump_json(), outcome.submission_id),
        ) == 1
