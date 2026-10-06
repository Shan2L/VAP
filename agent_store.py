"""SQLite persistence for agent conversations, pending approvals and audit events."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    conversation_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    message TEXT NOT NULL,
    PRIMARY KEY (conversation_id, seq)
);
CREATE TABLE IF NOT EXISTS pending_actions (
    approval_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    tool_calls TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""
RETENTION_SEC = 30 * 24 * 3600


class AgentStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, closing(self._connect()) as db, db:
            db.executescript(SCHEMA)
            stale = [
                row[0]
                for row in db.execute(
                    "SELECT id FROM conversations WHERE updated_at < ?",
                    (time.time() - RETENTION_SEC,),
                )
            ]
            for table, column in (
                ("messages", "conversation_id"),
                ("pending_actions", "conversation_id"),
                ("events", "conversation_id"),
                ("conversations", "id"),
            ):
                db.executemany(
                    f"DELETE FROM {table} WHERE {column} = ?", [(cid,) for cid in stale]
                )
        path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10)

    def create_conversation(self) -> str:
        conversation_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_id, now, now),
            )
        return conversation_id

    def has_conversation(self, conversation_id: str) -> bool:
        with self._lock, closing(self._connect()) as db:
            row = db.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row is not None

    def load_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as db:
            rows = db.execute(
                "SELECT message FROM messages WHERE conversation_id = ? ORDER BY seq",
                (conversation_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def append_messages(
        self, conversation_id: str, messages: list[dict[str, Any]]
    ) -> None:
        with self._lock, closing(self._connect()) as db, db:
            (last,) = db.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            db.executemany(
                "INSERT INTO messages (conversation_id, seq, message) VALUES (?, ?, ?)",
                [
                    (
                        conversation_id,
                        last + offset,
                        json.dumps(message, ensure_ascii=False),
                    )
                    for offset, message in enumerate(messages, start=1)
                ],
            )
            db.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (time.time(), conversation_id),
            )

    def save_pending(
        self, approval_id: str, conversation_id: str, tool_calls: list[dict[str, Any]]
    ) -> None:
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO pending_actions (approval_id, conversation_id, tool_calls, created_at)"
                " VALUES (?, ?, ?, ?)",
                (approval_id, conversation_id, json.dumps(tool_calls), time.time()),
            )

    def pop_pending(self, approval_id: str) -> tuple[str, list[dict[str, Any]]] | None:
        with self._lock, closing(self._connect()) as db, db:
            row = db.execute(
                "SELECT conversation_id, tool_calls FROM pending_actions WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "DELETE FROM pending_actions WHERE approval_id = ?", (approval_id,)
            )
        return row[0], json.loads(row[1])

    def drop_pending_for(self, conversation_id: str) -> int:
        with self._lock, closing(self._connect()) as db, db:
            return db.execute(
                "DELETE FROM pending_actions WHERE conversation_id = ?",
                (conversation_id,),
            ).rowcount

    def record_event(
        self, conversation_id: str | None, kind: str, payload: dict[str, Any]
    ) -> None:
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO events (conversation_id, ts, kind, payload) VALUES (?, ?, ?, ?)",
                (
                    conversation_id,
                    time.time(),
                    kind,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )

    def events(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as db:
            rows = db.execute(
                "SELECT ts, kind, payload FROM events WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [
            {"ts": ts, "kind": kind, **json.loads(payload)}
            for ts, kind, payload in rows
        ]
