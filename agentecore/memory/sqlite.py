import json
import sqlite3
import threading
import time
from pathlib import Path

from agentecore.type.type import Block, Memory, Role, Turn

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    blocks     TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, id);
"""


class SQLiteMemory(Memory):
    """Memory persistente su SQLite (modulo standard, nessuna dipendenza).

    È opzionale: un `Agent` funziona anche con la `Memory` normale. I turni
    della sessione vengono caricati in RAM all'apertura e ogni `add` li scrive
    anche su disco, quindi le letture non toccano il database.

    Più sessioni (es. un utente ciascuna) possono condividere lo stesso file
    usando `session_id` diversi. Non aprire la stessa sessione in due oggetti
    nello stesso momento: ognuno terrebbe una copia in RAM non aggiornata.
    """

    def __init__(self, path: str | Path = "memory.db", session_id: str = "default"):
        super().__init__()
        self.session_id = session_id
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self.turns = self._load()

    def _load(self) -> list[Turn]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, blocks FROM turns WHERE session_id = ? ORDER BY id",
                (self.session_id,),
            ).fetchall()
        return [
            Turn(Role(role), [Block.from_dict(b) for b in json.loads(blocks)])
            for role, blocks in rows
        ]

    def add(self, role: Role, blocks: list[Block]) -> None:
        payload = json.dumps([b.to_dict() for b in blocks], ensure_ascii=False)
        # Prima il disco, poi la RAM: se la scrittura fallisce le due non si disallineano.
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO turns (session_id, role, blocks, created_at) VALUES (?, ?, ?, ?)",
                (self.session_id, role.value, payload, time.time()),
            )
        self.turns.append(Turn(role, blocks))

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM turns WHERE session_id = ?", (self.session_id,))
        self.turns.clear()

    def list_sessions(self) -> list[dict]:
        """Le sessioni presenti nel file, con numero di turni e ultimo aggiornamento."""
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id, COUNT(*), MAX(created_at) FROM turns "
                "GROUP BY session_id ORDER BY MAX(created_at) DESC"
            ).fetchall()
        return [{"session_id": s, "turns": n, "updated_at": t} for s, n, t in rows]

    def delete_session(self, session_id: str) -> None:
        """Elimina un'altra sessione dal file. Per la sessione corrente usa clear()."""
        if session_id == self.session_id:
            self.clear()
            return
        with self._lock, self._db:
            self._db.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
