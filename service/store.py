"""SQLite dnevnik predikcija i povratnih informacija.

Zasto SQLite a ne samo log fajl: Dan 6 (monitoring) treba da postavlja pitanja tipa
"koliko je zahteva u poslednja 24 h imalo nisko poverenje" i "koje rase korisnici
najcesce ispravljaju". To je upit, ne grep. Jedan fajl, bez servera, bez zavisnosti.

Tabela `predictions` je ujedno i izvor za petlju unapredjivanja: zahtevi sa niskim
poverenjem + korisnicke ispravke su kandidati za doznacavanje labela.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id    TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    model_version TEXT NOT NULL,
    top1          TEXT,
    top3_json     TEXT NOT NULL,
    max_prob      REAL NOT NULL,
    top3_mass     REAL NOT NULL,
    entropy       REAL NOT NULL,
    dog_score     REAL,
    is_dog        INTEGER NOT NULL,
    confident     INTEGER NOT NULL,
    latency_ms    REAL NOT NULL,
    image_bytes   INTEGER,
    image_w       INTEGER,
    image_h       INTEGER
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id   TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    correct_breed TEXT NOT NULL,
    was_top1     INTEGER,
    was_top3     INTEGER,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + lock: uvicorn poziva iz threadpool-a
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def log_prediction(self, row: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO predictions (request_id, ts, model_version, top1,"
                " top3_json, max_prob, top3_mass, entropy, dog_score, is_dog, confident,"
                " latency_ms, image_bytes, image_w, image_h)"
                " VALUES (:request_id, :ts, :model_version, :top1, :top3_json, :max_prob,"
                " :top3_mass, :entropy, :dog_score, :is_dog, :confident, :latency_ms,"
                " :image_bytes, :image_w, :image_h)", row)
            self._conn.commit()

    def get_prediction(self, request_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM predictions WHERE request_id = ?", (request_id,)).fetchone()
        return dict(r) if r else None

    def log_feedback(self, row: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO feedback (request_id, ts, correct_breed, was_top1,"
                " was_top3, note) VALUES (:request_id, :ts, :correct_breed, :was_top1,"
                " :was_top3, :note)", row)
            self._conn.commit()

    def recent_signals(self, since_ts: float) -> list[dict]:
        """Signali za detekciju drifta iz zadatog prozora. Indeks na ts drzi ovo jeftinim."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, top1, max_prob, top3_mass, entropy, dog_score, confident, is_dog"
                " FROM predictions WHERE ts >= ? ORDER BY ts", (since_ts,)).fetchall()
        return [dict(r) for r in rows]

    def relabel_candidates(self, limit: int = 500, max_top3_mass: float = 1.0,
                           since_ts: float = 0.0) -> list[dict]:
        """Kandidati za doznacavanje labela: korisnicke ispravke + nisko poverenje.

        Ispravke idu prve (one su jedine sa poznatim tacnim odgovorom), pa zahtevi sa
        najnizim poverenjem. LEFT JOIN, ne dva upita, da se redosled odredi u bazi.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.request_id, p.ts, p.top1, p.top3_json, p.max_prob, p.top3_mass,"
                "       p.entropy, p.dog_score, p.is_dog, p.confident,"
                "       f.correct_breed, f.was_top1, f.was_top3, f.note"
                "  FROM predictions p LEFT JOIN feedback f USING (request_id)"
                " WHERE p.ts >= ? AND (f.correct_breed IS NOT NULL"
                "       OR p.confident = 0 OR p.is_dog = 0 OR p.top3_mass <= ?)"
                " ORDER BY (f.correct_breed IS NOT NULL) DESC, p.top3_mass ASC"
                " LIMIT ?", (since_ts, max_top3_mass, limit)).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict:
        with self._lock:
            p = self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            f = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return {"predictions": p, "feedback": f}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def top3_to_json(preds: list[dict]) -> str:
    return json.dumps(preds, ensure_ascii=False)
