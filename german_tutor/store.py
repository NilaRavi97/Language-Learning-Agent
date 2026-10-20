"""SQLite persistence for vocabulary (Leitner spaced repetition) and the learner profile."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# Leitner box -> days until the next review. Correct answers promote a word one box,
# wrong answers send it back to box 1.
INTERVALS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
MAX_BOX = max(INTERVALS)

SCHEMA = """
CREATE TABLE IF NOT EXISTS words (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    german        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    english       TEXT NOT NULL,
    example       TEXT,
    box           INTEGER NOT NULL DEFAULT 1,
    due_date      TEXT NOT NULL,
    times_correct INTEGER NOT NULL DEFAULT 0,
    times_wrong   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mistakes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    category    TEXT NOT NULL,
    wrong       TEXT NOT NULL,
    corrected   TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_mistakes_category ON mistakes(category);
"""


@dataclass(frozen=True)
class Word:
    id: int
    german: str
    english: str
    example: str | None
    box: int
    due_date: date
    times_correct: int
    times_wrong: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Word:
        return cls(
            id=row["id"],
            german=row["german"],
            english=row["english"],
            example=row["example"],
            box=row["box"],
            due_date=date.fromisoformat(row["due_date"]),
            times_correct=row["times_correct"],
            times_wrong=row["times_wrong"],
        )


@dataclass(frozen=True)
class ReviewResult:
    word: Word
    correct: bool
    new_box: int
    next_due: date


class VocabStore:
    """Thread-safe wrapper around one SQLite connection.

    LangGraph may execute tools on worker threads, so the connection is opened with
    ``check_same_thread=False`` and every access goes through a lock.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ---------- vocabulary ----------

    def add_word(self, german: str, english: str, example: str | None = None,
                 *, today: date | None = None) -> bool:
        """Insert a word due today. Returns False if it already exists (case-insensitive)."""
        today = today or date.today()
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO words (german, english, example, due_date) VALUES (?, ?, ?, ?)",
                    (german.strip(), english.strip(), example, today.isoformat()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_word(self, word_id: int) -> Word | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM words WHERE id = ?", (word_id,)).fetchone()
        return Word.from_row(row) if row else None

    def all_words(self) -> list[Word]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM words ORDER BY id").fetchall()
        return [Word.from_row(r) for r in rows]

    def due_words(self, limit: int = 5, *, today: date | None = None) -> list[Word]:
        """Words whose review is due, weakest (lowest box) and most overdue first."""
        today = today or date.today()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM words WHERE due_date <= ? ORDER BY box, due_date, id LIMIT ?",
                (today.isoformat(), limit),
            ).fetchall()
        return [Word.from_row(r) for r in rows]

    def record_review(self, word_id: int, correct: bool,
                      *, today: date | None = None) -> ReviewResult | None:
        today = today or date.today()
        word = self.get_word(word_id)
        if word is None:
            return None
        box = min(word.box + 1, MAX_BOX) if correct else 1
        due = today + timedelta(days=INTERVALS[box])
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE words
                   SET box = ?, due_date = ?,
                       times_correct = times_correct + ?,
                       times_wrong   = times_wrong + ?
                   WHERE id = ?""",
                (box, due.isoformat(), int(correct), int(not correct), word_id),
            )
        return ReviewResult(word=word, correct=correct, new_box=box, next_due=due)

    def import_json(self, path: str | Path) -> tuple[int, int]:
        """Import ``[{"german", "english", "example"}, ...]``. Returns (added, skipped)."""
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
        added = sum(self.add_word(e["german"], e["english"], e.get("example")) for e in entries)
        return added, len(entries) - added

    # ---------- learner profile ----------

    def log_mistake(self, category: str, wrong: str, corrected: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO mistakes (category, wrong, corrected) VALUES (?, ?, ?)",
                (category.strip().lower(), wrong, corrected),
            )

    def top_mistakes(self, limit: int = 3) -> list[tuple[str, int, str, str]]:
        """Most frequent mistake categories as (category, count, latest_wrong, latest_corrected)."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT m.category, c.n, m.wrong, m.corrected
                   FROM (SELECT category, COUNT(*) AS n, MAX(id) AS last_id
                         FROM mistakes GROUP BY category) c
                   JOIN mistakes m ON m.id = c.last_id
                   ORDER BY c.n DESC, c.last_id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [(r["category"], r["n"], r["wrong"], r["corrected"]) for r in rows]

    def stats(self, *, today: date | None = None) -> dict[str, int]:
        today = today or date.today()
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) AS total,
                          COALESCE(SUM(due_date <= ?), 0) AS due,
                          COALESCE(SUM(box = ?), 0) AS mastered,
                          COALESCE(SUM(times_correct), 0) AS correct,
                          COALESCE(SUM(times_wrong), 0) AS wrong
                   FROM words""",
                (today.isoformat(), MAX_BOX),
            ).fetchone()
            mistakes = self._conn.execute("SELECT COUNT(*) FROM mistakes").fetchone()[0]
        return {**dict(row), "mistakes": mistakes}

    def profile_summary(self) -> str:
        """Short plain-text learner profile injected into the system prompts each turn."""
        s = self.stats()
        lines = [
            f"- Vocabulary: {s['total']} words saved, {s['due']} due for review, "
            f"{s['mastered']} mastered."
        ]
        top = self.top_mistakes()
        if top:
            lines.append("- Recurring mistakes (focus on these):")
            lines += [f"  - {cat} ({n}x), e.g. \"{wrong}\" -> \"{fixed}\""
                      for cat, n, wrong, fixed in top]
        else:
            lines.append("- No recurring mistakes recorded yet.")
        return "\n".join(lines)
