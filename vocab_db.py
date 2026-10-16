import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path("tutor.db")
INTERVALS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}   # box -> days until next review


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row             # rows behave like dicts: row["german"]
    conn.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            german        TEXT NOT NULL UNIQUE COLLATE NOCASE,
            english       TEXT NOT NULL,
            example       TEXT,
            box           INTEGER NOT NULL DEFAULT 1,
            due_date      TEXT NOT NULL,
            times_correct INTEGER NOT NULL DEFAULT 0,
            times_wrong   INTEGER NOT NULL DEFAULT 0
        )
    """)
    return conn


def add_word(german, english, example):
    """Returns True if added, False if the word already exists."""
    with connect() as conn:
        try:
            conn.execute(
                "INSERT INTO words (german, english, example, due_date) VALUES (?, ?, ?, ?)",
                (german, english, example, date.today().isoformat()),
            )
            return True
        except sqlite3.IntegrityError:          # UNIQUE constraint → duplicate
            return False


def all_words():
    with connect() as conn:
        return conn.execute("SELECT * FROM words ORDER BY id").fetchall()


def due_words(limit=5):
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM words WHERE due_date <= ? ORDER BY box, due_date LIMIT ?",
            (date.today().isoformat(), limit),
        ).fetchall()


def record_review(word_id, correct):
    """Move the word between boxes. Returns (new_box, next_due_date) or None if not found."""
    with connect() as conn:
        row = conn.execute("SELECT box FROM words WHERE id = ?", (word_id,)).fetchone()
        if row is None:
            return None
        box = min(row["box"] + 1, 5) if correct else 1
        due = date.today() + timedelta(days=INTERVALS[box])
        conn.execute(
            """UPDATE words
               SET box = ?, due_date = ?,
                   times_correct = times_correct + ?,
                   times_wrong   = times_wrong + ?
               WHERE id = ?""",
            (box, due.isoformat(), int(correct), int(not correct), word_id),
        )
        return box, due


# Run this file directly once to import your words from step 4:
#   python vocab_db.py
if __name__ == "__main__":
    import json
    for w in json.loads(Path("words.json").read_text(encoding="utf-8")):
        added = add_word(w["german"], w["english"], w["example"])
        print(w["german"], "→", "added" if added else "already there")
