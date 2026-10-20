"""Runtime settings, read from environment variables (and .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


@dataclass(frozen=True)
class Settings:
    model: str = "gemini-3.6-flash"
    level: str = "A2"
    data_dir: Path = Path(".tutor")
    history_window: int = 30  # max messages sent to the model per call
    requests_per_minute: float | None = None  # client-side throttle, e.g. 5 for the free tier

    def __post_init__(self) -> None:
        if self.level not in CEFR_LEVELS:
            raise ValueError(f"level must be one of {CEFR_LEVELS}, got {self.level!r}")
        if self.requests_per_minute is not None and self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")

    @property
    def vocab_db(self) -> Path:
        return self.data_dir / "vocab.db"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @classmethod
    def from_env(cls) -> Settings:
        base = cls()
        rpm = os.getenv("TUTOR_RPM")
        return replace(
            base,
            model=os.getenv("TUTOR_MODEL", base.model),
            level=os.getenv("TUTOR_LEVEL", base.level).upper(),
            data_dir=Path(os.getenv("TUTOR_DATA_DIR", str(base.data_dir))),
            requests_per_minute=float(rpm) if rpm else None,
        )
