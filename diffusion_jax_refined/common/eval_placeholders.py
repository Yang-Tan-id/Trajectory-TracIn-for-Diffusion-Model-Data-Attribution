from __future__ import annotations

from pathlib import Path


def write_eval_note(path: str | Path, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

