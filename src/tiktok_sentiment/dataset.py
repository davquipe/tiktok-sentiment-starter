"""Lectura, validación y deduplicación de comentarios exportados con permiso."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Comment:
    id: str
    video_id: str
    text: str
    create_time: str = ""
    like_count: int = 0


def clean_text(value: str) -> str:
    """Conserva emojis, puntuación y acentos; normaliza espacios."""
    return re.sub(r"\s+", " ", value).strip()


def load_comments(path: Path) -> list[Comment]:
    """Acepta columnas de la Research API: id, video_id y text."""
    comments: list[Comment] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        missing = {"id", "video_id", "text"} - columns
        if missing:
            raise ValueError(f"Columnas faltantes: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            comment_id = (row.get("id") or "").strip()
            video_id = (row.get("video_id") or "").strip()
            content = clean_text(row.get("text") or "")
            if not comment_id or not video_id or not content:
                continue
            if comment_id in seen_ids:
                continue
            try:
                like_count = int((row.get("like_count") or "0").strip())
            except ValueError as error:
                raise ValueError(f"like_count inválido en línea {line_number}") from error
            if like_count < 0:
                raise ValueError(f"like_count negativo en línea {line_number}")
            comments.append(
                Comment(
                    id=comment_id,
                    video_id=video_id,
                    text=content,
                    create_time=(row.get("create_time") or "").strip(),
                    like_count=like_count,
                )
            )
            seen_ids.add(comment_id)
    return comments
