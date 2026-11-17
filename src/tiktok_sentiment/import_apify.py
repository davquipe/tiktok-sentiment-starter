"""Importación de exportaciones autorizadas de comentarios, sin scraping local."""

import csv
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from .dataset import clean_text

_VIDEO_PATH = re.compile(r"^/@[^/]+/video/(\d+)(?:/)?$")
_COLUMN_NAMES = ("id", "video_id", "text", "create_time", "like_count")


def extract_video_id(url: str) -> str:
    """Valida la URL de un video de TikTok, sin realizar solicitudes de red."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in {
        "tiktok.com", "www.tiktok.com"
    }:
        raise ValueError("--video-url debe ser una URL https://www.tiktok.com/@usuario/video/ID")
    match = _VIDEO_PATH.fullmatch(parsed.path)
    if not match:
        raise ValueError("No se encontró el ID en --video-url (se espera /@usuario/video/ID)")
    return match.group(1)


def _read_export(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo exportado: {path}")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict) and isinstance(raw.get("items"), list):
            rows = raw["items"]
        elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
            rows = raw["data"]
        else:
            raise ValueError("JSON no reconocido; se espera una lista de comentarios")
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("La exportación JSON contiene filas que no son objetos")
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise ValueError("CSV vacío o sin encabezado")
            return list(reader)
    raise ValueError("Formato no admitido; exporta CSV o JSON desde Apify")


def _first(row: dict, *fields: str) -> str:
    for key in fields:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def import_comments(source: Path, dest: Path, video_url: str) -> dict:
    """Importa los campos necesarios y descarta datos personales del exportador."""
    video_id = extract_video_id(video_url)
    raw = _read_export(source)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    skipped = 0
    for number, item in enumerate(raw, 1):
        if item.get("error") or item.get("errorCode"):
            skipped += 1
            continue
        comment_id = _first(item, "cid", "commentId", "comment_id", "id")
        text = clean_text(_first(item, "text", "commentText", "comment_text"))
        stated_video_id = _first(item, "videoId", "video_id")
        if stated_video_id and stated_video_id != video_id:
            raise ValueError(
                f"La fila {number} pertenece al video {stated_video_id}, no al {video_id}"
            )
        if not comment_id or not text or comment_id in seen:
            skipped += 1
            continue
        count = _first(item, "diggCount", "likeCount", "like_count") or "0"
        try:
            likes = int(count)
        except ValueError as error:
            raise ValueError(f"Likes inválidos en la fila {number}: {count!r}") from error
        if likes < 0:
            raise ValueError(f"Likes negativos en la fila {number}")
        rows.append({
            "id": comment_id,
            "video_id": video_id,
            "text": text,
            "create_time": _first(item, "createTime", "create_time"),
            "like_count": str(likes),
        })
        seen.add(comment_id)
    if not rows:
        raise ValueError("No hay comentarios importables. Comprueba el video y el formato exportado")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=_COLUMN_NAMES)
        writer.writeheader()
        writer.writerows(rows)
    return {"video_id": video_id, "imported": len(rows), "skipped": skipped, "output": str(dest)}
