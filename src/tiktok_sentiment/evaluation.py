"""Evaluación reproducible de predicciones frente a etiquetas humanas independientes."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .dataset import clean_text, load_comments
from .model import LABELS


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    id: str
    video_id: str
    text: str
    gold_label: str
    predicted_label: str
    confidence: float


def _csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: columnas faltantes: {', '.join(sorted(missing))}")
        result = []
        for row in reader:
            result.append({key: value or "" for key, value in row.items() if key is not None})
    return result


def make_label_template(input_csv: Path, output_csv: Path) -> int:
    """Plantilla de etiquetado ciego: no incluye predicciones del modelo."""
    comments = load_comments(input_csv)
    if not comments:
        raise ValueError("El CSV de comentarios no contiene filas válidas")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "video_id", "text", "gold_label", "notes"])
        writer.writeheader()
        for comment in comments:
            writer.writerow({
                "id": comment.id,
                "video_id": comment.video_id,
                "text": comment.text,
                "gold_label": "",
                "notes": "",
            })
    return len(comments)


def _read_pairs(predictions_csv: Path, labels_csv: Path) -> tuple[list[EvaluationRow], dict[str, int]]:
    predictions = _csv_rows(
        predictions_csv,
        {"id", "video_id", "text", "sentiment", "confidence"},
    )
    lookup: dict[str, dict[str, str]] = {}
    for row in predictions:
        identifier = row["id"].strip()
        if not identifier:
            raise ValueError("Hay una predicción con id vacío")
        if identifier in lookup:
            raise ValueError(f"Predicción con id duplicado: {identifier}")
        if row["sentiment"].strip().upper() not in LABELS:
            raise ValueError(f"Predicción inválida para id {identifier}")
        try:
            confidence = float(row["confidence"])
        except ValueError as error:
            raise ValueError(f"Confianza inválida para id {identifier}") from error
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confianza fuera de rango para id {identifier}")
        lookup[identifier] = row

    labels = _csv_rows(labels_csv, {"id", "gold_label"})
    rows: list[EvaluationRow] = []
    seen: set[str] = set()
    skipped = 0
    for item in labels:
        identifier = item["id"].strip()
        if not identifier:
            raise ValueError("Hay una etiqueta con id vacío")
        if identifier in seen:
            raise ValueError(f"Etiqueta con id duplicado: {identifier}")
        seen.add(identifier)
        if identifier not in lookup:
            raise ValueError(f"No existe predicción para el comentario etiquetado: {identifier}")
        prediction = lookup[identifier]
        if "text" in item and clean_text(item["text"]) != clean_text(prediction["text"]):
            raise ValueError(f"El texto no coincide para id {identifier}; comprueba la versión del dataset")
        if "video_id" in item and item["video_id"].strip() != prediction["video_id"].strip():
            raise ValueError(f"El video_id no coincide para id {identifier}")
        gold = item["gold_label"].strip().upper()
        if gold in {"", "SKIP"}:
            skipped += 1
            continue
        if gold not in LABELS:
            raise ValueError(f"Etiqueta inválida para id {identifier}: {gold!r}. Usa NEG, NEU, POS o SKIP")
        rows.append(EvaluationRow(
            id=identifier,
            video_id=prediction["video_id"],
            text=prediction["text"],
            gold_label=gold,
            predicted_label=prediction["sentiment"].strip().upper(),
            confidence=float(prediction["confidence"]),
        ))
    if not rows:
        raise ValueError("No hay etiquetas válidas para evaluar (NEG, NEU o POS)")
    return rows, {"labeled_rows": len(labels), "skipped_rows": skipped, "evaluated_rows": len(rows)}


def calculate_metrics(rows: list[EvaluationRow]) -> dict:
    if not rows:
        raise ValueError("Se requiere al menos una fila evaluable")
    total = len(rows)
    matrix = {
        gold: {predicted: sum(1 for row in rows if row.gold_label == gold and row.predicted_label == predicted)
               for predicted in LABELS}
        for gold in LABELS
    }
    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[gold][label] for gold in LABELS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    correct = sum(matrix[label][label] for label in LABELS)
    warnings = [f"No hay ejemplos reales de la clase {label}; el macro-F1 no es representativo"
                for label in LABELS if per_class[label]["support"] == 0]
    return {
        "samples": total,
        "accuracy": round(correct / total, 4),
        "macro_f1": round(sum(per_class[label]["f1"] for label in LABELS) / len(LABELS), 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "matrix_definition": "filas = etiqueta humana real; columnas = predicción del modelo",
        "warnings": warnings,
    }


def write_evaluation(predictions_csv: Path, labels_csv: Path, output_dir: Path) -> dict:
    rows, coverage = _read_pairs(predictions_csv, labels_csv)
    metrics = {**calculate_metrics(rows), "labeling": coverage}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", *LABELS])
        for label in LABELS:
            writer.writerow([label, *(metrics["confusion_matrix"][label][pred] for pred in LABELS)])
    with (output_dir / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "id", "video_id", "text", "gold_label", "predicted_label", "confidence"
        ])
        writer.writeheader()
        for row in sorted(
            (row for row in rows if row.gold_label != row.predicted_label),
            key=lambda item: item.confidence,
            reverse=True,
        ):
            writer.writerow({
                "id": row.id,
                "video_id": row.video_id,
                "text": row.text,
                "gold_label": row.gold_label,
                "predicted_label": row.predicted_label,
                "confidence": f"{row.confidence:.6f}",
            })
    return metrics
