"""Generación de resultados reproducibles y métricas descriptivas."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import Comment
from .model import LABELS, Sentiment


def build_summary(comments: list[Comment], predictions: list[Sentiment]) -> dict:
    if len(comments) != len(predictions):
        raise ValueError("Comentarios y predicciones deben tener la misma longitud")

    overall: Counter[str] = Counter()
    video_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for comment, sentiment in zip(comments, predictions, strict=True):
        overall[sentiment.label] += 1
        video_counts[comment.video_id][sentiment.label] += 1

    def metrics(counts: Counter[str]) -> dict:
        total = sum(counts.values())
        return {
            "total": total,
            "counts": {label: counts[label] for label in LABELS},
            "percentages": {
                label: round(counts[label] * 100 / total, 2) if total else 0.0
                for label in LABELS
            },
        }

    return {
        "sample_type": "comments_from_authorized_csv",
        "note": "Los porcentajes describen solo los comentarios de la muestra; no son opiniones de toda la población.",
        "overall": metrics(overall),
        "by_video": {video_id: metrics(counts) for video_id, counts in sorted(video_counts.items())},
    }


def write_reports(
    comments: list[Comment],
    predictions: list[Sentiment],
    output_csv: Path,
    summary_json: Path,
) -> None:
    summary = build_summary(comments, predictions)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "id", "video_id", "text", "create_time", "like_count",
                "sentiment", "confidence", "prob_neg", "prob_neu", "prob_pos",
            ],
        )
        writer.writeheader()
        for comment, prediction in zip(comments, predictions, strict=True):
            writer.writerow(
                {
                    "id": comment.id,
                    "video_id": comment.video_id,
                    "text": comment.text,
                    "create_time": comment.create_time,
                    "like_count": comment.like_count,
                    "sentiment": prediction.label,
                    "confidence": round(prediction.confidence, 6),
                    "prob_neg": round(prediction.probabilities["NEG"], 6),
                    "prob_neu": round(prediction.probabilities["NEU"], 6),
                    "prob_pos": round(prediction.probabilities["POS"], 6),
                }
            )
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
