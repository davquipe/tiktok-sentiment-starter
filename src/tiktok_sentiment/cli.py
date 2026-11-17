"""Interfaz de línea de comandos para ejecutar análisis reproducible."""

import argparse
from pathlib import Path

from .dataset import load_comments
from .model import MODEL_ID, classify
from .report import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis de sentimientos en español de comentarios CSV")
    parser.add_argument("--input", type=Path, default=Path("data/raw/comments_sample.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/summary.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", type=str, default=MODEL_ID)
    args = parser.parse_args()

    comments = load_comments(args.input)
    if not comments:
        parser.error("No se encontraron comentarios válidos")

    print(f"Comentarios válidos: {len(comments)}")
    print(f"Clasificando con: {args.model}")
    predictions = classify([comment.text for comment in comments], args.model, args.batch_size)
    write_reports(comments, predictions, args.output, args.summary)
    print(f"Predicciones: {args.output}")
    print(f"Resumen: {args.summary}")
    print("Proceso completado correctamente")


if __name__ == "__main__":
    main()
