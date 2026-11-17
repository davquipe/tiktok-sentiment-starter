"""Comandos para anotación humana y evaluación objetiva del modelo."""

import argparse
from pathlib import Path

from .evaluation import make_label_template, write_evaluation


def label_main() -> None:
    parser = argparse.ArgumentParser(description="Genera un CSV para etiquetado humano ciego")
    parser.add_argument("--input", type=Path, default=Path("data/raw/comments_sample.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/labeled/to_label.csv"))
    args = parser.parse_args()
    count = make_label_template(args.input, args.output)
    print(f"Plantilla preparada: {args.output} ({count} comentarios)")
    print("Rellena gold_label con NEG, NEU, POS o SKIP. No consultes las predicciones al etiquetar.")


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa predicciones contra etiquetas humanas")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/labeled/comments_labeled_sample.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    result = write_evaluation(args.predictions, args.labels, args.output)
    print(f"Evaluados: {result['samples']} (omitidos: {result['labeling']['skipped_rows']})")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    for label, details in result["per_class"].items():
        print(f"  {label}: precision={details['precision']:.4f}, "
              f"recall={details['recall']:.4f}, f1={details['f1']:.4f}, n={details['support']}")
    for warning in result["warnings"]:
        print(f"Advertencia: {warning}")
    print(f"Reportes: {args.output / 'metrics.json'}, confusion_matrix.csv, errors.csv")


if __name__ == "__main__":
    evaluate_main()
