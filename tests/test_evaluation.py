"""Pruebas de evaluación totalmente offline; jamás ejecutan el transformer."""

import csv
import tempfile
import unittest
from pathlib import Path

from tiktok_sentiment.evaluation import (
    EvaluationRow, calculate_metrics, make_label_template, write_evaluation,
)


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_matrix_orientation(self) -> None:
        rows = [
            EvaluationRow("1", "v", "a", "NEG", "NEG", 0.9),
            EvaluationRow("2", "v", "b", "NEU", "POS", 0.8),
            EvaluationRow("3", "v", "c", "POS", "POS", 0.7),
        ]
        result = calculate_metrics(rows)
        self.assertEqual(result["accuracy"], 0.6667)
        self.assertEqual(result["macro_f1"], 0.5556)
        self.assertEqual(result["confusion_matrix"]["NEU"]["POS"], 1)
        self.assertEqual(result["per_class"]["POS"]["precision"], 0.5)
        self.assertEqual(result["per_class"]["POS"]["recall"], 1.0)

    def test_template_is_blind(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            source = path / "raw.csv"
            source.write_text('id,video_id,text\n1,v,"Muy bueno 🔥"\n', encoding="utf-8")
            result = path / "labels.csv"
            self.assertEqual(make_label_template(source, result), 1)
            with result.open(encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["gold_label"], "")
            self.assertNotIn("predicted_label", row)
            self.assertEqual(row["text"], "Muy bueno 🔥")

    def test_evaluation_reports_errors_and_skips(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            predictions = path / "predictions.csv"
            predictions.write_text(
                'id,video_id,text,sentiment,confidence\n'
                '1,v,Excelente,POS,0.8\n'
                '2,v,Pésimo,POS,0.95\n'
                '3,v,Quizá,NEU,0.6\n', encoding="utf-8")
            labels = path / "labels.csv"
            labels.write_text(
                'id,video_id,text,gold_label\n'
                '2,v,Pésimo,NEG\n'
                '1,v,Excelente,POS\n'
                '3,v,Quizá,SKIP\n', encoding="utf-8")
            result = write_evaluation(predictions, labels, path / "out")
            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["accuracy"], 0.5)
            self.assertEqual(result["labeling"]["skipped_rows"], 1)
            with (path / "out/errors.csv").open(encoding="utf-8") as stream:
                errors = list(csv.DictReader(stream))
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["id"], "2")
            self.assertTrue((path / "out/confusion_matrix.csv").exists())
            self.assertTrue((path / "out/metrics.json").exists())

    def test_fails_on_mismatched_comment_text(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "pred.csv").write_text(
                'id,video_id,text,sentiment,confidence\n1,v,Texto original,POS,0.9\n',encoding="utf-8")
            (path / "labels.csv").write_text(
                'id,video_id,text,gold_label\n1,v,Texto cambiado,POS\n',encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "texto no coincide"):
                write_evaluation(path / "pred.csv", path / "labels.csv", path / "out")

    def test_fails_on_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "pred.csv").write_text(
                'id,video_id,text,sentiment,confidence\n1,v,Hola,POS,0.9\n',encoding="utf-8")
            (path / "labels.csv").write_text(
                'id,gold_label\n1,POS\n1,NEG\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicado"):
                write_evaluation(path / "pred.csv", path / "labels.csv", path / "out")


if __name__ == "__main__":
    unittest.main()
