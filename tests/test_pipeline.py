"""Pruebas offline: no descargan el modelo ni contactan TikTok."""

import tempfile
import unittest
from pathlib import Path

from tiktok_sentiment.dataset import Comment, clean_text, load_comments
from tiktok_sentiment.model import Sentiment
from tiktok_sentiment.report import build_summary, write_reports


class DatasetTests(unittest.TestCase):
    def test_clean_text_keeps_emojis_and_accents(self) -> None:
        self.assertEqual(clean_text("  Qué   genial  🔥  "), "Qué genial 🔥")

    def test_read_deduplicate_and_preserve_large_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "comments.csv"
            file.write_text(
                'id,video_id,text,like_count\n'
                '999999999999999999,8888888888888,"Muy bueno 🔥",2\n'
                '999999999999999999,8888888888888,Duplicado,3\n'
                '111111111111111111,8888888888888,"  ",0\n',
                encoding="utf-8",
            )
            comments = load_comments(file)
            self.assertEqual(len(comments), 1)
            self.assertEqual(comments[0].id, "999999999999999999")
            self.assertEqual(comments[0].text, "Muy bueno 🔥")

    def test_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "comments.csv"
            file.write_text("id,text\n1,Hola\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "video_id"):
                load_comments(file)


class ReportingTests(unittest.TestCase):
    def test_summary_and_csv(self) -> None:
        comments = [
            Comment(id="1", video_id="v1", text="Bueno"),
            Comment(id="2", video_id="v1", text="Malo"),
            Comment(id="3", video_id="v2", text="¿Cuánto cuesta?"),
        ]
        predictions = [
            Sentiment("POS", 0.9, {"POS": 0.9, "NEU": 0.08, "NEG": 0.02}),
            Sentiment("NEG", 0.8, {"POS": 0.1, "NEU": 0.1, "NEG": 0.8}),
            Sentiment("NEU", 0.7, {"POS": 0.1, "NEU": 0.7, "NEG": 0.2}),
        ]
        summary = build_summary(comments, predictions)
        self.assertEqual(summary["overall"]["total"], 3)
        self.assertEqual(summary["by_video"]["v1"]["total"], 2)
        self.assertEqual(summary["overall"]["counts"], {"NEG": 1, "NEU": 1, "POS": 1})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            write_reports(comments, predictions, target / "scores.csv", target / "summary.json")
            self.assertTrue((target / "scores.csv").is_file())
            self.assertIn('"total": 3', (target / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
