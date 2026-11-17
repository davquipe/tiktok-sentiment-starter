"""Pruebas de integración sin conectar a TikTok ni a Apify."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tiktok_sentiment.dataset import load_comments
from tiktok_sentiment.import_apify import extract_video_id, import_comments

URL = "https://www.tiktok.com/@keikofujimorih/video/7707793734842240277"


class ApifyImportTests(unittest.TestCase):
    def test_extract_video_id(self) -> None:
        self.assertEqual(extract_video_id(URL), "7707793734842240277")
        with self.assertRaises(ValueError):
            extract_video_id("https://example.org/@x/video/123")
        with self.assertRaises(ValueError):
            extract_video_id("https://www.tiktok.com/@x")

    def test_json_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "apify.json", Path(tmp) / "out.csv"
            source.write_text(json.dumps([
                {"cid": "7500000000000000001", "videoId": "7707793734842240277", "text": "  Qué   genial 🔥  ", "diggCount": 11, "createTime": 123},
                {"cid": "7500000000000000002", "text": "No me convenció", "diggCount": 3},
                {"cid": "7500000000000000002", "text": "Duplicado", "diggCount": 1},
                {"url": URL, "error": "Not found", "errorCode": "POST_NOT_FOUND_OR_PRIVATE"},
            ], ensure_ascii=False), encoding="utf-8")
            result = import_comments(source, target, URL)
            self.assertEqual((result["imported"], result["skipped"]), (2, 2))
            comments = load_comments(target)
            self.assertEqual(comments[0].text, "Qué genial 🔥")
            self.assertEqual(comments[0].id, "7500000000000000001")
            self.assertEqual(comments[0].like_count, 11)
            self.assertEqual(comments[0].video_id, "7707793734842240277")

    def test_csv_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "export.csv", Path(tmp) / "processed.csv"
            source.write_text('cid,text,videoId,diggCount,uniqueId\n55,"Me gusta 👍",7707793734842240277,9,persona\n', encoding="utf-8")
            import_comments(source, target, URL)
            with target.open(encoding="utf-8", newline="") as file:
                data = list(csv.DictReader(file))
            self.assertEqual(data[0]["text"], "Me gusta 👍")
            self.assertNotIn("uniqueId", data[0])

    def test_wrong_video_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "export.json", Path(tmp) / "processed.csv"
            source.write_text(json.dumps([{"cid": "123", "text": "Hola", "videoId": "999"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pertenece al video"):
                import_comments(source, target, URL)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
