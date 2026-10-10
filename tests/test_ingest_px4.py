
import tempfile
import unittest
from pathlib import Path

from src.skygrid_ai.language.ingest_px4 import (
    ingest_documents,
    normalize_markdown,
)


class TestPX4Ingestion(unittest.TestCase):

    def test_markdown_cleanup(self):
        value = normalize_markdown(
            "<!--remove-->\n"
            "See [telemetry](url) "
            "![photo](img.png)"
        )

        self.assertIn("telemetry", value)
        self.assertNotIn("<!--", value)
        self.assertNotIn("img.png", value)

    def test_source_manifest_and_deduplication(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder) / "repo"
            output = Path(folder) / "output"

            target = repo / "en" / "concept"
            target.mkdir(parents=True)

            text = (
                "Vehicle flight modes and telemetry "
                "are engineering concepts. "
            ) * 9

            (target / "first.md").write_text(
                text, encoding="utf-8"
            )

            (target / "duplicate.md").write_text(
                text, encoding="utf-8"
            )

            revision = "a" * 40

            items = ingest_documents(
                repo, output, revision
            )

            self.assertEqual(len(items), 1)

            self.assertIn(
                "/blob/" + revision,
                items[0]["source_url"],
            )

            self.assertTrue(
                (
                    output / "docs" / items[0]["file"]
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
