
import json
import tempfile
import unittest
from pathlib import Path

from src.skygrid_ai.language.build_corpus import (
    load_documents,
    split_documents,
)


class TestCorpus(unittest.TestCase):

    def make_fixture(self, folder):
        entries = []

        for n in range(5):
            name = f"topic_{n}.txt"

            (folder / name).write_text(
                f"Document {n} about safe flight science. " * 5,
                encoding="utf-8"
            )

            entries.append({
                "filename": name,
                "source": "tutorial example",
                "rights_status": "tutorial_demo"
            })

        (folder / "manifest.json").write_text(
            json.dumps(entries),
            encoding="utf-8"
        )

        return entries

    def test_document_split(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.make_fixture(folder)

            groups = split_documents(
                load_documents(folder)
            )

            self.assertEqual(
                [
                    len(groups[k])
                    for k in ("train", "val", "test")
                ],
                [3, 1, 1]
            )

            names = [
                doc["filename"]
                for group in groups.values()
                for doc in group
            ]

            self.assertEqual(
                len(names), len(set(names))
            )

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            entries = self.make_fixture(folder)

            original = (
                folder / entries[0]["filename"]
            ).read_text(encoding="utf-8")

            (
                folder / entries[1]["filename"]
            ).write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "Duplicate document"
            ):
                load_documents(folder)

    def test_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            entries = self.make_fixture(folder)

            entries[0]["source"] = ""

            (folder / "manifest.json").write_text(
                json.dumps(entries),
                encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "source"
            ):
                load_documents(folder)


if __name__ == "__main__":
    unittest.main()
