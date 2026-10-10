import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.skygrid_ai.language.build_large_corpus import (
    collect_documents, split_by_source, write_corpus, clean_markdown
)


class TestLargeCorpus(unittest.TestCase):
    def test_clean_markdown(self):
        self.assertEqual(clean_markdown("# Plane\n\nRead [telemetry](https://example.com).\n\n![pic](x.png)"), "# Plane\n\nRead telemetry.")

    def test_dedup_and_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / "repo" / "en"
            folder.mkdir(parents=True)
            (root / "repo" / ".git").mkdir()
            for n in range(10):
                (folder / f"topic{n}.md").write_text((f"# Title {n}\n" + f"Aviation telemetry and navigation document number {n}. " * 12), encoding="utf8")
            (folder / "duplicate.md").write_text((folder / "topic0.md").read_text(encoding="utf8"), encoding="utf8")
            source = {"name":"test", "folder":"repo", "directories":("en",), "repo":"Org/Repo", "license":"CC BY 4.0", "license_url":"https://creativecommons.org/licenses/by/4.0/"}
            with patch("src.skygrid_ai.language.build_large_corpus.subprocess.check_output", return_value="a" * 40):
                docs, counts = collect_documents((source,), repos=root)
            self.assertEqual(len(docs), 10)
            self.assertEqual(counts[("test", "duplicate_skip")], 1)
            groups = split_by_source(docs)
            self.assertEqual([len(groups[x]) for x in ("train", "val", "test")], [8, 1, 1])
            summary = write_corpus(groups, root / "out")
            self.assertEqual(summary["train"]["documents"], 8)
            self.assertTrue((root / "out" / "manifest.json").exists())
            self.assertEqual(len(set(d["sha256_cleaned"] for group in groups.values() for d in group)), 10)
