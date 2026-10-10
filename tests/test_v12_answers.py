import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.skygrid_ai.knowledge.grounded import (
    answer_from_evidence, load_verified_document, candidate_sentences,
)
from src.skygrid_ai.knowledge.retrieval import SearchIndex


class TestV12AnswerQuality(unittest.TestCase):
    def prepare(self, directory):
        root = Path(directory)
        (root / "docs").mkdir()
        docs = [
            ("mavlink", "en/services/heartbeat.md", "hb", (
                "# Heartbeat/Connection Protocol\n\n"
                "## Connecting to a GCS or MAVLink API {#gcs}\n\n"
                "The heartbeat protocol is used to advertise the existence of a system on the MAVLink network, along with its component identifier and flight mode.\n\n"
                "Components send heartbeat messages repeatedly to make network presence discoverable.\n"
            )),
            ("px4", "en/concept/control_allocation.md", "ca", (
                "# Control Allocation (Mixing)\n\n"
                "Control allocation replaces the legacy mixing approach used in PX4 v1.13 and earlier.\n\n"
                "PX4 takes desired torque and thrust commands from the core controllers and translates them to actuator commands which control motors or servos.\n"
            )),
        ]
        manifest, entries = [], []
        for source, path, name, body in docs:
            (root / "docs" / f"{name}.txt").write_text(body, encoding="utf-8")
            url = f"https://example.com/{source}/{name}"
            manifest.append({
                "source": source, "path": path, "text_file": f"docs/{name}.txt",
                "url": url,
                "sha256_cleaned": hashlib.sha256(
                    " ".join(body.split()).casefold().encode("utf-8")
                ).hexdigest(),
            })
            # Simulate the flattened, possibly truncated BM25 passage data.
            entries.append({
                "id": f"{source}:{path}:0", "source": source,
                "path": path, "url": url,
                "text": " ".join(body.split())[:190],
            })
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return SearchIndex(entries)

    def test_heartbeat_answer_not_heading_or_truncated_excerpt(self):
        with tempfile.TemporaryDirectory() as folder:
            index = self.prepare(folder)
            result = answer_from_evidence(
                index, "What does MAVLink HEARTBEAT do?", corpus=folder
            )
            self.assertEqual(result["status"], "excerpt_found")
            self.assertIn("advertise the existence", result["answer"])
            self.assertTrue(result["answer"].endswith("."))
            self.assertNotIn("##", result["answer"])
            self.assertEqual(result["citations"][0]["url"], "https://example.com/mavlink/hb")

    def test_px4_chooses_explanation_not_historical_heading(self):
        with tempfile.TemporaryDirectory() as folder:
            index = self.prepare(folder)
            result = answer_from_evidence(
                index, "How does PX4 handle control allocation?", corpus=folder
            )
            self.assertEqual(result["status"], "excerpt_found")
            self.assertIn("torque and thrust", result["answer"])
            self.assertIn("translates", result["answer"])

    def test_original_doc_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            index = self.prepare(folder)
            (Path(folder) / "docs/hb.txt").write_text("altered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                answer_from_evidence(
                    index, "What does MAVLink HEARTBEAT do?", corpus=folder
                )

    def test_broken_passage_fragment_is_not_returned(self):
        index = SearchIndex([{
            "id": "m:0", "source": "mavlink", "path": "en/services/heartbeat.md",
            "url": "https://example.com/m",
            "text": "## Connecting to a GCS or MAVLink API {#gcs} The HEARTBEAT may also used by GCS to determine if it",
        }])
        out = answer_from_evidence(index, "What does MAVLink HEARTBEAT do?")
        self.assertEqual(out["status"], "abstained")

    def test_separate_system_source_mismatch_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            index = self.prepare(folder)
            out = answer_from_evidence(
                index, "How does MAVLink handle control allocation?", corpus=folder
            )
            self.assertEqual(out["status"], "source_mismatch")
