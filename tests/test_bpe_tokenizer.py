
import tempfile
import unittest
from pathlib import Path

from src.skygrid_ai.language.bpe_tokenizer import (
    ByteBPETokenizer
)


class TestByteBPE(unittest.TestCase):

    def setUp(self):
        self.corpus = (
            "Flight telemetry and UAV navigation. " * 30
        )

        self.tokenizer = ByteBPETokenizer.train(
            self.corpus, 320
        )

    def test_unicode_roundtrip(self):
        samples = [
            "",
            "aircraft",
            "GPS, MAVLink, PX4",
            "Temp −5°C — café 🚁",
            "飞行高度",
        ]

        for text in samples:
            self.assertEqual(
                self.tokenizer.decode(
                    self.tokenizer.encode(text)
                ),
                text
            )

    def test_repeatable_merges(self):
        second = ByteBPETokenizer.train(
            self.corpus, 320
        )

        self.assertEqual(
            self.tokenizer.merges,
            second.merges
        )

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tokenizer.json"

            self.tokenizer.save(path)

            restored = ByteBPETokenizer.load(path)

            self.assertEqual(
                restored.encode("drone flight"),
                self.tokenizer.encode("drone flight")
            )

            self.assertEqual(
                restored.fingerprint(),
                self.tokenizer.fingerprint()
            )

    def test_special_tokens(self):
        tokens = self.tokenizer.encode(
            "flight", add_eos=True
        )

        self.assertEqual(
            tokens[-1],
            ByteBPETokenizer.EOS_ID
        )

        self.assertEqual(
            self.tokenizer.decode(tokens),
            "flight"
        )

        with self.assertRaises(ValueError):
            self.tokenizer.decode(
                tokens, skip_special=False
            )

    def test_compression(self):
        encoded = self.tokenizer.encode(
            self.corpus
        )

        self.assertLess(
            len(encoded),
            len(self.corpus.encode("utf-8"))
        )

    def test_invalid_merges(self):
        with self.assertRaises(ValueError):
            ByteBPETokenizer([
                (258, 3)
            ])


if __name__ == "__main__":
    unittest.main()
