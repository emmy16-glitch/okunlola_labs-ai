
import tempfile
import unittest
from pathlib import Path

from src.skygrid_ai.language.tokenizer import CharTokenizer
from src.skygrid_ai.language.prepare_text import split_paragraphs


class TestTokenizer(unittest.TestCase):

    def test_round_trip(self):
        text = "lift and drag."
        tokenizer = CharTokenizer.train(text)

        self.assertEqual(
            tokenizer.decode(tokenizer.encode(text)),
            text
        )

    def test_unknown_character(self):
        tokenizer = CharTokenizer.train("flight")

        self.assertEqual(
            tokenizer.encode("?"),
            [1]
        )

    def test_save_and_load(self):
        tokenizer = CharTokenizer.train("aircraft")

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vocab.json"

            tokenizer.save(path)
            loaded = CharTokenizer.load(path)

            self.assertEqual(
                loaded.encode("air"),
                tokenizer.encode("air")
            )

    def test_split_has_no_paragraph_overlap(self):
        sample = "\n\n".join(
            f"Paragraph {n}."
            for n in range(12)
        )

        train, val, test = split_paragraphs(sample)

        self.assertEqual(
            (len(train), len(val), len(test)),
            (8, 2, 2)
        )

        self.assertEqual(
            len(set(train + val + test)),
            12
        )


if __name__ == "__main__":
    unittest.main()
