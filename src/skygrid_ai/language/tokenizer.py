
import json
from pathlib import Path


class CharTokenizer:
    """Our own character-level tokenizer."""

    def __init__(self, vocabulary):
        self.vocab = list(vocabulary)

        self.char_to_id = {
            char: index
            for index, char in enumerate(self.vocab)
        }

    @classmethod
    def train(cls, text):
        characters = sorted(set(text))

        return cls([
            "<PAD>",
            "<UNK>",
            *characters
        ])

    def encode(self, text):
        return [
            self.char_to_id.get(char, 1)
            for char in text
        ]

    def decode(self, token_ids):
        return "".join(
            "?" if int(token_id) == 1
            else self.vocab[int(token_id)]
            for token_id in token_ids
            if int(token_id) != 0
        )

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {"vocab": self.vocab},
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        return cls(data["vocab"])
