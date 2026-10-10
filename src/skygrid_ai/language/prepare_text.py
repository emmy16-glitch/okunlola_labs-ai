
import random
from pathlib import Path

from .tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    ROOT / "data" / "aviation"
    / "examples" / "intro_to_flight.txt"
)

OUT = ROOT / "data" / "aviation" / "processed"


def split_paragraphs(text, seed=42):
    paragraphs = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]

    if len(paragraphs) < 10:
        raise ValueError(
            "Need at least 10 distinct paragraphs"
        )

    if len(set(paragraphs)) != len(paragraphs):
        raise ValueError(
            "Remove duplicate paragraphs first"
        )

    rng = random.Random(seed)
    rng.shuffle(paragraphs)

    n = len(paragraphs)
    n_test = max(1, int(n * 0.20))
    n_val = max(1, int(n * 0.20))

    return (
        paragraphs[n_test + n_val:],
        paragraphs[n_test:n_test + n_val],
        paragraphs[:n_test]
    )


def main():
    train, val, test = split_paragraphs(
        SOURCE.read_text(encoding="utf-8")
    )

    OUT.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": "\n\n".join(train),
        "val": "\n\n".join(val),
        "test": "\n\n".join(test)
    }

    # Learn vocabulary from training text only.
    tokenizer = CharTokenizer.train(splits["train"])

    tokenizer.save(OUT / "vocab.json")

    for name, text in splits.items():
        (OUT / f"{name}.txt").write_text(
            text,
            encoding="utf-8"
        )

        token_ids = tokenizer.encode(text)

        print(
            f"{name}: {len(text)} characters, "
            f"{len(token_ids)} tokens, "
            f"{token_ids.count(1)} unknown"
        )

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Files saved in: {OUT}")


if __name__ == "__main__":
    main()
