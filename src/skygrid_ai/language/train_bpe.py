
import json
from pathlib import Path

from .bpe_tokenizer import ByteBPETokenizer


ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    ROOT / "data" / "aviation" / "processed_v07"
)

OUTPUT = (
    ROOT / "data" / "aviation" / "processed_v08"
)


def main():
    train_path = SOURCE / "train.txt"

    if not train_path.exists():
        raise FileNotFoundError(
            "Run build_corpus.py before train_bpe.py"
        )

    train_text = train_path.read_text(
        encoding="utf-8"
    )

    # Learn merge rules ONLY from training data.
    tokenizer = ByteBPETokenizer.train(
        train_text,
        target_vocab_size=320
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save(
        OUTPUT / "bpe_tokenizer.json"
    )

    report = {}

    for split in ("train", "val", "test"):
        text = (
            SOURCE / f"{split}.txt"
        ).read_text(encoding="utf-8")

        tokens = tokenizer.encode(
            text, add_eos=True
        )

        # Check that encoding did not lose text.
        if tokenizer.decode(tokens) != text:
            raise RuntimeError(
                f"{split} round-trip failed"
            )

        (
            OUTPUT / f"{split}_tokens.json"
        ).write_text(
            json.dumps(tokens),
            encoding="utf-8"
        )

        byte_count = len(text.encode("utf-8"))
        token_count = len(tokens) - 1

        report[split] = {
            "bytes": byte_count,
            "tokens": token_count,
        }

        print(
            f"{split}: {byte_count} bytes -> "
            f"{token_count} BPE tokens"
        )

    print(
        f"Vocabulary size: {tokenizer.vocab_size}"
    )

    print(
        f"Tokenizer fingerprint: "
        f"{tokenizer.fingerprint()[:16]}"
    )

    print(f"Saved files to: {OUTPUT}")


if __name__ == "__main__":
    main()
