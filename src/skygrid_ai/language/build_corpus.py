
import hashlib
import json
import random
from pathlib import Path

from .tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "data" / "aviation" / "examples"
OUTPUT = ROOT / "data" / "aviation" / "processed_v07"


def load_documents(folder):
    entries = json.loads(
        (folder / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(entries, list):
        raise ValueError("Manifest must be a list")

    documents = []
    filenames = set()
    fingerprints = set()

    for entry in entries:
        filename = entry["filename"]
        source = entry["source"]
        permission = entry["rights_status"]

        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".txt")
        ):
            raise ValueError("Invalid document filename")

        if filename in filenames:
            raise ValueError("Repeated filename")

        if not isinstance(source, str) or not source.strip():
            raise ValueError("Every document needs a source")

        if permission not in ("tutorial_demo", "approved"):
            raise ValueError("Record document rights status")

        content = (
            folder / filename
        ).read_text(encoding="utf-8").strip()

        if len(content) < 80:
            raise ValueError(
                f"Document too short: {filename}"
            )

        digest = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        if digest in fingerprints:
            raise ValueError(
                f"Duplicate document: {filename}"
            )

        filenames.add(filename)
        fingerprints.add(digest)

        documents.append({
            "filename": filename,
            "source": source,
            "rights_status": permission,
            "sha256": digest,
            "text": content,
        })

    if len(documents) < 3:
        raise ValueError(
            "Add at least three documents"
        )

    return documents


def split_documents(documents, seed=42):
    if len(documents) < 3:
        raise ValueError(
            "At least three documents required"
        )

    shuffled = sorted(
        documents,
        key=lambda item: item["filename"]
    )

    random.Random(seed).shuffle(shuffled)

    n_holdout = max(
        1, int(len(shuffled) * 0.20)
    )

    return {
        "train": shuffled[2 * n_holdout:],
        "val": shuffled[n_holdout:2 * n_holdout],
        "test": shuffled[:n_holdout],
    }


def main():
    documents = load_documents(SOURCES)
    splits = split_documents(documents)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    train_text = "\n\n".join(
        doc["text"] for doc in splits["train"]
    )

    tokenizer = CharTokenizer.train(train_text)
    tokenizer.save(OUTPUT / "vocab.json")

    index = {}

    for group, docs in splits.items():
        combined = "\n\n".join(
            doc["text"] for doc in docs
        )

        (OUTPUT / f"{group}.txt").write_text(
            combined,
            encoding="utf-8"
        )

        token_ids = tokenizer.encode(combined)

        index[group] = [
            {
                key: value
                for key, value in doc.items()
                if key != "text"
            }
            for doc in docs
        ]

        print(
            f"{group}: {len(docs)} documents, "
            f"{len(token_ids)} characters, "
            f"{token_ids.count(1)} unknown tokens"
        )

    (OUTPUT / "manifest.json").write_text(
        json.dumps(index, indent=2),
        encoding="utf-8"
    )

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Output: {OUTPUT}")
    print("Demo corpus, not a trained aviation model.")


if __name__ == "__main__":
    main()
