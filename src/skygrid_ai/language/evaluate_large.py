"""SkyGrid AI v0.9d: full-document validation and held-out test evaluation.

Runs with v0.9c best/last checkpoints and v0.9b token binaries.
No training and no checkpoint mutation. Example:
python -m src.skygrid_ai.language.evaluate_large --split val
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .large_bpe import LargeByteBPE, PAD_ID, EOS_ID, read_token_ids
from .modern_transformer import SkyGridTransformerV08
from .train_large_transformer import DATA, OUTPUT as TRAINING_OUTPUT, sha256_file

OUTPUT = TRAINING_OUTPUT.parent / "v09d"


def separate_documents(token_ids):
    """Split at EOS and reject missing delimiters or unexpected padding."""
    documents = []
    current = []
    for token in token_ids:
        token = int(token)
        if token == PAD_ID:
            raise ValueError("Unexpected padding token in evaluation data")
        current.append(token)
        if token == EOS_ID:
            if len(current) < 2:
                raise ValueError("Document has no text tokens")
            documents.append(current)
            current = []
    if current:
        raise ValueError("Evaluation data ends without an EOS token")
    if not documents:
        raise ValueError("No evaluation documents")
    return documents


def validate_tokens(folder, split, tokenizer):
    """Validate local split against the original v0.9b preparation report."""
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("format") != "uint32_le":
        raise ValueError("Unexpected token-file format")
    if (report.get("vocabulary_size") != tokenizer.vocab_size or
            report.get("tokenizer_fingerprint") != tokenizer.fingerprint()):
        raise ValueError("Tokenizer and preparation report disagree")
    ids = read_token_ids(folder / f"{split}.bin")
    if any(int(t) < 0 or int(t) >= tokenizer.vocab_size for t in ids):
        raise ValueError("Token ID outside the vocabulary")
    expected = report["splits"][split]
    if len(ids) != expected["token_count_with_eos"]:
        raise ValueError("Prepared token count changed")
    documents = separate_documents(ids)
    if len(documents) != expected["eos_tokens"] or len(documents) != expected["documents"]:
        raise ValueError("Document or EOS count changed")
    sha = hashlib.sha256()
    total_bytes = 0
    for doc in documents:
        # Full documents must decode losslessly; verifies the original source-byte digest.
        raw = tokenizer.decode(doc[:-1]).encode("utf-8")
        sha.update(raw)
        total_bytes += len(raw)
    if (sha.hexdigest() != expected["sha256_utf8_text"] or
            total_bytes != expected["utf8_bytes"]):
        raise ValueError("Evaluation token text differs from preparation report")
    return documents, {"tokens": len(ids), "documents": len(documents), "utf8_bytes": total_bytes}


def score_documents(model, documents, tokenizer, *, context_size, batch_size=8):
    """Score every next-token target once, resetting context at document boundaries.

    First token of each document cannot be predicted without a BOS token, and is
    excluded. EOS targets are included. Full windows are evaluated in batches;
    shorter trailing windows are evaluated individually (never padded into loss).
    """
    if context_size < 1 or batch_size < 1:
        raise ValueError("context_size and batch_size must be positive")
    model.eval()
    loss_sum = 0.0
    target_count = 0
    scored_bytes = 0
    windows = 0
    pending_x, pending_y = [], []

    def run_batch(xs, ys):
        nonlocal loss_sum, target_count, windows
        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        _, loss = model(x, y)
        loss_sum += float(loss) * y.numel()
        target_count += y.numel()
        windows += len(xs)

    with torch.inference_mode():
        for doc in documents:
            if len(doc) < 2 or doc[-1] != EOS_ID:
                raise ValueError("Invalid EOS-terminated document")
            scored_bytes += sum(len(tokenizer.id_to_bytes[int(t)])
                                for t in doc[1:] if int(t) != EOS_ID)
            for start in range(0, len(doc) - 1, context_size):
                end = min(start + context_size, len(doc) - 1)
                x, y = doc[start:end], doc[start + 1:end + 1]
                if len(x) == context_size:
                    pending_x.append(x)
                    pending_y.append(y)
                    if len(pending_x) >= batch_size:
                        run_batch(pending_x, pending_y)
                        pending_x, pending_y = [], []
                else:
                    # This short window's position sequence resets to zero.
                    run_batch([x], [y])
        if pending_x:
            run_batch(pending_x, pending_y)
    if target_count == 0 or scored_bytes == 0:
        raise ValueError("Not enough evaluable text")
    nats = loss_sum / target_count
    return {
        "evaluated_targets": target_count,
        "scored_utf8_bytes_excluding_first_token_per_document": scored_bytes,
        "evaluated_windows": windows,
        "cross_entropy_nats_per_token": nats,
        "perplexity": math.exp(nats),
        "bits_per_scored_byte_including_eos_cost": loss_sum / (math.log(2) * scored_bytes),
        "uniform_token_baseline_nats": math.log(tokenizer.vocab_size),
    }


def evaluate(split="val", checkpoint_name="best", *, data=DATA,
             training_output=TRAINING_OUTPUT, output=OUTPUT, batch_size=8):
    """Evaluate an existing v0.9c checkpoint without updating training files."""
    if split not in ("val", "test") or checkpoint_name not in ("best", "last"):
        raise ValueError("Choose split val/test and checkpoint best/last")
    tokenizer = LargeByteBPE.load(Path(data) / "tokenizer.json")
    checkpoint_path = Path(training_output) / f"{checkpoint_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != "skygrid_v09c_1":
        raise ValueError("Expected a v0.9c checkpoint")
    if (checkpoint.get("vocab_size") != tokenizer.vocab_size or
            checkpoint.get("tokenizer_fingerprint") != tokenizer.fingerprint()):
        raise ValueError("Checkpoint tokenizer mismatch")
    # These checks prevent evaluating a checkpoint against swapped training splits.
    for name, expected in checkpoint["data_digests"].items():
        if sha256_file(Path(data) / f"{name}.bin") != expected:
            raise ValueError(f"Checkpoint training-data mismatch: {name}")
    documents, counts = validate_tokens(data, split, tokenizer)
    config = checkpoint["model_config"]
    model = SkyGridTransformerV08(vocab_size=tokenizer.vocab_size, **config)
    model.load_state_dict(checkpoint["model_state"])
    metrics = score_documents(model, documents, tokenizer,
                              context_size=config["context_size"], batch_size=batch_size)
    result = {
        "version": "skygrid_v09d_full_documents_v1",
        "checkpoint": checkpoint_name,
        "training_step": checkpoint["step"],
        "split": split,
        "evaluation_method": "All next-token targets within each document; reset context at EOS; exclude first token of each document",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_sha256": sha256_file(Path(data) / f"{split}.bin"),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        **counts,
        **metrics,
    }
    destination = Path(output) / f"{split}_{checkpoint_name}_step{checkpoint['step']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Evaluation already exists: {destination}")
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--final-test", action="store_true",
                        help="Acknowledge this is the held-out final test set")
    args = parser.parse_args()
    if args.split == "test" and not args.final_test:
        parser.error("Test is held out. Use --final-test only after choosing your model using validation.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result, destination = evaluate(args.split, args.checkpoint, batch_size=args.batch_size)
    print(f"SkyGrid v0.9d | {args.split} | {args.checkpoint} checkpoint step {result['training_step']}")
    print(f"Documents={result['documents']:,} | all scored targets={result['evaluated_targets']:,}")
    print(f"Full-document loss={result['cross_entropy_nats_per_token']:.4f} nats/token")
    print(f"Perplexity={result['perplexity']:.3f}")
    print(f"Bits per scored UTF-8 byte={result['bits_per_scored_byte_including_eos_cost']:.3f}")
    print("This is a next-token prediction benchmark, not proof of aviation expertise.")
    print("Saved:", destination)


if __name__ == "__main__":
    main()
