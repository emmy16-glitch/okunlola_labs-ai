"""SkyGrid AI v0.9c: train the original Transformer on v0.9b aviation tokens.

Run: python -m src.skygrid_ai.language.train_large_transformer --steps 100
Resume: python -m src.skygrid_ai.language.train_large_transformer --resume --steps 400

Weights start randomly initialized. Old checkpoints are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .large_bpe import LargeByteBPE, EOS_ID, read_token_ids
from .modern_transformer import SkyGridTransformerV08

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "aviation" / "tokenized_v09b"
OUTPUT = ROOT / "checkpoints" / "v09c"
CONFIG = {
    "context_size": 128,
    "d_model": 96,
    "n_heads": 4,
    "n_layers": 3,
    "hidden_width": 256,
}


def sha256_file(path):
    """Fingerprint a binary dataset without loading it into RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prepared_data(folder=DATA):
    """Load v0.9b assets; check vocabulary, EOS counts, and declared sizes."""
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    tok = LargeByteBPE.load(folder / "tokenizer.json")
    if report.get("format") != "uint32_le":
        raise ValueError("Expected little-endian uint32 v0.9b token files")
    if report.get("vocabulary_size") != tok.vocab_size:
        raise ValueError("Vocabulary size differs from the report")
    if report.get("tokenizer_fingerprint") != tok.fingerprint():
        raise ValueError("Tokenizer fingerprint differs from the report")

    arrays = {}
    digests = {}
    for split in ("train", "val"):
        path = folder / f"{split}.bin"
        ids = read_token_ids(path)
        expected = report["splits"][split]
        if len(ids) != expected["token_count_with_eos"]:
            raise ValueError(f"Wrong token count in {split}")
        if ids.count(EOS_ID) != expected["eos_tokens"]:
            raise ValueError(f"Missing or extra EOS tokens in {split}")
        if any(token < 0 or token >= tok.vocab_size for token in ids):
            raise ValueError(f"Invalid token ID in {split}")
        if len(ids) <= CONFIG["context_size"]:
            raise ValueError(f"{split} must have more than {CONFIG['context_size']} tokens")
        arrays[split] = torch.tensor(ids, dtype=torch.long)
        digests[split] = sha256_file(path)
    return tok, arrays["train"], arrays["val"], digests


def batch_from_starts(tokens, starts, context_size):
    """Each target is exactly the token following its input position."""
    x = torch.stack([tokens[int(i):int(i) + context_size] for i in starts])
    y = torch.stack([tokens[int(i) + 1:int(i) + context_size + 1] for i in starts])
    return x, y


def make_batch(tokens, context_size, batch_size):
    if len(tokens) <= context_size:
        raise ValueError("Training data is smaller than the context window")
    starts = torch.randint(0, len(tokens) - context_size, (batch_size,))
    return batch_from_starts(tokens, starts, context_size)


def evaluate_sample(model, tokens, context_size, windows=64, batch_size=8):
    """Fixed sampled windows, NOT full-corpus validation loss."""
    if windows < 1 or batch_size < 1:
        raise ValueError("Validation windows and batch size must be positive")
    if len(tokens) <= context_size:
        raise ValueError("Validation data is smaller than the context window")
    # Use spaced, repeatable positions; do not consume the training RNG.
    starts = torch.linspace(
        0, len(tokens) - context_size - 1, steps=windows
    ).long()
    model.eval()
    weighted_loss = 0.0
    checked = 0
    with torch.inference_mode():
        for i in range(0, len(starts), batch_size):
            x, y = batch_from_starts(tokens, starts[i:i + batch_size], context_size)
            _, loss = model(x, y)
            weighted_loss += float(loss) * y.numel()
            checked += y.numel()
    return weighted_loss / checked


def save_atomic(data, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".tmp")
    torch.save(data, temp_path)
    temp_path.replace(destination)


def train(train_tokens, val_tokens, *, vocab_size, tokenizer_fingerprint,
          data_digests, config, steps, output, resume=False, batch_size=8,
          lr=0.0003, eval_every=25, save_every=25, val_windows=64):
    """Train from scratch or resume exactly from a v0.9c state on same CPU setup."""
    if steps < 1 or batch_size < 1 or eval_every < 1 or save_every < 1 or lr <= 0:
        raise ValueError("Invalid training configuration")
    if val_windows < 1:
        raise ValueError("val_windows must be positive")
    config = dict(config)
    output = Path(output)
    last_path, best_path = output / "last.pt", output / "best.pt"
    metadata = {
        "format_version": "skygrid_v09c_1",
        "model_config": config,
        "vocab_size": vocab_size,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "data_digests": dict(data_digests),
        "batch_size": batch_size,
        "lr": lr,
        "eval_every": eval_every,
        "save_every": save_every,
        "val_windows": val_windows,
    }
    torch.manual_seed(42)
    model = SkyGridTransformerV08(vocab_size=vocab_size, **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    start, best_loss, best_step = 0, float("inf"), 0
    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        for key, value in metadata.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Checkpoint incompatibility: {key}")
        if not best_path.exists():
            raise FileNotFoundError("best.pt is missing")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["rng_state"])
        start = int(checkpoint["step"])
        best_loss = float(checkpoint["best_validation_loss"])
        best_step = int(checkpoint["best_step"])
    elif last_path.exists() or best_path.exists():
        raise FileExistsError("v0.9c checkpoints already exist. Use --resume.")
    if steps <= start:
        raise ValueError(f"Requested {steps} total steps but checkpoint is at {start}")
    count = sum(p.numel() for p in model.parameters())
    print(f"SkyGrid v0.9c | parameters={count:,} | vocab={vocab_size}")
    print(f"Training tokens={len(train_tokens):,}; validation tokens={len(val_tokens):,}")
    print(f"Resuming at step {start}; requested total steps={steps}")
    for step in range(start + 1, steps + 1):
        model.train()
        x, y = make_batch(train_tokens, config["context_size"], batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0:
            val_loss = evaluate_sample(
                model, val_tokens, config["context_size"], val_windows, batch_size
            )
            print(
                f"step {step:5d} | train_batch_loss={loss.detach().item():.4f} "
                f"| sampled_val_loss={val_loss:.4f}"
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step
                save_atomic({
                    **metadata, "model_state": model.state_dict(),
                    "step": step, "sampled_validation_loss": val_loss,
                }, best_path)
        if step % save_every == 0 or step == steps:
            save_atomic({
                **metadata, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "step": step, "best_validation_loss": best_loss,
                "best_step": best_step,
            }, last_path)
    print(f"Checkpoint step {steps}; best sampled validation step {best_step}")
    print(f"Saved: {output}")
    return last_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Total steps, not additional steps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--confirm-rights-reviewed", action="store_true",
                        help="You reviewed training documents, embedded rights, and attribution")
    args = parser.parse_args()
    if not args.confirm_rights_reviewed:
        parser.error("Review dataset rights and attribution, then pass --confirm-rights-reviewed")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer, training, validation, digests = load_prepared_data(DATA)
    train(
        training, validation, vocab_size=tokenizer.vocab_size,
        tokenizer_fingerprint=tokenizer.fingerprint(),
        data_digests=digests, config=CONFIG, steps=args.steps,
        output=OUTPUT, resume=args.resume,
        batch_size=args.batch_size, eval_every=args.eval_every,
        save_every=args.save_every, val_windows=args.val_windows,
    )


if __name__ == "__main__":
    main()
