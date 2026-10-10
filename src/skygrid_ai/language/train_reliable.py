import argparse
import hashlib
import json
from pathlib import Path

import torch

from .bpe_tokenizer import ByteBPETokenizer
from .modern_transformer import SkyGridTransformerV08

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "aviation" / "processed_v08"
OUTPUT = ROOT / "checkpoints" / "v08c"
CONFIG = {
    "context_size": 128,
    "d_model": 96,
    "n_heads": 4,
    "n_layers": 3,
    "hidden_width": 256,
}


def digest(tokens):
    """Identify precisely which token sequence trained the model."""
    payload = json.dumps(tokens.tolist(), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_batch(tokens, context_size, batch_size):
    if len(tokens) <= context_size:
        raise ValueError("Not enough tokens for the context window")
    starts = torch.randint(0, len(tokens) - context_size, (batch_size,))
    x = torch.stack([tokens[i:i + context_size] for i in starts])
    y = torch.stack([tokens[i + 1:i + context_size + 1] for i in starts])
    return x, y


def evaluate(model, tokens, context_size):
    model.eval()
    weighted_loss = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(tokens) - 1, context_size):
            segment = tokens[start:start + context_size + 1]
            if len(segment) < 2:
                continue
            _, loss = model(segment[:-1].unsqueeze(0), segment[1:].unsqueeze(0))
            weighted_loss += loss.item() * (len(segment) - 1)
            count += len(segment) - 1
    if count == 0:
        raise ValueError("Empty validation sequence")
    return weighted_loss / count


def save_atomic(payload, path):
    """Write a temporary file, then replace the target checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(train_tokens, val_tokens, tokenizer, *, config, steps, output,
          resume=False, batch_size=4, eval_every=20, save_every=20):
    config = dict(config)
    output = Path(output)
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    metadata = {
        "format_version": 1,
        "config": config,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "training_digest": digest(train_tokens),
        "validation_digest": digest(val_tokens),
        "batch_size": batch_size,
        "learning_rate": 0.0006,
    }

    torch.manual_seed(42)
    model = SkyGridTransformerV08(vocab_size=tokenizer.vocab_size, **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.01)
    start = 0
    best_loss = float("inf")
    best_step = 0

    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        for key, expected in metadata.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"Checkpoint mismatch: {key}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["rng_state"])
        start = checkpoint["step"]
        best_loss = checkpoint["best_loss"]
        best_step = checkpoint["best_step"]
        if not best_path.exists():
            raise FileNotFoundError("Saved best.pt is missing")
    elif last_path.exists():
        raise FileExistsError("Checkpoint already exists: use --resume")

    if steps <= start:
        raise ValueError(f"Requested {steps} total steps, already at {start}")

    for step in range(start + 1, steps + 1):
        model.train()
        x, y = make_batch(train_tokens, config["context_size"], batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % eval_every == 0:
            val_loss = evaluate(model, val_tokens, config["context_size"])
            print(f"Step {step:3d} | train={loss.item():.4f} | val={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step
                save_atomic({
                    **metadata,
                    "model_state": model.state_dict(),
                    "step": step,
                    "validation_loss": val_loss,
                }, best_path)

        if step % save_every == 0 or step == steps:
            save_atomic({
                **metadata,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "step": step,
                "best_loss": best_loss,
                "best_step": best_step,
            }, last_path)

    print(f"Training saved at step {steps}; best validation step: {best_step}")
    return last_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80,
                        help="Total steps, not additional steps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = ByteBPETokenizer.load(DATA / "bpe_tokenizer.json")
    train_tokens = torch.tensor(json.loads((DATA / "train_tokens.json").read_text()), dtype=torch.long)
    val_tokens = torch.tensor(json.loads((DATA / "val_tokens.json").read_text()), dtype=torch.long)
    train(train_tokens, val_tokens, tokenizer, config=CONFIG,
          steps=args.steps, output=OUTPUT, resume=args.resume)


if __name__ == "__main__":
    main()
