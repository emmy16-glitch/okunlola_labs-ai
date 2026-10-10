
import copy
import json
from pathlib import Path

import torch

from .bpe_tokenizer import ByteBPETokenizer
from .modern_transformer import SkyGridTransformerV08


ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / "data" / "aviation" / "processed_v08"

CHECKPOINT = (
    ROOT / "checkpoints" / "skygrid_transformer_v08b.pt"
)

CONFIG = {
    "context_size": 128,
    "d_model": 96,
    "n_heads": 4,
    "n_layers": 3,
    "hidden_width": 256,
}


def read_tokens(name):
    path = DATA / f"{name}_tokens.json"

    if not path.exists():
        raise FileNotFoundError(
            "Run train_bpe.py first"
        )

    numbers = json.loads(
        path.read_text(encoding="utf-8")
    )

    return torch.tensor(
        numbers, dtype=torch.long
    )


def make_batch(tokens, context_size, batch_size=4):
    if len(tokens) <= context_size:
        raise ValueError(
            "Not enough training tokens for context size"
        )

    starts = torch.randint(
        0,
        len(tokens) - context_size,
        (batch_size,)
    )

    inputs = torch.stack([
        tokens[i:i + context_size]
        for i in starts
    ])

    targets = torch.stack([
        tokens[i + 1:i + context_size + 1]
        for i in starts
    ])

    return inputs, targets


def evaluate(model, tokens, context_size):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for start in range(
            0, len(tokens) - 1, context_size
        ):
            segment = tokens[
                start:start + context_size + 1
            ]

            if len(segment) < 2:
                continue

            x = segment[:-1].unsqueeze(0)
            y = segment[1:].unsqueeze(0)

            _, loss = model(x, y)

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    if total_tokens == 0:
        raise ValueError(
            "Validation has no next-token examples"
        )

    return total_loss / total_tokens


def main():
    tokenizer = ByteBPETokenizer.load(
        DATA / "bpe_tokenizer.json"
    )

    train_tokens = read_tokens("train")
    val_tokens = read_tokens("val")

    torch.manual_seed(42)

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    model = SkyGridTransformerV08(
        vocab_size=tokenizer.vocab_size,
        **CONFIG
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0006,
        weight_decay=0.01
    )

    best_loss = float("inf")
    best_state = None
    best_step = 0

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("SkyGrid Transformer v0.8b")
    print(f"Parameters: {parameters:,}")
    print(f"Vocabulary: {tokenizer.vocab_size}")
    print(f"Training tokens: {len(train_tokens)}")
    print(f"Validation tokens: {len(val_tokens)}")

    for step in range(1, 81):
        model.train()

        x, y = make_batch(
            train_tokens,
            CONFIG["context_size"]
        )

        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        if step == 1 or step % 20 == 0:
            val_loss = evaluate(
                model,
                val_tokens,
                CONFIG["context_size"]
            )

            print(
                f"Step {step:3d} | "
                f"train loss={loss.item():.4f} | "
                f"val loss={val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(
                    model.state_dict()
                )
                best_step = step

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state": best_state,
            "config": CONFIG,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_fingerprint": (
                tokenizer.fingerprint()
            ),
            "best_step": best_step,
            "validation_loss": best_loss,
        },
        CHECKPOINT
    )

    print(f"Best step: {best_step}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Saved: {CHECKPOINT}")


if __name__ == "__main__":
    main()
