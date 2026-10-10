
import copy
from pathlib import Path

import torch

from .model import SkyGridLanguageModel
from .tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / "data" / "aviation" / "processed"

CHECKPOINT = (
    ROOT / "checkpoints" / "skygrid_language_v06.pt"
)

CONFIG = {
    "context_size": 32,
    "d_model": 48,
    "n_heads": 4,
    "n_layers": 2,
}


def make_batch(tokens, context_size, batch_size=8):
    if len(tokens) <= context_size:
        raise ValueError("Not enough training tokens")

    starts = torch.randint(
        0,
        len(tokens) - context_size,
        (batch_size,)
    )

    x = torch.stack([
        tokens[i:i + context_size]
        for i in starts
    ])

    y = torch.stack([
        tokens[i + 1:i + context_size + 1]
        for i in starts
    ])

    return x, y


def validation_loss(model, tokens, context_size):
    model.eval()
    losses = []

    with torch.inference_mode():
        for i in range(
            0,
            len(tokens) - context_size,
            context_size
        ):
            x = tokens[
                i:i + context_size
            ].unsqueeze(0)

            y = tokens[
                i + 1:i + context_size + 1
            ].unsqueeze(0)

            _, loss = model(x, y)

            losses.append(loss.item())

    if not losses:
        raise ValueError("Validation text is too short")

    return sum(losses) / len(losses)


def main():
    tokenizer = CharTokenizer.load(
        DATA / "vocab.json"
    )

    train_text = (
        DATA / "train.txt"
    ).read_text(encoding="utf-8")

    val_text = (
        DATA / "val.txt"
    ).read_text(encoding="utf-8")

    train_tokens = torch.tensor(
        tokenizer.encode(train_text),
        dtype=torch.long
    )

    val_tokens = torch.tensor(
        tokenizer.encode(val_text),
        dtype=torch.long
    )

    torch.manual_seed(42)

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    model = SkyGridLanguageModel(
        len(tokenizer.vocab),
        **CONFIG
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003
    )

    best_loss = float("inf")
    best_state = None
    best_step = 0

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    for step in range(1, 151):
        model.train()

        x, y = make_batch(
            train_tokens,
            CONFIG["context_size"]
        )

        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0:
            val_loss = validation_loss(
                model,
                val_tokens,
                CONFIG["context_size"]
            )

            print(
                f"Step {step:3d} | "
                f"train loss {loss.item():.3f} | "
                f"val loss {val_loss:.3f}"
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
            "vocab_size": len(tokenizer.vocab),
        },
        CHECKPOINT
    )

    print(f"Best step: {best_step}")
    print(f"Saved: {CHECKPOINT}")


if __name__ == "__main__":
    main()
