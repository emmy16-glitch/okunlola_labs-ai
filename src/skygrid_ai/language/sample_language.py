
from pathlib import Path

import torch

from .model import SkyGridLanguageModel
from .tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[3]

DATA = ROOT / "data" / "aviation" / "processed"

CHECKPOINT = (
    ROOT / "checkpoints" / "skygrid_language_v06.pt"
)


def main():
    tokenizer = CharTokenizer.load(
        DATA / "vocab.json"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    if len(tokenizer.vocab) != checkpoint["vocab_size"]:
        raise ValueError("Tokenizer does not match model")

    model = SkyGridLanguageModel(
        len(tokenizer.vocab),
        **checkpoint["config"]
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    torch.manual_seed(7)

    tokens = torch.tensor(
        [tokenizer.encode("air")],
        dtype=torch.long
    )

    with torch.inference_mode():
        for _ in range(100):
            context = tokens[:, -model.context_size:]

            logits, _ = model(context)

            probabilities = torch.softmax(
                logits[:, -1, :] / 0.8,
                dim=-1
            )

            # Do not generate padding or unknown IDs.
            probabilities[:, :2] = 0

            probabilities /= probabilities.sum(
                dim=-1,
                keepdim=True
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            tokens = torch.cat(
                (tokens, next_token),
                dim=1
            )

    print(tokenizer.decode(tokens[0].tolist()))

    print(
        "\nDemo text only: not factual aviation advice."
    )


if __name__ == "__main__":
    main()
