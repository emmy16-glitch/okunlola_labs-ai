import argparse
from pathlib import Path

import torch

from .bpe_tokenizer import ByteBPETokenizer
from .modern_transformer import SkyGridTransformerV08
from .train_reliable import DATA, OUTPUT


def decode_generated(tokenizer, token_ids):
    """Allow a sampled sequence to end in an incomplete UTF-8 character."""
    pieces = [tokenizer.id_to_bytes[token] for token in token_ids
              if token not in (tokenizer.PAD_ID, tokenizer.EOS_ID)]
    return b"".join(pieces).decode("utf-8", errors="replace")


def generate(prompt="air", max_new_tokens=100, temperature=0.8, top_k=20):
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    if max_new_tokens < 1 or temperature <= 0 or top_k < 1:
        raise ValueError("Invalid sampling parameters")

    tokenizer = ByteBPETokenizer.load(DATA / "bpe_tokenizer.json")
    checkpoint = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=True)
    if (checkpoint["vocab_size"] != tokenizer.vocab_size or
            checkpoint["tokenizer_fingerprint"] != tokenizer.fingerprint()):
        raise ValueError("Saved model and tokenizer do not match")

    model = SkyGridTransformerV08(tokenizer.vocab_size, **checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    torch.manual_seed(7)
    ids = tokenizer.encode(prompt)
    context = torch.tensor([ids], dtype=torch.long)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, _ = model(context[:, -model.context_size:])
            scores = logits[:, -1, :] / temperature
            scores[:, tokenizer.PAD_ID] = -float("inf")
            values, indices = torch.topk(scores, min(top_k, scores.size(-1)))
            probs = torch.softmax(values, dim=-1)
            next_id = indices.gather(1, torch.multinomial(probs, 1))
            if next_id.item() == tokenizer.EOS_ID:
                break
            context = torch.cat((context, next_id), dim=1)

    return decode_generated(tokenizer, context[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="air")
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()
    print(generate(args.prompt, args.tokens))
    print("\nExperimental generated text: not factual aviation guidance.")


if __name__ == "__main__":
    main()
