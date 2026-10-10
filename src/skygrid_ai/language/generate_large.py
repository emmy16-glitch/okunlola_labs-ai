"""Generate experimental text from the randomly initialized/trained v0.9c model."""
import argparse

import torch

from .large_bpe import LargeByteBPE, PAD_ID, EOS_ID
from .modern_transformer import SkyGridTransformerV08
from .train_large_transformer import DATA, OUTPUT


def decode_partial(tokenizer, tokens):
    """Handle a final incomplete UTF-8 byte sequence without crashing."""
    chunks = [tokenizer.id_to_bytes[int(t)] for t in tokens
              if int(t) not in (PAD_ID, EOS_ID)]
    return b"".join(chunks).decode("utf-8", errors="replace")


def generate(prompt, *, tokenizer, checkpoint, max_new_tokens=100,
             temperature=0.8, top_k=20, seed=7):
    if not prompt or max_new_tokens < 1 or temperature <= 0 or top_k < 1:
        raise ValueError("Invalid text generation settings")
    if tokenizer.vocab_size != checkpoint["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match checkpoint")
    if tokenizer.fingerprint() != checkpoint["tokenizer_fingerprint"]:
        raise ValueError("Tokenizer fingerprint does not match checkpoint")
    model = SkyGridTransformerV08(
        vocab_size=tokenizer.vocab_size, **checkpoint["model_config"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt)
    generated = list(ids)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated[-model.context_size:]], dtype=torch.long)
            logits, _ = model(x)
            scores = logits[0, -1, :] / temperature
            scores[PAD_ID] = -float("inf")
            top = min(top_k, len(scores))
            values, indices = torch.topk(scores, top)
            probabilities = torch.softmax(values, dim=-1)
            next_id = int(indices[torch.multinomial(probabilities, 1)].item())
            if next_id == EOS_ID:
                break
            generated.append(next_id)
    return decode_partial(tokenizer, generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Aircraft telemetry")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = LargeByteBPE.load(DATA / "tokenizer.json")
    checkpoint = torch.load(OUTPUT / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != "skygrid_v09c_1":
        raise ValueError("Not a SkyGrid v0.9c checkpoint")
    print(generate(
        args.prompt, tokenizer=tokenizer, checkpoint=checkpoint,
        max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k,
    ))
    print("\nExperimental output only; not reliable aviation guidance.")


if __name__ == "__main__":
    main()
