"""Generate experimental v1.0 text; not safety-critical aviation advice."""
import argparse
import torch
from .large_bpe import LargeByteBPE
from .generate_large import generate
from .train_large_transformer import DATA
from .train_v10 import OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', default='Aircraft telemetry')
    parser.add_argument('--tokens', type=int, default=100)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='best')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = LargeByteBPE.load(DATA / 'tokenizer.json')
    checkpoint = torch.load(OUTPUT / f'{args.checkpoint}.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 'skygrid_v09c_1':
        raise ValueError('Unexpected training checkpoint format')
    print(generate(args.prompt, tokenizer=tokenizer, checkpoint=checkpoint,
                   max_new_tokens=args.tokens, temperature=args.temperature,
                   top_k=args.top_k))
    print('\nResearch text only; verify every aviation claim independently.')


if __name__ == '__main__':
    main()
