"""Opt-in, from-scratch larger CPU Transformer; separate checkpoints/v10.

Default smoke: python -m src.skygrid_ai.language.train_v10 --steps 20 --confirm-rights-reviewed
Resume:        python -m src.skygrid_ai.language.train_v10 --resume --steps 1000 --confirm-rights-reviewed
"""
import argparse
from pathlib import Path
import torch

from .train_large_transformer import ROOT, load_prepared_data, train

OUTPUT = ROOT / 'checkpoints' / 'v10'
DEFAULT_CONFIG = {
    'context_size': 256, 'd_model': 128,
    'n_heads': 4, 'n_layers': 4, 'hidden_width': 384,
}


def validate_config(config):
    if config['context_size'] < 16 or config['context_size'] > 1024:
        raise ValueError('Context size must be between 16 and 1024')
    if config['d_model'] % config['n_heads'] or (config['d_model'] // config['n_heads']) % 2:
        raise ValueError('Head dimensions must be even')
    if min(config['d_model'], config['n_heads'], config['n_layers'], config['hidden_width']) < 1:
        raise ValueError('All model dimensions must be positive')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20, help='Final cumulative training step')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--context-size', type=int, default=256)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--hidden-width', type=int, default=384)
    parser.add_argument('--val-windows', type=int, default=32)
    parser.add_argument('--confirm-rights-reviewed', action='store_true')
    args = parser.parse_args()
    if not args.confirm_rights_reviewed:
        parser.error('Review source rights/attribution before running training')
    config = validate_config({
        'context_size': args.context_size, 'd_model': args.d_model,
        'n_heads': args.n_heads, 'n_layers': args.n_layers,
        'hidden_width': args.hidden_width,
    })
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer, training, validation, digests = load_prepared_data()
    if min(len(training), len(validation)) <= config['context_size']:
        raise ValueError('Insufficient tokens for configured context length')
    # Reuses audited training, deterministic resume, fingerprint checks, and safeguards.
    train(training, validation, vocab_size=tokenizer.vocab_size,
          tokenizer_fingerprint=tokenizer.fingerprint(), data_digests=digests,
          config=config, steps=args.steps, output=OUTPUT, resume=args.resume,
          batch_size=args.batch_size, val_windows=args.val_windows)


if __name__ == '__main__':
    main()
