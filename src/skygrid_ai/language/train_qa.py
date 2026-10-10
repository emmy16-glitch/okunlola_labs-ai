"""Supervise SkyGrid's OWN pretrained-from-scratch v10 checkpoint on QA.

Only answer/EOS tokens contribute to loss. Source weights and token data are
never overwritten. Small candidate dataset: requires an actual human review.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .large_bpe import LargeByteBPE, PAD_ID
from .modern_transformer import SkyGridTransformerV08
from .qa_pilot import BENCHMARKS, PROTOCOL, digest_files, encode_supervised, prepare_examples
from .train_large_transformer import DATA, ROOT, save_atomic, sha256_file

SOURCE = ROOT / 'checkpoints' / 'v10' / 'last.pt'
OUTPUT = ROOT / 'checkpoints' / 'qa_v15'
FORMAT = 'skygrid_qa_v15_1'


def load_starting_point(*, source=SOURCE, data=DATA):
    """Reject changed tokenizer, binary data, or a non-v10 source checkpoint."""
    source, data = Path(source), Path(data)
    cp = torch.load(source, map_location='cpu', weights_only=True)
    if cp.get('format_version') != 'skygrid_v09c_1' or not isinstance(cp.get('model_config'), dict):
        raise ValueError('Expected existing v10-compatible from-scratch checkpoint')
    if not isinstance(cp.get('step'), int) or cp['step'] < 1:
        raise ValueError('No trained source checkpoint')
    tok = LargeByteBPE.load(data / 'tokenizer.json')
    if cp.get('vocab_size') != tok.vocab_size or cp.get('tokenizer_fingerprint') != tok.fingerprint():
        raise ValueError('Source checkpoint tokenizer mismatch')
    if not isinstance(cp.get('data_digests'), dict) or set(cp['data_digests']) != {'train', 'val'}:
        raise ValueError('Training-data fingerprints missing')
    for split in ('train', 'val'):
        if sha256_file(data / f'{split}.bin') != cp['data_digests'][split]:
            raise ValueError(f'Source checkpoint {split} dataset fingerprint mismatch')
    return tok, cp, sha256_file(source)


def collate(examples, tokenizer, context_size):
    if not examples:
        raise ValueError('Cannot train on an empty batch')
    pairs = [encode_supervised(row, tokenizer, context_size) for row in examples]
    size = max(len(x) for x, _ in pairs)
    xs = torch.full((len(pairs), size), PAD_ID, dtype=torch.long)
    ys = torch.full((len(pairs), size), -100, dtype=torch.long)
    for i, (x, y) in enumerate(pairs):
        xs[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        ys[i, :len(y)] = torch.tensor(y, dtype=torch.long)
    return xs, ys


def masked_loss(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           y.reshape(-1), ignore_index=-100)


def evaluate_answers(model, rows, tokenizer, context_size):
    """Entire split, answer-target-only mean NLL. Never include prompt tokens."""
    if not rows:
        raise ValueError('No QA evaluation rows')
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for row in rows:
            x, y = collate([row], tokenizer, context_size)
            active = int((y != -100).sum().item())
            total += float(masked_loss(model, x, y)) * active
            count += active
    return total / count, count


def metadata_for(tokenizer, source, source_hash, qa_hash, batch_size, lr, eval_every):
    return {
        'format_version': FORMAT, 'model_config': dict(source['model_config']),
        'vocab_size': tokenizer.vocab_size,
        'tokenizer_fingerprint': tokenizer.fingerprint(),
        'data_digests': dict(source['data_digests']),
        'qa_protocol_sha256': qa_hash,
        'source_checkpoint_sha256': source_hash,
        'source_training_step': source['step'],
        'batch_size': batch_size, 'lr': lr, 'eval_every': eval_every,
        'loss_objective': 'next_token_cross_entropy_on_answer_and_eos_only',
    }


def train_qa(rows, tokenizer, source, source_hash, qa_hash, *, output=OUTPUT,
             steps=20, resume=False, batch_size=2, lr=0.0001, eval_every=10):
    if steps < 1 or batch_size < 1 or lr <= 0 or eval_every < 1:
        raise ValueError('Invalid training settings')
    if not rows.get('train') or not rows.get('val') or not rows.get('test'):
        raise ValueError('Expected document-disjoint train/val/test rows')
    output = Path(output)
    last_file, best_file = output / 'last.pt', output / 'best.pt'
    metadata = metadata_for(tokenizer, source, source_hash, qa_hash, batch_size, lr, eval_every)
    torch.manual_seed(2026)
    model = SkyGridTransformerV08(vocab_size=tokenizer.vocab_size, **metadata['model_config'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_loss, best_step, start = float('inf'), 0, 0
    if resume:
        previous = torch.load(last_file, map_location='cpu', weights_only=True)
        for key, expected in metadata.items():
            if previous.get(key) != expected:
                raise ValueError(f'QA checkpoint metadata mismatch: {key}')
        if not best_file.is_file():
            raise FileNotFoundError('QA best checkpoint missing')
        model.load_state_dict(previous['model_state'])
        optimizer.load_state_dict(previous['optimizer_state'])
        torch.set_rng_state(previous['rng_state'])
        best_loss, best_step, start = (float(previous['best_qa_val_loss']),
                                      int(previous['best_step']), int(previous['step']))
    else:
        if last_file.exists() or best_file.exists():
            raise FileExistsError('QA checkpoint already exists; use --resume, never overwrite')
        model.load_state_dict(source['model_state'])
    if steps <= start:
        raise ValueError(f'Requested total steps {steps} is not greater than {start}')
    context = metadata['model_config']['context_size']
    # Preflight all rows, including reserved test: checks fitting but does NOT
    # evaluate or use heldout answers in optimizer selection.
    for split in ('train', 'val'):
        for row in rows[split]:
            encode_supervised(row, tokenizer, context)
    print(f'SkyGrid QA pilot | source pretrained step={source["step"]} | '
          f'parameters={sum(p.numel() for p in model.parameters()):,}')
    print(f'Examples: train={len(rows["train"])} val={len(rows["val"])} '
          f'test={len(rows["test"])} (test never scored by training)')
    print(f'QA fine-tune step {start} -> {steps} (cumulative)')
    for step in range(start + 1, steps + 1):
        model.train()
        selected = torch.randint(len(rows['train']), (batch_size,)).tolist()
        x, y = collate([rows['train'][i] for i in selected], tokenizer, context)
        loss = masked_loss(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            val_loss, n = evaluate_answers(model, rows['val'], tokenizer, context)
            print(f'step {step:5d} | train_answer_loss={float(loss.detach()):.4f} '
                  f'| val_answer_loss={val_loss:.4f} | val_targets={n}', flush=True)
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
                save_atomic({**metadata, 'model_state': model.state_dict(),
                             'step': step, 'val_answer_loss': val_loss}, best_file)
        if step % eval_every == 0 or step == steps:
            save_atomic({**metadata, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'rng_state': torch.get_rng_state(), 'step': step,
                         'best_qa_val_loss': best_loss, 'best_step': best_step}, last_file)
    print(f'Saved separate QA checkpoints in {output}; best QA val step {best_step}')
    return last_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true', help='Review candidate QAs, provenance and splits (no training)')
    parser.add_argument('--steps', type=int, default=20, help='Cumulative QA-training steps, not pretraining steps')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--source-checkpoint', choices=('last', 'best'), default='last')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--confirm-rights-reviewed', action='store_true')
    parser.add_argument('--confirm-qa-reviewed', action='store_true')
    args = parser.parse_args()
    rows = prepare_examples()
    if args.audit:
        print(json.dumps({'status': 'candidate_qa_requires_human_review',
                          'counts': {k: len(v) for k, v in rows.items()},
                          'review_train_val': {s: rows[s] for s in ('train', 'val')},
                          'reserved_test': {'count': len(rows['test']),
                                            'reference_answers_withheld': True}}, indent=2, ensure_ascii=False))
        return
    if not args.confirm_rights_reviewed or not args.confirm_qa_reviewed:
        parser.error('Actually review original source/rights AND QA references; both explicit confirmations required')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    source_path = SOURCE.with_name(args.source_checkpoint + '.pt')
    tokenizer, source, source_hash = load_starting_point(source=source_path)
    qa_hash = digest_files((PROTOCOL, *BENCHMARKS))
    train_qa(rows, tokenizer, source, source_hash, qa_hash, steps=args.steps,
             resume=args.resume, batch_size=args.batch_size,
             lr=args.learning_rate, eval_every=args.eval_every)


if __name__ == '__main__':
    main()
