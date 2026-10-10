"""Early-stopped QA training of SkyGrid's own v10 checkpoint using reviewed data.

All examples use question-only live retrieval. Source v10/qa_v15 files stay untouched.
"""
import argparse
import json
from pathlib import Path

import torch

from .qa_v16 import CANDIDATES, REVIEWS, data_report, dataset_fingerprint, load_reviewed, encode_v16
from .train_qa import (masked_loss, load_starting_point,
                       SOURCE)
from .modern_transformer import SkyGridTransformerV08
from .train_large_transformer import ROOT, save_atomic

OUTPUT = ROOT / 'checkpoints' / 'qa_v16'
FORMAT = 'skygrid_qa_v16_1'



def collate_v16(rows, tok, context):
    from .large_bpe import PAD_ID
    pairs = [encode_v16(row, tok, context) for row in rows]
    size = max(len(x) for x, _ in pairs)
    x = torch.full((len(pairs), size), PAD_ID, dtype=torch.long)
    y = torch.full((len(pairs), size), -100, dtype=torch.long)
    for i, (a, b) in enumerate(pairs):
        x[i, :len(a)] = torch.tensor(a, dtype=torch.long)
        y[i, :len(b)] = torch.tensor(b, dtype=torch.long)
    return x, y


def evaluate_v16(model, rows, tok, context):
    if not rows:
        raise ValueError('No QA evaluation rows')
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for row in rows:
            x, y = collate_v16([row], tok, context)
            n = int((y != -100).sum())
            total += float(masked_loss(model, x, y)) * n
            count += n
    return total / count, count


def metadata_for(tok, source, source_hash, qa_hash, *, lr, batch_size,
                 eval_every, patience, min_delta):
    return {
        'format_version': FORMAT, 'model_config': dict(source['model_config']),
        'vocab_size': tok.vocab_size, 'tokenizer_fingerprint': tok.fingerprint(),
        'data_digests': dict(source['data_digests']),
        'source_checkpoint_sha256': source_hash,
        'source_training_step': source['step'],
        'qa_dataset_sha256': qa_hash,
        'loss_objective': 'next_token_cross_entropy_on_answer_and_eos_only',
        'evidence_selection': 'question_only_live_extraction',
        'lr': lr, 'batch_size': batch_size, 'eval_every': eval_every,
        'patience': patience, 'min_delta': min_delta,
    }


def fit(rows, tok, source, source_hash, qa_hash, *, output=OUTPUT, steps=100,
        resume=False, lr=0.00002, batch_size=2, eval_every=10,
        patience=4, min_delta=0.003, min_train=20):
    if not (1 <= steps and 1 <= batch_size and 1 <= eval_every and patience >= 1 and
            0 < lr < 0.01 and min_delta >= 0):
        raise ValueError('Invalid QA training parameters')
    if min(len(rows.get(s, [])) for s in ('val', 'test')) < 1 or len(rows.get('train', [])) < min_train:
        raise ValueError('Insufficient approved, document-disjoint QA examples')
    config = source['model_config']
    context = config['context_size']
    # Fail before touching checkpoints on any unencodable answer.
    for split in ('train', 'val'):
        for r in rows[split]:
            encode_v16(r, tok, context)
    output = Path(output)
    last_path, best_path = output / 'last.pt', output / 'best.pt'
    meta = metadata_for(tok, source, source_hash, qa_hash, lr=lr, batch_size=batch_size,
                        eval_every=eval_every, patience=patience, min_delta=min_delta)
    torch.manual_seed(2027)
    model = SkyGridTransformerV08(vocab_size=tok.vocab_size, **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    initial, best_loss, best_step, stale = 0, float('inf'), 0, 0
    if resume:
        cp = torch.load(last_path, map_location='cpu', weights_only=True)
        if not best_path.is_file():
            raise FileNotFoundError('Best QA checkpoint missing')
        for k, v in meta.items():
            if cp.get(k) != v:
                raise ValueError(f'Cannot resume changed QA run: {k}')
        if cp.get('stopped_early'):
            raise ValueError('Early-stopped QA run is complete; start a new approved experiment')
        model.load_state_dict(cp['model_state'])
        optimizer.load_state_dict(cp['optimizer_state'])
        torch.set_rng_state(cp['rng_state'])
        initial, best_loss, best_step, stale = (int(cp['step']), float(cp['best_qa_val_loss']),
                                                 int(cp['best_step']), int(cp['stale_evaluations']))
    else:
        if last_path.exists() or best_path.exists():
            raise FileExistsError('QA v16 checkpoint already exists; use --resume')
        model.load_state_dict(source['model_state'])
    if steps <= initial:
        raise ValueError('Total steps must be greater than saved step')
    print(f'Q&A v16 | train={len(rows["train"])} val={len(rows["val"])} '
          f'test={len(rows["test"])} | original v10 step {source["step"]}')
    stopped = False
    for step in range(initial + 1, steps + 1):
        model.train()
        picks = torch.randint(len(rows['train']), (batch_size,)).tolist()
        x, y = collate_v16([rows['train'][i] for i in picks], tok, context)
        loss = masked_loss(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0:
            val_loss, targets = evaluate_v16(model, rows['val'], tok, context)
            print(f'step {step:5d} | train_answer_loss={float(loss.detach()):.4f} '
                  f'| val_answer_loss={val_loss:.4f} | val_targets={targets}', flush=True)
            if val_loss < best_loss - min_delta:
                best_loss, best_step, stale = val_loss, step, 0
                save_atomic({**meta, 'model_state': model.state_dict(), 'step': step,
                             'val_answer_loss': val_loss}, best_path)
            elif step != 1:
                stale += 1
            stopped = stale >= patience
        if step % eval_every == 0 or step == steps or stopped:
            save_atomic({**meta, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(), 'rng_state': torch.get_rng_state(),
                         'step': step, 'best_qa_val_loss': best_loss, 'best_step': best_step,
                         'stale_evaluations': stale, 'stopped_early': stopped}, last_path)
        if stopped:
            print(f'Early stopping at step {step}; best validation checkpoint step {best_step}')
            break
    print(f'Saved separate QA v16 checkpoints: {output}; best validation step {best_step}')
    return {'last_step': step, 'best_step': best_step, 'stopped_early': stopped, 'best_val_loss': best_loss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true', help='Count approved and rejected cases without training')
    parser.add_argument('--candidates', default=str(CANDIDATES))
    parser.add_argument('--reviews', default=str(REVIEWS))
    parser.add_argument('--minimum-approved', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.00002)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--min-delta', type=float, default=0.003)
    parser.add_argument('--confirm-rights-reviewed', action='store_true')
    parser.add_argument('--confirm-qa-reviewed', action='store_true')
    args = parser.parse_args()
    if not args.audit and not (args.confirm_rights_reviewed and args.confirm_qa_reviewed):
        parser.error('Complete independent source/rights and Q&A human review before training')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rows, rejected = load_reviewed(candidates=args.candidates, reviews=args.reviews,
                                   minimum=args.minimum_approved, strict=not args.audit)
    print(json.dumps(data_report(rows, rejected), indent=2))
    if args.audit:
        return
    tok, source, source_hash = load_starting_point(source=SOURCE)
    digest = dataset_fingerprint(candidates=args.candidates, reviews=args.reviews)
    print(json.dumps(fit(rows, tok, source, source_hash, digest, steps=args.steps,
                         resume=args.resume, batch_size=args.batch_size,
                         lr=args.learning_rate, eval_every=args.eval_every,
                         patience=args.patience, min_delta=args.min_delta), indent=2))


if __name__ == '__main__':
    main()
