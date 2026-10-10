"""Compare QA answer-only losses and drafts; reserve test for explicit final run.

Token-loss and reference-term overlap are NOT aviation factual correctness.
Human reviewers must inspect evidence and output. Never use for flight control.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from .modern_transformer import SkyGridTransformerV08
from .qa_pilot import BENCHMARKS, PROTOCOL, digest_files, prepare_examples
from .train_large_transformer import ROOT, DATA, sha256_file
from .train_qa import FORMAT, OUTPUT, SOURCE, evaluate_answers, load_starting_point
from .large_bpe import EOS_ID, PAD_ID

REPORTS = ROOT / 'checkpoints' / 'qa_v15_evaluation'


def generate_answer(model, tokenizer, row, context_size, *, limit=64):
    """Deterministic greedy experimental draft; output may be nonsensical."""
    if limit < 1:
        raise ValueError('Generation limit must be positive')
    from .qa_pilot import prompt_for_example
    prompt = prompt_for_example(row, tokenizer, context_size, answer_room=1)
    generated = list(tokenizer.encode(prompt))
    out = []
    model.eval()
    with torch.inference_mode():
        for _ in range(limit):
            x = torch.tensor([generated[-context_size:]], dtype=torch.long)
            logits, _ = model(x)
            nxt = int(logits[0, -1].argmax().item())
            if nxt == EOS_ID:
                break
            if nxt == PAD_ID:
                break
            out.append(nxt)
            generated.append(nxt)
    # A partial final UTF-8 sequence is common in byte-token generation.
    return b''.join(tokenizer.id_to_bytes[int(t)] for t in out).decode('utf-8', errors='replace').strip()


def load_qa_checkpoint(path, tokenizer, source, source_hash, qa_hash):
    cp = torch.load(path, map_location='cpu', weights_only=True)
    expected = {
        'format_version': FORMAT,
        'model_config': source['model_config'],
        'vocab_size': tokenizer.vocab_size,
        'tokenizer_fingerprint': tokenizer.fingerprint(),
        'data_digests': source['data_digests'],
        'qa_protocol_sha256': qa_hash,
        'source_checkpoint_sha256': source_hash,
        'source_training_step': source['step'],
        'loss_objective': 'next_token_cross_entropy_on_answer_and_eos_only',
    }
    for field, value in expected.items():
        if cp.get(field) != value:
            raise ValueError(f'QA model provenance mismatch: {field}')
    return cp


def measure(split='val', checkpoint_name='best', *, root=ROOT, final_test=False,
            examples=None, sample_tokens=64):
    """Baseline and candidate use identical answer-only target sequences."""
    if split not in ('val', 'test') or checkpoint_name not in ('last', 'best'):
        raise ValueError('Invalid split or checkpoint')
    if split == 'test' and not final_test:
        raise ValueError('Final test requires --final-test; do not tune on test')
    root = Path(root)
    data = root / 'data' / 'aviation' / 'tokenized_v09b'
    cp_folder = root / 'checkpoints'
    reports = cp_folder / 'qa_v15_evaluation'
    if split == 'test' and (reports / 'final_test_locked.json').exists():
        raise FileExistsError('Final QA test already run. Do not re-evaluate or tune against it.')
    protocol = root / 'data' / 'qa' / 'aviation_v15_protocol.json'
    bench = (root / 'data' / 'benchmarks' / 'aviation_v10.json',
             root / 'data' / 'benchmarks' / 'aviation_v11_independent.json')
    rows = examples if examples is not None else prepare_examples(
        protocol=protocol, benchmarks=bench, corpus=root / 'data' / 'aviation' / 'processed_v09b')
    qa_hash = digest_files((protocol, *bench))
    qa_path = cp_folder / 'qa_v15' / (checkpoint_name + '.pt')
    if not qa_path.exists():
        raise FileNotFoundError('No supervised QA checkpoint: ' + str(qa_path))
    qa = torch.load(qa_path, map_location='cpu', weights_only=True)
    source_name = qa.get('source_checkpoint_sha256')
    candidates = [cp_folder / 'v10' / 'last.pt', cp_folder / 'v10' / 'best.pt']
    source_path = next((p for p in candidates if p.is_file() and sha256_file(p) == source_name), None)
    if source_path is None:
        raise ValueError('The original v10 source checkpoint changed or is missing')
    tok, source, source_hash = load_starting_point(source=source_path, data=data)
    qa = load_qa_checkpoint(qa_path, tok, source, source_hash, qa_hash)
    context = source['model_config']['context_size']
    reference = SkyGridTransformerV08(vocab_size=tok.vocab_size, **source['model_config'])
    reference.load_state_dict(source['model_state'])
    trained = SkyGridTransformerV08(vocab_size=tok.vocab_size, **qa['model_config'])
    trained.load_state_dict(qa['model_state'])
    original_loss, count_a = evaluate_answers(reference, rows[split], tok, context)
    trained_loss, count_b = evaluate_answers(trained, rows[split], tok, context)
    if count_a != count_b:
        raise ValueError('QA evaluation target-count mismatch')
    details = []
    for row in rows[split]:
        draft = generate_answer(trained, tok, row, context, limit=sample_tokens)
        baseline_draft = generate_answer(reference, tok, row, context, limit=sample_tokens)
        expected = row['answer'].strip()
        # Exact match is a brittle string check, NOT fact correctness.
        details.append({'id': row['id'], 'question': row['question'],
                        'kind': row['kind'], 'reference_answer': expected,
                        'baseline_draft_unverified': baseline_draft,
                        'qa_draft_unverified': draft,
                        'exact_string_match_baseline': baseline_draft == expected,
                        'exact_string_match_qa': draft == expected,
                        'citation': row.get('citation'),
                        'human_fact_review_required': True})
    result = {
        'split': split, 'evaluation': 'answer_only_teacher_forced_next_token_loss',
        'pilot_examples': len(rows[split]), 'answer_targets': count_a,
        'source_v10_step': source['step'], 'qa_training_step': qa['step'],
        'baseline_loss': original_loss, 'qa_loss': trained_loss,
        'loss_difference_qa_minus_v10': trained_loss - original_loss,
        'qa_protocol_sha256': qa_hash, 'qa_checkpoint_sha256': sha256_file(qa_path),
        'source_checkpoint_sha256': source_hash,
        'warning': ('Small curated pilot, not unseen aviation-knowledge testing; '
                    'teacher-forced loss and exact-string matches are not factual correctness. '
                    'All generated drafts require independent human fact review.'),
        'items': details,
    }
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / (('final_test_locked' if split == 'test' else
                         f'val_{checkpoint_name}_step{qa["step"]}') + '.json')
    # Exclusive final-test record: never silently overwrite/re-run the held-out set.
    if split == 'test':
        with output.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    else:
        if output.exists():
            raise FileExistsError('Evaluation report exists: ' + str(output))
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--checkpoint', choices=('last', 'best'), default='best')
    parser.add_argument('--final-test', action='store_true')
    parser.add_argument('--tokens', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result, output = measure(args.split, args.checkpoint, final_test=args.final_test,
                             sample_tokens=args.tokens)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print('Saved:', output)


if __name__ == '__main__':
    main()
