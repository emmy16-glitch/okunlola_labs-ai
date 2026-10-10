"""Compare a locally trained v1.0 Transformer against the frozen v0.9c validation baseline.

This command NEVER accesses test.bin, trains a model, or modifies checkpoints.
It evaluates the entire validation set using the same v0.9d evaluation protocol
for both models, and refuses stale or mismatched checkpoint/report fingerprints.
"""
import argparse
import json
import math
from pathlib import Path

from .evaluate_large import evaluate
from .train_large_transformer import ROOT, sha256_file

BASELINE = ROOT / 'checkpoints' / 'v09d' / 'val_last_step5000.json'


def validate_report(report, *, checkpoint_hash, val_hash, tokenizer_fingerprint,
                    checkpoint_name, training_step):
    """Validate cached metric provenance, not just the displayed numeric loss."""
    fixed = {
        'version': 'skygrid_v09d_full_documents_v1',
        'split': 'val', 'checkpoint': checkpoint_name,
        'training_step': training_step, 'checkpoint_sha256': checkpoint_hash,
        'dataset_sha256': val_hash,
        'tokenizer_fingerprint': tokenizer_fingerprint,
    }
    for field, expected in fixed.items():
        if report.get(field) != expected:
            raise ValueError(f'Mismatched evaluation report: {field}')
    if report.get('evaluation_method') != (
        'All next-token targets within each document; reset context at EOS; '
        'exclude first token of each document'
    ):
        raise ValueError('Evaluation protocols differ')
    for name in ('cross_entropy_nats_per_token', 'perplexity',
                 'bits_per_scored_byte_including_eos_cost'):
        value = report.get(name)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Missing or invalid metric: {name}')
    for name in ('evaluated_targets', 'scored_utf8_bytes_excluding_first_token_per_document'):
        value = report.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f'Missing or invalid target count: {name}')
    return report


def compare_reports(baseline, contender):
    """Compare only exactly comparable, validated full-document scores."""
    for field in ('split', 'evaluation_method', 'dataset_sha256',
                  'tokenizer_fingerprint', 'evaluated_targets',
                  'scored_utf8_bytes_excluding_first_token_per_document'):
        if baseline[field] != contender[field]:
            raise ValueError(f'Not comparable: {field}')
    metric = 'cross_entropy_nats_per_token'
    delta = contender[metric] - baseline[metric]
    return {
        'split': 'val',
        'baseline': {'version': 'v09c', 'step': baseline['training_step'],
                     'loss': baseline[metric], 'perplexity': baseline['perplexity']},
        'contender': {'version': 'v10', 'step': contender['training_step'],
                      'loss': contender[metric], 'perplexity': contender['perplexity']},
        'loss_difference_v10_minus_v09c': delta,
        'lower_loss_is_better': True,
        'caveat': ('Same validation examples and scoring protocol, but different '
                   'context lengths and training steps. This is a language-model '
                   'prediction comparison, not a factual aviation QA score.'),
    }


def run_comparison(*, root=ROOT, checkpoint_name='last', batch_size=4,
                   evaluator=evaluate):
    """Safely evaluate v10 and reuse matching cached evaluation reports."""
    if checkpoint_name not in ('last', 'best') or batch_size < 1:
        raise ValueError('Choose last/best and a positive evaluation batch size')
    import torch
    root = Path(root)
    data = root / 'data' / 'aviation' / 'tokenized_v09b'
    baseline_path = root / 'checkpoints' / 'v09d' / 'val_last_step5000.json'
    v09c_path = root / 'checkpoints' / 'v09c' / 'last.pt'
    v10_folder = root / 'checkpoints' / 'v10'
    v10_path = v10_folder / (checkpoint_name + '.pt')
    reports_folder = root / 'checkpoints' / 'v10_evaluation'
    if not baseline_path.is_file():
        raise FileNotFoundError('Missing frozen v0.9c validation report: ' + str(baseline_path))
    if not v09c_path.is_file():
        raise FileNotFoundError('Missing original v0.9c checkpoint: ' + str(v09c_path))
    if not v10_path.is_file():
        raise FileNotFoundError('Train v10 locally before comparison: ' + str(v10_path))

    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    old_model = torch.load(v09c_path, map_location='cpu', weights_only=True)
    model = torch.load(v10_path, map_location='cpu', weights_only=True)
    if (old_model.get('format_version') != 'skygrid_v09c_1' or
            model.get('format_version') != 'skygrid_v09c_1'):
        raise ValueError('Unexpected model checkpoint format')
    if old_model.get('step') != 5000 or not isinstance(model.get('step'), int) or model['step'] < 1:
        raise ValueError('Unrecognized original baseline step or new training step')
    if (old_model.get('vocab_size') != model.get('vocab_size') or
            old_model.get('tokenizer_fingerprint') != model.get('tokenizer_fingerprint') or
            old_model.get('data_digests') != model.get('data_digests')):
        raise ValueError('Models used different tokenizer or training/validation data')
    for split in ('train', 'val'):
        if sha256_file(data / (split + '.bin')) != model['data_digests'][split]:
            raise ValueError(f'Local {split} data no longer matches checkpoints')
    from .large_bpe import LargeByteBPE
    tokenizer = LargeByteBPE.load(data / 'tokenizer.json')
    if tokenizer.fingerprint() != model['tokenizer_fingerprint'] or tokenizer.vocab_size != model['vocab_size']:
        raise ValueError('Tokenizer does not match model metadata')

    val_hash = model['data_digests']['val']
    token_fp = model['tokenizer_fingerprint']
    validate_report(baseline, checkpoint_hash=sha256_file(v09c_path),
                    val_hash=val_hash, tokenizer_fingerprint=token_fp,
                    checkpoint_name='last', training_step=5000)

    report_path = reports_folder / f'val_{checkpoint_name}_step{model["step"]}.json'
    if report_path.exists():
        contender = json.loads(report_path.read_text(encoding='utf-8'))
        reused = True
    else:
        contender, created = evaluator('val', checkpoint_name, data=data,
                                       training_output=v10_folder, output=reports_folder,
                                       batch_size=batch_size)
        if Path(created) != report_path:
            raise ValueError('Evaluation wrote an unexpected report filename')
        reused = False
    validate_report(contender, checkpoint_hash=sha256_file(v10_path),
                    val_hash=val_hash, tokenizer_fingerprint=token_fp,
                    checkpoint_name=checkpoint_name, training_step=model['step'])
    summary = compare_reports(baseline, contender)
    summary['evaluation_report'] = str(report_path)
    summary['cached_report_reused'] = reused
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=('best', 'last'), default='last')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = run_comparison(checkpoint_name=args.checkpoint, batch_size=args.batch_size)
    print(json.dumps(result, indent=2))
    print('Validation ONLY. The previous held-out test remains frozen.')


if __name__ == '__main__':
    main()
