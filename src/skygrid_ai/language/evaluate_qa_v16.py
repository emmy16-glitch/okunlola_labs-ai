"""Evaluate approved question-only QA on held-out documents, then request human grading.

Does NOT automatically equate token loss or keyword matches with factual correctness.
Final-test requires an explicit flag and is locked after one run.
"""
import argparse
import json
from pathlib import Path
import torch

from .qa_v16 import CANDIDATES, REVIEWS, data_report, dataset_fingerprint, load_reviewed, save_new
from .train_large_transformer import ROOT, sha256_file
from .train_qa import load_starting_point, SOURCE
from .train_qa_v16 import evaluate_v16
from .qa_v16 import prompt_v16
from .train_qa_v16 import FORMAT, OUTPUT
from .modern_transformer import SkyGridTransformerV08
from .large_bpe import EOS_ID, PAD_ID

REPORTS = ROOT / 'checkpoints' / 'qa_v16_evaluation'



def generate_v16(model, tok, row, context, *, limit=64):
    prompt = prompt_v16(row['question'], row['evidence'], tok, context)
    generated = tok.encode(prompt)
    output = []
    model.eval()
    with torch.inference_mode():
        for _ in range(limit):
            ids = torch.tensor([generated[-context:]], dtype=torch.long)
            logits, _ = model(ids)
            token = int(logits[0, -1].argmax().item())
            if token in (EOS_ID, PAD_ID):
                break
            output.append(token)
            generated.append(token)
    return b''.join(tok.id_to_bytes[int(t)] for t in output).decode('utf-8', errors='replace').strip()


def compare(*, root=ROOT, candidates=CANDIDATES, reviews=REVIEWS,
            checkpoint='best', split='val', final_test=False, limit=64,
            rows=None, qa_hash=None):
    if split not in ('val', 'test') or checkpoint not in ('best', 'last') or limit < 1:
        raise ValueError('Invalid evaluation arguments')
    if split == 'test' and not final_test:
        raise ValueError('Test is reserved: use --final-test only once')
    root = Path(root)
    reports = root / 'checkpoints' / 'qa_v16_evaluation'
    destination = reports / ('final_test_locked.json' if split == 'test' else f'val_{checkpoint}.json')
    if destination.exists():
        raise FileExistsError('QA report already exists; never replace final test or prior evaluation')
    if rows is None:
        rows, _ = load_reviewed(candidates=candidates, reviews=reviews,
                                corpus=root / 'data/aviation/processed_v09b', minimum=100)
    digest = qa_hash or dataset_fingerprint(candidates=candidates, reviews=reviews,
                             corpus=root / 'data/aviation/processed_v09b',
                             index_path=root / 'data/aviation/index_v10/passages.json.gz')
    cp_path = root / 'checkpoints' / 'qa_v16' / (checkpoint + '.pt')
    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
    if cp.get('format_version') != FORMAT or cp.get('qa_dataset_sha256') != digest or cp.get('evidence_selection') != 'question_only_live_extraction':
        raise ValueError('QA source/protocol/retrieval fingerprint mismatch')
    source_path = root / 'checkpoints' / 'v10' / 'last.pt'
    if not source_path.is_file() or sha256_file(source_path) != cp.get('source_checkpoint_sha256'):
        raise ValueError('Original source v10 checkpoint changed')
    tokenizer, source, _ = load_starting_point(source=source_path, data=root/'data/aviation/tokenized_v09b')
    if (cp.get('model_config') != source.get('model_config') or
            cp.get('tokenizer_fingerprint') != tokenizer.fingerprint() or
            cp.get('data_digests') != source.get('data_digests')):
        raise ValueError('QA model does not match original tokenizer/training data')
    baseline = SkyGridTransformerV08(vocab_size=tokenizer.vocab_size, **source['model_config'])
    baseline.load_state_dict(source['model_state'])
    trained = SkyGridTransformerV08(vocab_size=tokenizer.vocab_size, **cp['model_config'])
    trained.load_state_dict(cp['model_state'])
    context = cp['model_config']['context_size']
    baseline_loss, base_targets = evaluate_v16(baseline, rows[split], tokenizer, context)
    qa_loss, qa_targets = evaluate_v16(trained, rows[split], tokenizer, context)
    if qa_targets != base_targets:
        raise ValueError('Mismatched validation targets')
    items = []
    for row in rows[split]:
        items.append({'id': row['id'], 'question': row['question'],
                      'reference_answer': row['answer'],
                      'question_only_evidence': row['evidence'],
                      'citation': row.get('citation'),
                      'source_draft': generate_v16(baseline, tokenizer, row, context, limit=limit),
                      'qa_draft': generate_v16(trained, tokenizer, row, context, limit=limit),
                      'needs_human_factual_support_relevance_and_refusal_review': True})
    result = {'split': split, 'qa_step': cp['step'], 'source_v10_step': source['step'],
              'source_loss': baseline_loss, 'qa_loss': qa_loss,
              'loss_difference': qa_loss - baseline_loss, 'answer_targets': qa_targets,
              'model_sha256': sha256_file(cp_path), 'dataset_sha256': digest,
              'warning': 'Loss is teacher-forced; it does not certify correct factual answers. Human scoring required.',
              'items': items}
    save_new(destination, result)
    return result, destination



def refusal_audit(*, root=ROOT, challenges=None, index=None, corpus=None):
    """Check retrieval gate and flag unsupported claims; no automatic truth labels."""
    from src.skygrid_ai.knowledge.retrieval import load_index
    from src.skygrid_ai.knowledge.grounded import answer_from_evidence
    root = Path(root)
    path = root / 'data/qa/refusal_challenges_v16.json'
    examples = json.loads(path.read_text(encoding='utf-8')) if challenges is None else challenges
    if index is None:
        index = load_index()
    if corpus is None:
        corpus = root / 'data/aviation/processed_v09b'
    output = []
    for row in examples:
        result = answer_from_evidence(index, row['question'], corpus=corpus)
        gated = result['status'] != 'excerpt_found'
        expected = row['expected_gate']
        passed = (gated if expected == 'refuse' else
                  not gated if expected == 'evidence' else None)
        output.append({'id': row['id'], 'expected_gate': expected,
                       'actual_status': result['status'], 'neural_generation_blocked': gated,
                       'check_passed': passed,
                       'human_entailment_review_required': True,
                       'citation': result.get('citations', [])})
    measured = [x for x in output if x['check_passed'] is not None]
    return {'gated_cases_checked': len(measured),
            'checks_passed': sum(x['check_passed'] for x in measured),
            'warning': 'Lexical source access is not proof of safety or accurate refusal; inspect review cases manually',
            'items': output}


def human_review_template(report):
    return {'report_sha256': report['model_sha256'], 'split': report['split'],
            'instructions': 'Check each generated claim against the original pinned source. Grade factuality, support, relevance, and refusal separately.',
            'grades': {item['id']: {'source_checked': False,
                                  'baseline': {'factuality': None, 'source_support': None,
                                               'relevance': None, 'refusal': None},
                                  'qa': {'factuality': None, 'source_support': None,
                                         'relevance': None, 'refusal': None},
                                  'notes': ''} for item in report['items']}}


def score_manual(report, ratings):
    if (ratings.get('report_sha256') != report['model_sha256'] or
            ratings.get('split') != report['split']):
        raise ValueError('Human review is for a different QA report')
    expected = {item['id'] for item in report['items']}
    if set(ratings.get('grades', {})) != expected or not expected:
        raise ValueError('Every example must be reviewed')
    dimensions = ('factuality', 'source_support', 'relevance', 'refusal')
    permitted = {'pass', 'partial', 'fail', 'not_applicable'}
    totals = {model: {dim: {'pass': 0, 'partial': 0, 'fail': 0, 'not_applicable': 0}
                      for dim in dimensions} for model in ('baseline', 'qa')}
    for key in expected:
        data = ratings['grades'][key]
        if data.get('source_checked') is not True:
            raise ValueError('Source has not been independently checked')
        for model in totals:
            row = data.get(model)
            if not isinstance(row, dict):
                raise ValueError('Missing human model grades')
            for dimension in dimensions:
                grade = row.get(dimension)
                if grade not in permitted:
                    raise ValueError('Incomplete human grade: ' + dimension)
                totals[model][dimension][grade] += 1
    return {'examples_human_reviewed': len(expected), 'split': report['split'],
            'grades': totals, 'note': 'Human ratings supplied by reviewer, not independent automated factual verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val','test'), default='val')
    parser.add_argument('--checkpoint', choices=('best','last'), default='best')
    parser.add_argument('--final-test', action='store_true')
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--refusal-audit', action='store_true')
    parser.add_argument('--review-report', help='Existing evaluation JSON for generating/scoring a human review')
    parser.add_argument('--review-template', help='Write a NEW review template file')
    parser.add_argument('--ratings', help='Human-completed grades JSON')
    args = parser.parse_args()
    if args.refusal_audit:
        print(json.dumps(refusal_audit(), indent=2))
        return
    if args.review_report:
        report = json.loads(Path(args.review_report).read_text(encoding='utf-8'))
        if bool(args.review_template) == bool(args.ratings):
            parser.error('Choose --review-template OR --ratings')
        if args.review_template:
            save_new(args.review_template, human_review_template(report))
            print('Review form:', args.review_template)
        else:
            values = json.loads(Path(args.ratings).read_text(encoding='utf-8'))
            print(json.dumps(score_manual(report, values), indent=2))
        return
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result, path = compare(split=args.split, checkpoint=args.checkpoint,
                           final_test=args.final_test, limit=args.tokens)
    print(json.dumps({'report': str(path), 'source_loss': result['source_loss'],
                      'qa_loss': result['qa_loss'], 'examples': len(result['items']),
                      'needs_human_review': True}, indent=2))


if __name__ == '__main__':
    main()
