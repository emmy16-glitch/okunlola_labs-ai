"""Opt-in, review-free WEAK SUPERVISION experiment from local verified docs.

Automatically extracted source sentences are NOT human-verified QA examples.
Never mixes these labels with qa_v16's approved rows or checkpoints.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from src.skygrid_ai.knowledge.grounded import answer_from_evidence, candidate_sentences, load_verified_document
from src.skygrid_ai.knowledge.retrieval import CORPUS, INDEX, ROOT, load_index
from .qa_v16 import split_for, save_new, encode_v16
from .train_qa import load_starting_point
from .train_qa_v16 import fit, evaluate_v16
from .modern_transformer import SkyGridTransformerV08
from .evaluate_qa_v16 import generate_v16
from .train_large_transformer import sha256_file

DATASET = ROOT / 'data' / 'qa' / 'auto_weak_v18.json'
OUTPUT = ROOT / 'checkpoints' / 'qa_auto_v18'
PATTERN = re.compile(
    r'^(?:The |A |An )?(?P<subject>[A-Za-z][\w/()\- ]{2,65}?) '
    r'(?P<verb>is used to|are used to|can be used to|allows|enables|provides|'
    r'sends|reports|advertises|supports|translates|converts|is responsible for)\b', re.I)
BAD_SUBJECT = {'it', 'this', 'that', 'these', 'those', 'they', 'we', 'you', 'there', 'following'}


def question_from_sentence(sentence, source):
    """Simple rule-based transformation; never asserts semantic correctness."""
    match = PATTERN.match(sentence)
    if not match:
        return None
    subject = match['subject'].strip()
    if not 1 <= len(subject.split()) <= 8 or subject.lower() in BAD_SUBJECT:
        return None
    if any(c in subject for c in '[]{}'):
        return None
    verb = match['verb'].lower()
    if 'used to' in verb or 'responsible for' in verb:
        q = f'what is {subject} used for?'
    elif verb in {'provides', 'sends', 'reports', 'advertises', 'supports', 'translates', 'converts', 'allows', 'enables'}:
        q = f'what does {subject} {verb.rstrip("s") if verb.endswith("s") else verb}?'
    else:
        return None
    return f'In {source.upper()}, {q}'


def create_weak_dataset(*, maximum=300, corpus=CORPUS, index=None):
    """Source-verified document-disjoint rows; no artificial approvals."""
    if maximum < 30:
        raise ValueError('Need at least 30 requested examples')
    folder = Path(corpus)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    index = load_index() if index is None else index
    rows = {'train': [], 'val': [], 'test': []}
    counts = {'documents_examined': 0, 'retrieval_or_quality_rejected': 0}
    seen_questions = set()
    for doc in sorted(manifest, key=lambda r: (r['source'], r['path'])):
        if sum(map(len, rows.values())) >= maximum:
            break
        text = load_verified_document(folder, doc['source'], doc['path'], manifest=manifest)
        counts['documents_examined'] += 1
        split = split_for(doc['source'], doc['path'])
        # A maximum of one QA pair per document prevents document overlap between splits.
        for _, sentence in list(candidate_sentences(text))[:40]:
            if not 55 <= len(sentence) <= 260:
                continue
            question = question_from_sentence(sentence, doc['source'])
            if not question or question.lower() in seen_questions:
                continue
            result = answer_from_evidence(index, question, corpus=folder)
            citation = result.get('citations') or []
            if (result['status'] != 'excerpt_found' or not citation or
                    (citation[0]['source'], citation[0]['path']) != (doc['source'], doc['path']) or
                    not 45 <= len(result['answer']) <= 260):
                counts['retrieval_or_quality_rejected'] += 1
                continue
            row = {'id': 'weak_' + hashlib.sha256((doc['source'] + ':' + doc['path']).encode()).hexdigest()[:16],
                   'question': question, 'answer': result['answer'],
                   'evidence': result['answer'], 'citation': citation[0],
                   'kind': 'weak_extractive', 'split': split,
                   'source_sha256': doc['sha256_cleaned']}
            rows[split].append(row)
            seen_questions.add(question.lower())
            break
    return {'format': 'skygrid_auto_weak_v18_1', 'status': 'AUTOMATIC_UNREVIEWED_NOT_FACT_CERTIFIED',
            'source_manifest_sha256': sha256_file(folder / 'manifest.json'),
            'index_sha256': sha256_file(INDEX),
            'retriever_sha256': sha256_file(ROOT / 'src/skygrid_ai/knowledge/retrieval.py'),
            'extractor_sha256': sha256_file(ROOT / 'src/skygrid_ai/knowledge/grounded.py'),
            'counts': counts, 'splits': rows}


def load_dataset(path=DATASET, *, corpus=CORPUS):
    bundle = json.loads(Path(path).read_text(encoding='utf-8'))
    if bundle.get('format') != 'skygrid_auto_weak_v18_1' or bundle.get('status') != 'AUTOMATIC_UNREVIEWED_NOT_FACT_CERTIFIED':
        raise ValueError('Unexpected weak-label dataset format/status')
    if sha256_file(Path(corpus) / 'manifest.json') != bundle.get('source_manifest_sha256'):
        raise ValueError('Source manifest changed; do not reuse old weak labels')
    for key, path_to_check in (('index_sha256', INDEX),
                               ('retriever_sha256', ROOT / 'src/skygrid_ai/knowledge/retrieval.py'),
                               ('extractor_sha256', ROOT / 'src/skygrid_ai/knowledge/grounded.py')):
        if sha256_file(path_to_check) != bundle.get(key):
            raise ValueError('Index/retrieval implementation changed: ' + key)
    rows = bundle['splits']
    if set(rows) != {'train', 'val', 'test'}:
        raise ValueError('Invalid data splits')
    docs, questions = set(), set()
    for split, items in rows.items():
        for row in items:
            citation = row['citation']
            key = (citation['source'], citation['path'])
            if key in docs or row['question'] in questions or row['split'] != split:
                raise ValueError('Duplicated document/question or split mismatch')
            if split_for(*key) != split or row['answer'] != row['evidence'] or row['kind'] != 'weak_extractive':
                raise ValueError('Corrupt weak-label example')
            # Local source files still need to match the pinned original manifest.
            load_verified_document(corpus, *key)
            docs.add(key)
            questions.add(row['question'])
    return rows


def run_training(path=DATASET, *, steps=100, resume=False):
    rows = load_dataset(path)
    counts = {name: len(group) for name, group in rows.items()}
    if sum(counts.values()) < 100 or min(counts['val'], counts['test']) < 8 or counts['train'] < 20:
        raise ValueError(f'Not enough automatic source-matched examples: {counts}; need 100 overall and 8 per evaluation split')
    tokenizer, source, source_hash = load_starting_point()
    context = source['model_config']['context_size']
    for split in ('train', 'val'):
        for row in rows[split]:
            encode_v16(row, tokenizer, context)
    # Explicit separate output and a hash of the AUTOMATIC dataset, not a human review signature.
    return fit(rows, tokenizer, source, source_hash, sha256_file(path), output=OUTPUT,
               steps=steps, resume=resume, min_train=20, lr=2e-5, patience=4)


def evaluate(path=DATASET, *, checkpoint='best'):
    if checkpoint not in ('best', 'last'):
        raise ValueError('Invalid checkpoint')
    rows = load_dataset(path)
    tokenizer, source, source_hash = load_starting_point()
    location = OUTPUT / (checkpoint + '.pt')
    cp = torch.load(location, map_location='cpu', weights_only=True)
    if (cp.get('qa_dataset_sha256') != sha256_file(path) or
            cp.get('source_checkpoint_sha256') != source_hash or
            cp.get('evidence_selection') != 'question_only_live_extraction'):
        raise ValueError('Auto QA checkpoint/data provenance mismatch')
    old = SkyGridTransformerV08(vocab_size=tokenizer.vocab_size, **source['model_config'])
    old.load_state_dict(source['model_state'])
    model = SkyGridTransformerV08(vocab_size=tokenizer.vocab_size, **cp['model_config'])
    model.load_state_dict(cp['model_state'])
    context = cp['model_config']['context_size']
    before, n = evaluate_v16(old, rows['val'], tokenizer, context)
    after, _ = evaluate_v16(model, rows['val'], tokenizer, context)
    preview = [{'question': row['question'], 'reference_extracted': row['answer'],
                'neural_draft_unverified': generate_v16(model, tokenizer, row, context, limit=64),
                'citation': row['citation']} for row in rows['val'][:6]]
    return {'split': 'val', 'label_status': 'unreviewed_weak_source_extraction',
            'validation_examples': len(rows['val']), 'target_tokens': n,
            'baseline_loss': before, 'trained_loss': after, 'step': cp['step'],
            'drafts': preview,
            'warning': 'No independent human factual accuracy grade; this is a weak supervision experiment, not flight advice.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--maximum', type=int, default=300)
    train = sub.add_parser('train')
    train.add_argument('--steps', type=int, default=100)
    train.add_argument('--resume', action='store_true')
    train.add_argument('--confirm-rights-reviewed', action='store_true')
    train.add_argument('--accept-unreviewed-weak-labels', action='store_true')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--checkpoint', choices=['last', 'best'], default='best')
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.command == 'build':
        out = create_weak_dataset(maximum=args.maximum)
        save_new(DATASET, out)
        print(json.dumps({'path': str(DATASET), 'counts': {k: len(v) for k, v in out['splits'].items()},
                          'status': out['status'], 'review_questions_required': 0}, indent=2))
    elif args.command == 'train':
        if not (args.confirm_rights_reviewed and args.accept_unreviewed_weak_labels):
            parser.error('Requires reviewed document rights and explicit acknowledgement of UNREVIEWED weak training labels')
        print(json.dumps(run_training(steps=args.steps, resume=args.resume), indent=2))
    else:
        print(json.dumps(evaluate(checkpoint=args.checkpoint), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
