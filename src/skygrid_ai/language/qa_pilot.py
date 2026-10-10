"""Auditable, document-disjoint pilot QA examples for SkyGrid's OWN model.

Grounded answers are candidate human-written reference summaries, not facts
certified by keyword matching. Source documents are fingerprint verified.
This is a tiny supervised-learning pilot, not a real-world UAV adviser.
"""
import hashlib
import json
from pathlib import Path

from src.skygrid_ai.knowledge.benchmark import load_questions
from src.skygrid_ai.knowledge.grounded import candidate_sentences, load_verified_document
from src.skygrid_ai.knowledge.retrieval import CORPUS, ROOT, words
from src.skygrid_ai.language.large_bpe import EOS_ID

PROTOCOL = ROOT / 'data' / 'qa' / 'aviation_v15_protocol.json'
BENCHMARKS = (
    ROOT / 'data' / 'benchmarks' / 'aviation_v10.json',
    ROOT / 'data' / 'benchmarks' / 'aviation_v11_independent.json',
)
NO_EVIDENCE = 'No supporting aviation source is available.'


def digest_files(paths):
    h = hashlib.sha256()
    for name in paths:
        p = Path(name)
        data = p.read_bytes()
        h.update(str(p.name).encode('utf-8') + b'\0')
        h.update(len(data).to_bytes(8, 'little') + data)
    return h.hexdigest()


def load_protocol(path=PROTOCOL, benchmarks=BENCHMARKS):
    """Validate explicit splits and prevent question/document leakage."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if raw.get('version') != 'skygrid_qa_pilot_v1':
        raise ValueError('Unknown QA protocol version')
    if raw.get('review_status') != 'candidate_examples_require_human_factual_and_rights_review':
        raise ValueError('QA review metadata must be preserved')
    splits = raw.get('splits')
    if not isinstance(splits, dict) or set(splits) != {'train', 'val', 'test'}:
        raise ValueError('Need train/val/test source-document splits')
    listed = [identifier for ids in splits.values() for identifier in ids]
    if len(listed) != len(set(listed)):
        raise ValueError('A question appears in more than one split')
    reference = [item for file in benchmarks for item in load_questions(file)]
    ids = {r['id']: r for r in reference}
    if len(ids) != len(reference) or set(listed) != set(ids):
        raise ValueError('QA splits must partition reference questions exactly')
    result = {'train': [], 'val': [], 'test': []}
    document_split = {}
    for split, selected in splits.items():
        if not selected:
            raise ValueError(f'Empty QA split: {split}')
        for identifier in selected:
            row = dict(ids[identifier])
            key = (row['source'], row['path'])
            if key in document_split and document_split[key] != split:
                raise ValueError('Source document leaks across QA splits')
            document_split[key] = split
            result[split].append({**row, 'split': split, 'kind': 'grounded',
                                  'answer': row['reference']})
    for row in raw.get('additional_grounded', []):
        if (not isinstance(row, dict) or row.get('split') not in result or
                not all(isinstance(row.get(k), str) and row[k].strip()
                        for k in ('id', 'source', 'path', 'question', 'answer')) or
                row['id'] in ids):
            raise ValueError('Invalid or duplicate additional grounded QA example')
        split = row['split']
        key = (row['source'], row['path'])
        if key in document_split and document_split[key] != split:
            raise ValueError('Source document leaks across QA splits')
        document_split[key] = split
        result[split].append({**row, 'kind': 'grounded'})
        ids[row['id']] = row
    for row in raw.get('unsupported', []):
        if (not isinstance(row, dict) or row.get('split') not in result or
                not all(isinstance(row.get(k), str) and row[k].strip()
                        for k in ('id', 'question', 'answer')) or
                row['id'] in ids):
            raise ValueError('Invalid unsupported question')
        ids[row['id']] = row
        result[row['split']].append({**row, 'kind': 'unsupported'})
    if len(ids) != len(listed) + len(raw.get('additional_grounded', [])) + len(raw.get('unsupported', [])):
        raise ValueError('Duplicate unsupported question')
    return result


def select_evidence(question, answer, source_text):
    """Pick a real sentence on the subject, for a human to review.

    Lexical overlap NEVER constitutes factual entailment. The source and its
    SHA validation are what protect provenance, not this ranking heuristic.
    """
    target = set(words(question)) | set(words(answer))
    candidates = list(candidate_sentences(source_text))
    if not candidates:
        raise ValueError('The pinned source contains no complete prose sentence')
    ranked = sorted(((len(set(words(s)) & target), -position, s)
                     for position, s in candidates), reverse=True)
    if ranked[0][0] < 2:
        raise ValueError('No sufficiently related source sentence for this QA candidate')
    return ranked[0][2]


def prepare_examples(*, protocol=PROTOCOL, benchmarks=BENCHMARKS, corpus=CORPUS):
    """Fail closed if any pinned document is absent/modified; no web downloads."""
    rows = load_protocol(protocol, benchmarks)
    manifest = json.loads((Path(corpus) / 'manifest.json').read_text(encoding='utf-8'))
    source_urls = {(r['source'], r['path']): r['url'] for r in manifest}
    for split in rows:
        for example in rows[split]:
            if example['kind'] == 'unsupported':
                example['evidence'] = NO_EVIDENCE
                example['citation'] = None
                continue
            text = load_verified_document(corpus, example['source'], example['path'])
            example['evidence'] = select_evidence(example['question'], example['answer'], text)
            example['citation'] = {'source': example['source'], 'path': example['path'],
                                   'url': source_urls[(example['source'], example['path'])]}
    return rows


def prompt_for_example(row, tokenizer, context_size, answer_room=1):
    if answer_room < 1 or context_size < 24:
        raise ValueError('Invalid prompt token budget')
    word_list = row['evidence'].split()
    while True:
        prompt = 'Question: ' + row['question'] + '\nEvidence: ' + ' '.join(word_list) + '\nAnswer:'
        if len(tokenizer.encode(prompt)) + answer_room <= context_size:
            return prompt
        if len(word_list) <= 3:
            raise ValueError('Question and evidence cannot fit context')
        word_list.pop()


def encode_supervised(row, tokenizer, context_size):
    """Train ONLY on answer/EOS tokens, never prompt/padding targets.

    Keep the entire reference answer; if it doesn't fit, shorten ONLY evidence
    at word boundaries. Do not silently clip answer tokens or split UTF-8.
    """
    if context_size < 24:
        raise ValueError('QA context window is too small')
    answer_tokens = tokenizer.encode(' ' + row['answer']) + [EOS_ID]
    if len(answer_tokens) < 2 or len(answer_tokens) >= context_size:
        raise ValueError('Reference answer exceeds context window')
    words_evidence = row['evidence'].split()
    while True:
        prefix = 'Question: ' + row['question'] + '\nEvidence: ' + ' '.join(words_evidence) + '\nAnswer:'
        prefix_ids = tokenizer.encode(prefix)
        sequence = prefix_ids + answer_tokens
        if len(prefix_ids) >= 2 and len(sequence) - 1 <= context_size:
            break
        if len(words_evidence) <= 3:
            raise ValueError('Question, answer and minimum evidence do not fit context')
        words_evidence.pop()
    x = sequence[:-1]
    y = [-100] * (len(prefix_ids) - 1) + sequence[len(prefix_ids):]
    assert len(x) == len(y)
    assert any(value != -100 for value in y)
    return x, y
