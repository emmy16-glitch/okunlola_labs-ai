"""Human-curated aviation QA data, using the EXACT live question-only retrieval gate.

Candidate creation is mechanical and does not constitute a factual review.
No expected answers are passed into search, evidence selection or generation.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from src.skygrid_ai.knowledge.grounded import answer_from_evidence, candidate_sentences, load_verified_document
from src.skygrid_ai.knowledge.retrieval import CORPUS, INDEX, ROOT, load_index, _sha
from .qa_pilot import NO_EVIDENCE, digest_files
from .large_bpe import EOS_ID

CANDIDATES = ROOT / 'data' / 'qa' / 'candidates_v16.json'
REVIEWS = ROOT / 'data' / 'qa' / 'reviews_v16.json'
VERSION = 'skygrid_qa_v16_1'
REFUSAL = 'The supplied documentation does not establish that claim.'


def split_for(source, path):
    """Stable document-level 70/15/15 split; never split one document."""
    h = hashlib.sha256(f'{source}:{path}'.encode('utf-8')).digest()[0]
    return 'train' if h < 179 else 'val' if h < 218 else 'test'


def _topic(path):
    return ' '.join(re.sub(r'[_-]+', ' ', Path(path).stem).split())


def generate_candidates(*, corpus=CORPUS, maximum=300):
    """Draft one source-linked question per document, for mandatory human review."""
    if maximum < 1:
        raise ValueError('maximum must be positive')
    folder = Path(corpus)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    result = []
    for row in sorted(manifest, key=lambda r: (r['source'], r['path'])):
        if len(result) == maximum:
            break
        # Keep one document per question, so future validation is document-disjoint.
        text = load_verified_document(folder, row['source'], row['path'], manifest=manifest)
        sentences = [s for _, s in candidate_sentences(text) if 55 <= len(s) <= 330]
        if not sentences:
            continue
        topic = _topic(row['path'])
        if not topic or topic in ('index', 'readme'):
            continue
        topic_terms = set(re.findall(r'[a-z0-9]+', topic.casefold()))
        answer = max(sentences[:16], key=lambda s: (
            len(set(re.findall(r'[a-z0-9]+', s.casefold())) & topic_terms),
            any(v in s.casefold() for v in (' allows ', ' is used ', ' enables ', ' provides ')),
            -sentences.index(s)))
        cid = hashlib.sha256(f"{row['source']}:{row['path']}".encode()).hexdigest()[:16]
        result.append({
            'id': 'draft_' + cid,
            'source': row['source'], 'path': row['path'], 'url': row['url'],
            'source_sha256': row['sha256_cleaned'], 'split': split_for(row['source'], row['path']),
            'question': f'What does the {row["source"]} documentation say about {topic}?',
            'answer': answer, 'kind': 'grounded',
            'review_status': 'pending_human_review',
        })
    # Negative examples also require a reviewer to establish that the
    # retrieved passage does NOT support the proposed safety guarantee.
    negatives = [
        ('Does MAVLink HEARTBEAT guarantee collision-free flight?', 'train'),
        ('Does PX4 control allocation guarantee every flight is safe?', 'train'),
        ('Does MAVLink message signing guarantee a flight will avoid obstacles?', 'val'),
        ('Does PX4 flight mode selection guarantee collision-free flight?', 'val'),
        ('Does MAVLink telemetry guarantee that all weather is safe?', 'test'),
        ('Does QGroundControl guarantee every drone has enough battery?', 'test'),
    ]
    for question, split in negatives:
        ident = hashlib.sha256(question.encode()).hexdigest()[:16]
        result.append({'id': 'refusal_' + ident, 'kind': 'unsupported',
                       'split': split, 'source': '', 'path': '', 'url': '',
                       'source_sha256': '', 'question': question,
                       'answer': REFUSAL, 'review_status': 'pending_human_review'})
    return {'version': VERSION, 'notice': 'UNREVIEWED DRAFTS. Rewrite and check against original pinned source before approving.',
            'candidates': result}


def review_template(bundle):
    if bundle.get('version') != VERSION:
        raise ValueError('Unsupported candidate format')
    return {'version': VERSION, 'candidate_sha256': hash_json(bundle),
            'instructions': 'Independently inspect original source and every question/answer. Approve only when accurate.',
            'reviews': {row['id']: {'approved': False, 'reviewer': '',
                                   'source_checked': False, 'answer_supported': False,
                                   'question_checked': False} for row in bundle['candidates']}}


def hash_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def save_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_reviewed(*, candidates=CANDIDATES, reviews=REVIEWS, corpus=CORPUS,
                  index=None, minimum=100, strict=True):
    """Include ONLY explicitly reviewed, citation-matched examples.

    A claimed approval cannot be independently verified by software; the
    reviewer remains responsible for checking claims and original documents.
    """
    candidates, reviews = Path(candidates), Path(reviews)
    bundle = json.loads(candidates.read_text(encoding='utf-8'))
    record = json.loads(reviews.read_text(encoding='utf-8'))
    if bundle.get('version') != VERSION or record.get('version') != VERSION:
        raise ValueError('Unsupported QA review format')
    if record.get('candidate_sha256') != hash_json(bundle):
        raise ValueError('Candidate file changed since human review')
    items = bundle.get('candidates')
    ratings = record.get('reviews')
    if not isinstance(items, list) or not isinstance(ratings, dict):
        raise ValueError('Invalid QA candidate or review rows')
    ids = [r['id'] for r in items]
    if len(ids) != len(set(ids)) or set(ratings) != set(ids):
        raise ValueError('Duplicate/missing QA reviews')
    index = load_index() if index is None else index
    splits = {'train': [], 'val': [], 'test': []}
    rejected = []
    seen_documents = set()
    seen_questions = set()
    manifest = json.loads((Path(corpus) / 'manifest.json').read_text(encoding='utf-8'))
    meta = {(r['source'], r['path']): r for r in manifest}
    for row in items:
        review = ratings[row['id']]
        if review.get('approved') is not True:
            continue
        if not (isinstance(review.get('reviewer'), str) and review['reviewer'].strip()
                and all(review.get(k) is True for k in
                        ('source_checked', 'answer_supported', 'question_checked'))):
            raise ValueError(f'Approved record has incomplete human review: {row["id"]}')
        if row.get('kind') == 'unsupported':
            question = row['question'].strip()
            if not question or question.casefold() in seen_questions or row['split'] not in splits:
                raise ValueError('Invalid or duplicate approved refusal question')
            seen_questions.add(question.casefold())
            # Only use a refusal training example when the live path would
            # actually provide evidence to the neural model. A missing source
            # already triggers the non-neural abstention gate in production.
            gate = answer_from_evidence(index, question, corpus=corpus)
            if gate['status'] != 'excerpt_found':
                rejected.append({'id': row['id'], 'reason': 'live_gate_already_refuses',
                                 'status': gate['status']})
                continue
            splits[row['split']].append({**row, 'evidence': gate['answer'],
                                        'citation': gate['citations'][0],
                                        'reviewer': review['reviewer']})
            continue
        key = (row['source'], row['path'])
        if key not in meta or (row['url'] != meta[key]['url'] or
                               row['source_sha256'] != meta[key]['sha256_cleaned']):
            raise ValueError(f'Candidate provenance changed: {row["id"]}')
        if key in seen_documents or row['question'].strip().casefold() in seen_questions:
            raise ValueError('Duplicate approved QA document or question')
        seen_documents.add(key)
        seen_questions.add(row['question'].strip().casefold())
        if row['split'] != split_for(*key):
            raise ValueError('QA document split was modified')
        if row.get('kind') != 'grounded' or not row['question'].strip() or not row['answer'].strip():
            raise ValueError('Malformed grounded QA record')
        # Check every source fingerprint before any training. The expected answer
        # is NEVER given to answer_from_evidence, preserving live behavior.
        original = load_verified_document(corpus, *key, manifest=manifest)
        if _sha(original) != row['source_sha256']:
            raise ValueError('Modified source document')
        result = answer_from_evidence(index, row['question'], corpus=corpus)
        citation = result.get('citations', [])
        if (result['status'] != 'excerpt_found' or not citation or
                (citation[0]['source'], citation[0]['path']) != key):
            rejected.append({'id': row['id'], 'reason': 'question_only_source_mismatch_or_abstention',
                             'status': result['status']})
            continue
        splits[row['split']].append({**row, 'evidence': result['answer'],
                                   'citation': citation[0], 'reviewer': review['reviewer']})
    count = sum(len(v) for v in splits.values())
    if strict and (count < minimum or min(len(v) for v in splits.values()) < 8):
        raise ValueError(f'Only {count} reviewed, retrieval-supported QAs '
                         f'(train={len(splits["train"])} val={len(splits["val"])} '
                         f'test={len(splits["test"])}). Need at least {minimum} and 8 per split. '
                         f'Rejected evidence matches: {len(rejected)}')
    return splits, rejected


def dataset_fingerprint(*, candidates=CANDIDATES, reviews=REVIEWS, corpus=CORPUS, index_path=INDEX):
    # Tie the model to ALL input files and both the retrieval algorithm and index.
    return digest_files([candidates, reviews, Path(corpus) / 'manifest.json', index_path,
                         ROOT / 'src/skygrid_ai/knowledge/retrieval.py',
                         ROOT / 'src/skygrid_ai/knowledge/grounded.py'])


def data_report(splits, rejected):
    return {'reviewed_retrieval_supported': {k: len(v) for k, v in splits.items()},
            'rejected_due_to_question_only_retrieval': rejected,
            'human_verified_factual_accuracy': 'not independently certified by software'}


ANSWER_RESERVE = 96


def prompt_v16(question, evidence, tokenizer, context_size, *, answer_reserve=ANSWER_RESERVE):
    """Identical prompt and evidence budget in training and live inference."""
    if not question.strip() or not evidence.strip() or context_size <= answer_reserve + 20:
        raise ValueError('Question/evidence/context not usable')
    evidence_words = evidence.split()
    while True:
        prompt = 'Question: ' + question + '\nEvidence: ' + ' '.join(evidence_words) + '\nAnswer:'
        if len(tokenizer.encode(prompt)) + answer_reserve <= context_size:
            return prompt
        if len(evidence_words) <= 3:
            raise ValueError('Question and evidence cannot fit context')
        evidence_words.pop()


def encode_v16(row, tokenizer, context_size):
    answer_ids = tokenizer.encode(' ' + row['answer']) + [EOS_ID]
    if len(answer_ids) > ANSWER_RESERVE:
        raise ValueError('Human-reviewed answer too long for reserved answer budget')
    prompt = prompt_v16(row['question'], row['evidence'], tokenizer, context_size)
    prompt_ids = tokenizer.encode(prompt)
    all_ids = prompt_ids + answer_ids
    x = all_ids[:-1]
    y = [-100] * (len(prompt_ids) - 1) + all_ids[len(prompt_ids):]
    if len(x) > context_size or len(x) != len(y):
        raise ValueError('Invalid supervised window')
    return x, y
