"""Interactive manual review of local v1.6 aviation Q&A candidates.

This tool CANNOT verify factual accuracy or approve anything by itself. It
shows provenance, preflights the current question-only retrieval gate, and
records explicit reviewer decisions in the existing review JSON. Rejected
candidates are skipped on subsequent runs. It never edits training data.

Windows Git Bash examples:
  python -m src.skygrid_ai.language.review_qa_v16 --status
  python -m src.skygrid_ai.language.review_qa_v16 --reviewer "My name" --limit 15
  python -m src.skygrid_ai.language.review_qa_v16 --reviewer "My name" --split val
"""
import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .qa_v16 import CANDIDATES, REVIEWS, VERSION, hash_json
from src.skygrid_ai.knowledge.grounded import answer_from_evidence, load_verified_document
from src.skygrid_ai.knowledge.retrieval import CORPUS, load_index


def load_queue(candidates=CANDIDATES, reviews=REVIEWS):
    """Check the review-file-to-candidates binding before ANY edit."""
    bundle = json.loads(Path(candidates).read_text(encoding='utf-8'))
    record = json.loads(Path(reviews).read_text(encoding='utf-8'))
    if bundle.get('version') != VERSION or record.get('version') != VERSION:
        raise ValueError('Expected v1.6 candidate and review files')
    if record.get('candidate_sha256') != hash_json(bundle):
        raise ValueError('Candidates changed after the review template was generated. Reconcile the files before reviewing.')
    items = bundle.get('candidates')
    reviews_by_id = record.get('reviews')
    if not isinstance(items, list) or not isinstance(reviews_by_id, dict):
        raise ValueError('Invalid candidate/review structure')
    ids = [r.get('id') for r in items if isinstance(r, dict)]
    if (len(ids) != len(items) or len(ids) != len(set(ids)) or
            set(ids) != set(reviews_by_id)):
        raise ValueError('Missing or duplicate candidate/review IDs')
    if any(r.get('split') not in ('train', 'val', 'test') or
           r.get('kind') not in ('grounded', 'unsupported') for r in items):
        raise ValueError('Invalid split or candidate type')
    return bundle, record


def decision_for(review):
    """Existing v1.6 approvals remain valid; extra rejection metadata is optional."""
    if review.get('approved') is True:
        return 'approved'
    if review.get('decision') == 'rejected':
        return 'rejected'
    return 'pending'


def counts(bundle, record):
    result = {s: {'approved': 0, 'rejected': 0, 'pending': 0} for s in ('train', 'val', 'test')}
    for row in bundle['candidates']:
        result[row['split']][decision_for(record['reviews'][row['id']])] += 1
    return {'total': len(bundle['candidates']), 'by_split': result,
            'approved_total': sum(row['approved'] for row in result.values()),
            'minimum_training_approval': '100 total and at least 8 in each split; live retrieval must also support each approved answer'}


def record_decision(record, identifier, decision, reviewer, *, checked=False, reason=''):
    """Only an explicit affirmative human response can produce approval flags."""
    if decision not in ('approved', 'rejected'):
        raise ValueError('Decision must be approved or rejected')
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('Reviewer name is required')
    entry = record['reviews'][identifier]
    if decision_for(entry) != 'pending':
        raise ValueError('This candidate was already reviewed; do not overwrite prior decisions')
    if decision == 'approved' and checked is not True:
        raise ValueError('Approval requires affirmative independent source/question/answer checks')
    entry.update({'approved': decision == 'approved', 'reviewer': reviewer.strip(),
                  'source_checked': bool(checked) if decision == 'approved' else False,
                  'answer_supported': bool(checked) if decision == 'approved' else False,
                  'question_checked': bool(checked) if decision == 'approved' else False,
                  'decision': decision})
    if decision == 'rejected':
        entry['review_note'] = reason.strip()
    return record


def save_reviews(path, record, *, previous_bytes):
    """Atomic save, with a backup. Refuse accidental concurrent editing."""
    path = Path(path)
    if path.read_bytes() != previous_bytes:
        raise ValueError('Review file changed while this session was open; refusing to overwrite')
    text = json.dumps(record, indent=2, ensure_ascii=False) + '\n'
    backup = path.with_name(path.name + '.bak')
    shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path.read_bytes()


def eligible_evidence(row, index, corpus=CORPUS):
    """The same question-only source gate used in training and inference."""
    if row['kind'] == 'unsupported':
        evidence = answer_from_evidence(index, row['question'], corpus=corpus)
        if evidence['status'] != 'excerpt_found':
            return None, 'live system already refuses this unsupported question'
        return evidence['answer'], None
    evidence = answer_from_evidence(index, row['question'], corpus=corpus)
    citation = evidence.get('citations') or []
    if evidence['status'] != 'excerpt_found' or not citation:
        return None, 'question-only retrieval did not produce source evidence'
    if (citation[0]['source'], citation[0]['path']) != (row['source'], row['path']):
        return None, 'question-only retrieval selected another document'
    return evidence['answer'], None


def yes(input_fn, prompt):
    return input_fn(prompt).strip().casefold() == 'yes'


def run_review(bundle, record, *, reviews_path, reviewer, limit=15, split='all',
               corpus=CORPUS, input_fn=input, print_fn=print, index=None):
    if not reviewer or not reviewer.strip() or limit < 1:
        raise ValueError('Provide a reviewer and a positive limit')
    if split not in ('all', 'train', 'val', 'test'):
        raise ValueError('Invalid split')
    reviews_path = Path(reviews_path)
    previous = reviews_path.read_bytes()
    index = load_index() if index is None else index
    completed, skipped_source = 0, 0
    for row in bundle['candidates']:
        if split != 'all' and row['split'] != split:
            continue
        if decision_for(record['reviews'][row['id']]) != 'pending':
            continue
        if completed >= limit:
            break
        excerpt, reason = eligible_evidence(row, index, corpus=corpus)
        if reason:
            skipped_source += 1
            continue
        print_fn('\n' + '-' * 65)
        print_fn(f'Candidate: {row["id"]}  |  {row["split"]}  |  {row["kind"]}')
        print_fn('Question: ' + row['question'])
        print_fn('Proposed answer (UNVERIFIED): ' + row['answer'])
        if row['kind'] == 'grounded':
            print_fn('Pinned original document: ' + row['url'])
            # Excerpt for discovery only. Reviewer must independently verify the
            # whole claim against the original source, not just this snippet.
            print_fn('Question-only retrieved excerpt: ' + excerpt)
            load_verified_document(corpus, row['source'], row['path'])
        else:
            print_fn('Refusal candidate: independently establish why evidence does NOT support the guarantee.')
            print_fn('Live retrieval excerpt requiring review: ' + excerpt)
        print_fn('Do NOT approve without inspecting the original source and the full claim.')
        decision = input_fn('Choose [a]pprove, [r]eject, [s]kip, [q]uit: ').strip().casefold()
        if decision == 'q':
            break
        if decision == 's':
            continue
        if decision == 'a':
            verified = (yes(input_fn, 'Have you independently inspected the original source? Type yes: ')
                        and yes(input_fn, 'Is the QUESTION accurate and unambiguous? Type yes: ')
                        and yes(input_fn, 'Is the full ANSWER supported (or refusal justified)? Type yes: '))
            if not verified:
                print_fn('Not approved; candidate remains pending.')
                continue
            record_decision(record, row['id'], 'approved', reviewer, checked=True)
        elif decision == 'r':
            note = input_fn('Optional reason for rejection: ')
            record_decision(record, row['id'], 'rejected', reviewer, reason=note)
        else:
            print_fn('Unrecognized decision; candidate remains pending.')
            continue
        previous = save_reviews(reviews_path, record, previous_bytes=previous)
        completed += 1
        print_fn('Saved decision. ' + json.dumps(counts(bundle, record)['by_split']))
    print_fn(f'Finished this review session: {completed} decisions saved; '
             f'{skipped_source} candidates skipped because current retrieval could not support them.')
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', default=str(CANDIDATES))
    parser.add_argument('--reviews', default=str(REVIEWS))
    parser.add_argument('--status', action='store_true', help='Show approval progress without modifying files')
    parser.add_argument('--reviewer', help='Identity of the human checking evidence; required for reviewing')
    parser.add_argument('--split', choices=('all', 'train', 'val', 'test'), default='all')
    parser.add_argument('--limit', type=int, default=15, help='Maximum decisions to save this session')
    args = parser.parse_args()
    bundle, record = load_queue(args.candidates, args.reviews)
    if args.status:
        print(json.dumps(counts(bundle, record), indent=2))
        return
    if not args.reviewer or not args.reviewer.strip():
        parser.error('Provide --reviewer to record your personal review decisions')
    run_review(bundle, record, reviews_path=args.reviews, reviewer=args.reviewer,
               limit=args.limit, split=args.split)
    print('Updated review file:', args.reviews)
    print('Next: python -m src.skygrid_ai.language.train_qa_v16 --audit')


if __name__ == '__main__':
    main()
