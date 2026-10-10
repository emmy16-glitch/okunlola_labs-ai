"""Source-specific reference-coverage evaluation; NOT a model-accuracy certification."""
import json
from pathlib import Path
from .retrieval import SearchIndex, words

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / 'data' / 'benchmarks' / 'aviation_v10.json'


def load_questions(path=BENCHMARK):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Benchmark must contain nonempty list')
    ids = set()
    for row in rows:
        if (not isinstance(row, dict) or not all(row.get(k) for k in
                ('id', 'question', 'source', 'path', 'reference', 'key_terms'))):
            raise ValueError('Incomplete benchmark item')
        if row['id'] in ids:
            raise ValueError('Duplicate benchmark id')
        ids.add(row['id'])
    return rows


def evaluate_retrieval(index, questions, top_k=5):
    """Measure provenance and reference-term coverage, not semantic truth."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    rows = []
    for question in questions:
        hits = index.search(question['question'], limit=top_k)
        relevant = [p for p in hits if p['source'] == question['source'] and p['path'] == question['path']]
        evidence_text = ' '.join(p['text'] for p in relevant).casefold()
        terms = [term.casefold() for term in question['key_terms']]
        covered = [term for term in terms if term in evidence_text]
        rows.append({
            'id': question['id'], 'question': question['question'],
            'source_hit': bool(relevant), 'reference_terms_found': covered,
            'reference_terms_total': len(terms),
            'reference_coverage': len(covered) / len(terms),
            'top_sources': [p['id'] for p in hits],
        })
    n = len(rows)
    return {
        'questions': n,
        'source_recall_at_k': sum(row['source_hit'] for row in rows) / n,
        'mean_reference_term_coverage': sum(row['reference_coverage'] for row in rows) / n,
        'note': 'Retrieval quality only; does not prove answers are true or the Transformer knows these facts',
        'items': rows,
    }


def evaluate_candidate_answers(questions, answers):
    """Conservative keyword-based candidate screening. Flags unverified answers for review."""
    results = []
    for row in questions:
        candidate = str(answers.get(row['id'], '')).strip()
        abstained = not candidate or candidate.casefold() in ('i do not know', 'unknown', 'abstain')
        expected = [x.casefold() for x in row['key_terms']]
        matched = [x for x in expected if x in candidate.casefold()] if not abstained else []
        results.append({'id': row['id'], 'abstained': abstained,
                        'reference_terms_present': matched,
                        'coverage': len(matched) / len(expected),
                        'needs_human_fact_check': not abstained})
    return {'questions': len(results),
            'abstention_rate': sum(x['abstained'] for x in results) / len(results),
            'mean_reference_term_coverage': sum(x['coverage'] for x in results) / len(results),
            'warning': 'Keyword coverage is not factual correctness; all non-abstaining answers require human verification',
            'items': results}
