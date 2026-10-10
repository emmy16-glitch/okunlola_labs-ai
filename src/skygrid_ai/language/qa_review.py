"""Manual factual QA grading: NO automated keyword-based truth claims.

A human must compare each draft with the pinned documentation and grade both
source v10 and supervised QA drafts. This script only tallies supplied reviews.
"""
import argparse
import json
from pathlib import Path

GRADES = {'correct', 'partial', 'incorrect', 'abstained'}


def make_template(report):
    if report.get('split') not in ('val', 'test'):
        raise ValueError('Not a SkyGrid QA validation/test report')
    return {
        'source_report_sha256': report['qa_checkpoint_sha256'],
        'split': report['split'],
        'instructions': ('Independently compare source and QA draft to the cited original docs. '
                         'Set source_checked=true only after checking factual evidence. '
                         'Grade correct, partial, incorrect, or abstained; no automatic grading.'),
        'ratings': {row['id']: {'source_checked': False, 'baseline': None, 'qa': None,
                               'reviewer_notes': ''} for row in report['items']},
    }


def score_human_reviews(report, reviews):
    if reviews.get('source_report_sha256') != report.get('qa_checkpoint_sha256'):
        raise ValueError('Review form does not match the evaluated QA model')
    if reviews.get('split') != report.get('split'):
        raise ValueError('Review split mismatch')
    ratings = reviews.get('ratings')
    expected = {r['id'] for r in report['items']}
    if not isinstance(ratings, dict) or set(ratings) != expected:
        raise ValueError('Review must cover all report examples, not a selected subset')
    totals = {'baseline': {g: 0 for g in GRADES}, 'qa': {g: 0 for g in GRADES}}
    by_kind = {}
    for item in report['items']:
        row = ratings[item['id']]
        if row.get('source_checked') is not True:
            raise ValueError('Sources must be reviewed for every example')
        kind = item['kind']
        by_kind.setdefault(kind, {'n': 0, 'baseline_correct': 0, 'qa_correct': 0})
        by_kind[kind]['n'] += 1
        for model in ('baseline', 'qa'):
            grade = row.get(model)
            if grade not in GRADES:
                raise ValueError('Complete factual grades for every model and example')
            totals[model][grade] += 1
            if grade == 'correct':
                by_kind[kind][model + '_correct'] += 1
    n = len(expected)
    return {
        'split': report['split'], 'human_reviewed_examples': n,
        'grades': totals,
        'correct_fraction': {m: totals[m]['correct'] / n for m in ('baseline', 'qa')},
        'by_kind': by_kind,
        'note': ('Manual review of a very small pilot; cannot establish broad aviation '
                 'competence, flight safety, or independent unseen knowledge.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--template', help='Write a blank human-review form to this NEW file')
    parser.add_argument('--ratings', help='Human-completed JSON review form')
    args = parser.parse_args()
    if bool(args.template) == bool(args.ratings):
        parser.error('Choose exactly one of --template or --ratings')
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    if args.template:
        path = Path(args.template)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(make_template(report), handle, indent=2, ensure_ascii=False)
        print('Review template:', path)
    else:
        reviews = json.loads(Path(args.ratings).read_text(encoding='utf-8'))
        print(json.dumps(score_human_reviews(report, reviews), indent=2))


if __name__ == '__main__':
    main()
