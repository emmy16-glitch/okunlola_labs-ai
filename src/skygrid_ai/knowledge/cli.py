"""SkyGrid v1.0 offline evidence tool: build-index, ask, benchmark, score-answers."""
import argparse
import json
from pathlib import Path

from .benchmark import evaluate_candidate_answers, evaluate_retrieval, load_questions
from .retrieval import INDEX, build_passages, load_index, save_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build-index')
    ask = sub.add_parser('ask')
    ask.add_argument('question')
    ask.add_argument('--top-k', type=int, default=3)
    benchmark = sub.add_parser('benchmark')
    benchmark.add_argument('--top-k', type=int, default=5)
    score = sub.add_parser('score-answers')
    score.add_argument('file', help='JSON map from benchmark IDs to generated answers')
    args = parser.parse_args()
    if args.command == 'build-index':
        passages, fingerprint = build_passages()
        destination = save_index(passages, fingerprint)
        print(f'Indexed {len(passages):,} source-linked passages to {destination}')
        return
    if args.command == 'score-answers':
        candidates = json.loads(Path(args.file).read_text(encoding='utf-8'))
        if not isinstance(candidates, dict):
            raise ValueError('Answers must be an id-to-text JSON object')
        result = evaluate_candidate_answers(load_questions(), candidates)
    else:
        index = load_index()
        if args.command == 'ask':
            result = index.evidence(args.question, limit=args.top_k)
        else:
            result = evaluate_retrieval(index, load_questions(), top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
