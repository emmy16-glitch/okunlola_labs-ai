"""Offline SkyGrid documentation search, cautious excerpts, and benchmarks."""
import argparse
import json
from pathlib import Path

from .benchmark import evaluate_candidate_answers, evaluate_retrieval, load_questions
from .grounded import answer_from_evidence
from .neural_bridge import answer_with_model
from .retrieval import CORPUS, INDEX, build_passages, load_index, save_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build-index')
    ask = sub.add_parser('ask')
    ask.add_argument('question')
    ask.add_argument('--top-k', type=int, default=3)
    answer = sub.add_parser('answer', help='Extract a source-backed sentence; abstain when evidence is insufficient')
    answer.add_argument('question')
    answer.add_argument('--top-k', type=int, default=8)
    neural = sub.add_parser('answer-neural', help='Show cited source excerpt AND an explicitly unverified local Transformer continuation')
    neural.add_argument('question')
    neural.add_argument('--model', choices=('v09c', 'v10'), default='v09c')
    neural.add_argument('--checkpoint', choices=('last', 'best'), default='last')
    neural.add_argument('--tokens', type=int, default=64)
    neural.add_argument('--temperature', type=float, default=0.7)
    neural.add_argument('--top-k-tokens', type=int, default=10)
    neural.add_argument('--seed', type=int, default=7)
    check = sub.add_parser('benchmark-independent', help='Evaluate new, untuned source-retrieval questions')
    check.add_argument('--top-k', type=int, default=5)
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
        elif args.command == 'answer':
            result = answer_from_evidence(index, args.question, top_k=args.top_k, corpus=CORPUS)
        elif args.command == 'answer-neural':
            result = answer_with_model(
                index, args.question, corpus=CORPUS, model_version=args.model,
                checkpoint_name=args.checkpoint, max_new_tokens=args.tokens,
                temperature=args.temperature, top_k_tokens=args.top_k_tokens,
                seed=args.seed)
        elif args.command == 'benchmark-independent':
            benchmark_path = Path(__file__).resolve().parents[3] / 'data' / 'benchmarks' / 'aviation_v11_independent.json'
            result = evaluate_retrieval(index, load_questions(benchmark_path), top_k=args.top_k)
            result['benchmark'] = 'aviation_v11_independent'
            result['note'] += '; this is a new development evaluation, not proof of factual QA capability'
        else:
            result = evaluate_retrieval(index, load_questions(), top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
