"""Create candidate aviation question/answer pairs and a SEPARATE blank human-review form.

Example: python -m src.skygrid_ai.language.curate_qa_v16 --maximum 300
Output files are ignored local data, never approved automatically.
"""
import argparse
import json
from .qa_v16 import CANDIDATES, REVIEWS, generate_candidates, review_template, save_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maximum', type=int, default=300)
    parser.add_argument('--candidates', default=str(CANDIDATES))
    parser.add_argument('--reviews', default=str(REVIEWS))
    parser.add_argument('--reviews-only', action='store_true', help='Regenerate a fresh review template after editing candidates')
    args = parser.parse_args()
    # Both outputs must be NEW to prevent destroying manual review work.
    from pathlib import Path
    if Path(args.reviews).exists() or (not args.reviews_only and Path(args.candidates).exists()):
        parser.error('Candidate/review output already exists. Refusing to overwrite it.')
    if args.reviews_only:
        bundle = json.loads(Path(args.candidates).read_text(encoding='utf-8'))
    else:
        bundle = generate_candidates(maximum=args.maximum)
        save_new(args.candidates, bundle)
    save_new(args.reviews, review_template(bundle))
    print(json.dumps({'candidates_created': len(bundle['candidates']),
                      'reviewed_approved': 0, 'candidates': args.candidates,
                      'reviews': args.reviews,
                      'note': 'Edit candidate wording before reviewing; after editing regenerate a blank review template with the new candidate SHA. Nothing is approved automatically.'}, indent=2))


if __name__ == '__main__':
    main()
