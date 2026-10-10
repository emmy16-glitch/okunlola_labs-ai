# SkyGrid AI v1.6 — Question-only, source-checked supervised research

This release replaces the **overfitting 24-example v1.5 pilot** with a review-gated, reproducible experiment. The earlier 10,000-step from-scratch v10 model remains the starting point, and all original `checkpoints/v10/`, `checkpoints/qa_v15/`, token datasets and retrieval indexes remain untouched.

## What is fixed

1. **No answer leakage into evidence selection.** Both training and live inference call the same existing `answer_from_evidence(index, question, corpus=...)` with the **question only**. An approved grounded example is excluded if retrieval abstains or cites another document. The reference answer never influences retrieval.
2. **Identical training/inference prompt layout.** Both use `Question: ...`, `Evidence: ...`, `Answer:` and reserve 96 tokens for the response in the 256-token v10 context. Training scores **answer/EOS tokens only**, never the question or padding. Excessively long reviewed answers are rejected, not clipped.
3. **Scalable review queue.** Candidate generation can produce ~300 *UNREVIEWED* question/answer drafts with pinned URLs and source fingerprints, plus six *UNREVIEWED* refusal candidates. None are automatically approved. Before training, a human must independently inspect and correct each question and answer against its source and explicitly approve it in a separate review record. The program cannot certify someone's manual approval or guarantee truth.
4. **Document-disjoint train/validation/test.** Splits are deterministic using a hash of source and path. Duplicate approved documents/questions are rejected, and source/index/retriever/candidates/reviews are fingerprint-bound to the new model. The original documentation was also seen by the pretrained model, so these are not truly novel external knowledge tests.
5. **Safer training.** A smaller learning rate (`2e-5`), early stopping (`--patience 4`, `--min-delta 0.003`), reproducible checkpoint resume and hard stop on changed data/retrieval code. It requires at least **100 approved, retrievable examples**, with at least eight in each split. Refusal examples without usable live evidence are excluded: production already refuses before invoking the model in those cases.
6. **Actual answer-quality review.** Side-by-side source v10 and QA v16 generations, teacher-forced answer-token loss, and a **blank** human grading form for factual correctness, source support, relevance, and refusal behavior. The one-time final test is locked. A separate refusal-gate challenge set highlights cases that may require human entailment review rather than inventing an automatic factual-accuracy score.

The legacy v15 **training CLI is disabled** to prevent repeating the overfitted experiment. Its original checkpoints, evaluation reports and `--audit` output remain available. This is continued training of *SkyGrid's own randomly initialized model*, not fine-tuning another provider's pretrained weights.

## Commands to run on your Windows Git Bash

Pull the commit and run all tests:

```bash
git pull origin main
python -m unittest discover -s tests -v
```

**Generate reviewable candidates** from your existing fingerprint-verified local source documents:

```bash
python -m src.skygrid_ai.language.curate_qa_v16 --maximum 300
```

This writes **ignored local files** under `data/qa/`:

- `candidates_v16.json` — machine-authored **drafts**, not verified facts. Rewrite unclear questions or answers against each pinned source URL. For grounded examples, keep `source`, `path`, `url`, `source_sha256`, and the assigned split intact.
- `reviews_v16.json` — all entries initially `approved: false`. For each genuinely reviewed item, fill a nonempty `reviewer`, set `source_checked`, `answer_supported`, and `question_checked` true, then set `approved` true. Reject incorrect or ambiguous examples instead of approving them.

If you edit the candidate JSON **before** starting reviews, keep the old blank review file only if the candidate has not changed. Otherwise rename/remove the **unused blank template** and regenerate it (never delete a completed review file):

```bash
python -m src.skygrid_ai.language.curate_qa_v16 --reviews-only
```

The revision hash binds reviews to the exact candidate file. Modifying candidates after reviews forces re-review. The software does not approve candidates for you; this human work is essential. A batch of 300 candidates is only a **review queue**, not 300 independently verified training answers.

Inspect how many examples were actually approved and accepted by live retrieval:

```bash
python -m src.skygrid_ai.language.train_qa_v16 --audit
```

Only **after reviewing at least 100 supported examples**, including enough in train, validation, and test, and checking source use/rights:

```bash
python -m src.skygrid_ai.language.train_qa_v16 --steps 100 --confirm-rights-reviewed --confirm-qa-reviewed
```

It saves under `checkpoints/qa_v16/`. It may stop before 100 if validation stops meaningfully improving. If not stopped, resume to a **cumulative** step count:

```bash
python -m src.skygrid_ai.language.train_qa_v16 --resume --steps 200 --confirm-rights-reviewed --confirm-qa-reviewed
```

**Evaluate validation, without opening the reserved test:**

```bash
python -m src.skygrid_ai.language.evaluate_qa_v16 --split val --checkpoint best
```

This writes a new `checkpoints/qa_v16_evaluation/val_best.json`. Reports cannot be silently overwritten. Review the drafts and create a blank grading form:

```bash
python -m src.skygrid_ai.language.evaluate_qa_v16 --review-report checkpoints/qa_v16_evaluation/val_best.json --review-template qa_v16_grades.json
```

Fill every human-review grade with `pass`, `partial`, `fail`, or `not_applicable`, and set `source_checked: true` after reviewing the cited documentation. Then score **the human grades, not keyword overlap**:

```bash
python -m src.skygrid_ai.language.evaluate_qa_v16 --review-report checkpoints/qa_v16_evaluation/val_best.json --ratings qa_v16_grades.json
```

Test its refusal gate independently:

```bash
python -m src.skygrid_ai.language.evaluate_qa_v16 --refusal-audit
```

Try the v16 model alongside independently labeled retrieved evidence:

```bash
python -m src.skygrid_ai.knowledge.cli answer-neural "What does MAVLink HEARTBEAT do?" --model qa_v16 --checkpoint best --tokens 64
```

**Do not run** the reserved test for routine iteration. Only after the entire training and validation process is frozen, a final once-only command exists:

```bash
python -m src.skygrid_ai.language.evaluate_qa_v16 --split test --checkpoint best --final-test
```

## Limits

- Source-linked candidate answers still require actual human review and relevant source evidence. A model or keyword match cannot certify their correctness.
- The 951,424-parameter Transformer has limited capacity, is not a competent flight-control system or dependable technical adviser, and may still generate nonsense.
- Differences in validation answer-token loss do not prove factual generalization. Human grading of drafts and refusal scenarios remains necessary.
- Keep all local review files, train/val/test assignments and reports private under `data/qa/` and `checkpoints/`. The new candidate/review files are ignored by Git.
- No real UAV or safety-critical action should depend on these experimental neural drafts.

## v1.7: Review the existing 816 candidates without editing JSON by hand

Your output of `candidates_created: 816`, `reviewed_approved: 0`, and a training
failure requiring 100 approved examples is **expected**. Do **not** regenerate
candidate data or pass review flags as a substitute for actual review. The
original source v10 checkpoint remains unchanged. The new interactive helper
uses the same question-only retrieval gate as training to **skip** candidates
whose cited document cannot be recovered; it never certifies factual correctness.

```bash
python -m src.skygrid_ai.language.review_qa_v16 --status
python -m src.skygrid_ai.language.review_qa_v16 --reviewer "Your Name" --split train --limit 15
python -m src.skygrid_ai.language.review_qa_v16 --reviewer "Your Name" --split val --limit 10
python -m src.skygrid_ai.language.review_qa_v16 --reviewer "Your Name" --split test --limit 10
python -m src.skygrid_ai.language.train_qa_v16 --audit
```

At each candidate the helper shows the question, **unreviewed** answer, source
URL, and the current question-only retrieved excerpt. Independently open and
inspect the source document before approving; the excerpt alone is not
proof. Choose `a` to approve **only after** affirming source, question, and
answer checks, `r` to reject a wrong or unclear candidate, `s` to skip
without saving a decision, or `q` to quit. All explicit decisions save to the
existing ignored `reviews_v16.json` with an additional `.bak` backup.
Run the command again to continue from pending candidates. The original
candidate file and its review-binding SHA are unchanged.

`--status` counts human decisions; `--audit` additionally checks which
approved rows **actually** pass the live retriever and the split minimums.
After at least 100 genuinely reviewed, retrieval-supported rows, including
at least 8 in each of the `train`, `val`, and `test` splits, and source rights
review, then run the **previously documented** `train_qa_v16` command. The
helper cannot approve sources on behalf of the user and cannot guarantee
truth or license clearance. Avoid inspecting final test model outputs during
development, and never use experimental neural drafts as flight guidance.
