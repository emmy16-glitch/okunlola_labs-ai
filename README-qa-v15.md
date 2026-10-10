# SkyGrid AI — from-scratch model, supervised aviation Q&A pilot

This pilot continues training **your own 951,424-parameter v1.0 model** from the 10,000-step `checkpoints/v10/last.pt` checkpoint. It does NOT import a pretrained model, touch the v10/ or v09c/ checkpoints, use online AI services, or retrain tokenization. Q&A checkpoints are written only to `checkpoints/qa_v15/`.

## Important experimental limitations

The pilot reuses 20 reference questions from `aviation_v10.json` and `aviation_v11_independent.json`, with four extra *synthetic refusal* examples. Those references are **candidate summaries requiring actual human review**, not independently verified truth labels. The QA supervision has 16 training, 4 validation, and 4 final-test examples. All 20 grounded items refer to distinct documents, and splits are **document-disjoint for QA supervision**. However, the original pretraining corpus may already contain all these documents; the test set is **not an unseen-document or independent aviation-expertise test**. There are far too few examples to expect robust instruction following, reasoning, or broadly correct UAV answers.

A GitHub checkout **does not contain your local training dataset or checkpoints**. The code needs the original processed v0.9b source files, tokenized binaries and original v10 last/best checkpoint on your computer.

Do not run this pilot or deploy it for controlling an aircraft, real-time flight operations or safety-critical advice.

## Human audit is required before starting

```bash
python -m src.skygrid_ai.language.train_qa --audit
```

This prints the **training and validation** source-linked questions, candidate answers, fingerprint-verified excerpts, URLs, and splits. It intentionally withholds reserved **test references** until the final evaluation. Review question/answer factual agreement against the linked, version-pinned documentation and check licenses/attribution/embedded third-party material. Overlap-based evidence extraction is not a factual verification engine. Fix reference answers or the training protocol if needed; the fingerprint includes all QA JSON inputs and prevents incompatible training resume.

Only **after actually reviewing both the QA answers and source rights**, run:

```bash
python -m src.skygrid_ai.language.train_qa --steps 20 --confirm-rights-reviewed --confirm-qa-reviewed
```

The training loss is masked: only reference **answer tokens plus EOS** contribute. The question, cited evidence and padding are excluded. The separate initial v10 checkpoint is never overwritten.

Next train to 100 **cumulative QA steps**, preserving the same settings:

```bash
python -m src.skygrid_ai.language.train_qa --resume --steps 100 --confirm-rights-reviewed --confirm-qa-reviewed
```

This is a **tiny overfitting-prone experiment**. Watch validation answer loss: if it stops decreasing or rises steadily, stop rather than forcing more steps.

## Compare on development/validation answers

```bash
python -m src.skygrid_ai.language.evaluate_qa --split val --checkpoint best
```

The evaluation loads the original v10 source checkpoint and the trained QA model; it computes answer-only teacher-forced loss on the **same** four validation examples. It also generates deterministic experimental drafts beside the reference answers and source IDs for **human review**. Lower loss and exact text matches do **not** certify aviation factual correctness.

To see the trained model in the existing citation-first workflow:

```bash
python -m src.skygrid_ai.knowledge.cli answer-neural "What does MAVLink HEARTBEAT do?" --model qa_v15 --checkpoint best --tokens 64
```

The model's draft remains explicitly marked *unverified* and is shown separately from the source-backed excerpt. No citation is asserted for the model-generated text. Existing mismatched-source and no-evidence refusals happen **before** model inference.

### Freeze the test set

Do not evaluate the final-test set during development. When you have made all training decisions, and after manually verifying the four reserved references, run this **once**:

```bash
python -m src.skygrid_ai.language.evaluate_qa --split test --checkpoint best --final-test
```

The first run creates `checkpoints/qa_v15_evaluation/final_test_locked.json`; subsequent test evaluations are refused, to avoid repeated tuning against a supposedly held-out benchmark. Even this final-test result is a small supervised-QA assessment on pretraining-exposed documentation, **not** an independent aviation knowledge certification.

## Data integrity and recoverability

- Each Q&A source text is loaded from the original local processed source corpus and SHA-verified against the pinned manifest.
- The train/val/test protocol prevents a document appearing in more than one QA-supervision split.
- The original v10 pretraining checkpoint SHA, tokenizer fingerprint, train/validation binary hashes, QA-protocol digest, and model architecture are checked on training and resume.
- The supervised `best.pt` and `last.pt` are kept under `checkpoints/qa_v15/`; an unrequested overwrite is refused.
- The original pretraining full-document validation result (2.093478 nats/token at v10 step 10,000) is **not directly comparable** to the new answer-only QA loss. Compare QA and v10 using the new `evaluate_qa` command on identical QA examples instead.

Tests:

```bash
python -m unittest discover -s tests -v
```

## Measure actual factual correctness by a human reviewer

The `evaluate_qa` report contains both **v10 baseline** and **QA-trained** drafts per question. For factual accuracy, use a manual source-based grading form. Example after a validation evaluation at QA step 20:

```bash
python -m src.skygrid_ai.language.qa_review --report checkpoints/qa_v15_evaluation/val_best_step20.json --template checkpoints/qa_v15_evaluation/review_blank.json
```

Open `review_blank.json`, verify **each** baseline and QA answer against the official pinned documentation link, set `source_checked` to `true`, and grade each output `correct`, `partial`, `incorrect` or `abstained`. Save it as a separate `review_completed.json`, then run:

```bash
python -m src.skygrid_ai.language.qa_review --report checkpoints/qa_v15_evaluation/val_best_step20.json --ratings checkpoints/qa_v15_evaluation/review_completed.json
```

Change the step number in the report filename to the actual best-step printed after training. **No keyword or loss metric automatically becomes a factual accuracy score.**

To expand the pilot after human review, add entries to `additional_grounded` in `data/qa/aviation_v15_protocol.json` with `id`, `split`, `question`, `answer`, `source` and `path`. The referenced document must exist in the locally pinned manifest. Multiple examples may share a source document **within** one split, but document-sharing across train/validation/test is rejected. Any edits change the QA dataset fingerprint and block incompatible resume. The bundled references remain candidate answers, not automatically approved labels.
