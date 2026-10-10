# SkyGrid AI v1.0 — Offline aviation knowledge and bigger training experiments

**Status:** Code and benchmark scaffolding. v1.0 does **not** turn the 406k-parameter v0.9 model into a reliable aviation assistant. No pretrained weights, external embedding APIs, or internet access are required.

## New functionality

- **Citation-first offline retrieval** from your existing `data/aviation/processed_v09b/` corpus using a home-built BM25 inverted index (standard Python only). Outputs *source excerpts with pinned URLs*, not synthesized claims. Abstains when lexical evidence is absent.
- **Nine source-specific aviation reference checks** across MAVLink and PX4. Compares retrieval with a cited reference document; reports source recall@5 and key-term coverage, **not factual correctness**. Benchmarks are a reproducible regression suite, not independently unseen subject matter: source documentation overlaps the corpus.
- **Candidate-answer screening** for human-authored/model-authored `id -> text` JSON responses, measuring key-term coverage and abstention, never certifying correctness without human review.
- **Optional larger Transformer from random initialization:** context 256, width 128, 4 layers, 4 attention heads, FF 384, batch 4, 32 sampled validation windows; isolated `checkpoints/v10/`. Existing v0.9c weights and v0.9d held-out test result remain untouched.

## Run on Windows Git Bash after `git pull origin main`

```bash
source .venv/Scripts/activate
python -m unittest discover -s tests -v
python -m src.skygrid_ai.knowledge.cli build-index
python -m src.skygrid_ai.knowledge.cli ask "What does MAVLink HEARTBEAT do?"
python -m src.skygrid_ai.knowledge.cli benchmark
```

The first command builds a local index at `data/aviation/index_v10/passages.json.gz`. It refuses to overwrite an existing index, and all indexed excerpts are tied to a source URL and checked against document content fingerprints. If the source corpus changes, archive the existing index and rebuild. The build works only on the original ignored source documents already on your laptop.

Candidate-answer screening (optional):

```bash
python -m src.skygrid_ai.knowledge.cli score-answers answers.json
```

`answers.json` must be a JSON object with keys matching the benchmark IDs in `data/benchmarks/aviation_v10.json` and free-text answers as values. Keyword matches are *not* proof of accurate claims, and the answers should be manually checked against the cited reference source.

## Optional architecture experiment (CPU)

Only after **real** source-rights and attribution review:

```bash
python -m src.skygrid_ai.language.train_v10 --steps 20 --confirm-rights-reviewed
python -m src.skygrid_ai.language.train_v10 --resume --steps 1000 --confirm-rights-reviewed
python -m src.skygrid_ai.language.generate_v10 --prompt "Aircraft telemetry" --tokens 80
```

The new model has more capacity but is randomly initialized and will initially generate poor text. Training **cannot** run in GitHub simply by pulling: its ignored 2.2-million-token dataset and checkpoints live on your PC. The new trainer intentionally reuses the v0.9c experiment's deterministic checkpointing and fingerprint validations; it writes only `checkpoints/v10/`. The resume step count is **cumulative**. Keep model hyperparameters identical when resuming.

## Comparing results without contaminating held-out testing

Use your `checkpoints/v09d/val_last_step5000.json` and `test_last_step5000.json` as the frozen reference metrics (the original test was already evaluated). For the new model, do **validation only** during experimentation. A full-document validation run can be invoked programmatically using `evaluate_large.evaluate('val', 'last', training_output=Path('checkpoints/v10'), output=Path('checkpoints/v10_evaluation'))` with the repository root as working directory. Do not repeatedly tune on the original 124-document test set; reserve a fresh external test corpus for a final comparison.

## Limits

- Local BM25 is lexical; it can miss paraphrases, false positives and sophisticated queries. A match is not a medical/flight/safety certification.
- Every retrieved snippet is only evidence; a system that synthesizes an answer must prove each claim against sources and have appropriate abstention behavior.
- The 9 regression questions are not a large independent knowledge exam, and source-term matching is **not answer accuracy**.
- Existing dataset license metadata still flags `rights_review_required`: a flag acknowledgement cannot substitute for license and embedded-content review.
- All artifacts created on your machine under `data/aviation/index_v10/` and `checkpoints/` should remain ignored by Git.
