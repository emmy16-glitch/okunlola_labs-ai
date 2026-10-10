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

## Retrieval ranking refinement

Search ranking now combines BM25 body matches with an editorial-filename topic
boost, and returns at most two passages per document in each result set. This
works with existing `passages.json.gz` indexes; **do not rebuild your index**.
Run `python -m src.skygrid_ai.knowledge.cli benchmark` again after pulling.
The initial 9-question benchmark scored 7/9 source recall; because those
misses motivated this change, improvement on these same questions would be
**development-set progress, not proof of generalization**. Evaluate new,
independently written questions as well. No generated claims are fact-checked
by this retrieval score.

## v1.1: Extractive answer preview and an untuned retrieval question set

After pulling the new code, **reuse your existing 10,034-passage index**:

```bash
python -m unittest discover -s tests -v
python -m src.skygrid_ai.knowledge.cli answer "What does MAVLink HEARTBEAT do?"
python -m src.skygrid_ai.knowledge.cli answer "How does PX4 handle control allocation?"
python -m src.skygrid_ai.knowledge.cli answer "How does MAVLink handle control allocation?"
python -m src.skygrid_ai.knowledge.cli benchmark-independent
```

The `answer` command **extracts a sentence verbatim** from an indexed source
and includes its pinned GitHub citation. It **does not** synthesize a
Transformer answer, imply source facts are correct, or offer flight instructions.
If a question names one software system but the strongest matching evidence
comes from another system, it refuses to silently conflate their roles.
Extractive selection is based on word overlap, so outputs may be awkward,
incomplete, or off-topic; compare the excerpt with the linked documentation.

`benchmark-independent` tests 11 additional source-specific questions that
were not used in the earlier nine-question ranking adjustment. It only
measures expected-source recall@5 and literal reference-term presence, **not**
correct answer quality. These new questions were authored after the ranking
change; avoid repeatedly tuning the retrieval ranker against them.

No model weights, tokenizers, local datasets, or indexes are overwritten.

## v1.0.2 — source-complete extractive answer repair

The `answer` command now opens the ORIGINAL audited document in
`data/aviation/processed_v09b/docs/` after document retrieval. This prevents
the old behavior where a 125-word flattened passage ended midway through a
sentence, or a heading was incorrectly presented as the answer. The command
requires the cleaned source fingerprint to match the pinned manifest before
emitting a citation-backed excerpt. Headings, tables, incomplete sentences,
and irrelevant passages are filtered; if evidence is insufficient it abstains.

The command chooses one explanatory sentence from the source; it is NOT neural
text generation or independently verified factual accuracy. The sentence may
have inline Markdown markers removed for readability. Provenance is the
document URL, not a claimed passage offset.

No index rebuild, retokenization, or training is necessary:

```bash
git pull origin main
python -m unittest discover -s tests -v
python -m src.skygrid_ai.knowledge.cli answer "What does MAVLink HEARTBEAT do?"
python -m src.skygrid_ai.knowledge.cli answer "How does PX4 handle control allocation?"
```

The original nine-question retrieval development benchmark remains 9/9.
The later eleven-question suite was 10/11 (90.9% recall@5); the missing
MAVLink PING source was noted, but this change intentionally does not tune
search ranking to that additional evaluation set. Neither score measures
answer correctness. Do not repeatedly tune evaluation questions into an
illusory held-out benchmark. Existing source-rights review requirements apply.

## v1.0.3 — optional neural continuation alongside source-backed evidence

The new `answer-neural` command **connects the original locally trained SkyGrid
Transformer to document retrieval** without claiming its output is trustworthy
aviation advice. It first uses the existing `answer` source-verification path,
then prepares a context-bounded prompt containing the question and the best
citation-backed excerpt. The model generates a **separate, unverified draft**;
the citation applies only to the original excerpt, **not** to neural words.

After a single pull, using the step-5,000 **v0.9c** checkpoint already on your PC:

```bash
git pull origin main
python -m unittest discover -s tests -v
python -m src.skygrid_ai.knowledge.cli answer-neural "What does MAVLink HEARTBEAT do?" --model v09c --checkpoint last --tokens 64
python -m src.skygrid_ai.knowledge.cli answer-neural "How does PX4 handle control allocation?" --model v09c --checkpoint last --tokens 64
python -m src.skygrid_ai.knowledge.cli answer-neural "How does MAVLink perform control allocation?"
```

The last query should return `source_mismatch` **without generating a neural
draft**. The default is v09c/last because step 5,000 performed marginally
better than step 4,900 on the complete validation set; the original test
result remains frozen. You may select `--checkpoint best` to compare locally.

To experiment with the optional larger **v10** architecture *after actually
training it*, use `--model v10`. If `checkpoints/v10/last.pt` has not been
created on your PC, the command raises a clear missing-checkpoint error. The
existing checkpoint formats, binary-dataset hashes and custom tokenizer
fingerprint are validated before generating; PyTorch loads weights-only.

This is **not yet a true instruction-following or evidence-entailing aviation
assistant**. It is a controlled integration experiment: a 406k-parameter
next-token predictor trained on documentation generally cannot understand
questions or reliably synthesize source-grounded answers. The source excerpt
remains the only reference-backed material and also requires human judgment
for safety-critical applications. The neural preview must not control UAVs or
be used for real-world flight decisions. No model files, source data, token
files, or retrieval indexes are modified by this command.

## v1.0.4 — reproducible CPU training and frozen-baseline comparison

Your earlier `answer-neural` experiment succeeded technically, but the
v0.9c model's neural continuation was incoherent. That is expected from a
406,176-parameter documentation next-token predictor, **not** evidence of
aviation question-answering ability. The cited **extractive** response is
still the useful output. The larger v10 model must be **trained locally from
random weights**; merely downloading code cannot train it.

The new `compare_v10` tool evaluates the entire existing **validation** split
using the same protocol as the v0.9d report, checks the actual checkpoint
hashes, tokenizer fingerprints and training/validation token-file hashes, and
compares results against the frozen step-5,000 v0.9c baseline. It never reads
the held-out test set or modifies any `.pt` file. A completed result is reused
only if its checkpoint digest and evaluation metadata still match. Results live
under ignored `checkpoints/v10_evaluation/`.

Run in Git Bash from the repository root:

```bash
 git pull origin main
 source .venv/Scripts/activate
 python -m unittest discover -s tests -v
```

After completing the **actual source-rights, embedded-content and attribution
review**, start a conservative local smoke test of the new model:

```bash
 python -m src.skygrid_ai.language.train_v10 --steps 20 --confirm-rights-reviewed
 python -m src.skygrid_ai.language.compare_v10 --checkpoint last
```

If it succeeds and CPU/memory behavior is acceptable, continue **cumulatively**
to step 1,000 and re-evaluate:

```bash
 python -m src.skygrid_ai.language.train_v10 --resume --steps 1000 --confirm-rights-reviewed
 python -m src.skygrid_ai.language.compare_v10 --checkpoint last
 python -m src.skygrid_ai.language.generate_v10 --prompt "Aircraft telemetry" --checkpoint last --tokens 80
```

You can also try `answer-neural ... --model v10 --checkpoint last` after the
model is trained; all neural words remain experimental and unverified. If
step 1,000 performs worse than v0.9c step 5,000, that does not prove the
larger architecture is inferior: it has had fewer optimization steps. Report
full validation numbers, training steps, and a few representative outputs
before deciding whether to train longer. The earlier **124-document test**
result must stay frozen; use an entirely new independently held-out dataset
for any eventual final generalization test.

Training requires your ignored `data/aviation/tokenized_v09b/` files and
existing `checkpoints/v09c/last.pt` and v0.9d validation report, all local to
your PC. GitHub does not have those files; no hosted training is implied. CPU
training on a 16-GB machine may be slow; checkpoints are saved regularly, and
the resume command continues rather than restarting.
