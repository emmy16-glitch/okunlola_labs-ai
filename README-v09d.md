# SkyGrid AI v0.9d — full-document validation and held-out test

This additive kit evaluates your trained **v0.9c** model without changing weights, old tokenizers, or checkpoints. It reads the prepared **v0.9b** binary dataset and its metadata.

## Install into your existing Git Bash repository

Extract `skygrid_ai_v09d_kit.zip` into the repo root, then:

```bash
source .venv/Scripts/activate
python -m unittest discover -s tests -v
python -m src.skygrid_ai.language.evaluate_large --split val
```

Expected test count: 60 (previous 55 + five new tests), assuming all previous tests remain passing.

After choosing the preferred checkpoint and evaluation protocol using validation only, run the **held-out final test** once:

```bash
python -m src.skygrid_ai.language.evaluate_large --split test --final-test
```

Use `--checkpoint last` to score the most recent step rather than the best sampled-validation checkpoint. Default is `best` (step 4900 based on your logs).

## What this changes relative to v0.9c

The v0.9c loss uses only 64 fixed samples of 128-token validation windows (8,192 token targets total, with possible overlap). This evaluator scores every possible next-token target within each complete document, in nonoverlapping context windows. It resets context at document boundaries rather than predicting straight across different documents. Because there is no BOS token, the first token of each document cannot be predicted and is excluded. The document EOS targets are scored.

It reports cross-entropy in nats per token, perplexity, and bits per **scored** UTF-8 byte (including cost of predicting EOS, but excluding the first UTF-8-token bytes of each document). These metrics describe token prediction, not technical factual accuracy.

It verifies token IDs, EOS counts, expected byte counts and original text SHA-256 from `report.json`, checks train/validation binary file digests against your checkpoint metadata, and refuses a mismatched tokenizer. Results are saved under `checkpoints/v09d/` (already covered by `.gitignore`). Saved result files aren't overwritten.

`--final-test` records the user's explicit decision to inspect a held-out test dataset; it cannot prevent manual peeking. Avoid repeatedly tuning against test results.

## Risks and limitations

- Earlier near-duplicate screening was heuristic; cross-document passage contamination may remain.
- Rights and source quality review are still needed, even if the training flag was used.
- A tiny 406,176-parameter model that achieves low prediction loss on documentation isn't necessarily capable of truthful aviation answers.
- Do not compare sampled-window loss and full-document loss as if they use identical protocols.

## Commit only the code

```bash
git add README-v09d.md src/skygrid_ai/language/evaluate_large.py tests/test_v09d_evaluation.py
git commit -m "Add full-document SkyGrid evaluation metrics"
git push origin main
```
