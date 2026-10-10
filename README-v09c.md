# SkyGrid AI v0.9c — Train our model on 2.2 million aviation token IDs

This kit adds new Python scripts and tests only; it doesn't modify existing language models, tokenizers, or previous checkpoints.

## Before training: approve the material

`processed_v09b/audit.json` and `manifest.json` currently say `rights_review_required`. Review the source paths, document rights, embedded third-party content, license notices, required CC BY attribution, and data quality before declaring the dataset approved for model training. `--confirm-rights-reviewed` is an acknowledgement of YOUR review, not an automatic license-clearing feature. Keep attribution details for relevant sources. This is research code, not aviation safety guidance.

## Run in Git Bash from the repository root

```bash
source .venv/Scripts/activate
python -m unittest discover -s tests -v
python -m src.skygrid_ai.language.train_large_transformer --steps 100 --confirm-rights-reviewed
python -m src.skygrid_ai.language.train_large_transformer --resume --steps 400 --confirm-rights-reviewed
python -m src.skygrid_ai.language.generate_large --prompt "Aircraft telemetry" --tokens 100
```

Only run the training commands after reviewing rights and source quality.

## What these files do

- `train_large_transformer.py` reads the uint32 little-endian `.bin` training and validation token files, validates their metadata, initializes the SkyGrid Transformer randomly, trains on CPU, evaluates **64 fixed sampled validation windows** (NOT a full validation epoch), and keeps `last.pt` and `best.pt` inside `checkpoints/v09c/`.
- `generate_large.py` loads `best.pt`, verifies the new tokenizer's fingerprint and samples continuation tokens. Text is still experimental and not validated aviation guidance.
- `test_v09c_training.py` checks next-token shifting, resume equivalence, dataset mismatches, checkpoint-based generation, and binary loader integrity.

Training samples 8 randomly selected 128-token windows per optimizer step by default. A checkpoint holds weights, optimizer state, PyTorch CPU RNG state, step, architecture, tokenizer ID, and binary-dataset fingerprints. Resuming uses `--steps` for the FINAL cumulative step count, not additional steps.

The new model has **406,176 trainable parameters** at vocabulary size 768; older v0.8 model has 363,168 at vocabulary 320. No pretrained weights are used.

The checkpoint filenames intentionally differ from `v08c` so older experiments remain available. Do not attempt to resume v0.8 training with the new tokens.

## Commit code (only after checking tests)

```bash
git add README-v09c.md src/skygrid_ai/language/train_large_transformer.py src/skygrid_ai/language/generate_large.py tests/test_v09c_training.py
git commit -m "Add large corpus Transformer training and generation"
git push origin main
```

These local models are very small compared to modern LLMs. Aviation expertise, generation quality, and real-world trustworthiness must be evaluated empirically. Source rights and cross-split leakage require more inspection than the current audit provides.
