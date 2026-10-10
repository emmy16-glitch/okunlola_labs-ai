# SkyGrid AI v0.8c — Reproducible training and generation

Additive files: existing v0.6/v0.8b source files are not modified.

## Requirements
- Existing `src/skygrid_ai/language/modern_transformer.py`
- Existing `src/skygrid_ai/language/bpe_tokenizer.py`
- Generated local `data/aviation/processed_v08/bpe_tokenizer.json`, `train_tokens.json`, `val_tokens.json`
- Python and PyTorch in your active `.venv`

## Run (Git Bash from repository root)

```bash
python -m unittest discover -s tests -v
python -m src.skygrid_ai.language.train_reliable --steps 40
python -m src.skygrid_ai.language.train_reliable --resume --steps 80
python -m src.skygrid_ai.language.generate_modern --prompt "aircraft" --tokens 80
```

`--steps` is the total target step, not additional steps. For example, 40 then `--resume --steps 80` trains 40 more steps. Last/best checkpoints are written to `checkpoints/v08c/` and should not be committed. A run can resume from the last completed save, not necessarily the precise interruption step.

## What is saved?

- `last.pt`: current model state, optimizer state, PyTorch CPU RNG state, step number, metadata and best-validation metrics.
- `best.pt`: validation-selected model weights plus metadata, used for inference.

Resume rejects model config, batch size, tokenizer, and training/validation token-data mismatches. This exercise targets reproducibility on CPU with the same software and environment; exact numeric reproducibility isn't guaranteed across different PyTorch/OS/hardware combinations.

## Interpretation

The corpus contains only a few synthetic demonstration documents. Text generation will be mostly incoherent. It is not aviation guidance or evidence of domain expertise.
