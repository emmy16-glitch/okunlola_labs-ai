# SkyGrid AI v0.9b — Near-duplicate audit and larger BPE vocabulary

This kit adds four files. It does **not** modify existing model code, old tokenizers, or checkpoints.

## Setup (Git Bash, from the repository root)

```bash
source .venv/Scripts/activate
printf 'data/aviation/processed_v09b/\ndata/aviation/tokenized_v09b/\n' >> .gitignore
python -m unittest discover -s tests -v
```

Expected test count: 50 if the previous 43 tests still pass.

## Run source audit

```bash
python -m src.skygrid_ai.language.audit_large_corpus
```

The audit verifies the revision and digest of each existing source document. It flags likely duplicate document families using hashed five-word shingles, groups each detected family into the same new train/val/test split, and writes cleaned per-document files and a manifest to `data/aviation/processed_v09b/`.

**Limitations:** bottom-k shingle candidate selection can miss similar documents, and cross-document repeated passages and semantic duplicates are not fully checked. This is a reproducible **screening**, not definitive proof of leakage-free evaluation. Review `audit.json`, source quality, and rights before using the corpus as approved training data. Document-level grouping also doesn't guarantee that similar snippets won't leak across splits.

## Train a new tokenizer (from scratch)

```bash
python -m src.skygrid_ai.language.prepare_large_tokens --vocab-size 768
```

Unlike v0.8, this tokenizer uses whitespace-aware chunks for scaling. It learns merge rules **only from the new training split**, using up to 12,000 most frequent distinct chunks, and encodes all documents without changing the learned rules. Every document gets a single EOS token. All Unicode text is reversible.

Output:

- `data/aviation/tokenized_v09b/tokenizer.json` — learned merge rules and fingerprint
- `train.bin`, `val.bin`, `test.bin` — uint32 little-endian token IDs, one EOS token per document
- `report.json` — token counts, byte counts, fingerprints and rights-review flag

**IMPORTANT:** `.bin` files are not compatible with the old v0.8 JSON-token training entrypoint, and the v0.9b tokenizer has a different merge/segmentation scheme than the v0.8 tokenizer. Never resume a v0.8 checkpoint against v0.9b tokens. We will wire a new v0.9c training entrypoint after checking these statistics.

## Keep previous experiments safe

Both new scripts refuse to overwrite nonempty existing output folders. To rebuild with different settings, first manually archive the existing output folder, then rerun. Do not delete `checkpoints/v08c/` or `data/aviation/processed_v09_large/`.

## Push source code only

```bash
git add .gitignore README-v09b.md src/skygrid_ai/language/audit_large_corpus.py src/skygrid_ai/language/large_bpe.py src/skygrid_ai/language/prepare_large_tokens.py tests/test_v09b_pipeline.py
git commit -m 'Add SkyGrid large-corpus audit and scalable BPE preparation'
git push origin main
```

Only push after testing on your computer. Source documentation clones, generated text and `.bin` files are ignored by Git.
