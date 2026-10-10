# SkyGrid AI v1.8 — Zero-review weak-supervision experiment

You do **not** need to manually answer or approve 816 questions to try this optional route. The old reviewed v1.6 workflow is still intact. The new command *does not mark machine-generated examples as human-reviewed*.

This approach takes a sentence from a locally verified MAVLink/PX4 document, constructs a **rule-based** subject-specific question, retrieves a relevant sentence using the **same question-only source gate** used in SkyGrid's ordinary answer path, and uses that retrieved sentence as a **weak** training label. It rejects pairs when retrieval points to another document. It gives **no assurance of factually correct or fluent neural answers**. Pretraining has already seen the source corpus; validation is document-disjoint at the new Q&A stage, not truly unseen source material.

## Commands (Windows Git Bash; Python venv active)

Pull the commit and run all tests:

```bash
git pull origin main
python -m unittest discover -s tests -v
python -m src.skygrid_ai.language.auto_qa_v18 build --maximum 300
```

The build makes `data/qa/auto_weak_v18.json` **once** (will not overwrite). Check the printed counts: you need at least **100 automatic source-matched examples**, at least **8 in validation and test**, and 20 in training. If counts are too low, do not train; the tool refuses rather than inventing labels. There is no review UI, review file, or per-question input.

If you have reviewed the licensing/rights of the local documents and accept that these labels are **unreviewed machine-generated weak supervision**:

```bash
python -m src.skygrid_ai.language.auto_qa_v18 train --steps 100 --confirm-rights-reviewed --accept-unreviewed-weak-labels
python -m src.skygrid_ai.language.auto_qa_v18 evaluate --checkpoint best
```

Checkpoints are saved **only** in `checkpoints/qa_auto_v18/`; your pretrained `checkpoints/v10/`, human-review `checkpoints/qa_v16/`, datasets, and indexes are not modified. `--steps` is cumulative; if the run has not early-stopped, continue with `--resume --steps 200` and the same two explicit acknowledgements.

The evaluation reads **validation only**, prints baseline vs new answer-token loss and six *unverified* generated drafts with citations. It intentionally does **not** report factual accuracy. Do not use neural drafts for flight control or safety decisions. For reliable offline use, prefer `python -m src.skygrid_ai.knowledge.cli answer "What does MAVLink HEARTBEAT do?"`, which shows an original retrieved source excerpt rather than synthesized claims.
