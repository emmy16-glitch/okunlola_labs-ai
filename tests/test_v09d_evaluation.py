import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.language.evaluate_large import (
    separate_documents, validate_tokens, score_documents,
)
from src.skygrid_ai.language.large_bpe import LargeByteBPE, write_token_ids


class ConstantLossModel(torch.nn.Module):
    def __init__(self, loss_value):
        super().__init__()
        self.loss_value = loss_value
        self.seen = 0

    def forward(self, x, targets=None):
        self.seen += targets.numel()
        return None, torch.tensor(self.loss_value)


class TestV09dEvaluation(unittest.TestCase):
    def test_separate_documents(self):
        self.assertEqual(separate_documents([65, 257, 66, 67, 257]),
                         [[65, 257], [66, 67, 257]])

    def test_missing_eos_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without an EOS"):
            separate_documents([65, 257, 66])

    def test_padding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "padding"):
            separate_documents([65, 256, 257])

    def test_full_score_targets_not_crossing_documents(self):
        tokenizer = LargeByteBPE()
        model = ConstantLossModel(2.0)
        docs = [[65, 66, 67, 257], [68, 69, 257]]
        report = score_documents(model, docs, tokenizer, context_size=2, batch_size=2)
        self.assertEqual(report["evaluated_targets"], 5)
        self.assertEqual(report["scored_utf8_bytes_excluding_first_token_per_document"], 3)
        self.assertEqual(model.seen, 5)
        self.assertAlmostEqual(report["cross_entropy_nats_per_token"], 2.0)
        self.assertAlmostEqual(report["perplexity"], math.exp(2.0))
        self.assertAlmostEqual(report["bits_per_scored_byte_including_eos_cost"], 10 / (3 * math.log(2)))

    def test_validation_rejects_changed_data(self):
        tokenizer = LargeByteBPE()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = "AB"
            ids = tokenizer.encode(text, add_eos=True)
            write_token_ids(root / "val.bin", ids)
            report = {
                "format": "uint32_le", "vocabulary_size": tokenizer.vocab_size,
                "tokenizer_fingerprint": tokenizer.fingerprint(),
                "splits": {"val": {
                    "token_count_with_eos": len(ids), "eos_tokens": 1,
                    "documents": 1, "sha256_utf8_text": hashlib.sha256(text.encode()).hexdigest(),
                    "utf8_bytes": len(text.encode()),
                }},
            }
            (root / "report.json").write_text(json.dumps(report))
            docs, meta = validate_tokens(root, "val", tokenizer)
            self.assertEqual(docs, [ids])
            self.assertEqual(meta["documents"], 1)
            write_token_ids(root / "val.bin", tokenizer.encode("AC", add_eos=True))
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_tokens(root, "val", tokenizer)


if __name__ == "__main__":
    unittest.main()
