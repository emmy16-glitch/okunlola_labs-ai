import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.language.large_bpe import (
    LargeByteBPE, EOS_ID, write_token_ids,
)
from src.skygrid_ai.language.train_large_transformer import (
    batch_from_starts, load_prepared_data, sha256_file, train,
)
from src.skygrid_ai.language.generate_large import generate


SMALL_CONFIG = {
    "context_size": 8, "d_model": 24,
    "n_heads": 4, "n_layers": 1, "hidden_width": 48,
}


class TestV09cTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.training = torch.tensor(([1, 2, 3, 4, EOS_ID] * 40), dtype=torch.long)
        self.validation = torch.tensor(([5, 6, 7, EOS_ID] * 20), dtype=torch.long)
        self.tokenizer = LargeByteBPE()
        self.kwargs = dict(
            vocab_size=self.tokenizer.vocab_size,
            tokenizer_fingerprint=self.tokenizer.fingerprint(),
            data_digests={"train": "x", "val": "y"},
            config=SMALL_CONFIG,
            batch_size=2, lr=0.0003, eval_every=2, save_every=2,
            val_windows=4,
        )

    def test_next_token_targets(self):
        x, y = batch_from_starts(self.training, [0, 2], 5)
        self.assertEqual(y[0].tolist(), self.training[1:6].tolist())
        self.assertEqual(y[1].tolist(), self.training[3:8].tolist())
        self.assertEqual(x.shape, (2, 5))

    def test_resume_equals_uninterrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            continuous = folder / "continuous"
            partial = folder / "partial"
            train(self.training, self.validation, output=continuous, steps=4, **self.kwargs)
            train(self.training, self.validation, output=partial, steps=2, **self.kwargs)
            train(self.training, self.validation, output=partial, steps=4, resume=True, **self.kwargs)
            a = torch.load(continuous / "last.pt", weights_only=True)
            b = torch.load(partial / "last.pt", weights_only=True)
            self.assertEqual(a["step"], b["step"])
            for name, weights in a["model_state"].items():
                self.assertTrue(torch.equal(weights, b["model_state"][name]), name)

    def test_changed_dataset_fingerprint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            train(self.training, self.validation, output=folder, steps=2, **self.kwargs)
            bad = dict(self.kwargs)
            bad["data_digests"] = {"train": "changed", "val": "y"}
            with self.assertRaisesRegex(ValueError, "data_digests"):
                train(self.training, self.validation, output=folder,
                      steps=4, resume=True, **bad)

    def test_saved_weights_are_usable_for_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            train(self.training, self.validation, output=folder, steps=2, **self.kwargs)
            best = torch.load(folder / "best.pt", weights_only=True)
            result = generate("MAVLink", tokenizer=self.tokenizer,
                              checkpoint=best, max_new_tokens=3)
            self.assertTrue(result.startswith("MAVLink"))
            wrong_tokenizer = LargeByteBPE([(65, 66)])
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                generate("MAVLink", tokenizer=wrong_tokenizer,
                         checkpoint=best, max_new_tokens=3)

    def test_binary_loader_detects_modified_report_and_data(self):
        import json
        from unittest.mock import patch
        from src.skygrid_ai.language import train_large_transformer as trainer
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            tokenizer = LargeByteBPE()
            tokenizer.save(folder / "tokenizer.json")
            for name in ("train", "val"):
                write_token_ids(folder / f"{name}.bin", [1] * 128 + [EOS_ID])
            report = {
                "format": "uint32_le",
                "vocabulary_size": tokenizer.vocab_size,
                "tokenizer_fingerprint": tokenizer.fingerprint(),
                "splits": {
                    name: {"token_count_with_eos": 129, "eos_tokens": 1}
                    for name in ("train", "val")
                },
            }
            (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
            tok, training, validation, fingerprints = load_prepared_data(folder)
            self.assertEqual(len(training), 129)
            self.assertEqual(fingerprints["train"], sha256_file(folder / "train.bin"))
            report["vocabulary_size"] += 1
            (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Vocabulary"):
                load_prepared_data(folder)


if __name__ == "__main__":
    unittest.main()
