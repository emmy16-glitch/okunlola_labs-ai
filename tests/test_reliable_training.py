import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.language.bpe_tokenizer import ByteBPETokenizer
from src.skygrid_ai.language.generate_modern import decode_generated
from src.skygrid_ai.language.train_reliable import train

SMALL = {
    "context_size": 8,
    "d_model": 16,
    "n_heads": 4,
    "n_layers": 1,
    "hidden_width": 32,
}


class TestReliableTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tokenizer = ByteBPETokenizer()
        self.train_tokens = torch.tensor([65, 66, 67, 68] * 8, dtype=torch.long)
        self.val_tokens = torch.tensor([65, 66, 67, 68] * 5, dtype=torch.long)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full"
            resumed = Path(folder) / "resumed"
            common = dict(config=SMALL, batch_size=2, eval_every=2, save_every=2)
            train(self.train_tokens, self.val_tokens, self.tokenizer,
                  steps=6, output=full, **common)
            train(self.train_tokens, self.val_tokens, self.tokenizer,
                  steps=3, output=resumed, **common)
            train(self.train_tokens, self.val_tokens, self.tokenizer,
                  steps=6, output=resumed, resume=True, **common)
            first = torch.load(full / "last.pt", weights_only=True)
            second = torch.load(resumed / "last.pt", weights_only=True)
            self.assertEqual(first["step"], second["step"])
            for key in first["model_state"]:
                self.assertTrue(torch.equal(first["model_state"][key], second["model_state"][key]))

    def test_mismatched_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            train(self.train_tokens, self.val_tokens, self.tokenizer,
                  config=SMALL, steps=2, output=path, eval_every=2)
            altered = self.train_tokens.clone()
            altered[0] = 99
            with self.assertRaisesRegex(ValueError, "training_digest"):
                train(altered, self.val_tokens, self.tokenizer,
                      config=SMALL, steps=4, output=path, resume=True, eval_every=2)

    def test_partial_utf8_is_safe_for_display(self):
        self.assertEqual(decode_generated(self.tokenizer, [0xE2]), "\ufffd")


if __name__ == "__main__":
    unittest.main()
