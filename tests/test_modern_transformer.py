
import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.language.modern_transformer import (
    SkyGridTransformerV08,
    apply_rope,
)


class TestModernTransformer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.model = SkyGridTransformerV08(
            vocab_size=320
        )

        self.model.eval()

    def test_output_and_loss(self):
        tokens = torch.randint(
            2, 320, (2, 16)
        )

        logits, loss = self.model(tokens, tokens)

        self.assertEqual(
            tuple(logits.shape),
            (2, 16, 320)
        )

        self.assertTrue(
            torch.isfinite(loss).item()
        )

    def test_future_tokens_cannot_leak(self):
        tokens = torch.randint(
            2, 320, (1, 12)
        )

        changed = tokens.clone()

        changed[0, -1] = (
            changed[0, -1] + 1
        ) % 320

        with torch.inference_mode():
            original, _ = self.model(tokens)
            altered, _ = self.model(changed)

        self.assertTrue(
            torch.allclose(
                original[:, :-1],
                altered[:, :-1],
                atol=1e-5
            )
        )

    def test_rope_keeps_first_position_unchanged(self):
        q = torch.randn(1, 4, 8, 24)
        k = torch.randn(1, 4, 8, 24)

        q_rot, k_rot = apply_rope(q, k)

        self.assertTrue(
            torch.allclose(
                q[:, :, 0],
                q_rot[:, :, 0],
                atol=1e-6
            )
        )

        self.assertTrue(
            torch.allclose(
                k[:, :, 0],
                k_rot[:, :, 0],
                atol=1e-6
            )
        )

    def test_tied_embeddings(self):
        self.assertIs(
            self.model.token_embedding.weight,
            self.model.output_head.weight
        )

    def test_gradients(self):
        self.model.train()

        tokens = torch.randint(
            2, 320, (2, 8)
        )

        _, loss = self.model(tokens, tokens)
        loss.backward()

        gradient = (
            self.model.blocks[0]
            .attention.qkv.weight.grad
        )

        self.assertIsNotNone(gradient)

        self.assertTrue(
            torch.isfinite(gradient).all().item()
        )

    def test_parameter_count(self):
        count = sum(
            p.numel()
            for p in self.model.parameters()
        )

        self.assertEqual(count, 363168)

    def test_checkpoint_round_trip(self):
        tokens = torch.randint(
            2, 320, (1, 8)
        )

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"

            torch.save(
                self.model.state_dict(),
                path
            )

            restored = SkyGridTransformerV08(
                vocab_size=320
            )

            restored.load_state_dict(
                torch.load(
                    path,
                    weights_only=True
                )
            )

            restored.eval()

            with torch.inference_mode():
                a, _ = self.model(tokens)
                b, _ = restored(tokens)

            self.assertTrue(
                torch.allclose(a, b, atol=1e-6)
            )

    def test_reject_too_long_input(self):
        tokens = torch.randint(
            2, 320, (1, 129)
        )

        with self.assertRaises(ValueError):
            self.model(tokens)


if __name__ == "__main__":
    unittest.main()
