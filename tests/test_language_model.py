
import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.language.model import (
    SkyGridLanguageModel
)


class TestLanguageModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

        self.model = SkyGridLanguageModel(
            vocab_size=38
        )

    def test_output_and_loss(self):
        x = torch.randint(2, 38, (2, 8))
        y = torch.randint(2, 38, (2, 8))

        logits, loss = self.model(x, y)

        self.assertEqual(
            tuple(logits.shape),
            (2, 8, 38)
        )

        self.assertTrue(
            torch.isfinite(loss).item()
        )

    def test_causal_mask(self):
        self.model.eval()

        x = torch.randint(2, 38, (1, 8))

        altered = x.clone()

        altered[0, -1] = (
            altered[0, -1] + 1
        ) % 38

        with torch.inference_mode():
            first, _ = self.model(x)
            second, _ = self.model(altered)

        self.assertTrue(
            torch.allclose(
                first[:, :-1],
                second[:, :-1],
                atol=1e-6
            )
        )

    def test_gradients_exist(self):
        x = torch.randint(2, 38, (2, 8))

        _, loss = self.model(x, x)

        loss.backward()

        gradient = (
            self.model.token_embedding.weight.grad
        )

        self.assertIsNotNone(gradient)

        self.assertTrue(
            torch.isfinite(gradient).all().item()
        )

    def test_checkpoint_roundtrip(self):
        self.model.eval()

        x = torch.randint(2, 38, (1, 8))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"

            torch.save(
                self.model.state_dict(),
                path
            )

            restored = SkyGridLanguageModel(
                vocab_size=38
            )

            restored.load_state_dict(
                torch.load(path, weights_only=True)
            )

            restored.eval()

            with torch.inference_mode():
                a, _ = self.model(x)
                b, _ = restored(x)

            self.assertTrue(
                torch.allclose(a, b)
            )


if __name__ == "__main__":
    unittest.main()
