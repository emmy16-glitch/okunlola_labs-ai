
import tempfile
import unittest
from pathlib import Path

import torch

from src.skygrid_ai.model import SkyGridTelemetryNet


class TestSkyGridTelemetryNet(unittest.TestCase):

    def test_output_shape(self):
        model = SkyGridTelemetryNet()

        inputs = torch.rand(5, 2)
        outputs = model(inputs)

        self.assertEqual(
            tuple(outputs.shape),
            (5, 1)
        )

    def test_parameter_count(self):
        model = SkyGridTelemetryNet()

        count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        self.assertEqual(count, 65)

    def test_saved_weights_load_correctly(self):
        original = SkyGridTelemetryNet()

        sample = torch.tensor([[0.12, 0.80]])

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.pt"

            torch.save(original.state_dict(), path)

            restored = SkyGridTelemetryNet()

            restored.load_state_dict(
                torch.load(path, weights_only=True)
            )

            self.assertTrue(
                torch.equal(
                    original(sample),
                    restored(sample)
                )
            )


if __name__ == "__main__":
    unittest.main()
