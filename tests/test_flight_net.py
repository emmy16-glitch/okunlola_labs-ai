
import unittest
import torch

from src.skygrid_ai.flight_data import generate_data
from src.skygrid_ai.flight_net import (
    SkyGridFlightNet,
    prepare_features,
)
from src.skygrid_ai.train_flight import split_by_scenario


class TestFlightNet(unittest.TestCase):

    def test_feature_normalization(self):
        row = {
            "battery_pct": 50,
            "gps_satellites": 16,
            "ground_speed_mps": 10,
            "vibration_index": 0.5,
        }

        features = prepare_features(row)

        for actual, expected in zip(
            features, [0.5, 0.8, 0.5, 0.1]
        ):
            self.assertAlmostEqual(actual, expected)

    def test_output_shape(self):
        model = SkyGridFlightNet()

        output = model(torch.rand(7, 4))

        self.assertEqual(tuple(output.shape), (7, 1))

    def test_parameter_count(self):
        model = SkyGridFlightNet()

        count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        self.assertEqual(count, 225)

    def test_flight_split(self):
        records = generate_data()
        train, val, test = split_by_scenario(records)

        self.assertEqual(len(train), 600)
        self.assertEqual(len(val), 200)
        self.assertEqual(len(test), 200)

        groups = [
            {row["flight_id"] for row in group}
            for group in (train, val, test)
        ]

        for a in range(3):
            for b in range(a + 1, 3):
                self.assertTrue(
                    groups[a].isdisjoint(groups[b])
                )


if __name__ == "__main__":
    unittest.main()
