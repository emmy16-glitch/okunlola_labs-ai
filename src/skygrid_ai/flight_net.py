
from torch import nn

FEATURE_NAMES = (
    "battery_pct",
    "gps_satellites",
    "ground_speed_mps",
    "vibration_index",
)

FEATURE_SCALES = (100.0, 20.0, 20.0, 5.0)


def prepare_features(record):
    values = [
        float(record[name])
        for name in FEATURE_NAMES
    ]

    if not (
        0 <= values[0] <= 100
        and 0 <= values[1]
        and 0 <= values[2]
        and 0 <= values[3]
    ):
        raise ValueError("Invalid telemetry values")

    return [
        value / scale
        for value, scale in zip(values, FEATURE_SCALES)
    ]


class SkyGridFlightNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, inputs):
        return self.layers(inputs)
