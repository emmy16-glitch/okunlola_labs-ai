
import argparse
from pathlib import Path

import torch

from .model import SkyGridTelemetryNet


ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_PATH = (
    ROOT / "checkpoints" / "skygrid_telemetry_v01.pt"
)


def predict(battery_percent: float, satellites: int) -> float:

    if not 0 <= battery_percent <= 100:
        raise ValueError("Battery percentage must be from 0 to 100")

    if not 0 <= satellites <= 20:
        raise ValueError(
            "For this demo, satellites must be from 0 to 20"
        )

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "Run first_model.py to train and save weights first"
        )

    # Reconstruct our neural network.
    model = SkyGridTelemetryNet()

    # Load the weights that our previous training produced.
    weights = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    # Normalize readings using the original data format.
    inputs = torch.tensor(
        [[battery_percent / 100, satellites / 20]],
        dtype=torch.float32
    )

    # Make a prediction without training.
    with torch.inference_mode():
        score = torch.sigmoid(model(inputs)).item()

    return score


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic SkyGrid telemetry demonstration"
    )

    parser.add_argument(
        "--battery",
        type=float,
        required=True
    )

    parser.add_argument(
        "--satellites",
        type=int,
        required=True
    )

    args = parser.parse_args()

    score = predict(args.battery, args.satellites)

    print(f"Synthetic attention score: {score:.4f}")
    print("Demo only: not a real aircraft health assessment.")


if __name__ == "__main__":
    main()
