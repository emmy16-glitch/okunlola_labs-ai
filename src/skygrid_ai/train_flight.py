
import copy
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from .flight_net import (
    FEATURE_NAMES,
    FEATURE_SCALES,
    SkyGridFlightNet,
    prepare_features,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "synthetic" / "flights.csv"

CHECKPOINT_PATH = (
    ROOT / "checkpoints" / "skygrid_flight_v04.pt"
)


def split_by_scenario(records, seed=42):
    """Split whole flights, retaining every scenario."""

    ids_by_scenario = defaultdict(set)

    for row in records:
        ids_by_scenario[row["scenario"]].add(
            row["flight_id"]
        )

    rng = random.Random(seed)

    train_ids = set()
    val_ids = set()
    test_ids = set()

    for scenario in sorted(ids_by_scenario):
        ids = sorted(ids_by_scenario[scenario])
        rng.shuffle(ids)

        if len(ids) < 5:
            raise ValueError(
                "Need at least five flights per scenario"
            )

        n = max(1, round(len(ids) * 0.20))

        test_ids.update(ids[:n])
        val_ids.update(ids[n:2 * n])
        train_ids.update(ids[2 * n:])

    return (
        [
            r for r in records
            if r["flight_id"] in train_ids
        ],
        [
            r for r in records
            if r["flight_id"] in val_ids
        ],
        [
            r for r in records
            if r["flight_id"] in test_ids
        ],
    )


def make_tensors(records):
    features = [
        prepare_features(row)
        for row in records
    ]

    labels = [
        [float(row["label"])]
        for row in records
    ]

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


def report(model, x, y, name):
    model.eval()

    with torch.inference_mode():
        predicted = torch.sigmoid(model(x)) >= 0.5
        actual = y >= 0.5

        tp = int((predicted & actual).sum())
        tn = int((~predicted & ~actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total
    recall = tp / max(tp + fn, 1)

    print(
        f"{name}: accuracy={accuracy:.1%}, "
        f"recall={recall:.1%}, "
        f"false alarms={fp}, "
        f"missed positives={fn}"
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "First run: python -m src.skygrid_ai.flight_data"
        )

    with DATA_PATH.open(
        newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    train, val, test = split_by_scenario(records)

    train_x, train_y = make_tensors(train)
    val_x, val_y = make_tensors(val)
    test_x, test_y = make_tensors(test)

    torch.manual_seed(42)

    model = SkyGridFlightNet()
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0

    print(
        f"Records: training={len(train)}, "
        f"validation={len(val)}, testing={len(test)}"
    )

    for epoch in range(1, 151):
        model.train()

        optimizer.zero_grad()

        loss = loss_fn(
            model(train_x),
            train_y
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.inference_mode():
            val_loss = loss_fn(
                model(val_x),
                val_y
            ).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_epoch = epoch

        if epoch in (1, 50, 100, 150):
            print(
                f"Epoch {epoch:3d} | "
                f"train loss={loss.item():.4f} | "
                f"validation loss={val_loss:.4f}"
            )

    model.load_state_dict(best_state)

    report(model, test_x, test_y, "Held-out test")

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state": best_state,
            "features": FEATURE_NAMES,
            "scales": FEATURE_SCALES,
            "best_epoch": best_epoch,
        },
        CHECKPOINT_PATH
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Saved: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
