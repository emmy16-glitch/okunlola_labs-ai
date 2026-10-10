
import csv
import random
from pathlib import Path


COLUMNS = [
    "flight_id",
    "second",
    "battery_pct",
    "gps_satellites",
    "ground_speed_mps",
    "vibration_index",
    "scenario",
    "label",
]

SCENARIOS = [
    "normal",
    "sim_low_battery",
    "sim_gps_dropout",
    "sim_high_vibration",
]


def generate_data(flights=40, readings=25, seed=42):
    """Generate reproducible synthetic flight records."""

    rng = random.Random(seed)
    records = []

    for flight_number in range(flights):
        flight_id = f"SIM-{flight_number + 1:04d}"
        scenario = SCENARIOS[flight_number % 4]

        if scenario == "sim_low_battery":
            starting_battery = rng.uniform(22, 35)
        else:
            starting_battery = rng.uniform(70, 95)

        for second in range(readings):
            drain = 0.5 if scenario == "sim_low_battery" else 0.2

            battery = max(
                0,
                starting_battery - second * drain
            )

            gps = (
                rng.randint(0, 4)
                if scenario == "sim_gps_dropout"
                else rng.randint(10, 18)
            )

            vibration = (
                rng.uniform(2.5, 4.5)
                if scenario == "sim_high_vibration"
                else rng.uniform(0.1, 0.8)
            )

            record = {
                "flight_id": flight_id,
                "second": second,
                "battery_pct": round(battery, 2),
                "gps_satellites": gps,
                "ground_speed_mps": round(
                    rng.uniform(3, 15), 2
                ),
                "vibration_index": round(vibration, 3),
                "scenario": scenario,
                "label": 0 if scenario == "normal" else 1,
            }

            records.append(record)

    return records


def validate_data(records):
    """Check basic dataset quality."""

    if not records:
        raise ValueError("Dataset is empty")

    for record in records:
        if not record["flight_id"]:
            raise ValueError("Missing flight ID")

        if not 0 <= float(record["battery_pct"]) <= 100:
            raise ValueError("Invalid battery percentage")

        if int(record["gps_satellites"]) < 0:
            raise ValueError("Invalid satellite count")

        if float(record["ground_speed_mps"]) < 0:
            raise ValueError("Invalid ground speed")

        if float(record["vibration_index"]) < 0:
            raise ValueError("Invalid vibration value")

        if int(record["label"]) not in (0, 1):
            raise ValueError("Invalid classification label")


def save_data(records, path):
    """Write records to a CSV file."""

    validate_data(records)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def split_by_flight(records, test_fraction=0.20, seed=42):
    """Keep entire flights together during splitting."""

    if not 0 < test_fraction < 1:
        raise ValueError("Invalid test fraction")

    flight_ids = sorted({
        record["flight_id"] for record in records
    })

    if len(flight_ids) < 2:
        raise ValueError("Need at least two flights")

    rng = random.Random(seed)
    rng.shuffle(flight_ids)

    test_count = max(1, int(len(flight_ids) * test_fraction))
    test_ids = set(flight_ids[:test_count])

    training = [
        row for row in records
        if row["flight_id"] not in test_ids
    ]

    testing = [
        row for row in records
        if row["flight_id"] in test_ids
    ]

    return training, testing


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / "data" / "synthetic" / "flights.csv"

    records = generate_data()
    save_data(records, output)

    training, testing = split_by_flight(records)

    print("SkyGrid Flight Dataset")
    print("----------------------")
    print(f"Total records: {len(records)}")
    print(f"Training records: {len(training)}")
    print(f"Testing records: {len(testing)}")
    print(f"Dataset saved: {output}")


if __name__ == "__main__":
    main()
