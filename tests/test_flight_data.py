
import unittest

from src.skygrid_ai.flight_data import (
    generate_data,
    validate_data,
    split_by_flight,
)


class TestFlightData(unittest.TestCase):

    def test_record_count(self):
        records = generate_data()
        self.assertEqual(len(records), 1000)

    def test_data_validation(self):
        records = generate_data()
        validate_data(records)

    def test_no_flight_leakage(self):
        records = generate_data()

        training, testing = split_by_flight(records)

        train_ids = {
            row["flight_id"] for row in training
        }

        test_ids = {
            row["flight_id"] for row in testing
        }

        self.assertTrue(train_ids.isdisjoint(test_ids))

        self.assertEqual(len(training), 800)
        self.assertEqual(len(testing), 200)

    def test_invalid_battery_rejected(self):
        records = generate_data()
        records[0]["battery_pct"] = -5

        with self.assertRaises(ValueError):
            validate_data(records)


if __name__ == "__main__":
    unittest.main()
