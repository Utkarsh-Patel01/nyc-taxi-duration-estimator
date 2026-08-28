import pandas as pd
import pytest
from src.features.engineer import (
    add_distance_feature,
    add_time_features,
    haversine_distance_km,
    valid_coordinate_mask,
    valid_passenger_count_mask,
    valid_speed_mask,
    valid_trip_duration_mask,
)


class TestHaversineDistance:
    def test_known_nyc_distance(self):
        distance = haversine_distance_km(40.7580, -73.9855, 40.7484, -73.9857)
        assert 0.9 < distance < 1.2

    def test_zero_distance_for_identical_points(self):
        distance = haversine_distance_km(40.75, -73.98, 40.75, -73.98)
        assert distance == pytest.approx(0.0, abs=1e-9)

    def test_add_distance_feature_adds_column(self):
        df = pd.DataFrame(
            {
                "pickup_latitude": [40.7580],
                "pickup_longitude": [-73.9855],
                "dropoff_latitude": [40.7484],
                "dropoff_longitude": [-73.9857],
            }
        )
        result = add_distance_feature(df.copy())
        assert "distance_km" in result.columns
        assert result["distance_km"].iloc[0] > 0


class TestTimeFeatures:
    def test_monday_evening(self):
        df = pd.DataFrame(
            {"pickup_datetime": pd.to_datetime(["2016-03-14 18:30:00"])}
        )
        result = add_time_features(df.copy())
        assert result["pickup_hour"].iloc[0] == 18
        assert result["pickup_dayofweek"].iloc[0] == 0
        assert result["is_weekend"].iloc[0] == 0

    def test_saturday_morning(self):
        df = pd.DataFrame(
            {"pickup_datetime": pd.to_datetime(["2016-03-12 09:00:00"])}
        )
        result = add_time_features(df.copy())
        assert result["pickup_hour"].iloc[0] == 9
        assert result["pickup_dayofweek"].iloc[0] == 5
        assert result["is_weekend"].iloc[0] == 1


class TestValidityMasks:
    def test_coordinate_mask_rejects_outside_nyc(self):
        df = pd.DataFrame(
            {
                "pickup_longitude": [-73.98, -121.0],
                "pickup_latitude": [40.75, 38.5],
                "dropoff_longitude": [-73.97, -121.0],
                "dropoff_latitude": [40.76, 38.5],
            }
        )
        assert valid_coordinate_mask(df).tolist() == [True, False]

    def test_passenger_count_mask_rejects_invalid_values(self):
        df = pd.DataFrame({"passenger_count": [0, 1, 6, 7]})
        assert valid_passenger_count_mask(df).tolist() == [False, True, True, False]

    def test_duration_mask_rejects_invalid_values(self):
        df = pd.DataFrame({"trip_duration": [10, 600, 79_200, 100_000]})
        assert valid_trip_duration_mask(df).tolist() == [False, True, True, False]

    def test_speed_mask_rejects_implausible_speed(self):
        df = pd.DataFrame(
            {"distance_km": [10.0, 200.0], "trip_duration": [3600, 3600]}
        )
        assert valid_speed_mask(df, max_kmh=100.0).tolist() == [True, False]
