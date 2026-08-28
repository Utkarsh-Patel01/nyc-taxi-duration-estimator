from pathlib import Path
import pandas as pd
import pytest

from src.features.engineer import add_distance_feature, add_time_features
from src.inference.predict import (
    InvalidTripInputError,
    build_feature_row,
    format_duration,
    load_model,
    predict,
    predict_trip_duration_seconds,
    validate_trip_input,
)
from src.models.train import FEATURE_COLUMNS

MODEL_PATH = Path("models/lightgbm_final.joblib")

VALID_TRIP = {
    "pickup_latitude": 40.7580,
    "pickup_longitude": -73.9855,
    "dropoff_latitude": 40.7484,
    "dropoff_longitude": -73.9857,
    "pickup_datetime": "2016-03-14 18:30:00",
    "passenger_count": 2,
}


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL_PATH)


class TestModelLoading:
    def test_loads_real_artifact(self, model):
        assert model is not None

    def test_raises_clear_error_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_model("models/does_not_exist.joblib")


class TestInputValidation:
    def test_rejects_coordinates_outside_nyc(self):
        with pytest.raises(InvalidTripInputError):
            validate_trip_input(38.5, -121.0, 38.5, -121.0, passenger_count=2)

    def test_accepts_valid_nyc_coordinates(self):
        validate_trip_input(40.7580, -73.9855, 40.7484, -73.9857, passenger_count=2)

    @pytest.mark.parametrize("passenger_count", [0, 9])
    def test_rejects_invalid_passenger_count(self, passenger_count):
        with pytest.raises(InvalidTripInputError):
            validate_trip_input(40.75, -73.98, 40.75, -73.98, passenger_count)


class TestPredictionSafety:
    def test_negative_prediction_is_clipped_to_zero(self):
        class FakeNegativeModel:
            def predict(self, _features):
                return [-42.0]

        dummy_row = pd.DataFrame([[0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
        assert predict_trip_duration_seconds(FakeNegativeModel(), dummy_row) == 0.0

    def test_predict_raises_typeerror_when_required_arg_missing(self, model):
        with pytest.raises(TypeError):
            predict(model, pickup_latitude=40.75)


class TestPredictionPipeline:
    def test_end_to_end_prediction_shape(self, model):
        result = predict(model, **VALID_TRIP)
        assert {"duration_seconds", "duration_formatted"} <= result.keys()
        assert result["duration_seconds"] >= 0

    def test_prediction_is_plausible_magnitude(self, model):
        result = predict(model, **VALID_TRIP)
        assert 0 < result["duration_seconds"] < 3600

    def test_invalid_input_raises_before_model_prediction(self, model):
        with pytest.raises(InvalidTripInputError):
            predict(
                model,
                pickup_latitude=38.5,
                pickup_longitude=-121.0,
                dropoff_latitude=38.5,
                dropoff_longitude=-121.0,
                pickup_datetime="2016-03-14 18:30:00",
                passenger_count=2,
            )


class TestFormatDuration:
    def test_under_a_minute(self):
        assert format_duration(30) == "Estimated Trip Duration: less than 1 minute"

    def test_typical_minutes(self):
        assert format_duration(360) == "Estimated Trip Duration: 6 minutes"

    def test_over_an_hour(self):
        assert format_duration(4500) == "Estimated Trip Duration: 1h 15m"


class TestFeatureConsistency:
    def test_inference_features_match_training_transforms(self):
        raw = pd.DataFrame(
            {
                "pickup_latitude": [40.7580],
                "pickup_longitude": [-73.9855],
                "dropoff_latitude": [40.7484],
                "dropoff_longitude": [-73.9857],
                "pickup_datetime": pd.to_datetime(["2016-03-14 18:30:00"]),
            }
        )
        expected = add_time_features(add_distance_feature(raw.copy()))
        inference_row = build_feature_row(
            pickup_latitude=40.7580,
            pickup_longitude=-73.9855,
            dropoff_latitude=40.7484,
            dropoff_longitude=-73.9857,
            pickup_datetime="2016-03-14 18:30:00",
            passenger_count=2,
            vendor_id=1,
        )

        assert inference_row["distance_km"].iloc[0] == pytest.approx(
            expected["distance_km"].iloc[0]
        )
        assert inference_row["pickup_hour"].iloc[0] == expected["pickup_hour"].iloc[0]
        assert inference_row["pickup_dayofweek"].iloc[0] == expected[
            "pickup_dayofweek"
        ].iloc[0]
        assert inference_row["is_weekend"].iloc[0] == expected["is_weekend"].iloc[0]
