"""Inference pipeline for the NYC Taxi Trip Duration project.

This is the only prediction module a UI should use. It imports the exact
feature transformations used during training to prevent train/serve skew.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.features.engineer import (
    add_distance_feature,
    add_time_features,
    valid_coordinate_mask,
    valid_passenger_count_mask,
)
from src.models.train import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class InvalidTripInputError(ValueError):
    """Raised when trip input is outside the model's supported domain."""


def load_model(path: str | Path):
    """Load the serialized model artifact from ``path``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    logger.info("Loading model from %s", path)
    return joblib.load(path)


def build_feature_row(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    pickup_datetime,
    passenger_count: int,
    vendor_id: int = 1,
) -> pd.DataFrame:
    """Build one feature row with the exact transforms and column order used in training."""
    row = pd.DataFrame(
        {
            "vendor_id": [vendor_id],
            "passenger_count": [passenger_count],
            "pickup_longitude": [pickup_longitude],
            "pickup_latitude": [pickup_latitude],
            "dropoff_longitude": [dropoff_longitude],
            "dropoff_latitude": [dropoff_latitude],
            "pickup_datetime": [pd.to_datetime(pickup_datetime)],
        }
    )
    row = add_distance_feature(row)
    row = add_time_features(row)
    return row[FEATURE_COLUMNS]


def validate_trip_input(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    passenger_count: int,
) -> None:
    """Raise ``InvalidTripInputError`` for values outside training-data bounds."""
    row = pd.DataFrame(
        {
            "pickup_latitude": [pickup_latitude],
            "pickup_longitude": [pickup_longitude],
            "dropoff_latitude": [dropoff_latitude],
            "dropoff_longitude": [dropoff_longitude],
            "passenger_count": [passenger_count],
        }
    )
    if not valid_coordinate_mask(row).iloc[0]:
        raise InvalidTripInputError(
            "Pickup/dropoff coordinates fall outside the NYC area this model was trained on."
        )
    if not valid_passenger_count_mask(row).iloc[0]:
        raise InvalidTripInputError("Passenger count must be between 1 and 6.")


def predict_trip_duration_seconds(model, feature_row: pd.DataFrame) -> float:
    """Return predicted trip duration in seconds, clipped at zero."""
    raw_prediction = model.predict(feature_row)[0]
    return max(0.0, float(raw_prediction))


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable estimated duration."""
    minutes = round(seconds / 60)
    if minutes < 1:
        return "Estimated Trip Duration: less than 1 minute"
    if minutes < 60:
        return f"Estimated Trip Duration: {minutes} minutes"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"Estimated Trip Duration: {hours}h {remaining_minutes}m"


def predict(
    model,
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    pickup_datetime,
    passenger_count: int,
    vendor_id: int = 1,
    validate: bool = True,
) -> dict:
    """Validate, featurize, predict, clip, and format one trip estimate."""
    if validate:
        validate_trip_input(
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude,
            passenger_count,
        )

    feature_row = build_feature_row(
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude,
        pickup_datetime,
        passenger_count,
        vendor_id,
    )
    duration_seconds = predict_trip_duration_seconds(model, feature_row)
    return {
        "duration_seconds": round(duration_seconds, 1),
        "duration_formatted": format_duration(duration_seconds),
    }
