import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians, (lat1, lon1, lat2, lon2)
    )
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["distance_km"] = haversine_distance_km(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
    )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        raise TypeError("pickup_datetime must be datetime-like before engineering features")

    df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype("int8")
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype("int8")
    return df


def valid_coordinate_mask(
    df: pd.DataFrame,
    min_lon: float = -74.03,
    max_lon: float = -73.75,
    min_lat: float = 40.63,
    max_lat: float = 40.85,
) -> pd.Series:
    pickup_ok = df["pickup_longitude"].between(min_lon, max_lon) & df[
        "pickup_latitude"
    ].between(min_lat, max_lat)
    dropoff_ok = df["dropoff_longitude"].between(min_lon, max_lon) & df[
        "dropoff_latitude"
    ].between(min_lat, max_lat)
    return pickup_ok & dropoff_ok


def valid_passenger_count_mask(
    df: pd.DataFrame, min_count: int = 1, max_count: int = 6
) -> pd.Series:
    return df["passenger_count"].between(min_count, max_count)


def valid_trip_duration_mask(
    df: pd.DataFrame, min_seconds: int = 60, max_seconds: int = 79_200
) -> pd.Series:
    return df["trip_duration"].between(min_seconds, max_seconds)


def valid_speed_mask(df: pd.DataFrame, max_kmh: float = 100.0) -> pd.Series:
    hours = df["trip_duration"] / 3600.0
    implied_speed = df["distance_km"] / hours.replace(0, np.nan)
    return implied_speed.le(max_kmh) & implied_speed.notna()




def clean_and_engineer_features(
    df: pd.DataFrame,
    coord_bounds: dict | None = None,
    passenger_bounds: dict | None = None,
    duration_bounds: dict | None = None,
    max_speed_kmh: float = 100.0,
) -> pd.DataFrame:
    coord_bounds = coord_bounds or {}
    passenger_bounds = passenger_bounds or {}
    duration_bounds = duration_bounds or {}

    out = add_time_features(add_distance_feature(df.copy()))
    n_start = len(out)
    logger.info("Starting cleaning pipeline with %d rows", n_start)

    filters = (
        ("Coordinate", valid_coordinate_mask(out, **coord_bounds)),
        ("Passenger-count", valid_passenger_count_mask(out, **passenger_bounds)),
        ("Duration", valid_trip_duration_mask(out, **duration_bounds)),
        ("Speed", valid_speed_mask(out, max_kmh=max_speed_kmh)),
    )
    for name, mask in filters:
        mask = mask.reindex(out.index, fill_value=False)
        logger.info("%s filter: dropping %d rows", name, (~mask).sum())
        out = out.loc[mask]

    out = out.reset_index(drop=True)
    logger.info(
        n_start,
        len(out),
        100 * (n_start - len(out)) / n_start if n_start else 0.0,
    )
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.data.load import load_raw_data

    raw = load_raw_data("data/raw/train.csv")
    processed = clean_and_engineer_features(raw)
    output_path = Path("data/processed/train_features.parquet")
    processed.to_parquet(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)
