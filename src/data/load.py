"""Data loading utilities for the NYC Taxi Trip Duration project.

Centralizes dtype and parsing decisions so every script/notebook
that loads the raw data gets identical, memory-efficient columns.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Explicit dtypes: reduces memory footprint vs. pandas' default inference,
# and documents the expected schema in one place.
RAW_DTYPES = {
    "id": "object",
    "vendor_id": "int8",
    "passenger_count": "int8",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "store_and_fwd_flag": "category",
    "trip_duration": "int32",
}

# Parsed separately as datetimes, not included in RAW_DTYPES
DATETIME_COLS = ["pickup_datetime", "dropoff_datetime"]

# This column is derivable directly from pickup_datetime + trip_duration.
# It must NEVER be used as a model feature — flagged here so every
# downstream consumer of this module is reminded.
LEAKAGE_COLUMNS = ["dropoff_datetime"]


def load_raw_data(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw NYC taxi trip CSV with explicit dtypes.

    Args:
        path: Path to train.csv.
        nrows: Optional row limit, useful for fast local development
            (e.g. nrows=50_000) before running on the full dataset.

    Returns:
        DataFrame with enforced dtypes and parsed datetime columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")

    logger.info("Loading raw data from %s (nrows=%s)", path, nrows)

    df = pd.read_csv(
        path,
        dtype=RAW_DTYPES,
        parse_dates=DATETIME_COLS,
        nrows=nrows,
    )

    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def inspect_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of column dtypes and non-null counts."""
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null_count": df.notna().sum(),
        "null_count": df.isna().sum(),
    })
    return summary


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages per column."""
    missing = df.isna().sum()
    pct = (missing / len(df)) * 100
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct.round(3)})
    return report[report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def validate_raw_data(df: pd.DataFrame) -> list[str]:
    """Run basic sanity checks on the raw dataset.

    This is NOT outlier handling (that's Phase 3) — just structural
    checks that would indicate a corrupted download or schema mismatch.

    Returns:
        A list of human-readable warning strings (empty if all checks pass).
    """
    warnings = []

    expected_cols = set(RAW_DTYPES) | set(DATETIME_COLS)
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        warnings.append(f"Missing expected columns: {missing_cols}")

    if "id" in df.columns and df["id"].duplicated().any():
        n_dupes = df["id"].duplicated().sum()
        warnings.append(f"{n_dupes} duplicate 'id' values found")

    if "trip_duration" in df.columns and (df["trip_duration"] <= 0).any():
        n_bad = (df["trip_duration"] <= 0).sum()
        warnings.append(f"{n_bad} rows have trip_duration <= 0 seconds")

    if "passenger_count" in df.columns and (df["passenger_count"] < 0).any():
        warnings.append("Negative passenger_count values found")

    present_leakage_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if present_leakage_cols:
        warnings.append(
            f"Leakage-risk column(s) present: {present_leakage_cols}. "
            "These must be excluded from model features."
        )

    return warnings