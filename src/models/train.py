"""Training utilities for the NYC Taxi Trip Duration project.

This module defines the canonical model inputs and the reproducible
train/validation split used by every experiment.
"""

import logging
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# This whitelist prevents identifiers, raw datetimes, and leakage-risk columns
# from silently becoming model inputs.
FEATURE_COLUMNS = [
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance_km",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
]

TARGET_COLUMN = "trip_duration"


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return canonical model features ``X`` and target ``y`` from processed data."""
    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Processed data is missing expected columns: {sorted(missing)}")

    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def make_train_val_split(
    X: pd.DataFrame, y: pd.Series, val_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible random train/validation split.

    A random split is appropriate for this fixed six-month competition dataset:
    its trip-level features describe a journey rather than a long-running trend.
    """
    if not 0 < val_size < 1:
        raise ValueError("val_size must be greater than 0 and less than 1")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=RANDOM_SEED
    )
    logger.info(
        "Split: %d train rows, %d validation rows (%.0f%% validation)",
        len(X_train),
        len(X_val),
        val_size * 100,
    )
    return X_train, X_val, y_train, y_val


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 15,
    min_samples_leaf: int = 5,
    n_jobs: int = -1,
) -> tuple[RandomForestRegressor, float]:
    """Fit a modest, regularized Random Forest and return it with fit time.

    Capping tree depth and requiring multiple observations per leaf prevents
    individual trees from memorizing GPS-coordinate noise in this large dataset.
    """
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    logger.info("Random Forest trained in %.1f seconds", train_time)
    return model, train_time


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 20,
) -> tuple[XGBRegressor, float]:
    """Fit XGBoost with validation-based early stopping.

    The validation set determines when to stop adding trees; only the
    training split is used to fit the trees themselves.
    """
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if not 0 < subsample <= 1 or not 0 < colsample_bytree <= 1:
        raise ValueError("subsample and colsample_bytree must be in (0, 1]")
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be at least 1")

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.perf_counter() - start

    logger.info(
        "XGBoost trained in %.1f seconds; best iteration: %d",
        train_time,
        model.best_iteration,
    )
    return model, train_time


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 20,
) -> tuple[LGBMRegressor, float]:
    """Fit LightGBM with validation-based early stopping.

    ``num_leaves`` constrains LightGBM's leaf-wise trees, helping prevent
    overfitting while retaining its efficient large-tabular-data training.
    """
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if num_leaves < 2:
        raise ValueError("num_leaves must be at least 2")
    if not 0 < subsample <= 1 or not 0 < colsample_bytree <= 1:
        raise ValueError("subsample and colsample_bytree must be in (0, 1]")
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be at least 1")

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ],
    )
    train_time = time.perf_counter() - start

    logger.info(
        "LightGBM trained in %.1f seconds; best iteration: %d",
        train_time,
        model.best_iteration_,
    )
    return model, train_time


def generate_oof_predictions(
    train_fn,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    **train_fn_kwargs,
) -> np.ndarray:
    """Generate leakage-safe out-of-fold predictions for one base model.

    Each row is predicted by a fresh model that was trained without that row.
    These predictions can therefore be used safely as meta-model features.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))

    for fold_idx, (train_idx, holdout_idx) in enumerate(kf.split(X), start=1):
        X_fold_train = X.iloc[train_idx]
        X_fold_holdout = X.iloc[holdout_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_holdout = y.iloc[holdout_idx]

        if "X_val" in train_fn_kwargs or train_fn.__name__ in {
            "train_xgboost",
            "train_lightgbm",
        }:
            fold_model, _ = train_fn(
                X_fold_train, y_fold_train, X_fold_holdout, y_fold_holdout
            )
        else:
            fold_model, _ = train_fn(X_fold_train, y_fold_train)

        oof_preds[holdout_idx] = fold_model.predict(X_fold_holdout)
        logger.info("Fold %d/%d complete (%s)", fold_idx, n_splits, train_fn.__name__)

    return oof_preds
