import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred_clipped)
    return float(np.sqrt(mean_squared_error(log_true, log_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Return Root Mean Squared Error in seconds.
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Return Mean Absolute Error in seconds.
    return float(mean_absolute_error(y_true, y_pred))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Return the project's standard set of validation metrics.
    return {
        "rmsle": rmsle(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
