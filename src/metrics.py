import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(
        2.0 * np.abs(y_pred - y_true),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom != 0,
    )
    return float(100.0 * np.mean(ratio))


def mase(y_true, y_pred, scale: float) -> float:
    if not np.isfinite(scale) or scale <= 0:
        return float('nan')
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)
