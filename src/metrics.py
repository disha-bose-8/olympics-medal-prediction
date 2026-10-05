import numpy as np


def rmse(y_true, y_pred):
    """Root of the average squared error, in medals. This is the author's 'average std dev'."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    """Average absolute error, in medals."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def top10_rmse(y_true, y_pred, k=10):
    """Author's version: RMSE over only the k countries with the most actual medals."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    top = np.argsort(-y_true)[:k]
    return rmse(y_true[top], y_pred[top])


def selection_score(y_true, y_pred):
    """Author's model-selection score: RMSE + 0.25 * top-10 RMSE (lower is better)."""
    return rmse(y_true, y_pred) + 0.25 * top10_rmse(y_true, y_pred)


def to_counts(y_pred):
    """Round to whole medals and clip at 0, as the author does."""
    return np.clip(np.rint(np.asarray(y_pred, dtype=float)), 0, None)


if __name__ == "__main__":
    y_true = [0, 10, 20]
    y_pred = [1, 8, 25]
    print("rmse:", round(rmse(y_true, y_pred), 4))
    print("mae:", round(mae(y_true, y_pred), 4))
    print("top10_rmse (k=2):", round(top10_rmse(y_true, y_pred, k=2), 4))
    print("selection_score:", round(selection_score(y_true, y_pred), 4))
    print("to_counts:", to_counts([-0.7, 2.4, 7.6]))