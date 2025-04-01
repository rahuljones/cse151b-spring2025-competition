"""
ClimateBench Evaluation Metric for Kaggle Competition

This module implements the evaluation metric for the ClimateBench Kaggle competition.
It evaluates climate predictions using area-weighted metrics that assess both mean climate state
and temporal variability at each location.
"""

import re

import numpy as np
import pandas as pd
from tqdm import tqdm


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Evaluates climate predictions using area-weighted metrics that account for different grid cell sizes.
    The score combines three components:
    1. Overall area-weighted RMSE
    2. Time-mean area-weighted RMSE (spatial patterns of mean climate)
    3. Time-standard deviation area-weighted MAE (temporal variability at each location)

    The final score is a weighted average of these metrics across both temperature and precipitation variables.

    Parameters:
    -----------
    solution : pd.DataFrame
        Ground truth values with columns: [row_id_column_name, "Prediction"]
    submission : pd.DataFrame
        Predicted values with columns: [row_id_column_name, "Prediction"]
    row_id_column_name : str
        Name of the column that identifies each row

    Returns:
    --------
    float
        The final score (lower is better)

    Examples:
    ---------
    >>> import pandas as pd
    >>> import numpy as np
    >>> row_id_column_name = "ID"
    >>> # Create dummy data with 2 variables (tas, pr), 3 time points, 2x2 spatial grid
    >>> # Format: t{time}_{variable}_{lat}_{lon}
    >>> ids = []
    >>> for t in range(3):
    ...     for var in ["tas", "pr"]:
    ...         for lat in [0.0, 30.0]:
    ...             for lon in [0.0, 30.0]:
    ...                 ids.append(f"t{t:03d}_{var}_{lat:.2f}_{lon:.2f}")
    >>> # Create sample solution values
    >>> y_true = np.zeros(len(ids))
    >>> # tas values (temperature) between 270-300K
    >>> y_true[::2] = np.linspace(270, 300, len(ids)//2)
    >>> # pr values (precipitation) between 0-10 mm/day
    >>> y_true[1::2] = np.linspace(0, 10, len(ids)//2)
    >>> # Create predictions with some error
    >>> y_pred = y_true + np.random.normal(0, 1, size=len(ids))
    >>> # Ensure precipitation is non-negative
    >>> y_pred[1::2] = np.maximum(y_pred[1::2], 0)
    >>> # Create DataFrames
    >>> solution = pd.DataFrame({row_id_column_name: ids, "Prediction": y_true})
    >>> submission = pd.DataFrame({row_id_column_name: ids, "Prediction": y_pred})
    >>> # Calculate score (rounded for test stability)
    >>> round(score(solution, submission, row_id_column_name), 4) > 0
    True
    """
    if not all(col in submission.columns for col in [row_id_column_name, "Prediction"]):
        raise ValueError(f"Submission must have columns: {row_id_column_name}, 'Prediction'")

    merged = solution.merge(submission, on=row_id_column_name, how="left", suffixes=("_true", "_pred"))

    if merged["Prediction_pred"].isna().any():
        raise ValueError("Submission is missing predictions for some IDs")

    pattern = r"t(\d+)_([a-z]+)_(-?\d+\.?\d*)_(-?\d+\.?\d*)"

    id_components = []
    for id_str in merged[row_id_column_name]:
        match = re.match(pattern, id_str)
        if match:
            time, variable, lat, lon = match.groups()
            id_components.append({"time": int(time), "variable": variable, "lat": float(lat), "lon": float(lon)})
        else:
            raise ValueError(f"Invalid ID format: {id_str}")

    for key in ["time", "variable", "lat", "lon"]:
        merged[key] = [comp[key] for comp in id_components]

    variables = merged["variable"].unique()
    times = sorted(merged["time"].unique())
    lats = sorted(merged["lat"].unique())
    lons = sorted(merged["lon"].unique())

    lat_weights = {lat: np.cos(np.radians(lat)) for lat in lats}
    total_weight = sum(lat_weights.values())
    for lat in lat_weights:
        lat_weights[lat] /= total_weight
    weights_arr = np.array([lat_weights[lat] for lat in lats])

    merged["weight"] = merged["lat"].map(lat_weights)

    var_weights = {"tas": 0.5, "pr": 0.5}
    metric_var_weights = {
        "tas": {"monthly_rmse": 0.1, "time_mean": 1.0, "time_std": 1.0},
        "pr": {"monthly_rmse": 0.1, "time_mean": 1.0, "time_std": 0.75},
    }

    var_scores = {}
    for var in tqdm(variables):
        var_data = merged[merged["variable"] == var]

        true_df = var_data.pivot_table(index=["time", "lat", "lon"], values="Prediction_true").sort_index()
        pred_df = var_data.pivot_table(index=["time", "lat", "lon"], values="Prediction_pred").sort_index()

        true_3d = true_df.values.reshape(len(times), len(lats), len(lons))
        pred_3d = pred_df.values.reshape(len(times), len(lats), len(lons))

        squared_diff = (true_3d - pred_3d) ** 2
        time_avg_squared_diff = np.mean(squared_diff, axis=0)
        weighted_avg_squared_diff = np.sum(time_avg_squared_diff * weights_arr[:, None], axis=0)
        monthly_rmse = np.sqrt(np.mean(weighted_avg_squared_diff))

        time_mean_true = np.mean(true_3d, axis=0)
        time_mean_pred = np.mean(pred_3d, axis=0)
        time_mean_squared_diff = (time_mean_true - time_mean_pred) ** 2
        weighted_time_mean_squared_diff = np.sum(time_mean_squared_diff * weights_arr[:, None], axis=0)
        time_mean_rmse = np.sqrt(np.mean(weighted_time_mean_squared_diff))

        time_std_true = np.std(true_3d, axis=0)
        time_std_pred = np.std(pred_3d, axis=0)
        time_std_abs_diff = np.abs(time_std_true - time_std_pred)
        weighted_time_std_abs_diff = np.sum(time_std_abs_diff * weights_arr[:, None], axis=0)
        time_std_mae = np.mean(weighted_time_std_abs_diff)

        weights = metric_var_weights[var]
        var_score = (
            weights["monthly_rmse"] * monthly_rmse
            + weights["time_mean"] * time_mean_rmse
            + weights["time_std"] * time_std_mae
        )

        var_scores[var] = var_score

    final_score = sum(var_weights[var] * var_scores[var] for var in variables)
    return final_score
