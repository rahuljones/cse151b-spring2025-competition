#!/usr/bin/env python3
"""
Test script to verify that the Kaggle metric implementation produces the same results
as the original xarray-based area-weighted metrics implementation.

This script:
1. Creates synthetic climate data (temperature and precipitation) as xarray DataArrays
2. Calculates metrics using the original xarray-based area-weighted functions
3. Converts the same data to the Kaggle CSV submission format
4. Calculates metrics using the Kaggle metric function
5. Compares the results to ensure they're equivalent

Run with: python _test_kaggle_metric.py
"""

import sys

import numpy as np
import xarray as xr

from _climate_kaggle_metric import score as kaggle_score
from src.utils import calculate_weighted_metric, create_climate_data_array, get_lat_weights


def test_metric_equivalence():
    """
    Test that the Kaggle metric implementation gives the same results as
    the original xarray-based area-weighted implementation.
    """
    print("Testing equivalence between xarray and Kaggle metrics...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define dimensions
    n_times = 10
    n_lats = 12  # Latitude points from -90 to 90
    n_lons = 24  # Longitude points from 0 to 360

    # Create coordinate arrays
    times = np.arange(n_times)
    lats = np.linspace(-90, 90, n_lats)
    lons = np.linspace(0, 360, n_lons, endpoint=False)

    # Create synthetic true data with realistic patterns
    # Temperature: higher at equator, lower at poles, seasonal variation
    tas_true = np.zeros((n_times, n_lats, n_lons))
    pr_true = np.zeros((n_times, n_lats, n_lons))

    # Latitude pattern: warm at equator, cold at poles
    lat_pattern = 273.15 + 30 * np.cos(np.radians(lats))
    # Longitude pattern: some east-west variation
    lon_pattern = 5 * np.sin(np.radians(lons * 2))

    # Time pattern: seasonal cycle
    time_pattern = 10 * np.sin(np.radians(times * 36))

    # Combine patterns for temperature
    for t in range(n_times):
        for lat_idx, lat in enumerate(lats):
            for lon_idx, lon in enumerate(lons):
                # Base temperature pattern
                tas_true[t, lat_idx, lon_idx] = (
                    lat_pattern[lat_idx]  # Latitude dependence
                    + lon_pattern[lon_idx]  # Longitude dependence
                    + time_pattern[t]  # Time dependence
                )

                # Precipitation: higher in tropics, seasonal variation
                # Based on temperature but with equatorial peak
                pr_factor = np.cos(np.radians(lat)) * np.cos(np.radians(lat))  # Peak at equator
                pr_true[t, lat_idx, lon_idx] = max(0, 5 * pr_factor * (1 + 0.5 * np.sin(np.radians(time_pattern[t]))))

    # Create predicted data with some error
    tas_pred = tas_true + np.random.normal(0, 2, size=tas_true.shape)
    pr_pred = pr_true + np.random.normal(0, 1, size=pr_true.shape)
    # Ensure precipitation is non-negative
    pr_pred = np.maximum(pr_pred, 0)

    # Create xarray objects for weighted calculations
    tas_true_xr = create_climate_data_array(tas_true, times, lats, lons, var_name="tas", var_unit="K")
    tas_pred_xr = create_climate_data_array(tas_pred, times, lats, lons, var_name="tas", var_unit="K")

    pr_true_xr = create_climate_data_array(pr_true, times, lats, lons, var_name="pr", var_unit="mm/day")
    pr_pred_xr = create_climate_data_array(pr_pred, times, lats, lons, var_name="pr", var_unit="mm/day")

    # Calculate area weights
    weights = get_lat_weights(lats)
    weights_da = xr.DataArray(weights, dims=["y"], coords={"y": lats}, name="area_weights")

    # Calculate metrics using xarray functions
    print("\nCalculating metrics using xarray...")

    # Temperature metrics
    tas_diff_squared = (tas_true_xr - tas_pred_xr) ** 2
    tas_overall_rmse = calculate_weighted_metric(tas_diff_squared, weights_da, ("time", "y", "x"), "rmse")

    tas_true_mean = tas_true_xr.mean(dim="time")
    tas_pred_mean = tas_pred_xr.mean(dim="time")
    tas_mean_diff_squared = (tas_true_mean - tas_pred_mean) ** 2
    tas_time_mean_rmse = calculate_weighted_metric(tas_mean_diff_squared, weights_da, ("y", "x"), "rmse")

    tas_true_std = tas_true_xr.std(dim="time")
    tas_pred_std = tas_pred_xr.std(dim="time")
    tas_std_abs_diff = np.abs(tas_true_std - tas_pred_std)
    tas_time_std_mae = calculate_weighted_metric(tas_std_abs_diff, weights_da, ("y", "x"), "mae")

    # Precipitation metrics
    pr_diff_squared = (pr_true_xr - pr_pred_xr) ** 2
    pr_overall_rmse = calculate_weighted_metric(pr_diff_squared, weights_da, ("time", "y", "x"), "rmse")

    pr_true_mean = pr_true_xr.mean(dim="time")
    pr_pred_mean = pr_pred_xr.mean(dim="time")
    pr_mean_diff_squared = (pr_true_mean - pr_pred_mean) ** 2
    pr_time_mean_rmse = calculate_weighted_metric(pr_mean_diff_squared, weights_da, ("y", "x"), "rmse")

    pr_true_std = pr_true_xr.std(dim="time")
    pr_pred_std = pr_pred_xr.std(dim="time")
    pr_std_abs_diff = np.abs(pr_true_std - pr_pred_std)
    pr_time_std_mae = calculate_weighted_metric(pr_std_abs_diff, weights_da, ("y", "x"), "mae")

    # Variable-specific metric weights as specified
    metric_weights = {
        "tas": {
            "monthly_rmse": 0.1,  # Monthly RMSE for temperature
            "time_mean": 1.0,  # Time-mean RMSE for temperature
            "time_std": 1.0,  # Time-stddev MAE for temperature
        },
        "pr": {
            "monthly_rmse": 0.1,  # Monthly RMSE for precipitation
            "time_mean": 1.0,  # Time-mean RMSE for precipitation
            "time_std": 0.75,  # Time-stddev MAE for precipitation
        },
    }

    # Apply normalization and weighting for temperature
    tas_monthly_rmse = tas_overall_rmse
    tas_time_mean_rmse_norm = tas_time_mean_rmse
    tas_time_std_mae_norm = tas_time_std_mae

    tas_score = (
        metric_weights["tas"]["monthly_rmse"] * tas_monthly_rmse
        + metric_weights["tas"]["time_mean"] * tas_time_mean_rmse_norm
        + metric_weights["tas"]["time_std"] * tas_time_std_mae_norm
    )

    # Apply normalization and weighting for precipitation
    pr_monthly_rmse = pr_overall_rmse
    pr_time_mean_rmse_norm = pr_time_mean_rmse
    pr_time_std_mae_norm = pr_time_std_mae

    pr_score = (
        metric_weights["pr"]["monthly_rmse"] * pr_monthly_rmse
        + metric_weights["pr"]["time_mean"] * pr_time_mean_rmse_norm
        + metric_weights["pr"]["time_std"] * pr_time_std_mae_norm
    )

    var_weights = {"tas": 0.5, "pr": 0.5}
    xarray_score = var_weights["tas"] * tas_score + var_weights["pr"] * pr_score

    print(
        f"Temperature metrics - Overall RMSE: {tas_overall_rmse:.4f}, "
        f"Time-Mean RMSE: {tas_time_mean_rmse:.4f}, Time-Std MAE: {tas_time_std_mae:.4f}"
    )
    print(
        f"Precipitation metrics - Overall RMSE: {pr_overall_rmse:.4f}, "
        f"Time-Mean RMSE: {pr_time_mean_rmse:.4f}, Time-Std MAE: {pr_time_std_mae:.4f}"
    )
    print(f"Combined xarray score: {xarray_score:.6f}")

    # Convert to Kaggle format
    print("\nConverting to Kaggle format and calculating Kaggle metric...")

    # Stack the data into the expected shape (time, channels, y, x)
    true_stacked = np.stack([tas_true, pr_true], axis=1)
    pred_stacked = np.stack([tas_pred, pr_pred], axis=1)
    var_names = ["tas", "pr"]

    # Import the common conversion function
    from src.utils import convert_predictions_to_kaggle_format

    # Convert true values to Kaggle format
    solution = convert_predictions_to_kaggle_format(true_stacked, times, lats, lons, var_names)

    # Convert predicted values to Kaggle format
    submission = convert_predictions_to_kaggle_format(pred_stacked, times, lats, lons, var_names)

    # Calculate Kaggle score
    kaggle_score_val = kaggle_score(solution, submission, "ID")
    print(f"Kaggle score: {kaggle_score_val:.6f}")

    # Compare the two scores
    print("\nComparing scores:")
    print(f"XArray score: {xarray_score:.6f}")
    print(f"Kaggle score: {kaggle_score_val:.6f}")

    # Check if they're close (allowing for small numeric differences)
    difference = abs(xarray_score - kaggle_score_val)
    relative_difference = difference / xarray_score

    print(f"Absolute difference: {difference:.6f}")
    print(f"Relative difference: {relative_difference:.6f}")

    # Define threshold for acceptable difference (0.1%)
    threshold = 0.001

    if relative_difference < threshold:
        print("\n✅ PASS: The metrics are equivalent within tolerance.")
        return True
    else:
        print("\n❌ FAIL: The metrics differ by more than the tolerance threshold.")
        return False


if __name__ == "__main__":
    try:
        success = test_metric_equivalence()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error during test: {str(e)}")
        sys.exit(1)
