# -*- coding: utf-8 -*-
"""
Created on Tue May 05 13:28:15 2026

@author: Shringi Vaibhav (12235235)
"""

import numpy as np
import pandas as pd

# Selecting the values for the user parameters
df_final = pd.read_parquet("Follo_Final.parquet", engine = "pyarrow")
chainage_column = "Station_meter"
standstill_columns = ["CH Penetration [mm/rot]", "CH Torque [MNm]", "Thrust Force [kN]", "CH Rotation [rpm]"]
blocks = [(4126, 4140)] # gap length = b - a + 1 (initial value is inclusive)
    
# Defining computation of error metrics
def compute_errors(df_true, df_pred, blocks, cols, station_col):
    errors = {}
    for (a, b) in blocks:
        mask = (df_true[station_col] >= a) & (df_true[station_col] <= b)
        for col in cols:
            true_vals = df_true.loc[mask, col].values
            pred_vals = df_pred.loc[mask, col].values
            mae = np.mean(np.abs(true_vals - pred_vals))
            rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
            errors[col] = {"MAE": mae, "RMSE": rmse}
    return errors

# Linear interpolation baseline
def linear_interpolation_fill(df_original, blocks, target_cols, station_col):
    df_interp = df_original.copy()

    for (a, b) in blocks:
        mask = (df_interp[station_col] >= a) & (df_interp[station_col] <= b)

        # Remove values in gap (simulate missing data)
        for col in target_cols:
            df_interp.loc[mask, col] = np.nan

    # Perform linear interpolation
    df_interp[target_cols] = df_interp[target_cols].interpolate(method='linear')

    # Handle edge cases (if any NaNs remain)
    df_interp[target_cols] = df_interp[target_cols].fillna(method='bfill').fillna(method='ffill')

    return df_interp

# Linear interpolation baseline
df_linear = linear_interpolation_fill(
    df_original=df_final,
    blocks=blocks,
    target_cols=standstill_columns,
    station_col=chainage_column
)

# Compute errors for linear interpolation
errors_linear = compute_errors(df_final, df_linear, blocks, standstill_columns, chainage_column)

print("\nLinear Interpolation Errors:")
for k, v in errors_linear.items():
    print(f"{k}: MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}")