# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 13:44:07 2025

@author: Shringi Vaibhav (12235235)
"""

import re
import pywt
import os as os
import numpy as np
import glob as glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import IsolationForest
from scipy.interpolate import PchipInterpolator, interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# Defining the folder path where CSVs are stored
folder_path = os.path.join(os.getcwd(), "Follobanen/S980_All_CSV_files")

# Fetching all CSV file paths in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
tab_delimited_files = []

# Specifying the required columns
required_columns = ["Speed [mm/min]", "CH Penetration [mm/rot]", "CH Torque [MNm]", "Thrust Force [kN]",
                    "CH Rotation [rpm]", "Tunnel Station [m]", "Double Shield Mode [-]", "Scale Flow 01 [t/h]",
                    "Scale Flow 02 [t/h]", "Excav. Ton 01 [t]", "Excav. Ton 02 [t]", "TBM Roll 01 [deg]",
                    "TBM Incline 01 [deg]", "RingNr []", "Timestamp [ms]", "State []"]

# Defining the column representing chainage of excavation
chainage_column = ("Tunnel Station [m]")

# Creating an empty list to store DataFrames
df_list = []

# Iterating through each CSV file to append its content to the list of DataFrames
for file in csv_files:
    print(file.split("\\")[5].split("_")[2]) # The progress is shown for Ring Nr
    temp_df = pd.read_csv(file, delimiter = ';', decimal = '.', header = [0])  # Read the CSV         
    # Skipping the tab delimited files if the chainage column is missing
    if chainage_column not in temp_df.columns:
        print(f" Skipping {file}: 'Chainage' column missing")
        tab_delimited_files.append(file)
        continue

    # Filtering out the rows where chainage = 0
    temp_df = temp_df.loc[(temp_df[chainage_column] != 0) & 
                       (temp_df[chainage_column] != '\\N') & 
                       ~temp_df[chainage_column].isna()]

    # Filtering only required columns (based on the first header row)
    temp_df = temp_df.loc[:, temp_df.columns.get_level_values(0).isin(required_columns)]
    
    # Appending to df_list
    df_list.append(temp_df)
    
# Appending to the global DataFrame
global_df = pd.concat(df_list, ignore_index=True)  # Append with index reset

# Applying the filter to replace all NaN values with 999999
global_df.replace('\\N', np.nan, inplace=True)
global_df = global_df.apply(pd.to_numeric, errors='coerce')
Follo_df = global_df.sort_values(by='Tunnel Station [m]', ascending=False).reset_index(drop=True)

# Determining the min and max chainage values
min_chainage = Follo_df[chainage_column].min()
max_chainage = Follo_df[chainage_column].max()

# Defining the valid range of Follobanen project (excluding first 1000m and last 200m, testing length, reversed as excavation is on the other side)
valid_range = (Follo_df[chainage_column] >= min_chainage + 200) & (Follo_df[chainage_column] < max_chainage - 1000)

# Applying the filter for testing length
Follo_df = Follo_df.loc[valid_range]

# Displaying the global Follobanen DataFrame
print(Follo_df.head())

# Saving to a new Parquet file
Follo_df.to_parquet("Follo_raw.parquet", engine = "pyarrow", index=False)
df_orig = pd.read_parquet("Follo_raw.parquet", engine = "pyarrow")

# Defining the standstill columns
standstill_columns = ["CH Penetration [mm/rot]", "CH Torque [MNm]", "Thrust Force [kN]", "CH Rotation [rpm]"]

# Filtering out the standstill condition
for col in standstill_columns:
    df_orig = df_orig.loc[df_orig[str(col)] != 0]

# Resetting the index after the operation
df_orig.reset_index(drop=True, inplace=True)

df_clean = df_orig.copy()

# Defining function for Wavelet Transform Filtering
def wavelet_denoise(signal, wavelet='db4', level=3, thresholding='soft'):
    # Decompose the signal using wavelet transform
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise standard deviation (sigma) for thresholding
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Median absolute deviation
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))  # Universal threshold
    
    # Apply thresholding to all detail coefficients (skip the first approximation)
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode=thresholding) for c in coeffs[1:]]
    
    # Reconstruct the signal from the thresholded coefficients
    denoised = pywt.waverec(coeffs, wavelet)
    
    # Trim or pad to match the original length
    return denoised[:len(signal)]

df_denoised = df_clean.copy()

# Applying denoising on the standstill columns
for col in standstill_columns:
    print(f"Applying wavelet denoising on {col}...")
    # Interpolating the NaNs in the signal
    signal = df_clean[str(col)].astype(float).copy()
    signal_interpolated = signal.interpolate(method = "linear", limit_direction = "both")
    denoised = wavelet_denoise(signal_interpolated.values, wavelet='db4', level=3)
    df_denoised[str(col)] = denoised[:len(df_clean)]  # Ensure lengths match

# Plotting denoised signal against raw signal for standstill columns
for col in standstill_columns:
    col_to_plot_safename = re.sub(r'[<>:"/\\|?*]', '_', str(col))  # Replaces invalid characters with '_'
    plt.figure(figsize=(10,6))
    plt.plot(df_clean[str(col)].iloc[:10000], label="Original", alpha=0.5)
    plt.plot(df_denoised[str(col)].iloc[:10000], label="Denoised", linewidth=2)
    plt.title(f"Wavelet Denoising - {str(col)}")
    plt.xlabel('Row number in the dataset')
    plt.ylabel(str(col))
    plt.legend()
    plt.savefig(f'Figures\\{col_to_plot_safename}_plotwvredsst.png')  # Saves with the column name in the file name

# Saving to a new Parquet file
df_denoised.to_parquet("Follo_denoised.parquet", engine = "pyarrow", index=False)
df_denoised = pd.read_parquet("Follo_denoised.parquet", engine = "pyarrow")

df_task2 = df_denoised.copy()

# Defining the statistical function used for resampling of standstill parameters
agg_rules = {str(standstill_columns[0]):"median",
             str(standstill_columns[1]):"max",
             str(standstill_columns[2]):"median",
             str(standstill_columns[3]):"mean"}

for col in df_denoised.columns:
    if col not in agg_rules and col != str(chainage_column):
        agg_rules[col] = "median"

# Applying the selected functions
df_task2["Station_meter"] = df_task2[str(chainage_column)].round().astype(int)
df_task2 = df_task2.groupby("Station_meter").agg(agg_rules).reset_index()
print(f"Original rows: {len(df_denoised)}, After resampling: {len(df_task2)}")

# Comaparing plots before and after per meter resampling
for col in standstill_columns:
    col_to_plot_safename = re.sub(r'[<>:"/\\|?*]', '_', str(col))  # Replaces invalid characters with '_'
    plt.figure(figsize=(10,6))
    plt.plot(df_denoised[str(chainage_column)], df_denoised[str(col)], '.', alpha=0.3, label="Original (per ~5s)")
    plt.plot(df_task2["Station_meter"], df_task2[str(col)], '-', label="Resampled (1/meter)")
    plt.xlabel("Chainage (m)")
    plt.ylabel(str(col))
    plt.title(f"Resampling per Tunnel Meter: {col}")
    plt.legend()
    plt.savefig(f'Figures\\final{col_to_plot_safename}_plot.png')  # Saves with the column name in the file name

# Saving to a new Parquet file
df_task2.to_parquet("Follo_task2.parquet", engine = "pyarrow", index=False)
df_task2 = pd.read_parquet("Follo_task2.parquet", engine = "pyarrow")

# Redefining chainage column (updated dataframes only contain unique values of chainage)
chainage_column = "Station_meter"

# Defining parameters for Hybrid equal data point spacing method
# Gap thresholds (in meters)
gap_threshold_linear = 20    # if gaps <= 20 m -> use linear/pchip
gap_threshold_gpr = 50       # if gaps >= 50 m -> consider GPR (if hybrid)

use_gpr = False              # set True if you want to enable GPR fallback
gpr_length_scale = 50.0
gpr_noise_level = 1.0
imp_cols = [str(sst_col) for sst_col in standstill_columns]

# Preparing observed (after resampling) grid
df_obs = df_task2[[str(chainage_column)] + imp_cols].copy()
df_obs = df_obs.drop_duplicates(subset=str(chainage_column))   # ensure unique station rows
df_obs = df_obs.sort_values(str(chainage_column))

station_min = int(df_obs[str(chainage_column)].min())
station_max = int(df_obs[str(chainage_column)].max())
station_grid = np.arange(station_min, station_max + 1)  # full desired spacing

# Initializing new dataframe with NaNs
df_interp = pd.DataFrame({str(chainage_column): station_grid})
df_interp.set_index(str(chainage_column), inplace=True)

X_obs = df_obs[str(chainage_column)].values

# interpolating only missing meters for each standstill columns, preserving resampled ones
for col in standstill_columns:
    print(f"Processing column: {str(col)}")
    y_obs = df_obs[str(col)].values

    # Creating array aligned with station_grid, fill observed positions
    series_pred = pd.Series(index=station_grid, data=np.nan)

    # Mapping observed values
    observed_map = dict(zip(X_obs, y_obs))
    for x, val in observed_map.items():
        series_pred[int(x)] = val

    # Finding indices of missing segments (consecutive runs of NaN)
    isnan = series_pred.isna().values
    # Finding start-end indices of each NaN run
    nan_runs = []
    i = 0
    N = len(station_grid)
    while i < N:
        if isnan[i]:
            start = i
            while i < N and isnan[i]:
                i += 1
            end = i - 1
            nan_runs.append((start, end))
        else:
            i += 1

    # Continuing, if there are no missing values
    if not nan_runs:
        df_interp[str(col)] = series_pred.values
        continue

    # Preparing arrays for interpolation functions using sorted non-NaN observations
    x_known = np.array(list(observed_map.keys()))
    y_known = np.array(list(observed_map.values()))
    order = np.argsort(x_known)
    x_known = x_known[order]
    y_known = y_known[order]

    # Interpolating requires at least two known points
    if len(x_known) >= 2:
        linear_interp = interp1d(x_known, y_known, kind='linear', bounds_error=False, fill_value="extrapolate")
        try:
            pchip_interp = PchipInterpolator(x_known, y_known, extrapolate=True)
        except Exception:
            pchip_interp = None
    else:
        linear_interp = None
        pchip_interp = None

    # If using GPR, preparing model fitted on known points (only if enabled and enough data)
    if use_gpr and len(x_known) >= 5:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=gpr_length_scale, length_scale_bounds=(1.0, 1e3)) \
                 + WhiteKernel(noise_level=gpr_noise_level, noise_level_bounds=(1e-6, 1e3))
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None)
        try:
            gpr.fit(x_known.reshape(-1, 1), y_known)
            gpr_ready = True
        except Exception as e:
            print("  GPR fit failed:", e)
            gpr_ready = False
    else:
        gpr_ready = False

    # Filling each missing run
    for (start_i, end_i) in nan_runs:
        start_meter = station_grid[start_i]
        end_meter = station_grid[end_i]
        gap_length = end_meter - start_meter + 1

        # Determining neighbors for the gap
        left_idx = start_i - 1
        right_idx = end_i + 1
        left_x = station_grid[left_idx] if left_idx >= 0 else None
        right_x = station_grid[right_idx] if right_idx < N else None

        if gap_length <= gap_threshold_linear and linear_interp is not None:
            # Using PCHIP if available else linear
            xs = np.arange(start_meter, end_meter + 1)
            if pchip_interp is not None:
                series_pred.loc[start_meter:end_meter] = pchip_interp(xs)
            else:
                series_pred.loc[start_meter:end_meter] = linear_interp(xs)
        elif use_gpr and gpr_ready and gap_length >= gap_threshold_gpr:
            # Using GPR to predict for the whole span (prefer local model)
            xs = np.arange(start_meter, end_meter + 1).reshape(-1, 1)
            y_pred, sigma = gpr.predict(xs, return_std=True)
            series_pred.loc[start_meter:end_meter] = y_pred
        else:
            # Falling back: linear if possible, else nearest neighbor fill
            xs = np.arange(start_meter, end_meter + 1)
            if linear_interp is not None:
                series_pred.loc[start_meter:end_meter] = linear_interp(xs)
            else:
                # Nearest: Filling with left or right neighbor if available
                fill_val = None
                if left_x is not None and not np.isnan(series_pred[left_x]):
                    fill_val = series_pred[left_x]
                elif right_x is not None and not np.isnan(series_pred[right_x]):
                    fill_val = series_pred[right_x]
                series_pred.loc[start_meter:end_meter] = fill_val

    # Saving into df_interp dataframe
    df_interp[str(col)] = series_pred.values

# Resetting index to column
df_interp = df_interp.reset_index().rename(columns={'index': str(chainage_column)})
print("Interpolation finished. Result shape:", df_interp.shape)

# Comparing plot before and after hybrid data point spacing
for col in standstill_columns:
    col_to_plot_safename = re.sub(r'[<>:"/\\|?*]', '_', str(col))  # Replaces invalid characters with '_'
    plt.figure(figsize=(10,6))
    plt.scatter(df_obs[str(chainage_column)], df_obs[str(col)], s=8, alpha=0.4, label="Observed (1 row/tunnel meter)")
    plt.plot(df_interp[str(chainage_column)], df_interp[str(col)], color='red', linewidth=0.8, label="Interpolated")
    plt.xlabel("Chainage (m)")
    plt.ylabel(str(col))
    plt.title(f"Comparison: observed vs interpolated for {str(col)}")
    plt.legend()
    plt.savefig(f'Figures\\final_{col_to_plot_safename}_plot.png')  # Saves with the column name in the file name

# Saving to a new Parquet file
df_interp.to_parquet("Follo_Final.parquet", engine = "pyarrow", index=False)
df_interp = pd.read_parquet("Follo_Final.parquet", engine = "pyarrow")
