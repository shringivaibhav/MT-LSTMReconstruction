# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:44:44 2026

@author: Shringi Vaibhav (12235235)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

param_short = {
    "CH Torque [MNm]": "torque",
    "Thrust Force [kN]": "thrust",
    "CH Penetration [mm/rot]": "pen",
    "CH Rotation [rpm]": "rot"
}

file_params = {
    "torque.csv": "CH Torque [MNm]",
    "thrust.csv": "Thrust Force [kN]",
    "penetration.csv": "CH Penetration [mm/rot]",
    "rotation.csv": "CH Rotation [rpm]"
}

param_colors = {
    "CH Torque [MNm]": "#1f77b4",      # blue
    "Thrust Force [kN]": "#ff7f0e",      # orange
    "CH Penetration [mm/rot]": "#2ca02c", # green
    "CH Rotation [rpm]": "#d62728"     # red
}

region_colors = {
    1: "#2ca02c",  # green
    2: "#1f77b4",  # blue
    3: "#d62728"   # red
}

output_folder = "plots_custom"
os.makedirs(output_folder, exist_ok=True)

sns.set(style="whitegrid", palette="muted", font_scale=1.2)

def clean_filename(text):
    return re.sub(r'[^A-Za-z0-9_\-]', '_', text)

# --- Function to plot line graphs ---
def plot_line_regions(df, x_col, parameter_name, x_log=False, y_log=False, x_ticks=None, xlabel_override=None):
    
    plt.figure(figsize=(12,5))
    
    # Loop over regions
    for region in sorted(df['Location'].unique()):
        df_region = df[df['Location'] == region]
        
        grouped = df_region.groupby(x_col)['RMSE'].agg(['mean','std']).reset_index()
        
        plt.errorbar(
            grouped[x_col], grouped['mean'], yerr=grouped['std'],
            fmt='o-', capsize=5, markersize=6, linewidth=2,
            color=region_colors[region],
            label=f"Region {region}"
        )
    
    # Labels
    xlabel = xlabel_override if xlabel_override else x_col
    plt.xlabel(xlabel)
    plt.ylabel(f"{parameter_name} RMSE")
    plt.title(f"{parameter_name} RMSE vs {xlabel}")
    
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    if x_ticks:
        plt.xticks(x_ticks)
    
    plt.legend()
    plt.tight_layout()
    
    # Short x names
    x_short_map = {"Gap Length [m]": "gap", "Training Length [m]": "train", "Ratio": "ratio"}
    param_short_name = param_short[parameter_name]
    x_short = x_short_map.get(x_col, clean_filename(x_col))

    save_path = os.path.join(output_folder, f"{param_short_name}_{x_short}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()


# --- Loop over all parameter CSVs ---
for csv_file, param_name in file_params.items():
    print(f"Processing {param_name}...")
    df = pd.read_csv(csv_file)
    
    # --- Gap Length ---
    plot_line_regions(df, "Gap Length [m]", param_name, x_ticks=[5,10,15,20])
    
    # --- Training Length (log x) ---
    y_log_flag = param_name in ["CH Torque [MNm]", "Thrust Force [kN]"]
    plot_line_regions(df, "Training Length [m]", param_name, x_log=True, y_log=y_log_flag)
    
    # --- Ratio (log x) ---
    xlabel_eta = "$\\eta$"
    plot_line_regions(df, "Ratio", param_name, x_log=True, y_log=y_log_flag, xlabel_override=xlabel_eta)

print("All customized plots generated and saved in the 'plots_custom' folder.")