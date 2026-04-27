# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:46:19 2026

@author: Shringi Vaibhav (12235235)
"""

import re
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import List, Tuple

# Selecting the values for the user parameters
df_final = pd.read_parquet("Follo_Final.parquet", engine = "pyarrow")
chainage_column = "Station_meter"
standstill_columns = ["CH Penetration [mm/rot]", "CH Torque [MNm]", "Thrust Force [kN]", "CH Rotation [rpm]"]
blocks = [(2666, 2685)] # gap length = b - a + 1 (initial value is inclusive)

# Selecting values for model hyperparameters
training_length = None # None: full length of tunnel acts as training dataset
input_len = 30
assert len(blocks) == 1, "Only one block supported for now"
block_start, block_end = blocks[0]
output_len = block_end - block_start + 1
hidden_size = 128
num_layers = 1
dropout = 0.1
batch_size = 64
epochs = 40
lr = 1e-4
weight_decay = 1e-6
lambda_smooth = 0.5   # can tune
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multivariate Seq2Seq LSTM reconstruction
class Seq2SeqLSTM_MV(nn.Module):
    def __init__(self, input_size, hidden_size, output_len, n_targets, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_len * n_targets)
        self.output_len = output_len
        self.n_targets = n_targets

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1]  # (batch, hidden)
        out = self.fc(h_last)  # (batch, output_len * n_targets)
        out = out.view(-1, self.output_len, self.n_targets)  # (batch, output_len, n_targets)
        return out

# Defining smoothness loss function
def smoothness_loss_mv(pred: torch.Tensor, true: torch.Tensor, lambda_smooth: float):
    mse = nn.MSELoss()(pred, true)
    if pred.size(1) >= 2:
        diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
        diff_true = true[:, 1:, :] - true[:, :-1, :]
        smooth_mse = nn.MSELoss()(diff_pred, diff_true)
    else:
        smooth_mse = torch.tensor(0.0, device=pred.device)
    return mse + lambda_smooth * smooth_mse

# Defining building sequences excluding function
def build_sequences_outside_blocks_mv(df: pd.DataFrame, cols: List[str], target_cols: List[str],
                                      blocks: List[Tuple[int,int]], input_len: int, output_len: int,
                                      station_col: str, training_length=None):
    n = len(df)
    stations = df[station_col].values
    allowed = np.ones(n, dtype=bool)
    if training_length is not None:
        allowed[:] = False
        half = training_length / 2
        for (a, b) in blocks:
            allowed |= (stations >= a - half) & (stations <= b + half)
    in_block = np.zeros(n, dtype=bool)
    for (a,b) in blocks:
        in_block |= (stations >= a) & (stations <= b)

    X_vals = df[cols].values
    Y_vals = df[target_cols].values  # shape (n, n_targets)

    X_list, Y_list = [], []
    max_start = n - input_len - output_len + 1
    for i in range(max_start):
        if training_length is not None:
            # center_station = stations[i + input_len // 2]
            if not allowed[i + input_len // 2]:
                continue
        window_idx = np.arange(i, i + input_len + output_len)
        if in_block[window_idx].any():
            continue
        if np.isnan(X_vals[i:i+input_len]).any() or np.isnan(Y_vals[i+input_len:i+input_len+output_len]).any():
            continue
        X_list.append(X_vals[i:i+input_len])
        Y_list.append(Y_vals[i+input_len:i+input_len+output_len])

    if len(X_list) == 0:
        raise RuntimeError("No training sequences available - adjust blocks/input_len/output_len or check data coverage.")

    X_arr = np.stack(X_list).astype(np.float32)
    Y_arr = np.stack(Y_list).astype(np.float32)
    return X_arr, Y_arr

# Defining helper function
def train_model_on_df_mv(df: pd.DataFrame, cols: List[str], target_cols: List[str], blocks: List[Tuple[int,int]],
                         input_len: int, output_len: int, hidden_size: int, num_layers: int, dropout: float,
                         batch_size: int, epochs: int, lr: float, weight_decay: float, lambda_smooth: float,
                         station_col: str, device: torch.device):
    X_arr, Y_arr = build_sequences_outside_blocks_mv(df, cols, target_cols, blocks, input_len, output_len, station_col, training_length=training_length)
    nsamples, _, n_features = X_arr.shape
    _, out_len, n_targets = Y_arr.shape
    print(f"Training arrays X:{X_arr.shape} Y:{Y_arr.shape}")

    # Flattening and fitting scalers
    X_flat = X_arr.reshape(-1, n_features)
    Y_flat = Y_arr.reshape(-1, n_targets)

    scaler_X = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    scaler_X.fit(X_flat)
    scaler_y.fit(Y_flat)
    print("Scalers fitted:", "X features:", scaler_X.n_features_in_, "Y features:", scaler_y.n_features_in_)

    # Transforming and reshaping
    X_scaled = scaler_X.transform(X_flat).reshape(nsamples, input_len, n_features).astype(np.float32)
    Y_scaled = scaler_y.transform(Y_flat).reshape(nsamples, output_len, n_targets).astype(np.float32)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_scaled), torch.tensor(Y_scaled))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = Seq2SeqLSTM_MV(input_size=n_features, hidden_size=hidden_size, output_len=output_len, n_targets=n_targets,
                           num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training on Adam optimizer
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for Xb, Yb in train_loader:
            Xb = Xb.to(device).float()
            Yb = Yb.to(device).float()
            optimizer.zero_grad()
            out = model(Xb)  # (B, output_len, n_targets)
            loss = smoothness_loss_mv(out, Yb, lambda_smooth)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss at epoch {epoch}. out min/max {out.min().item()}/{out.max().item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} avg loss: {np.mean(losses):.6e}")

    return model, scaler_X, scaler_y

# Defining chunkwise multivariate refilling of block
def fill_blocks_chunkwise_mv(df_original: pd.DataFrame, model: nn.Module, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler,
                             cols: List[str], target_cols: List[str], blocks: List[Tuple[int,int]],
                             input_len: int, output_len: int, station_col: str, device: torch.device):
    df = df_original.copy()
    stations = df[station_col].values
    model.eval()

    with torch.no_grad():
        for (start_s, end_s) in blocks:
            mask_block = (stations >= start_s) & (stations <= end_s)
            indices = np.where(mask_block)[0]
            if indices.size == 0:
                raise RuntimeError(f"Block {start_s}-{end_s} out of station range.")
            start_idx = int(indices.min()); end_idx = int(indices.max())
            block_len = end_idx - start_idx + 1
            print(f"Filling block {start_s}-{end_s}: idx {start_idx}..{end_idx}, len {block_len}")

            offset = 0
            while offset < block_len:
                chunk_len = min(output_len, block_len - offset)
                context_end = start_idx + offset - 1
                context_start = context_end - input_len + 1
                if context_start < 0:
                    raise RuntimeError("Not enough history to predict first chunk. Increase input_len or move block downwards.")

                seq_raw = df.loc[context_start:context_end, cols].values
                if np.isnan(seq_raw).any():
                    seq_df = pd.DataFrame(seq_raw, columns=cols)
                    seq_df = seq_df.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
                    seq_raw = seq_df.values

                if seq_raw.shape[1] != scaler_X.n_features_in_:
                    raise RuntimeError(f"Feature mismatch: seq has {seq_raw.shape[1]} columns but scaler expects {scaler_X.n_features_in_}")

                seq_scaled = scaler_X.transform(seq_raw).astype(np.float32)
                seq_tensor = torch.tensor(seq_scaled).unsqueeze(0).to(device)
                pred_scaled = model(seq_tensor).cpu().numpy().squeeze(axis=0)
                pred_chunk_scaled = pred_scaled[:chunk_len, :]
                pred_chunk = scaler_y.inverse_transform(pred_chunk_scaled)

                # Writing back into dataframe for each target column
                write_idx = np.arange(start_idx + offset, start_idx + offset + chunk_len)
                for j, colname in enumerate(target_cols):
                    df.loc[write_idx, colname] = pred_chunk[:, j]
                offset += chunk_len
    return df

# Defining reverse dataframe function for backward model
def reverse_df_time(df: pd.DataFrame, station_col: str):
    return df.sort_values(by=station_col, ascending=False).reset_index(drop=True).copy()

# Defining weighted average reconstruction model for the block
def weighted_average_forward_backward_mv(df_forward, df_backward, blocks, station_col, target_cols):
    df_avg = df_forward.copy()
    for (a,b) in blocks:
        mask_f = (df_forward[station_col] >= a) & (df_forward[station_col] <= b)
        stations_f = df_forward.loc[mask_f, station_col].values
        mask_b = (df_backward[station_col] >= a) & (df_backward[station_col] <= b)
        stations_b = df_backward.loc[mask_b, station_col].values
        stations_union = np.unique(np.concatenate([stations_f, stations_b]))
        if stations_union.size == 0:
            raise RuntimeError(f"No stations found in block {a}-{b}")

        f_series = df_forward.set_index(station_col)[target_cols].reindex(stations_union)
        b_series = df_backward.set_index(station_col)[target_cols].reindex(stations_union)

        f_series = f_series.fillna(method='ffill').fillna(method='bfill')
        b_series = b_series.fillna(method='ffill').fillna(method='bfill')
        if f_series.isna().any().any() or b_series.isna().any().any():
            raise RuntimeError(f"Remaining NaNs when aligning forward/backward for block {a}-{b}")

        n = len(stations_union)
        if n == 1:
            w_f = np.array([0.5]); w_b = 1.0 - w_f
        else:
            stations_sorted_idx = np.argsort(stations_union)
            pos = np.arange(n)
            w_f_sorted = (n - 1 - pos) / (n - 1)
            w_b_sorted = 1.0 - w_f_sorted
            w_f = np.empty_like(w_f_sorted, dtype=float)
            w_b = np.empty_like(w_b_sorted, dtype=float)
            w_f[stations_sorted_idx] = w_f_sorted
            w_b[stations_sorted_idx] = w_b_sorted

        f_vals = f_series.values.astype(float)
        b_vals = b_series.values.astype(float)
        avg_vals = (w_f[:,None] * f_vals) + (w_b[:,None] * b_vals)

        for i, station in enumerate(stations_union):
            for j, colname in enumerate(target_cols):
                df_avg.loc[df_avg[station_col] == station, colname] = avg_vals[i, j]

    return df_avg

# Defining the bidirectional pipeline
def forward_backward_fill_pipeline_mv(df_orig: pd.DataFrame, cols: List[str], target_cols: List[str], blocks: List[Tuple[int,int]],
                                      input_len, output_len, hidden_size, num_layers, dropout,
                                      batch_size, epochs, lr, weight_decay, lambda_smooth,
                                      station_col, device, training_length=None):
    df = df_orig.sort_values(by=station_col).reset_index(drop=True).copy()
    for c in target_cols:
        df[f"Original {c}"] = df[c].copy()

    # Forward model
    f_model, f_scaler_X, f_scaler_y = train_model_on_df_mv(
        df=df, cols=cols, target_cols=target_cols, blocks=blocks,
        input_len=input_len, output_len=output_len, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout, batch_size=batch_size, epochs=epochs,
        lr=lr, weight_decay=weight_decay, lambda_smooth=lambda_smooth,
        station_col=station_col, device=device
    )
    df_forward = fill_blocks_chunkwise_mv(df, f_model, f_scaler_X, f_scaler_y, cols, target_cols, blocks, input_len, output_len, station_col, device)

    # Backward model (trained on reversed dataframe)
    df_rev = reverse_df_time(df, station_col)
    b_model, b_scaler_X, b_scaler_y = train_model_on_df_mv(
        df=df_rev, cols=cols, target_cols=target_cols, blocks=blocks,
        input_len=input_len, output_len=output_len, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout, batch_size=batch_size, epochs=epochs,
        lr=lr, weight_decay=weight_decay, lambda_smooth=lambda_smooth,
        station_col=station_col, device=device
    )
    df_rev_filled = fill_blocks_chunkwise_mv(df_rev, b_model, b_scaler_X, b_scaler_y, cols, target_cols, blocks, input_len, output_len, station_col, device)
    df_backward = df_rev_filled.sort_values(by=station_col, ascending=True).reset_index(drop=True).copy()

    df_avg = weighted_average_forward_backward_mv(df_forward, df_backward, blocks, station_col, target_cols)
    return df_forward, df_backward, df_avg, f_model, b_model

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

# Defining helper function for clean file names
def clean_filename(text):
    return re.sub(r'[^A-Za-z0-9_\-]', '_', text)

# Defining function to plot and save the reconstruction curves
def plot_and_save(df_orig, df_fwd, df_bwd, df_avg, blocks, target, station_col, save_path, title):
    df0 = df_orig.copy()
    for (a,b) in blocks:
        stations = df0[station_col].values
        start_idx = np.where(stations >= a)[0][0]
        end_idx = np.where(stations <= b)[0][-1]
        block_len = end_idx - start_idx + 1
        plot_start = max(0, start_idx - block_len)
        plot_end = min(len(df0)-1, end_idx + block_len)
        x = df0.loc[plot_start:plot_end, station_col].values
        plt.figure(figsize=(12,5))
        plt.plot(x, df0.loc[plot_start:plot_end, target].values, label="Original")
        plt.plot(x, df_fwd.loc[plot_start:plot_end, target].values, label="Forward")
        plt.plot(x, df_bwd.loc[plot_start:plot_end, target].values, label="Backward")
        plt.plot(x, df_avg.loc[plot_start:plot_end, target].values, linestyle='--', label="Weighted")
        plt.axvspan(a,b, alpha=0.2)
        plt.legend()
        plt.xlabel(station_col)
        plt.ylabel(target)
        plt.title(title)
        filename = f"{save_path}_{clean_filename(target)}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

# Running the pipeline on TBM dataset
if __name__ == "__main__":

    # Determing the parameter values
    gap_len = output_len
    train_len = training_length if training_length is not None else 8000
    ratio = round(train_len / gap_len, 2)

    experiment_name = f"Train{train_len}_Gap{gap_len}_Ratio{ratio}"

    print(f"\nRunning Experiment: {experiment_name}\n")

    os.makedirs("results", exist_ok=True)

    # Running bidirectional model for reconstruction
    df_fwd, df_bwd, df_avg, f_model, b_model = forward_backward_fill_pipeline_mv(
        df_orig=df_final,
        cols=standstill_columns,
        target_cols=standstill_columns,
        blocks=blocks,
        input_len=input_len,
        output_len=output_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_smooth=lambda_smooth,
        station_col=chainage_column,
        device=device
    )

    # Error metrics being computed
    errors = compute_errors(df_final, df_avg, blocks, standstill_columns, chainage_column)

    # Printing errors as output
    print("Errors:")
    for k, v in errors.items():
        print(f"{k}: MAE={v['MAE']:.4f}, RMSE={v['RMSE']:.4f}")

    # Exporting errors in a CSV file
    results_list = []
    for param, vals in errors.items():
        results_list.append({
            "Experiment": experiment_name,
            "Parameter": param,
            "Training Length": train_len,
            "Gap Length": gap_len,
            "Ratio": ratio,
            "MAE": vals["MAE"],
            "RMSE": vals["RMSE"]
        })

    results_df = pd.DataFrame(results_list)
    results_file = "results/results_summary.csv"
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)

    # Saving reconstruction plots
    for param in standstill_columns:
        plot_and_save(
            df_final,
            df_fwd,
            df_bwd,
            df_avg,
            blocks,
            param,
            chainage_column,
            save_path=f"results/{experiment_name}",
            title=experiment_name
        )

    print(f"\nSaved results for {experiment_name}\n")