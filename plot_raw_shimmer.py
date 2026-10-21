# ./Experiment in Cognitive Science 02455/plot_raw_shimmer.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_shimmer_raw(csv_filepath, sampling_rate_hz: float = 512.0):
    csv_path = Path(csv_filepath)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    # 1. Read the first line to check for 'sep=' directive
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    skiprows = 1 if "sep=" in first_line else 0

    # 2. Read dataframe (Shimmer files are tab-separated)
    df = pd.read_csv(csv_path, sep="\t", skiprows=skiprows, low_memory=False)

    # Clean column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop the units row if present (row 0 contains 'ms', 'mV', etc.)
    if len(df) > 0 and any(u in str(df.iloc[0, 0]).lower() for u in ["ms", "sec", "mv", "cal", "no_units"]):
        df = df.iloc[1:].reset_index(drop=True)

    # 3. Identify Timestamp and EMG columns
    time_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    emg_cols = [c for c in df.columns if "emg" in c.lower() and "status" not in c.lower()]

    if not emg_cols:
        raise ValueError(f"Could not find EMG columns. Available columns: {list(df.columns)}")

    # 4. Generate time axis (handles offline -1 SD card timestamps gracefully)
    if time_col and time_col in df.columns:
        time_raw = pd.to_numeric(df[time_col], errors="coerce")
        if time_raw.notna().sum() > 0 and time_raw.median() > 0:
            if time_raw.median() > 1e10:  # milliseconds
                time_s = (time_raw - time_raw.iloc[0]) / 1000.0
            else:                         # seconds
                time_s = time_raw - time_raw.iloc[0]
        else:
            # Fallback: create time axis from sample index and sampling rate
            print(f"[INFO] Unix timestamps are unset (-1). Generating time axis assuming {sampling_rate_hz} Hz sampling rate.")
            time_s = pd.Series(range(len(df))) / sampling_rate_hz
    else:
        time_s = pd.Series(range(len(df))) / sampling_rate_hz

    # 5. Plotting
    fig, axes = plt.subplots(len(emg_cols), 1, figsize=(12, 3 * len(emg_cols)), sharex=True)
    if len(emg_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(emg_cols):
        emg_vals = pd.to_numeric(df[col], errors="coerce")
        axes[idx].plot(time_s, emg_vals, color="#0072B2" if idx == 0 else "#D55E00", lw=0.6)
        axes[idx].set_ylabel(f"{col}\n(mV)")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title(f"Raw Shimmer Signal: {col}")

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    target_file = "data/raw_shimmer/sub-E41CB9_Shimmer_9BF9_Calibrated_SD.csv"
    plot_shimmer_raw(target_file, sampling_rate_hz=512.0)