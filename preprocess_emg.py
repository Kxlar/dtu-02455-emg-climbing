# ./Experiment in Cognitive Science 02455/preprocess_emg.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

import config


def load_shimmer_file(filepath: Path, default_fs: float = config.DEFAULT_SAMPLING_RATE):
    """
    Loads a Shimmer Consensys SD-card CSV/TSV file matching the exact structure of:
      "sep=\\t"
      Shimmer_9BF9_Timestamp_Unix_CAL  ...  Shimmer_9BF9_EMG_CH1_16BIT_CAL  Shimmer_9BF9_EMG_CH2_16BIT_CAL
      ms                               ...  mV                              mV
      -1                               ...  -19.07915                       -1.95099
    Extracts:
      - sample_idx (0 .. N-1)
      - timestamp_raw (-1 if offline SD mode, or Unix ms if docked/synced)
      - emg_ch1 (FDS: Flexor Digitorum Superficialis)
      - emg_ch2 (ECR: Extensor Carpi Radialis)
      - accel_mag (3-axis accelerometer magnitude for movement/tap inspection)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Shimmer file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    skiprows = 1 if "sep=" in first_line else 0
    df_raw = pd.read_csv(filepath, sep="\t", skiprows=skiprows, low_memory=False)

    # Drop empty trailing columns (Shimmer often exports a trailing tab creating 'Unnamed: 8')
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("^Unnamed")]
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # Drop the units row (row 0 contains 'ms', 'm/(s^2)', 'mV', 'no_units')
    if len(df_raw) > 0 and any(u in str(df_raw.iloc[0, 0]).lower() for u in ["ms", "sec", "mv", "no_units"]):
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

    # 1. Identify Timestamp Column
    time_col = next((c for c in df_raw.columns if "timestamp" in c.lower()), None)

    # 2. Identify CH1 (FDS) and CH2 (ECR) EMG Columns
    ch1_col = next(
        (c for c in df_raw.columns if any(k.lower() in c.lower() for k in config.EMG_CHANNEL_MAP["ch1"]["col_keywords"])),
        None
    )
    ch2_col = next(
        (c for c in df_raw.columns if any(k.lower() in c.lower() for k in config.EMG_CHANNEL_MAP["ch2"]["col_keywords"])),
        None
    )

    if ch1_col is None and ch2_col is None:
        raise ValueError(
            f"Could not find EMG_CH1 or EMG_CH2 columns in {filepath.name}.\n"
            f"Available columns: {list(df_raw.columns)}"
        )

    clean_df = pd.DataFrame()
    if time_col:
        clean_df["timestamp_raw"] = pd.to_numeric(df_raw[time_col], errors="coerce").fillna(-1.0)
    else:
        clean_df["timestamp_raw"] = -1.0

    if ch1_col:
        clean_df["emg_ch1"] = pd.to_numeric(df_raw[ch1_col], errors="coerce")
    if ch2_col:
        clean_df["emg_ch2"] = pd.to_numeric(df_raw[ch2_col], errors="coerce")

    # 3. Extract 3-Axis Low-Noise Accelerometer Magnitude (if present)
    acc_cols = [c for c in df_raw.columns if "accel" in c.lower()]
    if len(acc_cols) >= 3:
        ax = pd.to_numeric(df_raw[acc_cols[0]], errors="coerce").fillna(0.0)
        ay = pd.to_numeric(df_raw[acc_cols[1]], errors="coerce").fillna(0.0)
        az = pd.to_numeric(df_raw[acc_cols[2]], errors="coerce").fillna(0.0)
        clean_df["accel_mag"] = np.sqrt(ax**2 + ay**2 + az**2)

    emg_present = [c for c in ["emg_ch1", "emg_ch2"] if c in clean_df.columns]
    clean_df = clean_df.dropna(subset=emg_present).reset_index(drop=True)
    clean_df["sample_idx"] = np.arange(len(clean_df), dtype=np.int64)

    # 4. Check if timestamps are offline (-1) or valid Unix timestamps (> 0)
    valid_ts_mask = clean_df["timestamp_raw"] > 0
    if valid_ts_mask.sum() > len(clean_df) * 0.5:
        offline_sd_mode = False
        med_ts = clean_df.loc[valid_ts_mask, "timestamp_raw"].median()
        clean_df["timestamp_unix_ms"] = (
            clean_df["timestamp_raw"] * 1000.0 if med_ts < 1e11 else clean_df["timestamp_raw"]
        )
        dt_ms = np.median(np.diff(clean_df["timestamp_unix_ms"].values))
        fs_est = round(1000.0 / dt_ms, 2) if dt_ms > 0 else float(default_fs)
    else:
        offline_sd_mode = True
        fs_est = float(default_fs)
        clean_df["timestamp_unix_ms"] = -1.0

    print(f"[SHIMMER LOADED] {filepath.name}")
    print(f"  -> CH1 (FDS) Col : {ch1_col}")
    print(f"  -> CH2 (ECR) Col : {ch2_col}")
    print(f"  -> Total Samples : {len(clean_df):,} ({len(clean_df)/fs_est:.1f}s @ {fs_est} Hz)")
    print(f"  -> Clock Mode    : {'OFFLINE SD (-1 timestamps -> Squeeze Sync)' if offline_sd_mode else 'UNIX TIMESTAMP'}")

    return clean_df, fs_est, offline_sd_mode


def compute_combined_squeeze_snr(shimmer_df: pd.DataFrame, fs: float) -> np.ndarray:
    """
    Computes a combined Signal-to-Noise Ratio (SNR) burst envelope across CH1 (FDS) and CH2 (ECR)
    to reliably locate the high-amplitude calibration fist squeezes (t=0 Start Squeeze & End Squeeze).
    Steps:
      1. 20 Hz High-Pass Butterworth filter (removes the -19 mV / -2 mV DC offsets & drift)
      2. 200 ms Moving RMS window
      3. Robust Z-score against the recording's median resting noise floor (MAD)
    """
    sos_hp = signal.butter(4, 20.0, btype="highpass", fs=fs, output="sos")
    win_samples = max(int(0.20 * fs), 10)
    kernel = np.ones(win_samples) / win_samples

    combined_snr = np.zeros(len(shimmer_df), dtype=np.float64)
    n_chs = 0

    for col in ["emg_ch1", "emg_ch2"]:
        if col in shimmer_df.columns:
            raw = shimmer_df[col].values.astype(np.float64)
            hp = signal.sosfiltfilt(sos_hp, raw - np.mean(raw))
            rms = np.sqrt(np.convolve(hp**2, kernel, mode="same"))
            med = np.median(rms)
            robust_sd = max(np.median(np.abs(rms - med)) * 1.4826, 1e-6)
            combined_snr += (rms - med) / robust_sd
            n_chs += 1

    return combined_snr / max(n_chs, 1)


def detect_squeeze_anchor_and_fs(
    shimmer_df: pd.DataFrame,
    events_df: pd.DataFrame,
    initial_fs: float,
    manual_t0_sec: float = None,
    manual_tend_sec: float = None
) -> tuple[int, float]:
    """
    Locates the exact sample index of:
      1. The Start Squeeze (t = 0.0s) in the first 35% of the Shimmer recording
      2. The End Squeeze ('sync_end') in the last 30% of the Shimmer recording
    and computes the exact empirical sampling rate `calibrated_fs` so that even a 5-10%
    difference between nominal fs (e.g. 2048 Hz) and true hardware fs (~1954-2000 Hz)
    is 100% eliminated across all trials.
    """
    env_snr = compute_combined_squeeze_snr(shimmer_df, fs=initial_fs)
    n_total = len(env_snr)
    reaction_comp_samples = int(0.25 * initial_fs)

    # --- 1. LOCATE START SQUEEZE (t = 0.0s) ---
    if manual_t0_sec is not None:
        t0_anchor_idx = int(round(manual_t0_sec * initial_fs))
        print(f"  [MANUAL START OVERRIDE] Using t=0 at {manual_t0_sec:.2f}s (Sample Index = {t0_anchor_idx:,})")
    else:
        search_end_idx = min(int(120.0 * initial_fs), int(n_total * 0.35))
        search_end_idx = max(search_end_idx, min(n_total, int(10.0 * initial_fs)))
        early_env = env_snr[:search_end_idx]

        max_early = float(np.max(early_env))
        thresh_peak = max(max_early * 0.35, 6.0)
        peaks, _ = signal.find_peaks(early_env, height=thresh_peak, distance=int(1.0 * initial_fs))

        first_peak_idx = int(peaks[0]) if len(peaks) > 0 else int(np.argmax(early_env))

        peak_height = env_snr[first_peak_idx]
        onset_threshold = max(peak_height * 0.20, 3.0)
        lookback_start = max(0, first_peak_idx - int(1.2 * initial_fs))
        pre_segment = env_snr[lookback_start:first_peak_idx]
        below_indices = np.where(pre_segment <= onset_threshold)[0]

        if len(below_indices) > 0:
            start_squeeze_idx = lookback_start + int(below_indices[-1])
        else:
            start_squeeze_idx = max(0, first_peak_idx - int(0.35 * initial_fs))

        t0_anchor_idx = max(0, start_squeeze_idx - reaction_comp_samples)
        print(
            f"  [START SQUEEZE DETECTED] Rising edge at idx {start_squeeze_idx:,} "
            f"({start_squeeze_idx / initial_fs:.2f}s @ nominal fs) -> Screen t=0 anchor = idx {t0_anchor_idx:,}"
        )

    # --- 2. LOCATE END SQUEEZE ('sync_end') & RECALIBRATE SAMPLING RATE ---
    calibrated_fs = float(initial_fs)
    end_rows = events_df[events_df["event_type"] == "sync_end"]

    if not end_rows.empty and "rel_from_sync_onset_s" in end_rows.columns:
        expected_rel_end_s = float(end_rows.iloc[0]["rel_from_sync_onset_s"])

        if manual_tend_sec is not None:
            end_anchor_idx = int(round(manual_tend_sec * initial_fs))
            calibrated_fs = (end_anchor_idx - t0_anchor_idx) / expected_rel_end_s
            print(
                f"  [MANUAL END OVERRIDE] End Squeeze at {manual_tend_sec:.2f}s (idx {end_anchor_idx:,}) "
                f"-> Calibrated Shimmer fs = {calibrated_fs:.3f} Hz"
            )
            return t0_anchor_idx, calibrated_fs

        # Search the final 30% of the recording (and allow +/- 15% sampling rate tolerance around expected_end_idx)
        expected_end_idx = t0_anchor_idx + int(round(expected_rel_end_s * initial_fs))
        tol_samples = max(int(45.0 * initial_fs), int(0.15 * expected_rel_end_s * initial_fs))

        w_start = max(int(n_total * 0.70), expected_end_idx - tol_samples)
        w_start = min(w_start, max(t0_anchor_idx + int(15.0 * initial_fs), n_total - int(60.0 * initial_fs)))
        w_end = n_total

        if w_end > w_start + int(1.0 * initial_fs):
            end_env = env_snr[w_start:w_end]
            max_end = float(np.max(end_env))

            if max_end > 5.0:
                # Find the prominent calibration squeeze peak in the tail of the file
                end_peaks, _ = signal.find_peaks(end_env, height=max(max_end * 0.45, 5.0), distance=int(1.0 * initial_fs))
                end_peak_local = int(end_peaks[-1]) if len(end_peaks) > 0 else int(np.argmax(end_env))
                end_peak_idx = w_start + end_peak_local

                ep_height = env_snr[end_peak_idx]
                eb_lookback = max(w_start, end_peak_idx - int(1.2 * initial_fs))
                eb_below = np.where(env_snr[eb_lookback:end_peak_idx] <= max(ep_height * 0.20, 3.0))[0]
                end_rising_idx = (eb_lookback + int(eb_below[-1])) if len(eb_below) > 0 else end_peak_idx
                end_anchor_idx = max(w_start, end_rising_idx - reaction_comp_samples)

                calibrated_fs = (end_anchor_idx - t0_anchor_idx) / expected_rel_end_s
                print(
                    f"  [BOOKEND DRIFT CALIBRATED] End Squeeze detected at idx {end_rising_idx:,} "
                    f"({end_rising_idx / initial_fs:.2f}s on nominal axis | PsychoPy Δt = {expected_rel_end_s:.2f}s)\n"
                    f"  -> True Empirical Shimmer Sampling Rate = {calibrated_fs:.3f} Hz "
                    f"(Nominal was {initial_fs:.1f} Hz)"
                )
            else:
                print("  [WARNING] Could not find a clear End Squeeze peak (SNR > 5) in the final 30% of the file.")

    return t0_anchor_idx, calibrated_fs


def slice_emg_epochs(
    shimmer_df: pd.DataFrame,
    events_df: pd.DataFrame,
    initial_fs: float,
    manual_t0_sec: float = None,
    manual_tend_sec: float = None
) -> tuple[pd.DataFrame, float]:
    events_df = events_df.copy()

    if "rel_from_sync_onset_s" not in events_df.columns:
        sync_rows = events_df[events_df["event_type"].isin(["sync_start", "sync_marker"])]
        t0_psy = float(sync_rows["onset_psychopy_s"].iloc[0]) if not sync_rows.empty else float(events_df["onset_psychopy_s"].iloc[0])
        events_df["rel_from_sync_onset_s"] = events_df["onset_psychopy_s"] - t0_psy
        events_df["rel_from_sync_offset_s"] = events_df["offset_psychopy_s"] - t0_psy

    t0_anchor_idx, calibrated_fs = detect_squeeze_anchor_and_fs(
        shimmer_df,
        events_df,
        initial_fs=initial_fs,
        manual_t0_sec=manual_t0_sec,
        manual_tend_sec=manual_tend_sec
    )

    n_total_samples = len(shimmer_df)
    emg_cols = [c for c in ["emg_ch1", "emg_ch2"] if c in shimmer_df.columns]
    has_accel = "accel_mag" in shimmer_df.columns

    sliced_epochs = []

    for _, row in events_df.iterrows():
        rel_onset_s = float(row["rel_from_sync_onset_s"])
        rel_offset_s = float(row["rel_from_sync_offset_s"])

        idx_start = t0_anchor_idx + int(round(rel_onset_s * calibrated_fs))
        idx_end = t0_anchor_idx + int(round(rel_offset_s * calibrated_fs))

        idx_start_clamped = max(0, min(idx_start, n_total_samples))
        idx_end_clamped = max(0, min(idx_end, n_total_samples))

        if idx_end_clamped <= idx_start_clamped:
            print(
                f"  [WARNING] Epoch '{row['event_type']}' (trial {row['trial_num']}) at "
                f"t=+{rel_onset_s:.1f}..{rel_offset_s:.1f}s is beyond Shimmer recording end. Skipping."
            )
            continue

        segment = shimmer_df.iloc[idx_start_clamped:idx_end_clamped]
        n_samples = len(segment)
        rel_time_s = np.arange(n_samples, dtype=np.float64) / calibrated_fs

        epoch_record = {
            "participant_id": row["participant_id"],
            "expertise_group": row["expertise_group"],
            "dominant_hand": row["dominant_hand"],
            "seed": int(row["seed"]),
            "event_type": row["event_type"],
            "trial_num": int(row["trial_num"]),
            "condition": row["condition"],
            "stimulus_file": row["stimulus_file"],
            "rel_from_sync_onset_s": round(rel_onset_s, 4),
            "rel_from_sync_offset_s": round(rel_offset_s, 4),
            "target_duration_s": float(row["target_duration_s"]),
            "actual_duration_s": float(row["actual_duration_s"]),
            "start_sample_idx": idx_start_clamped,
            "end_sample_idx": idx_end_clamped,
            "sampling_rate_hz": round(calibrated_fs, 3),
            "n_samples": n_samples,
            "time_s": rel_time_s,
        }

        for col in emg_cols:
            epoch_record[col] = segment[col].values.astype(np.float64)

        if has_accel:
            epoch_record["accel_mag"] = segment["accel_mag"].values.astype(np.float64)

        sliced_epochs.append(epoch_record)

    return pd.DataFrame(sliced_epochs), calibrated_fs


def save_sync_alignment_plot(shimmer_df: pd.DataFrame, epochs_df: pd.DataFrame, fs: float, sub_id: str):
    """
    Saves a 2-panel diagnostic figure in ./data/epochs/<sub_id>_sync_alignment_check.png
    showing the full CH1 (FDS) and CH2 (ECR) traces with shaded overlays for:
      - Red   : Start & End Calibration Squeezes
      - Green : 30s Resting Baseline
      - Orange: Climbing Video Trials
      - Purple: Everyday Video Trials
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    t_full = np.arange(len(shimmer_df)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.2), sharex=True)
    fig.suptitle(
        f"Squeeze-Locked Synchronization & Dual-Channel Slicing Verification — {sub_id} (fs = {fs:.2f} Hz)",
        fontsize=12, fontweight="bold"
    )

    ch_specs = [
        ("emg_ch1", "CH1: FDS (mV)\n[Finger Flexor]", "#0072B2"),
        ("emg_ch2", "CH2: ECR (mV)\n[Wrist Extensor]", "#D55E00")
    ]

    for idx, (col, label, color) in enumerate(ch_specs):
        ax = axes[idx]
        if col not in shimmer_df.columns:
            continue
        # Remove static DC offset (-19mV / -2mV) for clean visual inspection
        sig = shimmer_df[col].values - np.mean(shimmer_df[col].values)
        ax.plot(t_full, sig, color=color, lw=0.45, alpha=0.85)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)

        for _, ep in epochs_df.iterrows():
            t0 = ep["start_sample_idx"] / fs
            t1 = ep["end_sample_idx"] / fs
            ev = ep["event_type"]
            if ev in ("sync_start", "sync_end", "sync_marker"):
                ax.axvspan(t0, t1, color="#d62728", alpha=0.30)
            elif ev == "baseline":
                ax.axvspan(t0, t1, color="#2ca02c", alpha=0.22)
            elif ev == "trial_video":
                span_col = "#ff7f0e" if ep["condition"] == "climbing" else "#9467bd"
                ax.axvspan(t0, t1, color=span_col, alpha=0.25)

    legend_patches = [
        mpatches.Patch(color="#d62728", alpha=0.35, label="Sync Squeeze (t=0 / End)"),
        mpatches.Patch(color="#2ca02c", alpha=0.30, label="30s Resting Baseline"),
        mpatches.Patch(color="#ff7f0e", alpha=0.35, label="Climbing Video Trial"),
        mpatches.Patch(color="#9467bd", alpha=0.35, label="Everyday Video Trial"),
    ]
    axes[0].legend(handles=legend_patches, loc="upper right", ncol=4, fontsize=8.5, frameon=True)
    axes[-1].set_xlabel("Shimmer Recording Time from File Start (seconds)")

    plt.tight_layout()
    out_plot = config.EPOCHS_DIR / f"{sub_id}_sync_alignment_check.png"
    plt.savefig(out_plot, dpi=160)
    plt.close(fig)
    print(f"  [SYNC CHECK PLOT] {out_plot.name}")


def preprocess_participant(
    events_csv_path: Path,
    shimmer_file_path: Path,
    fs_override: float = None,
    manual_t0_sec: float = None,
    manual_tend_sec: float = None,
    export_individual_csvs: bool = False
):
    events_df = pd.read_csv(events_csv_path)
    sub_id = str(events_df["participant_id"].iloc[0])
    group = str(events_df["expertise_group"].iloc[0])

    if fs_override is not None:
        cfg_fs = float(fs_override)
    elif "shimmer_fs_configured_hz" in events_df.columns:
        cfg_fs = float(events_df["shimmer_fs_configured_hz"].iloc[0])
    else:
        cfg_fs = config.DEFAULT_SAMPLING_RATE

    print(f"\n=======================================================")
    print(f"Preprocessing Participant: {sub_id} ({group})")
    print(f"=======================================================")

    shimmer_df, fs_est, _ = load_shimmer_file(shimmer_file_path, default_fs=cfg_fs)
    epochs_df, calibrated_fs = slice_emg_epochs(
        shimmer_df,
        events_df,
        initial_fs=fs_est,
        manual_t0_sec=manual_t0_sec,
        manual_tend_sec=manual_tend_sec
    )

    if epochs_df.empty:
        raise RuntimeError(f"No valid epochs were extracted for {sub_id}!")

    out_pkl = config.EPOCHS_DIR / f"{sub_id}_{group}_epochs.pkl"
    epochs_df.to_pickle(out_pkl)

    summary_cols = [
        c for c in epochs_df.columns
        if not c.startswith("emg_ch") and c not in ("time_s", "accel_mag")
    ]
    out_summary_csv = config.EPOCHS_DIR / f"{sub_id}_{group}_epochs_summary.csv"
    epochs_df[summary_cols].to_csv(out_summary_csv, index=False)

    save_sync_alignment_plot(shimmer_df, epochs_df, calibrated_fs, sub_id)

    print(f"  [SAVED] {len(epochs_df)} dual-channel epochs (CH1 FDS + CH2 ECR) -> {out_pkl.name}")
    print(f"  [SAVED] Epoch summary table -> {out_summary_csv.name}")

    if export_individual_csvs:
        sub_epoch_dir = config.EPOCHS_DIR / f"{sub_id}_csv_segments"
        sub_epoch_dir.mkdir(parents=True, exist_ok=True)
        for _, ep in epochs_df.iterrows():
            seg_dict = {"time_s": ep["time_s"]}
            if "emg_ch1" in ep:
                seg_dict["emg_ch1_FDS_mV"] = ep["emg_ch1"]
            if "emg_ch2" in ep:
                seg_dict["emg_ch2_ECR_mV"] = ep["emg_ch2"]
            fname = f"trial{ep['trial_num']:02d}_{ep['event_type']}_{ep['condition']}_{Path(ep['stimulus_file']).stem}.csv"
            pd.DataFrame(seg_dict).to_csv(sub_epoch_dir / fname, index=False)
        print(f"  [SAVED] Individual CSV segments -> {sub_epoch_dir}")

    return epochs_df


def simulate_demo_raw_files(n_per_group: int = 4, fs: float = 512.0, n_trials_per_cond: int = 5):
    """
    DEMO MODE: Generates synthetic dual-channel Shimmer files matching the exact header & -1 timestamps
    of sub-E41CB9_Shimmer_9BF9_Calibrated_SD.csv AND matching PsychoPy event logs.
    """
    rng = np.random.default_rng(2026)
    print(f"\n[DEMO SIMULATOR] Generating {2 * n_per_group} dual-channel participants (-1 timestamps, CH1 FDS + CH2 ECR)...")

    for g_idx, group in enumerate(["climber", "non_climber"]):
        for s in range(1, n_per_group + 1):
            sub_id = f"sub-DEMO{g_idx + 1}{s}"
            seed = 1000 + g_idx * 100 + s
            rel_cursor = 0.0
            events = []

            def add_ev(ev_type, t_num, cond, stim, dur):
                nonlocal rel_cursor
                events.append({
                    "participant_id": sub_id, "expertise_group": group, "dominant_hand": "right",
                    "shimmer_fs_configured_hz": fs, "seed": seed, "trigger_mode": "none",
                    "event_type": ev_type, "trial_num": t_num, "condition": cond, "stimulus_file": stim,
                    "rel_from_sync_onset_s": round(rel_cursor, 4),
                    "rel_from_sync_offset_s": round(rel_cursor + dur, 4),
                    "target_duration_s": dur, "actual_duration_s": dur,
                    "onset_psychopy_s": round(rel_cursor + 5.0, 4),
                    "offset_psychopy_s": round(rel_cursor + 5.0 + dur, 4),
                    "onset_unix_ms": -1, "offset_unix_ms": -1,
                    "onset_trigger_code": 0, "offset_trigger_code": 0
                })
                rel_cursor += dur

            # Continuous timeline locked to Start Squeeze (t = 0.0s)
            add_ev("sync_start", 0, "calibration_start", "fist_squeeze_start", 1.5)
            add_ev("post_sync_settle", 0, "muscle_settle", "relax_screen", 8.0)
            add_ev("baseline", 0, "resting_baseline", "fixation_cross", 10.0)

            cond_list = (["climbing"] * n_trials_per_cond) + (["everyday"] * n_trials_per_cond)
            rng.shuffle(cond_list)
            for idx, cond in enumerate(cond_list, start=1):
                stim_name = f"{'crimp' if cond == 'climbing' else 'book'}_{idx:02d}.mp4"
                add_ev("trial_video", idx, cond, stim_name, 4.0)
                add_ev("isi", idx, f"post_{cond}_isi", "fixation_cross", round(float(rng.uniform(4.0, 7.0)), 3))

            add_ev("sync_end", len(cond_list) + 1, "calibration_end", "fist_squeeze_end", 1.5)

            ev_path = config.LOG_DIR / f"{sub_id}_{group}_20261021_events.csv"
            pd.DataFrame(events).to_csv(ev_path, index=False)

            # Simulate Shimmer recording starting 5.0s before the Start Squeeze
            pre_buffer_s = 5.0
            total_dur_s = pre_buffer_s + rel_cursor + 4.0
            n_samples = int(total_dur_s * fs)
            t_shimmer_s = np.arange(n_samples) / fs
            t_rel_sync_s = t_shimmer_s - (pre_buffer_s + 0.25)  # +0.25s reaction time

            env_ch1 = np.ones(n_samples) * 0.012 * np.linspace(1.0, 1.07, n_samples)  # FDS resting floor
            env_ch2 = np.ones(n_samples) * 0.015 * np.linspace(1.0, 1.05, n_samples)  # ECR resting floor

            for ev in events:
                mask = (t_rel_sync_s >= ev["rel_from_sync_onset_s"]) & (t_rel_sync_s <= ev["rel_from_sync_offset_s"])
                if ev["event_type"] in ("sync_start", "sync_end"):
                    env_ch1[mask] *= 30.0  # ~0.36 mV burst like your test file!
                    env_ch2[mask] *= 25.0
                elif ev["event_type"] == "trial_video":
                    if ev["condition"] == "everyday":
                        gain_fds = rng.uniform(1.18, 1.32)
                        gain_ecr = rng.uniform(1.15, 1.28)
                    else:
                        gain_fds = rng.uniform(1.42, 1.68) if group == "climber" else rng.uniform(0.98, 1.08)
                        gain_ecr = rng.uniform(1.32, 1.55) if group == "climber" else rng.uniform(0.99, 1.08)

                    t_loc = t_rel_sync_s[mask] - ev["rel_from_sync_onset_s"]
                    burst = np.exp(-0.5 * ((t_loc - 2.0) / 0.85) ** 2)
                    env_ch1[mask] *= (1.0 + (gain_fds - 1.0) * burst)
                    env_ch2[mask] *= (1.0 + (gain_ecr - 1.0) * burst)

            hum = 0.006 * np.sin(2 * np.pi * 50.0 * t_shimmer_s)
            sway = 0.010 * np.sin(2 * np.pi * 1.1 * t_shimmer_s)
            ch1_mv = -19.141 + (env_ch1 * rng.normal(0, 1, n_samples)) + hum + sway
            ch2_mv = -2.062 + (env_ch2 * rng.normal(0, 1, n_samples)) + hum + sway

            shimmer_path = config.RAW_SHIMMER_DIR / f"{sub_id}_Shimmer_9BF9_Calibrated_SD.csv"
            with open(shimmer_path, "w", encoding="utf-8") as f:
                f.write('"sep=\t"\n')
                f.write(
                    "Shimmer_9BF9_Timestamp_Unix_CAL\tShimmer_9BF9_Accel_LN_X_CAL\t"
                    "Shimmer_9BF9_Accel_LN_Y_CAL\tShimmer_9BF9_Accel_LN_Z_CAL\t"
                    "Shimmer_9BF9_ECG_EMG_Status1_CAL\tShimmer_9BF9_EMG_CH1_16BIT_CAL\t"
                    "Shimmer_9BF9_EMG_CH2_16BIT_CAL\tShimmer_9BF9_Ext_Exp_A7_CAL\n"
                )
                f.write("ms\tm/(s^2)\tm/(s^2)\tm/(s^2)\tno_units\tmV\tmV\tmV\n")
                out_df = pd.DataFrame({
                    "ts": -1,
                    "ax": -1.25, "ay": -4.40, "az": 8.90,
                    "status": 128.0,
                    "ch1": np.round(ch1_mv, 6),
                    "ch2": np.round(ch2_mv, 6),
                    "a7": 1150.0
                })
                out_df.to_csv(f, sep="\t", index=False, header=False)

            print(f"  -> Created: {ev_path.name} & {shimmer_path.name}")


def match_and_preprocess_all(fs_override: float = None, export_csvs: bool = False):
    event_files = sorted(config.LOG_DIR.glob("*_events.csv"))
    if not event_files:
        print(f"[INFO] No '*_events.csv' files found in {config.LOG_DIR}")
        return

    shimmer_files = sorted([
        f for f in config.RAW_SHIMMER_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in (".csv", ".tsv", ".txt")
    ])

    for ev_file in event_files:
        sub_id = ev_file.name.split("_")[0]
        matched_shimmer = [f for f in shimmer_files if sub_id.lower() in f.name.lower()]
        if not matched_shimmer:
            print(f"[SKIP] Found log '{ev_file.name}', but no Shimmer file matching '{sub_id}' in {config.RAW_SHIMMER_DIR.name}/.")
            continue

        preprocess_participant(
            events_csv_path=ev_file,
            shimmer_file_path=matched_shimmer[0],
            fs_override=fs_override,
            export_individual_csvs=export_csvs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess & Slice Dual-Channel (FDS + ECR) Shimmer sEMG Recordings")
    parser.add_argument("--events", type=str, default=None, help="Path to a specific PsychoPy *_events.csv file")
    parser.add_argument("--shimmer", type=str, default=None, help="Path to a specific raw Shimmer CSV file")
    parser.add_argument("--fs", type=float, default=None, help="Override Shimmer sampling rate in Hz (default: 512.0)")
    parser.add_argument(
        "--manual-t0",
        type=float,
        default=None,
        help="Optional manual override for Start Squeeze t=0 time (in seconds from start of Shimmer file)"
    )
    parser.add_argument("--export-csvs", action="store_true", help="Export each sliced epoch as an individual CSV")
    parser.add_argument("--simulate-demo", action="store_true", help="Generate synthetic dual-channel Shimmer files and preprocess them")
    args = parser.parse_args()

    if args.simulate_demo:
        simulate_demo_raw_files(n_per_group=4, fs=args.fs or 512.0, n_trials_per_cond=5)
        match_and_preprocess_all(fs_override=args.fs, export_csvs=args.export_csvs)
    elif args.events and args.shimmer:
        preprocess_participant(
            Path(args.events),
            Path(args.shimmer),
            fs_override=args.fs,
            manual_t0_sec=args.manual_t0,
            export_individual_csvs=args.export_csvs
        )
    else:
        match_and_preprocess_all(fs_override=args.fs, export_csvs=args.export_csvs)