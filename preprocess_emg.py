# ./Experiment in Cognitive Science 02455/preprocess_emg.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import config


def load_shimmer_file(filepath: Path, time_col_override: str = None, emg_col_override: str = None):
    """
    Robustly loads a Shimmer Consensys exported CSV/TSV file.
    Handles 'sep=' headers, multi-row headers (column names + units row),
    and auto-detects the Unix timestamp (ms) and EMG (mV) columns.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Shimmer file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_lines = [f.readline().strip() for _ in range(5)]

    skiprows = 0
    sep = "\t" if "\t" in first_lines[1] else ","

    if first_lines[0].startswith("sep="):
        sep_char = first_lines[0].split("=")[1].strip()
        if sep_char == "\\t" or sep_char == "":
            sep = "\t"
        else:
            sep = sep_char
        skiprows = 1

    df_raw = pd.read_csv(filepath, sep=sep, skiprows=skiprows, low_memory=False)

    # Shimmer Consensys exports often put units (e.g. 'ms', 'mV', 'no_units') in row 0
    first_val = str(df_raw.iloc[0, 0]).strip().lower()
    if any(u in first_val for u in ["ms", "sec", "no_units", "mv", "cal", "unix"]):
        df_raw = df_raw.iloc[1:].reset_index(drop=True)

    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # 1. Identify Timestamp Column
    time_col = time_col_override
    if time_col is None:
        for candidate in config.SHIMMER_TIME_COLS:
            matches = [c for c in df_raw.columns if candidate.lower() in c.lower()]
            if matches:
                time_col = matches[0]
                break
    if time_col is None or time_col not in df_raw.columns:
        raise ValueError(
            f"Could not auto-detect timestamp column in {filepath.name}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    # 2. Identify EMG Channel Column(s)
    emg_cols = []
    if emg_col_override:
        emg_cols = [emg_col_override]
    else:
        for candidate in config.SHIMMER_EMG_COLS:
            matches = [c for c in df_raw.columns if candidate.lower() in c.lower() and "status" not in c.lower()]
            for m in matches:
                if m not in emg_cols:
                    emg_cols.append(m)

    if not emg_cols:
        raise ValueError(
            f"Could not auto-detect EMG channel column in {filepath.name}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    clean_df = pd.DataFrame()
    clean_df["timestamp_raw"] = pd.to_numeric(df_raw[time_col], errors="coerce")

    for idx, col in enumerate(emg_cols, start=1):
        clean_df[f"emg_ch{idx}"] = pd.to_numeric(df_raw[col], errors="coerce")

    trig_matches = [c for c in df_raw.columns if any(k in c.lower() for k in ["marker", "trigger", "event"])]
    if trig_matches:
        clean_df["shimmer_trigger"] = pd.to_numeric(df_raw[trig_matches[0]], errors="coerce").fillna(0).astype(int)

    clean_df = clean_df.dropna(subset=["timestamp_raw", "emg_ch1"]).reset_index(drop=True)

    median_ts = clean_df["timestamp_raw"].median()
    if 1e9 < median_ts < 1e11:
        clean_df["timestamp_unix_ms"] = clean_df["timestamp_raw"] * 1000.0
    else:
        clean_df["timestamp_unix_ms"] = clean_df["timestamp_raw"]

    dt_ms = np.median(np.diff(clean_df["timestamp_unix_ms"].values))
    fs_estimated = round(1000.0 / dt_ms, 2) if dt_ms > 0 else config.DEFAULT_SAMPLING_RATE

    print(f"[SHIMMER LOADED] {filepath.name}")
    print(f"  -> Time Column : '{time_col}' | EMG Columns: {emg_cols}")
    print(f"  -> Total Samples: {len(clean_df):,} | Estimated Sampling Rate: {fs_estimated} Hz")

    return clean_df, fs_estimated, emg_cols


def compute_clock_offset_from_sync_squeeze(
    shimmer_df: pd.DataFrame,
    events_df: pd.DataFrame,
    fs: float,
    search_window_s: float = 15.0
) -> float:
    """
    Finds the high-amplitude calibration fist squeeze in the Shimmer trace around the
    expected 'sync_marker' timestamp and returns the clock offset (ms) to add to
    PsychoPy timestamps so they align with the Shimmer clock.
    """
    sync_rows = events_df[events_df["event_type"] == "sync_marker"]
    if sync_rows.empty:
        print("  [SYNC] No 'sync_marker' row in PsychoPy log. Using direct Unix clock (offset = 0 ms).")
        return 0.0

    expected_onset_ms = float(sync_rows.iloc[0]["onset_unix_ms"])
    expected_offset_ms = float(sync_rows.iloc[0]["offset_unix_ms"])
    expected_center_ms = 0.5 * (expected_onset_ms + expected_offset_ms)

    win_ms = search_window_s * 1000.0
    sub_df = shimmer_df[
        (shimmer_df["timestamp_unix_ms"] >= expected_onset_ms - win_ms) &
        (shimmer_df["timestamp_unix_ms"] <= expected_offset_ms + win_ms)
    ].copy()

    if sub_df.empty:
        print("  [WARNING] Sync squeeze window outside Shimmer recording range! Falling back to offset = 0 ms.")
        return 0.0

    sig = sub_df["emg_ch1"].values - np.median(sub_df["emg_ch1"].values)
    win_samples = max(int(0.25 * fs), 10)
    rms_env = np.sqrt(np.convolve(sig ** 2, np.ones(win_samples) / win_samples, mode="same"))

    peak_idx = int(np.argmax(rms_env))
    actual_peak_ms = float(sub_df.iloc[peak_idx]["timestamp_unix_ms"])

    offset_ms = actual_peak_ms - expected_center_ms
    print(f"  [SYNC SQUEEZE ALIGNED] Detected burst peak drift: {offset_ms:+.1f} ms")
    return offset_ms


def slice_emg_epochs(
    shimmer_df: pd.DataFrame,
    events_df: pd.DataFrame,
    fs: float,
    sync_method: str = "unix"
) -> pd.DataFrame:
    """
    Slices the continuous Shimmer EMG DataFrame into labeled segments corresponding to
    every row in the PsychoPy events log (baseline, trial_video, isi).
    """
    clock_offset_ms = 0.0
    if sync_method == "sync_squeeze":
        clock_offset_ms = compute_clock_offset_from_sync_squeeze(shimmer_df, events_df, fs)

    ts_array = shimmer_df["timestamp_unix_ms"].values
    emg_cols = [c for c in shimmer_df.columns if c.startswith("emg_ch")]

    sliced_epochs = []

    for _, row in events_df.iterrows():
        ev_type = row["event_type"]
        onset_ms = float(row["onset_unix_ms"]) + clock_offset_ms
        offset_ms = float(row["offset_unix_ms"]) + clock_offset_ms

        idx_start = int(np.searchsorted(ts_array, onset_ms, side="left"))
        idx_end = int(np.searchsorted(ts_array, offset_ms, side="right"))

        segment = shimmer_df.iloc[idx_start:idx_end].copy()
        n_samples = len(segment)

        if n_samples == 0:
            print(
                f"  [WARNING] Empty EMG slice for {ev_type} (trial {row['trial_num']}, "
                f"{row['stimulus_file']}). Check clock synchronization!"
            )
            continue

        rel_time_s = (segment["timestamp_unix_ms"].values - onset_ms) / 1000.0

        epoch_record = {
            "participant_id": row["participant_id"],
            "expertise_group": row["expertise_group"],
            "dominant_hand": row["dominant_hand"],
            "seed": int(row["seed"]),
            "event_type": ev_type,
            "trial_num": int(row["trial_num"]),
            "condition": row["condition"],
            "stimulus_file": row["stimulus_file"],
            "target_duration_s": float(row["target_duration_s"]),
            "actual_duration_s": float(row["actual_duration_s"]),
            "onset_unix_ms_aligned": onset_ms,
            "offset_unix_ms_aligned": offset_ms,
            "clock_offset_applied_ms": clock_offset_ms,
            "sampling_rate_hz": fs,
            "n_samples": n_samples,
            "time_s": rel_time_s,
        }

        for col in emg_cols:
            epoch_record[col] = segment[col].values

        sliced_epochs.append(epoch_record)

    return pd.DataFrame(sliced_epochs)


def preprocess_participant(
    events_csv_path: Path,
    shimmer_file_path: Path,
    sync_method: str = "unix",
    export_individual_csvs: bool = False
):
    events_df = pd.read_csv(events_csv_path)
    sub_id = str(events_df["participant_id"].iloc[0])
    group = str(events_df["expertise_group"].iloc[0])

    print(f"\n==========================================")
    print(f"Preprocessing Participant: {sub_id} ({group})")
    print(f"==========================================")

    shimmer_df, fs, _ = load_shimmer_file(shimmer_file_path)
    epochs_df = slice_emg_epochs(shimmer_df, events_df, fs, sync_method=sync_method)

    if epochs_df.empty:
        raise RuntimeError(f"No valid epochs were extracted for {sub_id}!")

    out_pkl = config.EPOCHS_DIR / f"{sub_id}_{group}_epochs.pkl"
    epochs_df.to_pickle(out_pkl)

    summary_cols = [c for c in epochs_df.columns if not c.startswith("emg_ch") and c != "time_s"]
    out_summary_csv = config.EPOCHS_DIR / f"{sub_id}_{group}_epochs_summary.csv"
    epochs_df[summary_cols].to_csv(out_summary_csv, index=False)

    print(f"  [SAVED] {len(epochs_df)} epochs -> {out_pkl.name}")
    print(f"  [SAVED] Quality summary -> {out_summary_csv.name}")

    if export_individual_csvs:
        sub_epoch_dir = config.EPOCHS_DIR / f"{sub_id}_csv_segments"
        sub_epoch_dir.mkdir(parents=True, exist_ok=True)
        for _, ep in epochs_df.iterrows():
            ep_df = pd.DataFrame({
                "time_s": ep["time_s"],
                "emg_ch1": ep["emg_ch1"]
            })
            fname = f"trial{ep['trial_num']:02d}_{ep['event_type']}_{ep['condition']}_{Path(ep['stimulus_file']).stem}.csv"
            ep_df.to_csv(sub_epoch_dir / fname, index=False)
        print(f"  [SAVED] Individual epoch CSVs -> {sub_epoch_dir}")

    return epochs_df


def simulate_demo_raw_files(n_per_group: int = 3, fs: float = 512.0, n_trials_per_cond: int = 4):
    """
    DEMO MODE: Generates realistic synthetic PsychoPy *_events.csv files in ./data/logs/
    AND matching raw Shimmer Consensys CSV files in ./data/raw_shimmer/ so you can test
    the entire pipeline end-to-end (preprocess -> postprocess -> stats -> plot) right now!
    """
    rng = np.random.default_rng(2026)
    base_unix_s = 1792490000.0  # Fixed starting Unix timestamp

    print(f"\n[DEMO SIMULATOR] Generating {2 * n_per_group} synthetic participants (Shimmer + PsychoPy logs)...")

    for g_idx, group in enumerate(["climber", "non_climber"]):
        for s in range(1, n_per_group + 1):
            sub_id = f"sub-DEMO{g_idx + 1}{s}"
            seed = 1000 + g_idx * 100 + s
            t_cursor = base_unix_s + (g_idx * 10 + s) * 1000.0
            psy_cursor = 0.0

            events = []

            # 1. Calibration Sync Squeeze (1.5s) + 5s settle
            t_Cursor_start = t_cursor
            events.append({
                "participant_id": sub_id, "expertise_group": group, "dominant_hand": "right",
                "seed": seed, "trigger_mode": "none", "event_type": "sync_marker",
                "trial_num": 0, "condition": "calibration", "stimulus_file": "fist_squeeze",
                "onset_trigger_code": 1, "offset_trigger_code": 2,
                "onset_unix_s": t_cursor, "onset_unix_ms": int(round(t_cursor * 1000)),
                "offset_unix_s": t_cursor + 1.5, "offset_unix_ms": int(round((t_cursor + 1.5) * 1000)),
                "onset_psychopy_s": psy_cursor, "offset_psychopy_s": psy_cursor + 1.5,
                "target_duration_s": 1.5, "actual_duration_s": 1.5
            })
            t_cursor += 6.5
            psy_cursor += 6.5

            # 2. Resting Baseline (10s in demo for speed, or 30s)
            base_dur = 10.0
            events.append({
                "participant_id": sub_id, "expertise_group": group, "dominant_hand": "right",
                "seed": seed, "trigger_mode": "none", "event_type": "baseline",
                "trial_num": 0, "condition": "resting_baseline", "stimulus_file": "fixation_cross",
                "onset_trigger_code": 10, "offset_trigger_code": 11,
                "onset_unix_s": t_cursor, "onset_unix_ms": int(round(t_cursor * 1000)),
                "offset_unix_s": t_cursor + base_dur, "offset_unix_ms": int(round((t_cursor + base_dur) * 1000)),
                "onset_psychopy_s": psy_cursor, "offset_psychopy_s": psy_cursor + base_dur,
                "target_duration_s": base_dur, "actual_duration_s": base_dur
            })
            t_cursor += base_dur
            psy_cursor += base_dur

            # 3. Video Trials + Jittered ISIs
            cond_list = (["climbing"] * n_trials_per_cond) + (["everyday"] * n_trials_per_cond)
            rng.shuffle(cond_list)

            for idx, cond in enumerate(cond_list, start=1):
                vid_dur = 5.0
                isi_dur = round(float(rng.uniform(4.0, 7.0)), 3)
                stim_name = f"{'crimp' if cond == 'climbing' else 'book'}_{idx:02d}.mp4"

                events.append({
                    "participant_id": sub_id, "expertise_group": group, "dominant_hand": "right",
                    "seed": seed, "trigger_mode": "none", "event_type": "trial_video",
                    "trial_num": idx, "condition": cond, "stimulus_file": stim_name,
                    "onset_trigger_code": 21 if cond == "climbing" else 22, "offset_trigger_code": 30,
                    "onset_unix_s": t_cursor, "onset_unix_ms": int(round(t_cursor * 1000)),
                    "offset_unix_s": t_cursor + vid_dur, "offset_unix_ms": int(round((t_cursor + vid_dur) * 1000)),
                    "onset_psychopy_s": psy_cursor, "offset_psychopy_s": psy_cursor + vid_dur,
                    "target_duration_s": vid_dur, "actual_duration_s": vid_dur
                })
                t_cursor += vid_dur
                psy_cursor += vid_dur

                events.append({
                    "participant_id": sub_id, "expertise_group": group, "dominant_hand": "right",
                    "seed": seed, "trigger_mode": "none", "event_type": "isi",
                    "trial_num": idx, "condition": f"post_{cond}_isi", "stimulus_file": "fixation_cross",
                    "onset_trigger_code": 30, "offset_trigger_code": 31,
                    "onset_unix_s": t_cursor, "onset_unix_ms": int(round(t_cursor * 1000)),
                    "offset_unix_s": t_cursor + isi_dur, "offset_unix_ms": int(round((t_cursor + isi_dur) * 1000)),
                    "onset_psychopy_s": psy_cursor, "offset_psychopy_s": psy_cursor + isi_dur,
                    "target_duration_s": isi_dur, "actual_duration_s": isi_dur
                })
                t_cursor += isi_dur
                psy_cursor += isi_dur

            events_df = pd.DataFrame(events)
            ev_path = config.LOG_DIR / f"{sub_id}_{group}_20261020_events.csv"
            events_df.to_csv(ev_path, index=False)

            # Generate continuous Shimmer recording spanning the entire session (+2s buffer on each end)
            total_dur_s = (t_cursor - t_Cursor_start) + 4.0
            n_samples = int(total_dur_s * fs)
            t_vec_s = np.linspace(t_Cursor_start - 2.0, t_cursor + 2.0, n_samples, endpoint=False)
            ts_unix_ms = t_vec_s * 1000.0

            # Base resting MUAP carrier (band-limited 30-350 Hz stochastic process) + 50Hz hum + 1Hz cable sway
            resting_sd = 0.018  # 18 uV resting noise floor
            envelope_profile = np.ones(n_samples) * resting_sd

            # Add slow linear drift across session (+10% over session) to test ISI drift correction
            slow_drift = np.linspace(1.0, 1.10, n_samples)
            envelope_profile *= slow_drift

            # Modulate envelope during each event
            for ev in events:
                mask = (ts_unix_ms >= ev["onset_unix_ms"]) & (ts_unix_ms <= ev["offset_unix_ms"])
                if ev["event_type"] == "sync_marker":
                    envelope_profile[mask] *= 15.0  # Huge 15x fist squeeze spike
                elif ev["event_type"] == "trial_video":
                    # Realistic H1 interaction: Climbers resonate strongly to climbing (+45%),
                    # while both groups show mild resonance to everyday grasping (+20%)
                    if ev["condition"] == "everyday":
                        gain = rng.uniform(1.15, 1.30)
                    else:
                        gain = rng.uniform(1.35, 1.60) if group == "climber" else rng.uniform(0.98, 1.08)

                    # Smooth Gaussian burst peaking in the middle of the 5s clip
                    t_local = (ts_unix_ms[mask] - ev["onset_unix_ms"]) / 1000.0
                    burst = 1.0 + (gain - 1.0) * np.exp(-0.5 * ((t_local - 2.5) / 1.0) ** 2)
                    envelope_profile[mask] *= burst

            carrier = rng.normal(0.0, 1.0, size=n_samples)
            hum_50hz = 0.010 * np.sin(2 * np.pi * 50.0 * t_vec_s)
            motion_1hz = 0.015 * np.sin(2 * np.pi * 1.2 * t_vec_s)
            raw_emg_mv = (envelope_profile * carrier) + hum_50hz + motion_1hz

            # Write authentic Shimmer Consensys format (sep=\t + header + units row)
            shimmer_path = config.RAW_SHIMMER_DIR / f"{sub_id}_Shimmer3_EMG_Session.csv"
            with open(shimmer_path, "w", encoding="utf-8") as f:
                f.write("sep=\t\n")
                f.write("Shimmer_1_Timestamp_Unix_CAL\tShimmer_1_EMG_Ch1_CAL\n")
                f.write("ms\tmV\n")
                shimmer_out = pd.DataFrame({
                    "ts": np.round(ts_unix_ms, 3),
                    "emg": np.round(raw_emg_mv, 6)
                })
                shimmer_out.to_csv(f, sep="\t", index=False, header=False)

            print(f"  -> Created Demo Files: {ev_path.name} & {shimmer_path.name}")


def match_and_preprocess_all(sync_method: str = "unix", export_csvs: bool = False):
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
            print(
                f"[SKIP] Found log '{ev_file.name}', but no matching Shimmer file containing "
                f"'{sub_id}' in '{config.RAW_SHIMMER_DIR.name}/'."
            )
            continue

        preprocess_participant(
            events_csv_path=ev_file,
            shimmer_file_path=matched_shimmer[0],
            sync_method=sync_method,
            export_individual_csvs=export_csvs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess & Slice Shimmer sEMG Recordings")
    parser.add_argument("--events", type=str, default=None, help="Path to a specific PsychoPy *_events.csv file")
    parser.add_argument("--shimmer", type=str, default=None, help="Path to a specific raw Shimmer CSV file")
    parser.add_argument(
        "--sync",
        type=str,
        choices=["unix", "sync_squeeze"],
        default="unix",
        help="Synchronization method: 'unix' (direct RTC match) or 'sync_squeeze' (auto-aligns on calibration squeeze peak)"
    )
    parser.add_argument(
        "--export-csvs",
        action="store_true",
        help="Also export every single sliced epoch as its own standalone CSV file"
    )
    parser.add_argument(
        "--simulate-demo",
        action="store_true",
        help="Generate synthetic Shimmer CSVs + PsychoPy logs and run preprocessing on them"
    )
    args = parser.parse_args()

    if args.simulate_demo:
        simulate_demo_raw_files(n_per_group=4, fs=512.0, n_trials_per_cond=5)
        match_and_preprocess_all(sync_method=args.sync, export_csvs=args.export_csvs)
    elif args.events and args.shimmer:
        preprocess_participant(
            events_csv_path=Path(args.events),
            shimmer_file_path=Path(args.shimmer),
            sync_method=args.sync,
            export_individual_csvs=args.export_csvs
        )
    else:
        match_and_preprocess_all(sync_method=args.sync, export_csvs=args.export_csvs)