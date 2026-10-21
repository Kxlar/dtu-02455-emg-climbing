# ./Experiment in Cognitive Science 02455/postprocess_emg.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid

import config


def apply_clancy_cascade(
    raw_emg: np.ndarray,
    fs: float,
    demod_power: int = 1,
    apply_harmonic_notches: bool = True
) -> dict:
    """
    Implements the sequential sEMG amplitude estimation cascade (Clancy et al., 2002;
    Cifrek et al., 2009 Fig. 3; Protocol Section 5):
      Stage 1: Noise Reject Filter -> 20-500 Hz Butterworth Bandpass + 50 Hz Notch
      Stage 2: Demodulate / Detect -> Full-wave rectification |x|^d (d=1 for MAV, d=2 for RMS)
      Stage 3: Smooth              -> Low-pass Butterworth filter at 8 Hz (Linear Envelope)
      Stage 4: Relinearize         -> Raise smoothed signal to (.)^(1/d)
    """
    x = np.asarray(raw_emg, dtype=np.float64)
    # Remove static DC offset prior to filtering
    x = x - np.mean(x)

    nyq = 0.5 * fs
    # Safety check: if sampling rate is e.g. 1000 Hz, Nyquist is 500 Hz.
    # Clamp high-frequency cutoff slightly below Nyquist so digital Butterworth is stable.
    bp_low = config.BANDPASS_LOW
    bp_high = min(config.BANDPASS_HIGH, nyq * 0.95)

    # --- STAGE 1A: Bandpass Filter (20 Hz to 500 Hz) ---
    sos_bp = signal.butter(
        N=config.FILTER_ORDER,
        Wn=[bp_low, bp_high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )
    # Pad length prevents transient filter ringing at the start/end of 5s epochs
    padlen = min(int(1.0 * fs), len(x) // 3 - 1)
    x_bp = signal.sosfiltfilt(sos_bp, x, padlen=max(padlen, 0))

    # --- STAGE 1B: 50 Hz Power-Line Notch Filter (+ optional 150/250 Hz odd harmonics) ---
    x_notched = x_bp.copy()
    notch_freqs = [config.NOTCH_FREQ]
    if apply_harmonic_notches:
        notch_freqs = [f for f in (50.0, 100.0, 150.0, 250.0, 350.0, 450.0) if f < bp_high]

    for f0 in notch_freqs:
        b_notch, a_notch = signal.iirnotch(w0=f0, Q=config.NOTCH_Q, fs=fs)
        x_notched = signal.filtfilt(b_notch, a_notch, x_notched, padlen=max(padlen, 0))

    # --- STAGE 2: Detect / Full-Wave Rectification (|x|^d) ---
    # Non-linear demodulation shifts amplitude envelope energy down into the 0-10 Hz band
    if demod_power == 1:
        x_detected = np.abs(x_notched)
    elif demod_power == 2:
        x_detected = np.square(x_notched)
    else:
        x_detected = np.power(np.abs(x_notched), demod_power)

    # --- STAGE 3: Smooth / Linear Envelope Generation (8 Hz Low-Pass) ---
    sos_lp = signal.butter(
        N=config.FILTER_ORDER,
        Wn=config.ENVELOPE_LOWPASS,
        btype="lowpass",
        fs=fs,
        output="sos"
    )
    x_smoothed = signal.sosfiltfilt(sos_lp, x_detected, padlen=max(padlen, 0))
    # Prevent microscopic negative numerical overshoot from zero-phase filter on strictly positive input
    x_smoothed = np.maximum(x_smoothed, 0.0)

    # --- STAGE 4: Relinearize (.)^(1/d) ---
    if demod_power == 1:
        envelope = x_smoothed
    elif demod_power == 2:
        envelope = np.sqrt(x_smoothed)
    else:
        envelope = np.power(x_smoothed, 1.0 / demod_power)

    return {
        "bandpassed": x_bp,
        "notched": x_notched,
        "rectified": np.abs(x_notched),
        "envelope": envelope
    }


def compute_robust_baseline_stats(baseline_envelope: np.ndarray, fs: float) -> dict:
    """
    Computes artifact-resistant resting baseline metrics from the 30-second baseline envelope.
    Instead of a naive global mean (which can be skewed by a single swallow or postural twitch),
    we slide a 500ms window across the middle 28 seconds (trimming 1s edge transients)
    and compute:
      1. Trimmed Mean (10th to 80th percentile of windows) -> guards against micro-twitches
      2. Median & Robust SD (1.4826 * MAD)                -> for Z-score SNR normalization
      3. Standard Mean & SD                               -> standard protocol reference
    """
    # Trim first and last 1.0s of the 30s baseline to avoid participant button-press settling
    trim_samples = int(1.0 * fs)
    if len(baseline_envelope) > 4 * trim_samples:
        env_clean = baseline_envelope[trim_samples:-trim_samples]
    else:
        env_clean = baseline_envelope

    # Standard metrics
    mean_std = float(np.mean(env_clean))
    sd_std = float(np.std(env_clean, ddof=1))

    # Sliding 500ms non-overlapping window medians
    win_len = max(int(0.5 * fs), 1)
    n_wins = len(env_clean) // win_len
    if n_wins >= 5:
        win_means = np.array([
            np.mean(env_clean[i * win_len:(i + 1) * win_len])
            for i in range(n_wins)
        ])
        p10, p80 = np.percentile(win_means, [10, 80])
        quiet_wins = win_means[(win_means >= p10) & (win_means <= p80)]
        trimmed_mean = float(np.mean(quiet_wins)) if len(quiet_wins) > 0 else mean_std
    else:
        trimmed_mean = mean_std

    median_val = float(np.median(env_clean))
    mad = float(np.median(np.abs(env_clean - median_val)))
    robust_sd = max(1.4826 * mad, 1e-12)  # Consistent estimator of SD under Gaussian noise

    return {
        "base_mean": mean_std,
        "base_sd": max(sd_std, 1e-12),
        "base_trimmed_mean": max(trimmed_mean, 1e-12),
        "base_median": max(median_val, 1e-12),
        "base_robust_sd": robust_sd
    }


def extract_trial_features(
    time_s: np.ndarray,
    trial_envelope: np.ndarray,
    base_stats: dict,
    local_isi_floor: float = None
) -> dict:
    """
    Extracts Peak Amplitude and Area Under the Curve (AUC) from a video trial's linear envelope,
    along with all 3 baseline normalization strategies:
      1. Protocol % Variation against 30s Baseline (both standard mean & robust trimmed mean)
      2. Z-Score SNR Normalization against 30s Baseline
      3. Local Pre-Trial ISI Drift-Corrected Normalization
    """
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 5.0

    # --- 1. Raw Unnormalized Envelope Metrics ---
    raw_peak = float(np.max(trial_envelope))
    raw_mean = float(np.mean(trial_envelope))
    raw_auc = float(trapezoid(y=trial_envelope, x=time_s))
    # Duration-corrected AUC (AUC per second) in case video lengths vary by a few milliseconds
    raw_auc_per_sec = raw_auc / duration_s if duration_s > 0 else raw_mean

    # Reference baseline floors
    mu_base = base_stats["base_trimmed_mean"]       # Robust resting floor (primary)
    mu_base_raw = base_stats["base_mean"]           # Un-trimmed resting mean (secondary)
    sigma_base = base_stats["base_robust_sd"]       # Robust baseline noise spread

    # Expected baseline AUC over an identical duration T
    expected_base_auc = mu_base * duration_s
    expected_base_auc_raw = mu_base_raw * duration_s

    # --- 2. Protocol Normalization: Percentage of Variation vs. 30s Baseline ---
    # Formula: ((Metric_trial - Metric_baseline) / Metric_baseline) * 100
    # And Ratio form (where 1.0 = baseline level, matching Section 3.5 Expected Outcomes)
    peak_pct_change = ((raw_peak - mu_base) / mu_base) * 100.0
    auc_pct_change = ((raw_auc - expected_base_auc) / expected_base_auc) * 100.0

    peak_ratio_robust = raw_peak / mu_base
    auc_ratio_robust = raw_auc / expected_base_auc

    peak_pct_change_untrimmed = ((raw_peak - mu_base_raw) / mu_base_raw) * 100.0
    auc_pct_change_untrimmed = ((raw_auc - expected_base_auc_raw) / expected_base_auc_raw) * 100.0

    # --- 3. Z-Score SNR Normalization (Net Activation in units of Baseline SD) ---
    z_env = (trial_envelope - mu_base) / sigma_base
    peak_zscore = float(np.max(z_env))
    mean_zscore = float(np.mean(z_env))
    auc_zscore = float(trapezoid(y=z_env, x=time_s))

    # --- 4. Local Pre-Trial ISI Drift-Corrected Metrics ---
    if local_isi_floor is not None and local_isi_floor > 0:
        local_expected_auc = local_isi_floor * duration_s
        peak_pct_local_isi = ((raw_peak - local_isi_floor) / local_isi_floor) * 100.0
        auc_pct_local_isi = ((raw_auc - local_expected_auc) / local_expected_auc) * 100.0
    else:
        peak_pct_local_isi = peak_pct_change
        auc_pct_local_isi = auc_pct_change

    return {
        "duration_s": round(duration_s, 4),
        # Raw Envelope Features
        "raw_peak_mv": raw_peak,
        "raw_mean_mv": raw_mean,
        "raw_auc_mv_s": raw_auc,
        "raw_auc_per_sec_mv": raw_auc_per_sec,
        # Baseline Reference Values
        "baseline_trimmed_mean_mv": mu_base,
        "baseline_untrimmed_mean_mv": mu_base_raw,
        "baseline_robust_sd_mv": sigma_base,
        "local_pre_isi_floor_mv": local_isi_floor if local_isi_floor else np.nan,
        # Primary Dependent Variables (Protocol % Variation & Ratio vs. Robust Baseline)
        "norm_peak_pct": peak_pct_change,
        "norm_auc_pct": auc_pct_change,
        "norm_peak_ratio": peak_ratio_robust,
        "norm_auc_ratio": auc_ratio_robust,
        # Secondary Un-trimmed Baseline % Variation
        "norm_peak_pct_untrimmed": peak_pct_change_untrimmed,
        "norm_auc_pct_untrimmed": auc_pct_change_untrimmed,
        # Z-Score (SNR) Dependent Variables
        "zscore_peak": peak_zscore,
        "zscore_mean": mean_zscore,
        "zscore_auc": auc_zscore,
        # Local ISI Drift-Corrected Dependent Variables
        "local_isi_norm_peak_pct": peak_pct_local_isi,
        "local_isi_norm_auc_pct": auc_pct_local_isi,
    }


def postprocess_participant(epoch_pkl_path: Path, demod_power: int = 1, save_qc_plot: bool = True) -> pd.DataFrame:
    """
    Runs the full post-processing cascade on a single participant's sliced epochs (.pkl).
    1. Processes the 30s baseline through the 4-stage filter cascade
    2. Processes all ISI segments (to extract local pre-trial resting floors)
    3. Processes all 30 video trials (15 climbing, 15 everyday) and extracts normalized DVs
    """
    epochs_df = pd.read_pickle(epoch_pkl_path)
    sub_id = str(epochs_df["participant_id"].iloc[0])
    group = str(epochs_df["expertise_group"].iloc[0])
    fs = float(epochs_df["sampling_rate_hz"].iloc[0])

    print(f"\n--------------------------------------------------")
    print(f"Post-processing Participant: {sub_id} ({group}) | fs = {fs} Hz")
    print(f"--------------------------------------------------")

    # 1. Process 30s Resting Baseline Epoch
    base_rows = epochs_df[epochs_df["event_type"] == "baseline"]
    if base_rows.empty:
        raise ValueError(f"No 'baseline' epoch found in {epoch_pkl_path.name}!")

    base_raw = base_rows.iloc[0]["emg_ch1"]
    base_time = base_rows.iloc[0]["time_s"]
    base_casc = apply_clancy_cascade(base_raw, fs=fs, demod_power=demod_power)
    base_stats = compute_robust_baseline_stats(base_casc["envelope"], fs=fs)

    print(
        f"  [BASELINE] Trimmed Mean: {base_stats['base_trimmed_mean']:.6f} mV | "
        f"Untrimmed Mean: {base_stats['base_mean']:.6f} mV | "
        f"Robust SD: {base_stats['base_robust_sd']:.6f} mV"
    )

    # 2. Pre-compute envelopes for all ISI epochs to allow Local Pre-Trial Drift Correction
    #    Note: ISI_k occurs right AFTER trial_k, so ISI_(k-1) is the resting window right BEFORE trial_k.
    isi_floors = {}
    isi_rows = epochs_df[epochs_df["event_type"] == "isi"]
    for _, isi_row in isi_rows.iterrows():
        t_num = int(isi_row["trial_num"])
        isi_env = apply_clancy_cascade(isi_row["emg_ch1"], fs=fs, demod_power=demod_power)["envelope"]
        # Take the quiet last 2.0 seconds of the ISI (after previous trial's activity has settled)
        tail_samples = min(int(2.0 * fs), len(isi_env))
        isi_floors[t_num] = float(np.median(isi_env[-tail_samples:]))

    # 3. Process Video Observation Trials
    trial_rows = epochs_df[epochs_df["event_type"] == "trial_video"].sort_values("trial_num")
    trial_features_list = []
    processed_envelopes = []

    for _, row in trial_rows.iterrows():
        t_num = int(row["trial_num"])
        cond = str(row["condition"])
        stim = str(row["stimulus_file"])
        raw_sig = row["emg_ch1"]
        t_vec = row["time_s"]

        casc = apply_clancy_cascade(raw_sig, fs=fs, demod_power=demod_power)

        # For Trial 1, preceding resting state is the tail of the 30s baseline;
        # For Trial k > 1, preceding resting state is the tail of ISI_{k-1}
        local_floor = isi_floors.get(t_num - 1, base_stats["base_trimmed_mean"])

        feats = extract_trial_features(
            time_s=t_vec,
            trial_envelope=casc["envelope"],
            base_stats=base_stats,
            local_isi_floor=local_floor
        )

        record = {
            "participant_id": sub_id,
            "expertise_group": group,
            "dominant_hand": row["dominant_hand"],
            "seed": int(row["seed"]),
            "trial_num": t_num,
            "condition": cond,
            "stimulus_file": stim,
            "sampling_rate_hz": fs,
            **feats
        }
        trial_features_list.append(record)

        processed_envelopes.append({
            **record,
            "time_s": t_vec,
            "raw_emg": raw_sig,
            "filtered_emg": casc["notched"],
            "rectified_emg": casc["rectified"],
            "envelope_emg": casc["envelope"]
        })

    features_df = pd.DataFrame(trial_features_list)

    # Save participant's processed trial features CSV & full waveforms Pickle
    out_csv = config.PROCESSED_DIR / f"{sub_id}_{group}_features.csv"
    out_pkl = config.PROCESSED_DIR / f"{sub_id}_{group}_waveforms.pkl"
    features_df.to_csv(out_csv, index=False)
    pd.DataFrame(processed_envelopes).to_pickle(out_pkl)

    # Print quick condition summary for this participant
    cond_summary = features_df.groupby("condition")[["norm_peak_pct", "norm_auc_pct", "zscore_auc"]].mean()
    print("  [CONDITION AVERAGES]")
    for cond_name, vals in cond_summary.iterrows():
        print(
            f"    -> {cond_name:<10}: Peak = {vals['norm_peak_pct']:+6.2f}% | "
            f"AUC = {vals['norm_auc_pct']:+6.2f}% | Z-AUC = {vals['zscore_auc']:+6.2f}"
        )

    if save_qc_plot and len(processed_envelopes) > 0:
        generate_qc_cascade_plot(sub_id, group, processed_envelopes[0], base_stats)

    return features_df


def generate_qc_cascade_plot(sub_id: str, group: str, sample_trial: dict, base_stats: dict):
    """
    Saves a 4-panel Quality Control plot mirroring Figure 3 of Cifrek et al. (2009) / Clancy et al. (2002)
    so you can visually verify all 4 stages of the cascade for each participant.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t = sample_trial["time_s"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Clancy et al. (2002) sEMG Cascade — {sub_id} ({group}) | "
        f"Trial {sample_trial['trial_num']}: {sample_trial['condition']} ({sample_trial['stimulus_file']})",
        fontsize=11, fontweight="bold"
    )

    axes[0].plot(t, sample_trial["raw_emg"], color="black", lw=0.6)
    axes[0].set_ylabel("1. Raw EMG\n(mV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, sample_trial["filtered_emg"], color="#1f77b4", lw=0.6)
    axes[1].set_ylabel("2. Bandpass 20-500Hz\n+ 50Hz Notch (mV)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, sample_trial["rectified_emg"], color="#ff7f0e", lw=0.6)
    axes[2].set_ylabel("3. Full-Wave\nRectified |x| (mV)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, sample_trial["envelope_emg"], color="#d62728", lw=1.8, label="8 Hz Linear Envelope")
    axes[3].axhline(
        base_stats["base_trimmed_mean"], color="green", ls="--", lw=1.5,
        label=f"30s Resting Baseline ({base_stats['base_trimmed_mean']:.4f} mV)"
    )
    axes[3].fill_between(
        t, base_stats["base_trimmed_mean"], sample_trial["envelope_emg"],
        color="#d62728", alpha=0.2, label="Net AUC Integrated Region"
    )
    axes[3].set_ylabel("4. Linear Envelope\n(8 Hz Low-Pass, mV)")
    axes[3].set_xlabel("Time from Video Onset (seconds)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = config.PLOTS_DIR / f"{sub_id}_{group}_cascade_qc.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [QC PLOT SAVED] {plot_path.name}")


def run_batch_postprocessing(demod_power: int = 1):
    """
    Processes all *_epochs.pkl files in ./data/epochs/ and compiles a master
    group-level dataset (master_trial_level_features.csv & master_subject_level_means.csv)
    ready for the 2x2 Mixed ANOVA in stats_analysis.py.
    """
    epoch_files = sorted(config.EPOCHS_DIR.glob("*_epochs.pkl"))
    if not epoch_files:
        print(f"[INFO] No '*_epochs.pkl' files found in {config.EPOCHS_DIR}. Run preprocess_emg.py first.")
        return

    all_participants_dfs = []
    for ep_file in epoch_files:
        df_sub = postprocess_participant(ep_file, demod_power=demod_power)
        all_participants_dfs.append(df_sub)

    master_trials_df = pd.concat(all_participants_dfs, ignore_index=True)
    master_trials_path = config.PROCESSED_DIR / "master_trial_level_features.csv"
    master_trials_df.to_csv(master_trials_path, index=False)

    # Also aggregate within-subject condition means (1 row per participant x condition) for 2x2 Mixed ANOVA
    dv_cols = [
        "raw_peak_mv", "raw_auc_mv_s",
        "norm_peak_pct", "norm_auc_pct",
        "norm_peak_ratio", "norm_auc_ratio",
        "zscore_peak", "zscore_auc",
        "local_isi_norm_peak_pct", "local_isi_norm_auc_pct"
    ]
    subject_means_df = (
        master_trials_df
        .groupby(["participant_id", "expertise_group", "dominant_hand", "condition"], as_index=False)[dv_cols]
        .mean()
    )
    subject_means_path = config.PROCESSED_DIR / "master_subject_level_anova_table.csv"
    subject_means_df.to_csv(subject_means_path, index=False)

    print(f"\n==================================================")
    print(f"[BATCH POST-PROCESSING COMPLETE]")
    print(f"  -> Master Trial-Level CSV   : {master_trials_path}")
    print(f"  -> Master 2x2 ANOVA Table   : {subject_means_path}")
    print(f"==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process sEMG Epochs (Filter Cascade + Baseline Normalization)")
    parser.add_argument("--file", type=str, default=None, help="Specific *_epochs.pkl file to process (optional)")
    parser.add_argument(
        "--demod",
        type=int,
        choices=[1, 2],
        default=1,
        help="Demodulation law d in Clancy et al. (2002): 1 = Full-Wave Rectification (MAV), 2 = Square-Law (RMS)"
    )
    parser.add_argument(
        "--simulate-demo",
        action="store_true",
        help="Auto-generate demo Shimmer epochs if none exist and run full postprocessing"
    )
    args = parser.parse_args()

    if args.simulate_demo:
        existing_epochs = list(config.EPOCHS_DIR.glob("*_epochs.pkl"))
        if not existing_epochs:
            print("[DEMO MODE] No epoch files found yet — invoking preprocess_emg demo generator first...")
            from preprocess_emg import simulate_demo_raw_files, match_and_preprocess_all
            simulate_demo_raw_files(n_per_group=4, fs=512.0, n_trials_per_cond=5)
            match_and_preprocess_all(sync_method="sync_squeeze")
        run_batch_postprocessing(demod_power=args.demod)
    elif args.file:
        postprocess_participant(Path(args.file), demod_power=args.demod)
    else:
        run_batch_postprocessing(demod_power=args.demod)