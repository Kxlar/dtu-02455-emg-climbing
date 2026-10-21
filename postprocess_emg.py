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
      Stage 0: Remove static half-cell DC offset (e.g., -19.1 mV on CH1, -2.1 mV on CH2)
      Stage 1: Noise Reject Filter -> 20-500 Hz Butterworth Bandpass + 50 Hz Notch
               (Note: At fs=512 Hz, Nyquist is 256 Hz, so upper cutoff safely clamps to 0.95 * Nyq = 243.2 Hz)
      Stage 2: Demodulate / Detect -> Full-wave rectification |x|^d (d=1 for MAV, d=2 for RMS)
      Stage 3: Smooth              -> Low-pass Butterworth filter at 8 Hz (Linear Envelope)
      Stage 4: Relinearize         -> Raise smoothed signal to (.)^(1/d)
    """
    x = np.asarray(raw_emg, dtype=np.float64)
    if len(x) < 12:
        zeros = np.zeros_like(x)
        return {"bandpassed": zeros, "notched": zeros, "rectified": zeros, "envelope": zeros}

    # Stage 0: Center around 0 mV before filtering to prevent edge transients
    x = x - np.mean(x)

    nyq = 0.5 * fs
    bp_low = config.BANDPASS_LOW
    bp_high = min(config.BANDPASS_HIGH, nyq * 0.95)

    # Reflective padding length to eliminate filter ringing at t=0.0s and t=4.0s
    padlen = min(int(1.0 * fs), (len(x) // 3) - 1)
    padlen = max(padlen, 0)

    # --- STAGE 1A: Bandpass Filter (20 Hz to min(500 Hz, 0.95*Nyquist)) ---
    sos_bp = signal.butter(
        N=config.FILTER_ORDER,
        Wn=[bp_low, bp_high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )
    x_bp = signal.sosfiltfilt(sos_bp, x, padlen=padlen)

    # --- STAGE 1B: 50 Hz Power-Line Notch Filter (+ harmonics below bp_high) ---
    x_notched = x_bp.copy()
    notch_freqs = [config.NOTCH_FREQ]
    if apply_harmonic_notches:
        notch_freqs = [f for f in (50.0, 100.0, 150.0, 200.0, 250.0, 350.0, 450.0) if f < bp_high]

    for f0 in notch_freqs:
        b_notch, a_notch = signal.iirnotch(w0=f0, Q=config.NOTCH_Q, fs=fs)
        x_notched = signal.filtfilt(b_notch, a_notch, x_notched, padlen=padlen)

    # --- STAGE 2: Detect / Full-Wave Rectification (|x|^d) ---
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
    x_smoothed = signal.sosfiltfilt(sos_lp, x_detected, padlen=padlen)
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
    Computes artifact-resistant resting baseline metrics from the 30-second baseline envelope
    for a specific muscle channel (FDS or ECR):
      - Trims 1.0s at start and end to avoid transition transients
      - Slides a 500ms window across the baseline and computes the 10th-80th percentile
        trimmed mean (guards against accidental swallows or postural micro-twitches)
      - Computes Median & Robust SD (1.4826 * MAD) for Z-score SNR normalization
    """
    trim_samples = int(1.0 * fs)
    if len(baseline_envelope) > 4 * trim_samples:
        env_clean = baseline_envelope[trim_samples:-trim_samples]
    else:
        env_clean = baseline_envelope

    mean_std = float(np.mean(env_clean))
    sd_std = float(np.std(env_clean, ddof=1)) if len(env_clean) > 1 else 1e-12

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
    robust_sd = max(1.4826 * mad, 1e-12)

    return {
        "base_mean": max(mean_std, 1e-12),
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
    Extracts Peak Amplitude and Area Under the Curve (AUC) for a single trial and channel,
    normalized via:
      1. Percentage of Variation vs. 30s Resting Baseline (Protocol Section 5)
      2. Ratio vs. 30s Resting Baseline (1.0 = Resting Baseline, Protocol Section 3.5)
      3. Z-Score SNR vs. 30s Resting Baseline ((E(t) - mu_base) / sigma_base)
      4. Local Pre-Trial ISI Drift-Corrected % Variation (using the last 2.0s of ISI_{k-1})
    """
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 4.0

    # 1. Raw Envelope Features
    raw_peak = float(np.max(trial_envelope))
    raw_mean = float(np.mean(trial_envelope))
    raw_auc = float(trapezoid(y=trial_envelope, x=time_s))
    raw_auc_per_sec = raw_auc / duration_s if duration_s > 0 else raw_mean

    mu_base = base_stats["base_trimmed_mean"]
    mu_base_raw = base_stats["base_mean"]
    sigma_base = base_stats["base_robust_sd"]

    expected_base_auc = mu_base * duration_s
    expected_base_auc_raw = mu_base_raw * duration_s

    # 2. Protocol % Variation & Ratio against 30s Baseline
    peak_pct_change = ((raw_peak - mu_base) / mu_base) * 100.0
    auc_pct_change = ((raw_auc - expected_base_auc) / expected_base_auc) * 100.0

    peak_ratio_robust = raw_peak / mu_base
    auc_ratio_robust = raw_auc / expected_base_auc

    peak_pct_change_untrimmed = ((raw_peak - mu_base_raw) / mu_base_raw) * 100.0
    auc_pct_change_untrimmed = ((raw_auc - expected_base_auc_raw) / expected_base_auc_raw) * 100.0

    # 3. Z-Score SNR Normalization
    z_env = (trial_envelope - mu_base) / sigma_base
    peak_zscore = float(np.max(z_env))
    mean_zscore = float(np.mean(z_env))
    auc_zscore = float(trapezoid(y=z_env, x=time_s))

    # 4. Local Pre-Trial ISI Drift-Corrected Normalization
    if local_isi_floor is not None and local_isi_floor > 0:
        local_expected_auc = local_isi_floor * duration_s
        peak_pct_local_isi = ((raw_peak - local_isi_floor) / local_isi_floor) * 100.0
        auc_pct_local_isi = ((raw_auc - local_expected_auc) / local_expected_auc) * 100.0
    else:
        peak_pct_local_isi = peak_pct_change
        auc_pct_local_isi = auc_pct_change

    return {
        "duration_s": round(duration_s, 4),
        "raw_peak_mv": raw_peak,
        "raw_mean_mv": raw_mean,
        "raw_auc_mv_s": raw_auc,
        "raw_auc_per_sec_mv": raw_auc_per_sec,
        "baseline_trimmed_mean_mv": mu_base,
        "baseline_untrimmed_mean_mv": mu_base_raw,
        "baseline_robust_sd_mv": sigma_base,
        "local_pre_isi_floor_mv": local_isi_floor if local_isi_floor else np.nan,
        "norm_peak_pct": peak_pct_change,
        "norm_auc_pct": auc_pct_change,
        "norm_peak_ratio": peak_ratio_robust,
        "norm_auc_ratio": auc_ratio_robust,
        "norm_peak_pct_untrimmed": peak_pct_change_untrimmed,
        "norm_auc_pct_untrimmed": auc_pct_change_untrimmed,
        "zscore_peak": peak_zscore,
        "zscore_mean": mean_zscore,
        "zscore_auc": auc_zscore,
        "local_isi_norm_peak_pct": peak_pct_local_isi,
        "local_isi_norm_auc_pct": auc_pct_local_isi,
    }


def generate_qc_cascade_plot(sub_id: str, group: str, muscle_code: str, sample_trial: dict, base_stats: dict):
    """
    Saves a 4-panel Quality Control plot for each muscle (FDS and ECR) showing:
      1. Raw sEMG (DC offset removed)
      2. Bandpass (20-500 Hz) + 50 Hz Notch Filtered
      3. Full-Wave Rectified |x|
      4. 8 Hz Linear Envelope vs. 30s Resting Baseline & Integrated AUC
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t = sample_trial["time_s"]
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.0), sharex=True)
    fig.suptitle(
        f"Clancy et al. (2002) sEMG Cascade — Muscle: {muscle_code} ({sample_trial['channel'].upper()})\n"
        f"Participant: {sub_id} ({group}) | Trial {sample_trial['trial_num']}: "
        f"{sample_trial['condition']} ({sample_trial['stimulus_file']})",
        fontsize=11, fontweight="bold"
    )

    raw_zero_mean = sample_trial["raw_emg"] - np.mean(sample_trial["raw_emg"])
    axes[0].plot(t, raw_zero_mean, color="#222222", lw=0.55)
    axes[0].set_ylabel("1. Raw sEMG\n(DC removed, mV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, sample_trial["filtered_emg"], color="#0072B2", lw=0.55)
    axes[1].set_ylabel("2. Bandpass + Notch\n(20-500Hz, 50Hz, mV)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, sample_trial["rectified_emg"], color="#E69F00", lw=0.55)
    axes[2].set_ylabel("3. Full-Wave\nRectified |x| (mV)")
    axes[2].grid(True, alpha=0.3)

    env = sample_trial["envelope_emg"]
    base_mu = base_stats["base_trimmed_mean"]
    axes[3].plot(t, env, color="#D55E00", lw=1.9, label=f"{muscle_code} 8 Hz Linear Envelope")
    axes[3].axhline(
        base_mu, color="#009E73", ls="--", lw=1.5,
        label=f"30s Resting Baseline ({base_mu:.4f} mV)"
    )
    axes[3].fill_between(
        t, base_mu, env,
        color="#D55E00", alpha=0.22,
        label=f"Integrated AUC ({sample_trial['norm_auc_pct']:+.1f}% vs. Baseline)"
    )
    peak_idx = int(np.argmax(env))
    axes[3].scatter(
        [t[peak_idx]], [env[peak_idx]], color="black", marker="*", s=95, zorder=5,
        label=f"Peak Amplitude ({sample_trial['norm_peak_pct']:+.1f}%)"
    )
    axes[3].set_ylabel("4. Linear Envelope\n(8 Hz Low-Pass, mV)")
    axes[3].set_xlabel("Time from Video Stimulus Onset (seconds)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = config.PLOTS_DIR / f"{sub_id}_{group}_{muscle_code}_cascade_qc.png"
    plt.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"    [QC PLOT SAVED] {plot_path.name}")


def postprocess_participant(epoch_pkl_path: Path, demod_power: int = 1, save_qc_plot: bool = True) -> pd.DataFrame:
    """
    Runs the post-processing pipeline on a single participant's sliced epochs (.pkl).
    Automatically processes both CH1 (FDS) and CH2 (ECR) independently.
    """
    epochs_df = pd.read_pickle(epoch_pkl_path)
    sub_id = str(epochs_df["participant_id"].iloc[0])
    group = str(epochs_df["expertise_group"].iloc[0])
    fs = float(epochs_df["sampling_rate_hz"].iloc[0])

    print(f"\n------------------------------------------------------------")
    print(f"Post-processing Participant: {sub_id} ({group}) | fs = {fs:.2f} Hz")
    print(f"------------------------------------------------------------")

    base_rows = epochs_df[epochs_df["event_type"] == "baseline"]
    if base_rows.empty:
        raise ValueError(f"No 'baseline' epoch found in {epoch_pkl_path.name}!")

    trial_features_list = []
    processed_envelopes = []

    # Iterate through each configured EMG channel: ch1 -> FDS, ch2 -> ECR
    for ch_key, ch_meta in config.EMG_CHANNEL_MAP.items():
        col_name = f"emg_{ch_key}"
        if col_name not in epochs_df.columns:
            print(f"  [SKIP] Column '{col_name}' not present in {epoch_pkl_path.name}")
            continue

        muscle_code = ch_meta["muscle_code"]   # 'FDS' or 'ECR'
        muscle_name = ch_meta["muscle_name"]

        # 1. Process 30s Resting Baseline for this muscle channel
        base_raw = base_rows.iloc[0][col_name]
        base_casc = apply_clancy_cascade(base_raw, fs=fs, demod_power=demod_power)
        base_stats = compute_robust_baseline_stats(base_casc["envelope"], fs=fs)

        print(
            f"  -> Muscle: {muscle_code:<3} ({ch_key.upper()}) | "
            f"30s Baseline Trimmed Mean = {base_stats['base_trimmed_mean']:.6f} mV | "
            f"Robust SD = {base_stats['base_robust_sd']:.6f} mV"
        )

        # 2. Pre-compute settled ISI resting floors (last 2.0s of each ISI) for this muscle channel
        isi_floors = {}
        isi_rows = epochs_df[epochs_df["event_type"] == "isi"]
        for _, isi_row in isi_rows.iterrows():
            t_num = int(isi_row["trial_num"])
            isi_env = apply_clancy_cascade(isi_row[col_name], fs=fs, demod_power=demod_power)["envelope"]
            tail_samples = min(int(2.0 * fs), len(isi_env))
            if tail_samples > 0:
                isi_floors[t_num] = float(np.median(isi_env[-tail_samples:]))

        # 3. Process all Video Observation Trials for this muscle channel
        trial_rows = epochs_df[epochs_df["event_type"] == "trial_video"].sort_values("trial_num")
        sample_climb_trial_for_qc = None

        for _, row in trial_rows.iterrows():
            t_num = int(row["trial_num"])
            cond = str(row["condition"])
            stim = str(row["stimulus_file"])
            raw_sig = row[col_name]
            t_vec = row["time_s"]

            casc = apply_clancy_cascade(raw_sig, fs=fs, demod_power=demod_power)
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
                "channel": ch_key,
                "muscle": muscle_code,
                "muscle_full_name": muscle_name,
                "seed": int(row["seed"]),
                "trial_num": t_num,
                "condition": cond,
                "stimulus_file": stim,
                "sampling_rate_hz": fs,
                **feats
            }
            trial_features_list.append(record)

            wave_rec = {
                **record,
                "time_s": t_vec,
                "raw_emg": raw_sig,
                "filtered_emg": casc["notched"],
                "rectified_emg": casc["rectified"],
                "envelope_emg": casc["envelope"]
            }
            processed_envelopes.append(wave_rec)

            if sample_climb_trial_for_qc is None and cond == "climbing":
                sample_climb_trial_for_qc = wave_rec

        if save_qc_plot and sample_climb_trial_for_qc is not None:
            generate_qc_cascade_plot(sub_id, group, muscle_code, sample_climb_trial_for_qc, base_stats)

    features_df = pd.DataFrame(trial_features_list)

    # Save participant's dual-channel trial features CSV & full waveforms Pickle
    out_csv = config.PROCESSED_DIR / f"{sub_id}_{group}_features.csv"
    out_pkl = config.PROCESSED_DIR / f"{sub_id}_{group}_waveforms.pkl"
    features_df.to_csv(out_csv, index=False)
    pd.DataFrame(processed_envelopes).to_pickle(out_pkl)

    # Print quick condition summary per muscle
    cond_summary = (
        features_df
        .groupby(["muscle", "condition"])[["norm_peak_pct", "norm_auc_pct", "zscore_auc", "local_isi_norm_auc_pct"]]
        .mean()
    )
    print("  [CONDITION AVERAGES BY MUSCLE]")
    for (m_code, cond_name), vals in cond_summary.iterrows():
        print(
            f"    [{m_code}] {cond_name:<10} -> Peak: {vals['norm_peak_pct']:+6.2f}% | "
            f"AUC: {vals['norm_auc_pct']:+6.2f}% | Z-AUC: {vals['zscore_auc']:+5.2f} | "
            f"ISI-Corrected AUC: {vals['local_isi_norm_auc_pct']:+6.2f}%"
        )

    return features_df


def run_batch_postprocessing(demod_power: int = 1):
    """
    Processes all *_epochs.pkl files in ./data/epochs/ and builds the two master tables:
      1. master_trial_level_features.csv (all trials x 2 muscles per participant)
      2. master_subject_level_anova_table.csv (subject condition means per muscle for 2x2 ANOVA)
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

    dv_cols = [
        "raw_peak_mv", "raw_auc_mv_s", "raw_auc_per_sec_mv",
        "norm_peak_pct", "norm_auc_pct",
        "norm_peak_ratio", "norm_auc_ratio",
        "norm_peak_pct_untrimmed", "norm_auc_pct_untrimmed",
        "zscore_peak", "zscore_mean", "zscore_auc",
        "local_isi_norm_peak_pct", "local_isi_norm_auc_pct"
    ]
    subject_means_df = (
        master_trials_df
        .groupby(
            ["participant_id", "expertise_group", "dominant_hand", "channel", "muscle", "condition"],
            as_index=False
        )[dv_cols]
        .mean()
    )
    subject_means_path = config.PROCESSED_DIR / "master_subject_level_anova_table.csv"
    subject_means_df.to_csv(subject_means_path, index=False)

    print(f"\n============================================================")
    print(f"[BATCH DUAL-CHANNEL POST-PROCESSING COMPLETE]")
    print(f"  -> Participants Processed   : {len(epoch_files)} (x2 channels: FDS & ECR)")
    print(f"  -> Master Trial-Level CSV   : {master_trials_path}")
    print(f"  -> Master 2x2 ANOVA Table   : {subject_means_path}")
    print(f"============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process Dual-Channel sEMG Epochs (FDS & ECR)")
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
        help="Auto-generate demo dual-channel Shimmer epochs if none exist and run full postprocessing"
    )
    args = parser.parse_args()

    if args.simulate_demo:
        existing_epochs = list(config.EPOCHS_DIR.glob("*_epochs.pkl"))
        if not existing_epochs:
            print("[DEMO MODE] No epoch files found yet — running preprocess_emg demo generator first...")
            from preprocess_emg import simulate_demo_raw_files, match_and_preprocess_all
            simulate_demo_raw_files(n_per_group=4, fs=512.0, n_trials_per_cond=5)
            match_and_preprocess_all()
        run_batch_postprocessing(demod_power=args.demod)
    elif args.file:
        postprocess_participant(Path(args.file), demod_power=args.demod)
    else:
        run_batch_postprocessing(demod_power=args.demod)