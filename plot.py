# ./Experiment in Cognitive Science 02455/plot.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import config

# --- PUBLICATION STYLE CONFIGURATION (Colorblind-Safe Okabe-Ito Palette) ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150
})

PALETTE = {
    "climber": "#D55E00",      # Vermilion / Orange-Red
    "non_climber": "#0072B2"   # Deep Sky / Navy Blue
}
GROUP_LABELS = {
    "climber": "Expert Climbers (Group A)",
    "non_climber": "Non-Climbers (Group B)"
}


# ==============================================================================
# MODE 1: RAW SHIMMER SD FILE INSPECTOR (--raw-shimmer <filepath>)
# ==============================================================================
def plot_raw_shimmer_file(
    csv_filepath: str,
    sampling_rate_hz: float = config.DEFAULT_SAMPLING_RATE,
    show_window: bool = True
):
    """
    Directly loads and plots any raw Shimmer Consensys CSV/TSV file (handles 'sep=\\t',
    units row, and '-1' offline SD timestamps).
    Generates TWO figures:
      1. Overview Inspector (Raw CH1/CH2, 8 Hz Envelopes, and 3-Axis Accelerometer)
      2. 4x2 Side-by-Side Clancy Filter Cascade Showcase for both FDS (CH1) and ECR (CH2)
    """
    from preprocess_emg import load_shimmer_file
    from postprocess_emg import apply_clancy_cascade

    csv_path = Path(csv_filepath)
    clean_df, fs_est, offline_mode = load_shimmer_file(csv_path, default_fs=sampling_rate_hz)

    if not offline_mode and "timestamp_unix_ms" in clean_df.columns:
        time_s = (clean_df["timestamp_unix_ms"].values - clean_df["timestamp_unix_ms"].iloc[0]) / 1000.0
        time_label = "Time from Recording Start (seconds, Unix Clock)"
    else:
        time_s = np.arange(len(clean_df), dtype=np.float64) / fs_est
        time_label = f"Time from Recording Start (seconds, Offline SD @ {fs_est:g} Hz)"

    has_accel = "accel_mag" in clean_df.columns
    n_rows = 3 if has_accel else 2

    fig, axes = plt.subplots(n_rows, 1, figsize=(13.5, 3.1 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(
        f"Raw Shimmer Dual-Channel sEMG Inspector — {csv_path.name} (fs = {fs_est:g} Hz)",
        fontsize=13, fontweight="bold"
    )

    casc_ch1, casc_ch2 = None, None

    # --- Panel 1: Zero-Mean Raw sEMG (CH1 = FDS, CH2 = ECR) ---
    if "emg_ch1" in clean_df.columns:
        ch1_raw = clean_df["emg_ch1"].values
        dc1 = float(np.mean(ch1_raw))
        casc_ch1 = apply_clancy_cascade(ch1_raw, fs=fs_est)
        axes[0].plot(
            time_s, ch1_raw - dc1, color="#0072B2", lw=0.5, alpha=0.85,
            label=f"CH1: FDS Raw (DC offset {dc1:+.2f} mV removed)"
        )
    if "emg_ch2" in clean_df.columns:
        ch2_raw = clean_df["emg_ch2"].values
        dc2 = float(np.mean(ch2_raw))
        casc_ch2 = apply_clancy_cascade(ch2_raw, fs=fs_est)
        axes[0].plot(
            time_s, ch2_raw - dc2, color="#D55E00", lw=0.5, alpha=0.80,
            label=f"CH2: ECR Raw (DC offset {dc2:+.2f} mV removed)"
        )

    axes[0].set_ylabel("Raw sEMG\n(Zero-Mean, mV)")
    axes[0].set_title("1. Raw Unfiltered sEMG Waveforms (Static Half-Cell DC Offset Subtracted)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Post-Processed 8 Hz Clancy Linear Envelopes ---
    if casc_ch1 is not None:
        env1 = casc_ch1["envelope"]
        axes[1].plot(
            time_s, env1, color="#0072B2", lw=1.6,
            label=f"CH1: FDS 8 Hz Linear Envelope (Peak = {np.max(env1):.3f} mV)"
        )
    if casc_ch2 is not None:
        env2 = casc_ch2["envelope"]
        axes[1].plot(
            time_s, env2, color="#D55E00", lw=1.6, alpha=0.90,
            label=f"CH2: ECR 8 Hz Linear Envelope (Peak = {np.max(env2):.3f} mV)"
        )

    axes[1].set_ylabel("Clancy Envelope\n(mV)")
    axes[1].set_title(
        f"2. Clancy et al. (2002) Linear Envelopes "
        f"({config.BANDPASS_LOW:g}-{config.BANDPASS_HIGH:g} Hz Bandpass -> {config.NOTCH_FREQ:g} Hz Notch -> |x| -> {config.ENVELOPE_LOWPASS:g} Hz Low-Pass)"
    )
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Onboard 3-Axis Accelerometer Magnitude ---
    if has_accel:
        axes[2].plot(
            time_s, clean_df["accel_mag"].values, color="#009E73", lw=0.9,
            label="3-Axis Low-Noise Accelerometer |a| (m/s²)"
        )
        axes[2].set_ylabel("Accel |a|\n(m/s²)")
        axes[2].set_title("3. Onboard Accelerometer Magnitude (Physical Movement & Tap-Sync Verification)")
        axes[2].legend(loc="upper right", fontsize=8.5)
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel(time_label)
    plt.tight_layout()

    config.STATS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.STATS_PLOTS_DIR / f"raw_shimmer_inspection_{csv_path.stem}.png"
    plt.savefig(out_path, dpi=180)
    print(f"[RAW SHIMMER PLOT SAVED] {out_path}")

    # --- Also plot the 4x2 Filter Cascade Showcase for FDS & ECR on this raw file ---
    if casc_ch1 is not None or casc_ch2 is not None:
        fig_c, axes_c = plt.subplots(4, 2, figsize=(15, 9.2), sharex=True)
        fig_c.suptitle(
            f"Clancy et al. (2002) 4-Stage Filter Cascade Showcase (FDS vs. ECR) — {csv_path.name}",
            fontsize=13, fontweight="bold"
        )
        ch_items = [
            (0, "FDS (CH1: Finger Flexor)", clean_df.get("emg_ch1"), casc_ch1, "#0072B2"),
            (1, "ECR (CH2: Wrist Extensor)", clean_df.get("emg_ch2"), casc_ch2, "#D55E00"),
        ]
        for col_idx, title_str, raw_series, casc, col_hex in ch_items:
            if raw_series is None or casc is None:
                continue
            raw_v = raw_series.values
            raw_zm = raw_v - np.mean(raw_v)

            axes_c[0, col_idx].plot(time_s, raw_zm, color="#222222", lw=0.45)
            axes_c[0, col_idx].set_title(f"{title_str}\nStage 1: Raw sEMG (DC Offset Subtracted)")
            axes_c[0, col_idx].set_ylabel("mV")
            axes_c[0, col_idx].grid(True, alpha=0.3)

            axes_c[1, col_idx].plot(time_s, casc["notched"], color=col_hex, lw=0.45)
            axes_c[1, col_idx].set_title(f"Stage 2: Bandpass ({config.BANDPASS_LOW:g}-{config.BANDPASS_HIGH:g} Hz) + {config.NOTCH_FREQ:g} Hz Notch")
            axes_c[1, col_idx].set_ylabel("mV")
            axes_c[1, col_idx].grid(True, alpha=0.3)

            axes_c[2, col_idx].plot(time_s, casc["rectified"], color="#E69F00", lw=0.45)
            axes_c[2, col_idx].set_title("Stage 3: Full-Wave Rectification |x|")
            axes_c[2, col_idx].set_ylabel("mV")
            axes_c[2, col_idx].grid(True, alpha=0.3)

            axes_c[3, col_idx].plot(time_s, casc["envelope"], color=col_hex, lw=1.7, label=f"{config.ENVELOPE_LOWPASS:g} Hz Linear Envelope")
            axes_c[3, col_idx].set_title(f"Stage 4: Linear Envelope ({config.ENVELOPE_LOWPASS:g} Hz Low-Pass Butterworth)")
            axes_c[3, col_idx].set_ylabel("mV")
            axes_c[3, col_idx].set_xlabel(time_label)
            axes_c[3, col_idx].legend(loc="upper right", fontsize=8.5)
            axes_c[3, col_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        cascade_out = config.STATS_PLOTS_DIR / f"raw_shimmer_cascade_FDS_ECR_{csv_path.stem}.png"
        fig_c.savefig(cascade_out, dpi=180)
        print(f"[RAW SHIMMER CASCADE SAVED] {cascade_out}")

    if show_window:
        plt.show()
    plt.close("all")


# ==============================================================================
# MODE 2: PUBLICATION FIGURES FOR BOTH FDS (CH1) AND ECR (CH2)
# ==============================================================================
def draw_half_violin(ax, data, pos, color, width=0.22, side="left", alpha=0.40):
    """Draws a half-violin kernel density cloud on the specified side of `pos`."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 3 or np.std(data) == 0:
        return

    parts = ax.violinplot(
        data, positions=[pos], widths=width * 2,
        showmeans=False, showextrema=False, showmedians=False
    )
    for b in parts["bodies"]:
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        if side == "left":
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        else:
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_facecolor(color)
        b.set_edgecolor(color)
        b.set_linewidth(1.2)
        b.set_alpha(alpha)


def plot_raincloud_interaction_by_muscle(
    df_subj: pd.DataFrame,
    muscle: str,
    out_dir: Path,
    use_zscore: bool = False
):
    """
    FIGURE 1 (per muscle: FDS and ECR):
    2-Panel Publication Raincloud + 2x2 Factorial Interaction Plot for:
      - Panel A: Area Under the Curve (AUC)
      - Panel B: Peak Amplitude
    """
    sub_m = df_subj[df_subj["muscle"] == muscle].copy()
    if sub_m.empty:
        return

    muscle_desc = "Finger Flexor (CH1)" if muscle == "FDS" else "Wrist Extensor (CH2)"

    if use_zscore:
        dvs = [
            ("zscore_auc", f"[{muscle}] Normalized AUC (Z-Score SNR Units)", 0.0),
            ("zscore_peak", f"[{muscle}] Normalized Peak Amplitude (Z-Score SNR Units)", 0.0)
        ]
        suffix = "zscore"
    else:
        dvs = [
            ("norm_auc_pct", f"[{muscle}] Normalized AUC (% Change vs. 30s Baseline)", 0.0),
            ("norm_peak_pct", f"[{muscle}] Normalized Peak Amplitude (% Change vs. 30s Baseline)", 0.0)
        ]
        suffix = "pct"

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))
    fig.suptitle(
        f"Muscle: {muscle} ({muscle_desc}) — 2×2 Motor Resonance Interaction & Raincloud Distributions",
        fontsize=13.5, fontweight="bold", y=0.98
    )

    rng = np.random.default_rng(42)

    for ax_idx, (dv, ylabel, null_val) in enumerate(dvs):
        if dv not in sub_m.columns:
            continue
        ax = axes[ax_idx]
        wide = sub_m.pivot(
            index=["participant_id", "expertise_group"],
            columns="condition",
            values=dv
        ).dropna().reset_index()

        x_base = {"everyday": 0.0, "climbing": 1.0}
        grp_offset = {"non_climber": -0.06, "climber": +0.06}

        for grp in ["non_climber", "climber"]:
            sub_w = wide[wide["expertise_group"] == grp]
            if sub_w.empty or "everyday" not in sub_w or "climbing" not in sub_w:
                continue

            c = PALETTE[grp]
            x_ev = x_base["everyday"] + grp_offset[grp]
            x_cl = x_base["climbing"] + grp_offset[grp]

            y_ev = sub_w["everyday"].values
            y_cl = sub_w["climbing"].values

            draw_half_violin(ax, y_ev, pos=x_ev - 0.10, color=c, width=0.20, side="left", alpha=0.38)
            draw_half_violin(ax, y_cl, pos=x_cl + 0.10, color=c, width=0.20, side="right", alpha=0.38)

            jitter_ev = rng.uniform(-0.025, 0.025, size=len(sub_w))
            jitter_cl = rng.uniform(-0.025, 0.025, size=len(sub_w))

            for i in range(len(sub_w)):
                ax.plot(
                    [x_ev + jitter_ev[i], x_cl + jitter_cl[i]],
                    [y_ev[i], y_cl[i]],
                    color=c, alpha=0.18, lw=0.9, zorder=2
                )

            ax.scatter(x_ev + jitter_ev, y_ev, color=c, alpha=0.55, s=24, zorder=3)
            ax.scatter(x_cl + jitter_cl, y_cl, color=c, alpha=0.55, s=24, zorder=3)

            bp = ax.boxplot(
                [y_ev, y_cl],
                positions=[x_ev - 0.08, x_cl + 0.08],
                widths=0.05, patch_artist=True, showfliers=False, zorder=4
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("white")
                patch.set_edgecolor(c)
                patch.set_linewidth(1.3)
            for elem in ["whiskers", "caps", "medians"]:
                for item in bp[elem]:
                    item.set_color(c)
                    item.set_linewidth(1.3)

            mean_ev = float(np.mean(y_ev))
            sem_ev = float(stats.sem(y_ev)) if len(y_ev) > 1 else 0.0
            mean_cl = float(np.mean(y_cl))
            sem_cl = float(stats.sem(y_cl)) if len(y_cl) > 1 else 0.0
            slope = mean_cl - mean_ev

            ax.errorbar(
                [x_ev, x_cl], [mean_ev, mean_cl], yerr=[sem_ev, sem_cl],
                color=c, marker="o", markersize=9, markerfacecolor=c,
                markeredgecolor="white", markeredgewidth=1.5,
                lw=3.2, capsize=6, capthick=2.0,
                label=f"{GROUP_LABELS[grp]} (Δ Slope = {slope:+.1f})",
                zorder=6
            )

        ax.axhline(null_val, color="#444444", ls="--", lw=1.3, alpha=0.8, label="30s Resting Baseline Floor")
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(
            ["Everyday Grasping\n(Condition B: Control)", "Climbing Grips\n(Condition A: Target)"],
            fontweight="bold"
        )
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel(ylabel)
        metric_title = "Area Under the Curve (AUC)" if ax_idx == 0 else "Peak Amplitude"
        ax.set_title(f"{'A' if ax_idx == 0 else 'B'}. {muscle} {metric_title}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / f"fig1_{muscle}_raincloud_interaction_{suffix}.png"
    plt.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def plot_fds_vs_ecr_coactivation(df_subj: pd.DataFrame, df_trials: pd.DataFrame, out_dir: Path):
    """
    FIGURE 2:
    3-Panel Dual-Channel Comparison (Agonist FDS vs. Wrist Stabilizer ECR)
    """
    muscles = set(df_subj["muscle"].unique())
    if not {"FDS", "ECR"}.issubset(muscles):
        return

    has_trials = df_trials is not None and not df_trials.empty and "muscle" in df_trials.columns
    n_cols = 3 if has_trials else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 5.0))
    fig.suptitle(
        "Dual-Channel Forearm Comparison: Finger Flexor (FDS, CH1) vs. Wrist Extensor Stabilizer (ECR, CH2)",
        fontsize=13, fontweight="bold"
    )

    for idx, m_code in enumerate(["FDS", "ECR"]):
        ax = axes[idx]
        sub_m = df_subj[df_subj["muscle"] == m_code]
        for grp in ["climber", "non_climber"]:
            means, sems = [], []
            for cond in ["everyday", "climbing"]:
                vals = sub_m[(sub_m["expertise_group"] == grp) & (sub_m["condition"] == cond)]["norm_auc_pct"].dropna().values
                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                sems.append(float(stats.sem(vals)) if len(vals) > 1 else 0.0)

            offset = -0.03 if grp == "climber" else 0.03
            ax.errorbar(
                [0 + offset, 1 + offset], means, yerr=sems,
                marker="o", markersize=8, lw=2.6, capsize=5,
                color=PALETTE[grp], label=GROUP_LABELS[grp]
            )

        ax.axhline(0.0, color="gray", ls="--", lw=1.2, label="30s Resting Baseline")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Everyday Grasping\n(Control)", "Climbing Grips\n(Target)"])
        role = "CH1 Finger Flexor" if m_code == "FDS" else "CH2 Wrist Extensor"
        ax.set_title(f"{'A' if idx == 0 else 'B'}. {m_code} ({role})")
        ax.set_ylabel("Normalized AUC (% Change vs. Baseline)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.25)

    if has_trials:
        ax_c = axes[2]
        climb_trials = df_trials[df_trials["condition"] == "climbing"]
        piv = climb_trials.pivot_table(
            index=["participant_id", "expertise_group", "trial_num"],
            columns="muscle",
            values="norm_auc_pct"
        ).dropna().reset_index()

        for grp in ["non_climber", "climber"]:
            sub_p = piv[piv["expertise_group"] == grp]
            if len(sub_p) < 3:
                continue
            x_fds = sub_p["FDS"].values
            y_ecr = sub_p["ECR"].values
            r_val, p_val = stats.pearsonr(x_fds, y_ecr)

            ax_c.scatter(
                x_fds, y_ecr, color=PALETTE[grp], alpha=0.45, s=22,
                label=f"{GROUP_LABELS[grp]}\n(r = {r_val:.2f}, p = {p_val:.3f})"
            )
            if np.std(x_fds) > 0:
                m_slope, b_int = np.polyfit(x_fds, y_ecr, 1)
                x_line = np.linspace(np.min(x_fds), np.max(x_fds), 50)
                ax_c.plot(x_line, m_slope * x_line + b_int, color=PALETTE[grp], lw=2.2)

        ax_c.axhline(0.0, color="gray", ls=":", lw=1.0)
        ax_c.axvline(0.0, color="gray", ls=":", lw=1.0)
        ax_c.set_title("C. Agonist–Stabilizer Co-Activation\n(Climbing Trials: FDS vs. ECR)")
        ax_c.set_xlabel("FDS Normalized AUC (% vs. Baseline)")
        ax_c.set_ylabel("ECR Normalized AUC (% vs. Baseline)")
        ax_c.legend(loc="upper left", fontsize=8)
        ax_c.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = out_dir / "fig2_FDS_vs_ECR_coactivation.png"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def plot_grand_average_envelopes_by_muscle(waveforms_list: list, out_dir: Path):
    """
    FIGURE 3 (per muscle: FDS and ECR):
    Time-resolved Grand Average 8 Hz sEMG Linear Envelope (Mean ± SEM) across the 4.0s video window.
    """
    if not waveforms_list:
        return

    common_t = np.linspace(0.0, 4.0, 400)
    muscles = sorted(list({w.get("muscle", "FDS") for w in waveforms_list}), reverse=True)

    for m_code in muscles:
        traces = {
            ("climber", "climbing"): [],
            ("climber", "everyday"): [],
            ("non_climber", "climbing"): [],
            ("non_climber", "everyday"): []
        }

        for row in waveforms_list:
            if row.get("muscle", "FDS") != m_code:
                continue
            key = (row["expertise_group"], row["condition"])
            if key not in traces:
                continue
            t_vec = row["time_s"]
            env = row["envelope_emg"]
            base_mu = row["baseline_trimmed_mean_mv"]
            if len(t_vec) < 10 or base_mu <= 0:
                continue

            pct_trace = ((env - base_mu) / base_mu) * 100.0
            traces[key].append(np.interp(common_t, t_vec, pct_trace))

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
        fig.suptitle(
            f"Muscle: {m_code} — Time-Resolved sEMG Linear Envelope During Action Observation (Mean ± SEM)",
            fontsize=13, fontweight="bold"
        )

        for idx, grp in enumerate(["climber", "non_climber"]):
            ax = axes[idx]
            c = PALETTE[grp]

            for cond, ls_style in [("climbing", "-"), ("everyday", "--")]:
                mat = np.array(traces[(grp, cond)])
                if mat.ndim != 2 or len(mat) == 0:
                    continue
                mean_curve = np.mean(mat, axis=0)
                sem_curve = stats.sem(mat, axis=0) if len(mat) > 1 else np.zeros_like(mean_curve)

                line_col = c if cond == "climbing" else "#555555"
                label_str = f"{cond.capitalize()} Videos (n={len(mat)} trials)"
                ax.plot(common_t, mean_curve, color=line_col, ls=ls_style, lw=2.2, label=label_str)
                ax.fill_between(
                    common_t,
                    mean_curve - sem_curve,
                    mean_curve + sem_curve,
                    color=line_col,
                    alpha=0.22 if cond == "climbing" else 0.12
                )

            ax.axhline(0.0, color="black", ls=":", lw=1.2, label="30s Resting Baseline (0%)")
            ax.set_title(f"{'A' if idx == 0 else 'B'}. {GROUP_LABELS[grp]}")
            ax.set_xlabel("Time from Video Stimulus Onset (seconds)")
            if idx == 0:
                ax.set_ylabel(f"{m_code} Envelope Modulation (% vs. Baseline)")
            ax.set_xlim(0.0, 4.0)
            ax.legend(loc="upper right", fontsize=8.5)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        out_path = out_dir / f"fig3_{m_code}_grand_average_timecourse.png"
        plt.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"[SAVED] {out_path}")


def plot_isi_drift_robustness_by_muscle(df_trials: pd.DataFrame, out_dir: Path):
    """
    FIGURE 4 (per muscle: FDS and ECR):
    Chronological Trial Sequence (Trials 1 -> 30) comparing standard 30s baseline
    normalization against local Pre-Trial ISI normalization.
    """
    if df_trials is None or df_trials.empty:
        return

    muscles = sorted(df_trials["muscle"].unique(), reverse=True) if "muscle" in df_trials.columns else ["FDS"]

    for m_code in muscles:
        sub_m = df_trials[df_trials["muscle"] == m_code] if "muscle" in df_trials.columns else df_trials
        if sub_m.empty or "local_isi_norm_auc_pct" not in sub_m.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), sharey=True)
        fig.suptitle(
            f"Muscle: {m_code} — Chronological Stability & Pre-Trial ISI Drift Check Across Trials",
            fontsize=12.5, fontweight="bold"
        )

        metrics = [
            ("norm_auc_pct", "A. Standard 30s Baseline Normalization"),
            ("local_isi_norm_auc_pct", "B. Local Pre-Trial ISI Drift-Corrected Normalization")
        ]

        for idx, (col, title) in enumerate(metrics):
            ax = axes[idx]
            for grp in ["climber", "non_climber"]:
                sub_g = sub_m[sub_m["expertise_group"] == grp]
                if sub_g.empty:
                    continue
                grouped = sub_g.groupby("trial_num")[col].agg(["mean", "sem"]).reset_index()
                ax.plot(
                    grouped["trial_num"], grouped["mean"],
                    color=PALETTE[grp], marker="o", markersize=4, lw=1.8,
                    label=GROUP_LABELS[grp]
                )
                ax.fill_between(
                    grouped["trial_num"],
                    grouped["mean"] - grouped["sem"].fillna(0),
                    grouped["mean"] + grouped["sem"].fillna(0),
                    color=PALETTE[grp], alpha=0.18
                )

            ax.axhline(0.0, color="gray", ls="--", lw=1.2)
            ax.set_title(title)
            ax.set_xlabel("Chronological Trial Number")
            if idx == 0:
                ax.set_ylabel(f"{m_code} Normalized AUC (% Change)")
            ax.legend(loc="upper right", fontsize=8.5)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        out_path = out_dir / f"fig4_{m_code}_chronological_isi_drift_check.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[SAVED] {out_path}")


def plot_filter_cascade_showcase(waveforms_list: list, out_dir: Path):
    """
    FIGURE 5:
    Clancy et al. (2002) 4-Stage Filter Cascade Showcase for BOTH FDS (CH1) and ECR (CH2).
    Generates:
      1. `fig5_FDS_ECR_filter_cascade_showcase.png` (4 rows x 2 columns side-by-side comparison)
      2. `fig5a_FDS_filter_cascade.png` & `fig5b_ECR_filter_cascade.png` (individual 4-panel plots)
    Stages shown:
      - Stage 1: Raw sEMG (DC offset removed)
      - Stage 2: Bandpass (20-500 Hz) + 50 Hz Notch Filtered
      - Stage 3: Full-Wave Rectified |x|
      - Stage 4: 8 Hz Low-Pass Linear Envelope + 30s Resting Baseline + Shaded Net AUC + Peak Marker
    """
    from postprocess_emg import apply_clancy_cascade

    samples = {}

    # 1. Try to pick a representative climber 'climbing' trial with full cascade waveforms from waveforms_list
    if waveforms_list:
        for m_code in ["FDS", "ECR"]:
            candidates = [
                w for w in waveforms_list
                if w.get("muscle") == m_code
                and w.get("expertise_group") == "climber"
                and w.get("condition") == "climbing"
                and "raw_emg" in w
                and "filtered_emg" in w
            ]
            if not candidates:
                candidates = [w for w in waveforms_list if w.get("muscle") == m_code and "raw_emg" in w]
            if candidates:
                samples[m_code] = candidates[0]

    # 2. If running in --simulate-demo (where full raw waveforms weren't on disk), synthesize realistic 4.0s traces
    if not {"FDS", "ECR"}.issubset(samples.keys()):
        rng = np.random.default_rng(2026)
        fs = config.DEFAULT_SAMPLING_RATE
        t_vec = np.arange(0.0, 4.0, 1.0 / fs)
        n_pts = len(t_vec)

        for m_code, dc_off, base_sd, burst_gain in [
            ("FDS", -19.14, 0.015, 2.2),
            ("ECR", -2.06, 0.018, 1.8)
        ]:
            if m_code in samples:
                continue
            # Realistic motor unit action potential train + 50 Hz power-line hum + low-freq motion sway
            mod_profile = 1.0 + burst_gain * (
                np.exp(-0.5 * ((t_vec - 1.45) / 0.35) ** 2) +
                0.85 * np.exp(-0.5 * ((t_vec - 2.65) / 0.42) ** 2)
            )
            emg_band = rng.normal(0.0, base_sd, n_pts) * mod_profile
            hum_50hz = 0.010 * np.sin(2.0 * np.pi * 50.0 * t_vec)
            motion_sway = 0.018 * np.sin(2.0 * np.pi * 1.2 * t_vec)
            raw_sig = dc_off + emg_band + hum_50hz + motion_sway

            casc = apply_clancy_cascade(raw_sig, fs=fs, demod_power=1)
            base_mu = float(base_sd * 0.80)
            env = casc["envelope"]
            raw_auc = float(np.trapezoid(env, t_vec))
            exp_auc = base_mu * 4.0
            norm_auc_pct = ((raw_auc - exp_auc) / exp_auc) * 100.0
            norm_peak_pct = ((float(np.max(env)) - base_mu) / base_mu) * 100.0

            samples[m_code] = {
                "participant_id": "sub-DEMO",
                "expertise_group": "climber",
                "muscle": m_code,
                "channel": "ch1" if m_code == "FDS" else "ch2",
                "trial_num": 1,
                "condition": "climbing",
                "stimulus_file": "crimp_01.mp4",
                "time_s": t_vec,
                "raw_emg": raw_sig,
                "filtered_emg": casc["notched"],
                "rectified_emg": casc["rectified"],
                "envelope_emg": env,
                "baseline_trimmed_mean_mv": base_mu,
                "norm_auc_pct": norm_auc_pct,
                "norm_peak_pct": norm_peak_pct
            }

    # --- A. Combined 4x2 Side-by-Side Showcase Figure (FDS Left Column, ECR Right Column) ---
    fig, axes = plt.subplots(4, 2, figsize=(15.0, 9.2), sharex=True)
    fig.suptitle(
        "Clancy et al. (2002) 4-Stage sEMG Filter Cascade Showcase — CH1: FDS (Finger Flexor) vs. CH2: ECR (Wrist Extensor)",
        fontsize=13, fontweight="bold", y=0.98
    )

    col_specs = [
        (0, "FDS", "CH1: Flexor Digitorum Superficialis (FDS)", "#0072B2"),
        (1, "ECR", "CH2: Extensor Carpi Radialis (ECR)", "#D55E00")
    ]

    for col_idx, m_code, m_title, theme_color in col_specs:
        s = samples[m_code]
        t = s["time_s"]
        raw_zm = s["raw_emg"] - np.mean(s["raw_emg"])
        filt = s["filtered_emg"]
        rect = s["rectified_emg"]
        env = s["envelope_emg"]
        base_mu = float(s["baseline_trimmed_mean_mv"])

        # Row 1: Stage 1 Raw (DC Removed)
        axes[0, col_idx].plot(t, raw_zm, color="#222222", lw=0.55)
        axes[0, col_idx].set_title(f"{m_title}\n1. Raw Unfiltered sEMG (Static DC Offset Subtracted + 50 Hz Hum)")
        axes[0, col_idx].set_ylabel("Raw (mV)")
        axes[0, col_idx].grid(True, alpha=0.3)

        # Row 2: Stage 2 Bandpass + Notch
        axes[1, col_idx].plot(t, filt, color=theme_color, lw=0.55)
        axes[1, col_idx].set_title(
            f"2. Noise-Rejected: {config.BANDPASS_LOW:g}–{config.BANDPASS_HIGH:g} Hz Butterworth Bandpass + {config.NOTCH_FREQ:g} Hz Notch"
        )
        axes[1, col_idx].set_ylabel("Filtered (mV)")
        axes[1, col_idx].grid(True, alpha=0.3)

        # Row 3: Stage 3 Full-Wave Rectification
        axes[2, col_idx].plot(t, rect, color="#E69F00", lw=0.55)
        axes[2, col_idx].set_title("3. Demodulated: Full-Wave Rectification |x|")
        axes[2, col_idx].set_ylabel("|x| (mV)")
        axes[2, col_idx].grid(True, alpha=0.3)

        # Row 4: Stage 4 Linear Envelope + Baseline + Shaded AUC + Peak Marker
        axes[3, col_idx].plot(
            t, env, color=theme_color, lw=2.0,
            label=f"{m_code} {config.ENVELOPE_LOWPASS:g} Hz Linear Envelope"
        )
        axes[3, col_idx].axhline(
            base_mu, color="#009E73", ls="--", lw=1.5,
            label=f"30s Resting Baseline ({base_mu:.4f} mV)"
        )
        axes[3, col_idx].fill_between(
            t, base_mu, env, color=theme_color, alpha=0.22,
            label=f"Integrated Net AUC ({s['norm_auc_pct']:+.1f}% vs. Baseline)"
        )
        pk_idx = int(np.argmax(env))
        axes[3, col_idx].scatter(
            [t[pk_idx]], [env[pk_idx]], color="black", marker="*", s=100, zorder=5,
            label=f"Peak Amplitude ({s['norm_peak_pct']:+.1f}%)"
        )
        axes[3, col_idx].set_title(
            f"4. Smoothed Linear Envelope ({config.ENVELOPE_LOWPASS:g} Hz Low-Pass) & Feature Extraction"
        )
        axes[3, col_idx].set_ylabel("Envelope (mV)")
        axes[3, col_idx].set_xlabel("Time from Video Stimulus Onset (seconds)")
        axes[3, col_idx].set_xlim(float(t[0]), float(t[-1]))
        axes[3, col_idx].legend(loc="upper right", fontsize=8)
        axes[3, col_idx].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    combined_path = out_dir / "fig5_FDS_ECR_filter_cascade_showcase.png"
    plt.savefig(combined_path, dpi=240)
    plt.close(fig)
    print(f"[SAVED] {combined_path}")

    # --- B. Also Save Individual 4-Panel Cascade Figures for FDS and ECR ---
    for tag, (_, m_code, m_title, theme_color) in zip(["a", "b"], col_specs):
        s = samples[m_code]
        t = s["time_s"]
        raw_zm = s["raw_emg"] - np.mean(s["raw_emg"])
        env = s["envelope_emg"]
        base_mu = float(s["baseline_trimmed_mean_mv"])

        fig_s, ax_s = plt.subplots(4, 1, figsize=(9.5, 8.2), sharex=True)
        fig_s.suptitle(
            f"Clancy et al. (2002) 4-Stage Filter Cascade — {m_title}",
            fontsize=12.5, fontweight="bold"
        )
        ax_s[0].plot(t, raw_zm, color="#222222", lw=0.55)
        ax_s[0].set_ylabel("1. Raw sEMG\n(DC removed, mV)")
        ax_s[0].grid(True, alpha=0.3)

        ax_s[1].plot(t, s["filtered_emg"], color=theme_color, lw=0.55)
        ax_s[1].set_ylabel("2. Bandpass + Notch\n(20-500Hz, 50Hz, mV)")
        ax_s[1].grid(True, alpha=0.3)

        ax_s[2].plot(t, s["rectified_emg"], color="#E69F00", lw=0.55)
        ax_s[2].set_ylabel("3. Full-Wave\nRectified |x| (mV)")
        ax_s[2].grid(True, alpha=0.3)

        ax_s[3].plot(t, env, color=theme_color, lw=2.0, label=f"{m_code} {config.ENVELOPE_LOWPASS:g} Hz Linear Envelope")
        ax_s[3].axhline(base_mu, color="#009E73", ls="--", lw=1.5, label=f"30s Resting Baseline ({base_mu:.4f} mV)")
        ax_s[3].fill_between(
            t, base_mu, env, color=theme_color, alpha=0.22,
            label=f"Integrated Net AUC ({s['norm_auc_pct']:+.1f}%)"
        )
        pk_idx = int(np.argmax(env))
        ax_s[3].scatter(
            [t[pk_idx]], [env[pk_idx]], color="black", marker="*", s=95, zorder=5,
            label=f"Peak Amplitude ({s['norm_peak_pct']:+.1f}%)"
        )
        ax_s[3].set_ylabel("4. Linear Envelope\n(8 Hz Low-Pass, mV)")
        ax_s[3].set_xlabel("Time from Video Stimulus Onset (seconds)")
        ax_s[3].set_xlim(float(t[0]), float(t[-1]))
        ax_s[3].legend(loc="upper right", fontsize=8.5)
        ax_s[3].grid(True, alpha=0.3)

        plt.tight_layout()
        indiv_path = out_dir / f"fig5{tag}_{m_code}_filter_cascade.png"
        fig_s.savefig(indiv_path, dpi=220)
        plt.close(fig_s)
        print(f"[SAVED] {indiv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Plotting Suite for 02455 Dual-Channel (FDS & ECR) sEMG Experiment"
    )
    parser.add_argument(
        "--raw-shimmer",
        type=str,
        default=None,
        help="Path to a raw Shimmer CSV file (e.g. sub-E41CB9_Shimmer_9BF9_Calibrated_SD.csv) to inspect raw CH1/CH2 & cascade"
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=config.DEFAULT_SAMPLING_RATE,
        help="Sampling rate in Hz when inspecting a raw Shimmer file (default: 512.0)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the --raw-shimmer inspection figures to disk without opening an interactive window"
    )
    parser.add_argument(
        "--subj-table",
        type=str,
        default=str(config.PROCESSED_DIR / "master_subject_level_anova_table.csv"),
        help="Path to master_subject_level_anova_table.csv"
    )
    parser.add_argument(
        "--trial-table",
        type=str,
        default=str(config.PROCESSED_DIR / "master_trial_level_features.csv"),
        help="Path to master_trial_level_features.csv"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(config.STATS_PLOTS_DIR),
        help="Directory to save generated publication figures"
    )
    parser.add_argument(
        "--simulate-demo",
        action="store_true",
        help="Generate all dual-channel (FDS & ECR) publication figures using simulated data"
    )
    args = parser.parse_args()

    # --- MODE 1: Inspect a single raw Shimmer SD file (+ FDS/ECR Filter Cascade) ---
    if args.raw_shimmer:
        plot_raw_shimmer_file(
            csv_filepath=args.raw_shimmer,
            sampling_rate_hz=args.fs,
            show_window=not args.no_show
        )
        return

    # --- MODE 2: Generate full dual-channel publication figure suite ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subj_path = Path(args.subj_table)
    trial_path = Path(args.trial_table)
    waveform_pkls = sorted(config.PROCESSED_DIR.glob("*_waveforms.pkl"))

    if args.simulate_demo or not subj_path.exists():
        print("[PLOT] Using simulated N=64 dual-channel (FDS & ECR) dataset for publication figures...")
        from stats_analysis import simulate_demo_dataset
        df_subj, df_trials = simulate_demo_dataset(n_per_group=32)

        waveforms_list = []
        if waveform_pkls:
            for pkl in waveform_pkls:
                waveforms_list.extend(pd.read_pickle(pkl).to_dict(orient="records"))
        else:
            rng = np.random.default_rng(2026)
            t_vec = np.linspace(0.0, 4.0, 2048)
            burst_profile = np.exp(-0.5 * ((t_vec - 2.0) / 0.75) ** 2)
            for m_code in ["FDS", "ECR"]:
                for grp in ["climber", "non_climber"]:
                    for cond in ["climbing", "everyday"]:
                        gain = (0.48 if m_code == "FDS" else 0.36) if (grp == "climber" and cond == "climbing") else (
                            (0.26 if m_code == "FDS" else 0.21) if cond == "everyday" else 0.04
                        )
                        for _ in range(25):
                            base_mu = 0.012 if m_code == "FDS" else 0.015
                            noise = np.convolve(rng.normal(0, 0.08, len(t_vec)), np.ones(40) / 40, mode="same")
                            env = base_mu * (1.0 + gain * burst_profile + noise)
                            waveforms_list.append({
                                "muscle": m_code,
                                "expertise_group": grp,
                                "condition": cond,
                                "time_s": t_vec,
                                "envelope_emg": np.maximum(env, 1e-5),
                                "baseline_trimmed_mean_mv": base_mu
                            })
    else:
        print(f"[PLOT] Loading real processed dual-channel data from {subj_path.name}...")
        df_subj = pd.read_csv(subj_path)
        df_trials = pd.read_csv(trial_path) if trial_path.exists() else None
        waveforms_list = []
        for pkl in waveform_pkls:
            waveforms_list.extend(pd.read_pickle(pkl).to_dict(orient="records"))

    if "muscle" not in df_subj.columns:
        df_subj["muscle"] = "FDS"

    muscles = sorted(df_subj["muscle"].unique(), reverse=True)  # ['FDS', 'ECR']
    for muscle in muscles:
        plot_raincloud_interaction_by_muscle(df_subj, muscle, out_dir, use_zscore=False)
        plot_raincloud_interaction_by_muscle(df_subj, muscle, out_dir, use_zscore=True)

    plot_fds_vs_ecr_coactivation(df_subj, df_trials, out_dir)
    plot_grand_average_envelopes_by_muscle(waveforms_list, out_dir)
    plot_isi_drift_robustness_by_muscle(df_trials, out_dir)
    plot_filter_cascade_showcase(waveforms_list, out_dir)

    print(f"\n[DONE] All dual-channel (FDS & ECR) figures saved to: {out_dir}")


if __name__ == "__main__":
    main()