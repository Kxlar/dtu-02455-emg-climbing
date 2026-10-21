# ./Experiment in Cognitive Science 02455/plot.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

import config

# Clean publication styling
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
    "climber": "#D55E00",      # Warm vermilion/orange-red (Colorblind-safe Okabe-Ito)
    "non_climber": "#0072B2"   # Deep sky/navy blue (Colorblind-safe Okabe-Ito)
}
GROUP_LABELS = {
    "climber": "Expert Climbers (Group A)",
    "non_climber": "Non-Climbers (Group B)"
}


# ==============================================================================
# HELPER: HALF-VIOLIN (RAINCLOUD) DRAWER IN PURE MATPLOTLIB
# ==============================================================================
def draw_half_violin(ax, data, pos, color, width=0.22, side="left", alpha=0.45):
    """Draws a smooth half-violin (kernel density cloud) on either the left or right of `pos`."""
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


# ==============================================================================
# FIGURE 1: PAIRED RAINCLOUD + INTERACTION SLOPE PLOT (AUC & PEAK AMPLITUDE)
# ==============================================================================
def plot_raincloud_interaction(df_subj: pd.DataFrame, out_dir: Path, use_zscore: bool = False):
    """
    Plots the modern Paired Raincloud + 2x2 Interaction Graph for both:
      - Panel A: Area Under the Curve (AUC)
      - Panel B: Peak Amplitude
    Includes half-violins, boxplots, individual paired participant lines, and
    bold group-mean interaction lines (whose slope difference visualizes H1 vs H0).
    """
    if use_zscore:
        dvs = [
            ("zscore_auc", "Normalized AUC (Z-Score SNR Units)", 0.0),
            ("zscore_peak", "Normalized Peak Amplitude (Z-Score SNR Units)", 0.0)
        ]
        suffix = "zscore"
    else:
        dvs = [
            ("norm_auc_pct", "Normalized AUC (% Change vs. 30s Baseline)", 0.0),
            ("norm_peak_pct", "Normalized Peak Amplitude (% Change vs. 30s Baseline)", 0.0)
        ]
        suffix = "pct"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=False)
    fig.suptitle(
        "Motor Resonance 2×2 Interaction & Distribution Analysis (Raincloud Plot)",
        fontsize=14, fontweight="bold", y=0.98
    )

    rng = np.random.default_rng(42)

    for ax_idx, (dv, ylabel, null_val) in enumerate(dvs):
        ax = axes[ax_idx]
        wide = df_subj.pivot(
            index=["participant_id", "expertise_group"],
            columns="condition",
            values=dv
        ).dropna().reset_index()

        # X-coordinates: 0 = Everyday (Control), 1 = Climbing (Target)
        # We slightly offset the two groups horizontally so distributions don't collide
        x_base = {"everyday": 0.0, "climbing": 1.0}
        grp_offset = {"non_climber": -0.06, "climber": +0.06}

        for grp in ["non_climber", "climber"]:
            sub_w = wide[wide["expertise_group"] == grp]
            if sub_w.empty:
                continue

            c = PALETTE[grp]
            x_ev = x_base["everyday"] + grp_offset[grp]
            x_cl = x_base["climbing"] + grp_offset[grp]

            y_ev = sub_w["everyday"].values
            y_cl = sub_w["climbing"].values

            # 1. Draw Half-Violins ("Clouds") on the outer edges
            draw_half_violin(ax, y_ev, pos=x_ev - 0.10, color=c, width=0.20, side="left", alpha=0.40)
            draw_half_violin(ax, y_cl, pos=x_cl + 0.10, color=c, width=0.20, side="right", alpha=0.40)

            # 2. Draw Individual Participant Dots + Faint Within-Subject Lines ("Rain")
            jitter_ev = rng.uniform(-0.025, 0.025, size=len(sub_w))
            jitter_cl = rng.uniform(-0.025, 0.025, size=len(sub_w))

            for i in range(len(sub_w)):
                ax.plot(
                    [x_ev + jitter_ev[i], x_cl + jitter_cl[i]],
                    [y_ev[i], y_cl[i]],
                    color=c, alpha=0.18, lw=0.9, zorder=2
                )

            ax.scatter(x_ev + jitter_ev, y_ev, color=c, alpha=0.55, s=24, edgecolors="none", zorder=3)
            ax.scatter(x_cl + jitter_cl, y_cl, color=c, alpha=0.55, s=24, edgecolors="none", zorder=3)

            # 3. Draw Slim Interquartile Box/Whisker Indicators
            bp = ax.boxplot(
                [y_ev, y_cl],
                positions=[x_ev - 0.08, x_cl + 0.08],
                widths=0.05,
                patch_artist=True,
                showfliers=False,
                zorder=4
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("white")
                patch.set_edgecolor(c)
                patch.set_linewidth(1.3)
            for elem in ["whiskers", "caps", "medians"]:
                for item in bp[elem]:
                    item.set_color(c)
                    item.set_linewidth(1.3)

            # 4. Draw BOLD Group Mean Interaction Line (+/- SEM) linking Everyday -> Climbing
            mean_ev, sem_ev = np.mean(y_ev), stats.sem(y_ev) if len(y_ev) > 1 else 0.0
            mean_cl, sem_cl = np.mean(y_cl), stats.sem(y_cl) if len(y_cl) > 1 else 0.0
            slope = mean_cl - mean_ev

            ax.errorbar(
                [x_ev, x_cl], [mean_ev, mean_cl],
                yerr=[sem_ev, sem_cl],
                color=c, marker="o", markersize=9, markerfacecolor=c, markeredgecolor="white",
                markeredgewidth=1.5, lw=3.2, capsize=6, capthick=2.0,
                label=f"{GROUP_LABELS[grp]} (Δ Slope = {slope:+.1f})",
                zorder=6
            )

        # Reference line for 30s resting baseline
        ax.axhline(null_val, color="#444444", ls="--", lw=1.3, alpha=0.8, label="30s Resting Baseline Floor")

        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([
            "Everyday Grasping\n(Condition B: Control)",
            "Climbing Grips\n(Condition A: Target)"
        ], fontweight="bold")
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel(ylabel)
        panel_letter = "A" if ax_idx == 0 else "B"
        metric_title = "Area Under the Curve (AUC)" if ax_idx == 0 else "Peak Amplitude"
        ax.set_title(f"{panel_letter}. {metric_title} — Interaction & Distributions")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = out_dir / f"fig1_raincloud_interaction_{suffix}.png"
    plt.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==============================================================================
# FIGURE 2: TIME-RESOLVED GRAND AVERAGE ENVELOPE (0 to 5 SECONDS)
# ==============================================================================
def plot_grand_average_envelopes(waveforms_list: list, out_dir: Path):
    """
    Interpolates every trial's baseline-normalized envelope onto a common 0.0 to 5.0s grid
    and plots the Grand Average Time-Course (+/- SEM shaded ribbon) for all 4 cells
    of the 2x2 design. Shows the temporal dynamics of muscle resonance during video observation.
    """
    if not waveforms_list:
        return

    common_t = np.linspace(0.0, 5.0, 500)  # 100 Hz common visualization grid
    traces = {
        ("climber", "climbing"): [],
        ("climber", "everyday"): [],
        ("non_climber", "climbing"): [],
        ("non_climber", "everyday"): []
    }

    for row in waveforms_list:
        key = (row["expertise_group"], row["condition"])
        if key not in traces:
            continue
        t_vec = row["time_s"]
        env = row["envelope_emg"]
        base_mu = row["baseline_trimmed_mean_mv"]
        if len(t_vec) < 10 or base_mu <= 0:
            continue
        # Convert instantaneous envelope to % change over resting baseline
        pct_trace = ((env - base_mu) / base_mu) * 100.0
        interp_trace = np.interp(common_t, t_vec, pct_trace)
        traces[key].append(interp_trace)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    fig.suptitle(
        "Time-Resolved sEMG Linear Envelope During 5s Action Observation (Mean ± SEM)",
        fontsize=13, fontweight="bold"
    )

    cond_styles = {
        "climbing": {"ls": "-", "lw": 2.4, "label": "Climbing Grips (Target)"},
        "everyday": {"ls": "--", "lw": 2.0, "label": "Everyday Grasping (Control)"}
    }

    for idx, grp in enumerate(["climber", "non_climber"]):
        ax = axes[idx]
        c = PALETTE[grp]

        for cond in ["climbing", "everyday"]:
            mat = np.array(traces[(grp, cond)])
            if mat.ndim != 2 or len(mat) == 0:
                continue
            mean_curve = np.mean(mat, axis=0)
            sem_curve = stats.sem(mat, axis=0) if len(mat) > 1 else np.zeros_like(mean_curve)

            style = cond_styles[cond]
            fill_alpha = 0.28 if cond == "climbing" else 0.12
            line_color = c if cond == "climbing" else "#555555"

            ax.plot(
                common_t, mean_curve,
                color=line_color, ls=style["ls"], lw=style["lw"],
                label=f"{style['label']} (n={len(mat)} trials)"
            )
            ax.fill_between(
                common_t, mean_curve - sem_curve, mean_curve + sem_curve,
                color=line_color, alpha=fill_alpha
            )

        ax.axhline(0.0, color="black", ls=":", lw=1.2, label="Resting Baseline (0%)")
        ax.set_title(f"{'A' if idx == 0 else 'B'}. {GROUP_LABELS[grp]}")
        ax.set_xlabel("Time from Video Stimulus Onset (seconds)")
        if idx == 0:
            ax.set_ylabel("Forearm Flexor sEMG Modulation (% vs. Baseline)")
        ax.set_xlim(0.0, 5.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "fig2_grand_average_envelope_timecourse.png"
    plt.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==============================================================================
# FIGURE 3: CLANCY ET AL. (2002) SIGNAL PROCESSING CASCADE COMPARISON
# ==============================================================================
def plot_clancy_cascade_comparison(waveforms_list: list, out_dir: Path):
    """
    Side-by-side 4-stage signal processing comparison (Raw -> Filtered -> Rectified -> 8Hz Envelope)
    contrasting a representative Climber trial vs. Non-Climber trial observing a Climbing video.
    Directly illustrates how the filter cascade & AUC/Peak extraction work for the report.
    """
    climber_trials = [w for w in waveforms_list if w["expertise_group"] == "climber" and w["condition"] == "climbing"]
    non_climber_trials = [w for w in waveforms_list if w["expertise_group"] == "non_climber" and w["condition"] == "climbing"]

    if not climber_trials or not non_climber_trials:
        return

    # Pick representative trial near the 75th percentile of each group
    climber_trials.sort(key=lambda x: x["norm_auc_pct"])
    non_climber_trials.sort(key=lambda x: x["norm_auc_pct"])
    ex_climb = climber_trials[int(0.75 * len(climber_trials))]
    ex_non = non_climber_trials[int(0.50 * len(non_climber_trials))]

    fig, axes = plt.subplots(4, 2, figsize=(13, 8.5), sharex=True)
    fig.suptitle(
        "sEMG Signal Processing Cascade (Clancy et al., 2002) — Observing Climbing Grip Stimulus",
        fontsize=13, fontweight="bold"
    )

    examples = [
        (0, ex_climb, "Expert Climber Observing Climbing Grip", PALETTE["climber"]),
        (1, ex_non, "Non-Climber Observing Climbing Grip", PALETTE["non_climber"])
    ]

    for col_idx, ex, col_title, color in examples:
        t = ex["time_s"]
        base_mu = ex["baseline_trimmed_mean_mv"]

        # Row 0: Raw EMG
        axes[0, col_idx].plot(t, ex["raw_emg"], color="#333333", lw=0.55)
        axes[0, col_idx].set_title(f"{col_title}\n(Stimulus: {ex['stimulus_file']})")

        # Row 1: Bandpass 20-500 Hz + 50 Hz Notch
        axes[1, col_idx].plot(t, ex["filtered_emg"], color="#1f77b4", lw=0.55)

        # Row 2: Full-Wave Rectified |x|
        axes[2, col_idx].plot(t, ex["rectified_emg"], color="#ff7f0e", lw=0.55)

        # Row 3: 8 Hz Linear Envelope + Peak Star + Shaded AUC
        env = ex["envelope_emg"]
        axes[3, col_idx].plot(t, env, color=color, lw=2.0, label="8 Hz Linear Envelope")
        axes[3, col_idx].axhline(base_mu, color="green", ls="--", lw=1.4, label=f"30s Baseline ({base_mu:.4f} mV)")
        axes[3, col_idx].fill_between(
            t, base_mu, env, where=(env >= base_mu),
            color=color, alpha=0.28, label=f"Net Positive AUC ({ex['norm_auc_pct']:+.1f}%)"
        )

        # Highlight Peak Amplitude with a star marker
        peak_idx = int(np.argmax(env))
        axes[3, col_idx].scatter(
            [t[peak_idx]], [env[peak_idx]],
            color="black", marker="*", s=110, zorder=5,
            label=f"Peak Amplitude ({ex['norm_peak_pct']:+.1f}%)"
        )
        axes[3, col_idx].set_xlabel("Time from Video Onset (seconds)")
        axes[3, col_idx].legend(loc="upper right", fontsize=8, frameon=True)

        for r in range(4):
            axes[r, col_idx].grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("1. Raw sEMG\n(mV)")
    axes[1, 0].set_ylabel("2. Bandpass 20-500Hz\n+ 50Hz Notch (mV)")
    axes[2, 0].set_ylabel("3. Full-Wave\nRectified |x| (mV)")
    axes[3, 0].set_ylabel("4. Linear Envelope\n(8Hz Low-Pass, mV)")

    plt.tight_layout()
    out_path = out_dir / "fig3_clancy_cascade_showcase.png"
    plt.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==============================================================================
# FIGURE 4: VIDEO STIMULUS ITEM RANKING (WHICH CLIPS TRIGGERED MOST RESONANCE?)
# ==============================================================================
def plot_stimulus_item_ranking(df_trials: pd.DataFrame, out_dir: Path):
    """
    Ranks all 30 individual video clips (15 climbing, 15 everyday) by the mean normalized AUC
    they elicited in Climbers vs. Non-Climbers. Useful for checking if specific climbing holds
    (e.g., full crimps vs. slopers) drove stronger motor resonance.
    """
    if df_trials is None or df_trials.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharex=True)
    fig.suptitle(
        "Item-Level Analysis: Normalized sEMG AUC by Individual Video Clip (Mean ± SEM)",
        fontsize=13, fontweight="bold"
    )

    for idx, cond in enumerate(["climbing", "everyday"]):
        ax = axes[idx]
        sub = df_trials[df_trials["condition"] == cond]
        if sub.empty:
            continue

        # Order clips by Climber mean activation
        climber_order = (
            sub[sub["expertise_group"] == "climber"]
            .groupby("stimulus_file")["norm_auc_pct"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        if not climber_order:
            climber_order = sorted(sub["stimulus_file"].unique())

        y_pos = np.arange(len(climber_order))

        for grp, offset in [("non_climber", -0.15), ("climber", +0.15)]:
            grp_df = sub[sub["expertise_group"] == grp]
            means, sems = [], []
            for stim in climber_order:
                vals = grp_df[grp_df["stimulus_file"] == stim]["norm_auc_pct"].dropna().values
                means.append(np.mean(vals) if len(vals) else np.nan)
                sems.append(stats.sem(vals) if len(vals) > 1 else 0.0)

            ax.errorbar(
                means, y_pos + offset, xerr=sems,
                fmt="o", color=PALETTE[grp], markersize=6, lw=1.4, capsize=3,
                label=GROUP_LABELS[grp]
            )

        ax.axvline(0.0, color="gray", ls="--", lw=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(climber_order, fontsize=8.5)
        ax.set_xlabel("Normalized AUC (% Change vs. 30s Baseline)")
        title_str = "A. Climbing Grip Videos (Target)" if cond == "climbing" else "B. Everyday Grasping Videos (Control)"
        ax.set_title(title_str)
        ax.grid(True, axis="x", alpha=0.25)
        if idx == 0:
            ax.legend(loc="lower right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    out_path = out_dir / "fig4_stimulus_item_ranking.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==============================================================================
# FIGURE 5: TRIAL-BY-TRIAL HABITUATION / FATIGUE DRIFT CHECK (TRIALS 1 TO 30)
# ==============================================================================
def plot_habituation_drift_check(df_trials: pd.DataFrame, out_dir: Path):
    """
    Plots normalized sEMG activation across sequential trial number (1..30) and
    tracks the local pre-trial ISI resting floor across the 15-minute session.
    Verifies whether the 4-7s jittered ISI successfully prevented fatigue and signal contamination.
    """
    if df_trials is None or df_trials.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    fig.suptitle(
        "Quality Control: Temporal Habituation & Inter-Stimulus Interval (ISI) Recovery Across 30 Trials",
        fontsize=12.5, fontweight="bold"
    )

    # Top Panel: Normalized Trial AUC across Trial 1..30 by Group
    ax1 = axes[0]
    for grp in ["climber", "non_climber"]:
        sub = df_trials[df_trials["expertise_group"] == grp]
        stats_by_t = sub.groupby("trial_num")["norm_auc_pct"].agg(["mean", "sem"]).reset_index()
        ax1.plot(
            stats_by_t["trial_num"], stats_by_t["mean"],
            marker="o", markersize=5, lw=1.8, color=PALETTE[grp], label=GROUP_LABELS[grp]
        )
        ax1.fill_between(
            stats_by_t["trial_num"],
            stats_by_t["mean"] - stats_by_t["sem"].fillna(0),
            stats_by_t["mean"] + stats_by_t["sem"].fillna(0),
            color=PALETTE[grp], alpha=0.20
        )

    ax1.axhline(0.0, color="gray", ls="--", lw=1.2, label="30s Baseline Reference")
    ax1.set_ylabel("Trial Normalized AUC (%)")
    ax1.set_title("A. Mean Trial Activation Across Experimental Session (Habituation Check)")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, alpha=0.25)

    # Bottom Panel: Global vs. Local ISI Drift-Corrected Comparison
    ax2 = axes[1]
    if "local_isi_norm_auc_pct" in df_trials.columns:
        drift_diff = df_trials["norm_auc_pct"] - df_trials["local_isi_norm_auc_pct"]
        df_trials = df_trials.copy()
        df_trials["baseline_drift_pct"] = drift_diff
        for grp in ["climber", "non_climber"]:
            sub = df_trials[df_trials["expertise_group"] == grp]
            stats_by_t = sub.groupby("trial_num")["baseline_drift_pct"].agg(["mean", "sem"]).reset_index()
            ax2.plot(
                stats_by_t["trial_num"], stats_by_t["mean"],
                marker="s", markersize=4, lw=1.5, color=PALETTE[grp], label=f"{GROUP_LABELS[grp]} ISI Drift"
            )
    ax2.axhline(0.0, color="black", ls=":", lw=1.2)
    ax2.set_xlabel("Sequential Trial Number (1 to 30)")
    ax2.set_ylabel("Global vs. Local ISI Floor Δ (%)")
    ax2.set_title("B. Pre-Trial ISI Baseline Return Stability (Checks for Cumulative Muscle Tension)")
    ax2.set_xticks(range(1, 31))
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = out_dir / "fig5_habituation_fatigue_check.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[SAVED] {out_path}")


# ==============================================================================
# DEMO WAVEFORM GENERATOR (FOR TESTING plot.py BEFORE DATA COLLECTION)
# ==============================================================================
def simulate_waveforms_for_demo(df_trials: pd.DataFrame, fs: float = 1024.0) -> list:
    """Generates realistic synthetic 5s sEMG waveforms & Clancy cascades for demo plotting."""
    from postprocess_emg import apply_clancy_cascade

    rng = np.random.default_rng(123)
    t_vec = np.linspace(0.0, 5.0, int(5.0 * fs), endpoint=False)
    waveforms_list = []

    # Generate representative waveforms for a subset of participants so demo runs in ~2 seconds
    demo_subs = df_trials["participant_id"].unique()[:8]
    sub_df = df_trials[df_trials["participant_id"].isin(demo_subs)]

    for _, row in sub_df.iterrows():
        base_mu = 0.015  # 15 uV resting baseline envelope
        target_pct = float(row["norm_auc_pct"])
        gain = max(1.0 + (target_pct / 100.0), 0.2)

        # Simulate realistic burst profile peaking between t = 1.2s and 3.8s (when grip occurs in video)
        burst_profile = 1.0 + (gain - 1.0) * np.exp(-0.5 * ((t_vec - 2.5) / 0.9) ** 2)
        carrier = rng.normal(0.0, base_mu * 1.25, size=len(t_vec))
        powerline_hum = 0.008 * np.sin(2 * np.pi * 50.0 * t_vec)
        motion_drift = 0.012 * np.sin(2 * np.pi * 1.5 * t_vec)

        raw_emg = (burst_profile * carrier) + powerline_hum + motion_drift
        casc = apply_clancy_cascade(raw_emg, fs=fs, demod_power=1)

        waveforms_list.append({
            **row.to_dict(),
            "time_s": t_vec,
            "baseline_trimmed_mean_mv": float(np.mean( np.abs(carrier) )),
            "raw_emg": raw_emg,
            "filtered_emg": casc["notched"],
            "rectified_emg": casc["rectified"],
            "envelope_emg": casc["envelope"]
        })

    return waveforms_list


def main():
    parser = argparse.ArgumentParser(description="Generate Publication Figures for 02455 sEMG Experiment")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(config.STATS_PLOTS_DIR),
        help="Directory to save generated figures"
    )
    parser.add_argument(
        "--simulate-demo",
        action="store_true",
        help="Generate all 5 figures immediately using simulated N=64 data"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subj_path = config.PROCESSED_DIR / "master_subject_level_anova_table.csv"
    trial_path = config.PROCESSED_DIR / "master_trial_level_features.csv"
    waveform_pkls = sorted(config.PROCESSED_DIR.glob("*_waveforms.pkl"))

    if args.simulate_demo or not subj_path.exists():
        if not args.simulate_demo:
            print("[INFO] Real processed data not found yet; running in --simulate-demo mode to preview plots.")
        from stats_analysis import simulate_demo_dataset
        df_subj, df_trials = simulate_demo_dataset(n_per_group=32)
        waveforms_list = simulate_waveforms_for_demo(df_trials)
    else:
        df_subj = pd.read_csv(subj_path)
        df_trials = pd.read_csv(trial_path) if trial_path.exists() else None
        waveforms_list = []
        for pkl in waveform_pkls:
            df_w = pd.read_pickle(pkl)
            waveforms_list.extend(df_w.to_dict(orient="records"))

    print(f"\nGenerating figures into: {out_dir}")
    # 1. Primary Raincloud + Interaction Slope Graphs (both % Baseline and Z-score SNR)
    plot_raincloud_interaction(df_subj, out_dir, use_zscore=False)
    plot_raincloud_interaction(df_subj, out_dir, use_zscore=True)

    # 2. Grand Average Time-Course Envelope (0 to 5s)
    plot_grand_average_envelopes(waveforms_list, out_dir)

    # 3. Clancy et al. (2002) 4-Stage Filter Cascade Comparison
    plot_clancy_cascade_comparison(waveforms_list, out_dir)

    # 4. Stimulus Item-Level Ranking (15 Climbing vs 15 Everyday Clips)
    plot_stimulus_item_ranking(df_trials, out_dir)

    # 5. Session Habituation & ISI Baseline Recovery Check
    plot_habituation_drift_check(df_trials, out_dir)
    print("\n[ALL FIGURES COMPLETE]")


if __name__ == "__main__":
    main()