# ./Experiment in Cognitive Science 02455/stats_analysis.py
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm

import config

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. ASSUMPTION CHECKS: SHAPIRO-WILK (NORMALITY) & LEVENE'S (HOMOSCEDASTICITY)
# ==============================================================================
def check_anova_assumptions(df_subj: pd.DataFrame, dv: str, muscle: str) -> dict:
    """
    Evaluates the two parametric assumptions specified in Section 3.5 of the report:
      1. Shapiro-Wilk Test for normality across each of the 4 (Group x Condition) cells.
      2. Levene's Test for equality of variance between groups within each condition.
    """
    shapiro_rows = []
    normality_passed = True

    for (group, cond), sub_df in df_subj.groupby(["expertise_group", "condition"]):
        vals = sub_df[dv].dropna().values
        if len(vals) >= 3:
            stat_w, p_val = stats.shapiro(vals)
            passed = bool(p_val >= config.ALPHA_LEVEL)
            if not passed:
                normality_passed = False
        else:
            stat_w, p_val, passed = np.nan, np.nan, True

        shapiro_rows.append({
            "muscle": muscle,
            "dv": dv,
            "expertise_group": group,
            "condition": cond,
            "n": len(vals),
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "median": float(np.median(vals)) if len(vals) else np.nan,
            "iqr": float(stats.iqr(vals)) if len(vals) else np.nan,
            "shapiro_W": float(stat_w) if not np.isnan(stat_w) else np.nan,
            "shapiro_p": float(p_val) if not np.isnan(p_val) else np.nan,
            "normality_met": passed
        })

    levene_rows = []
    homoscedasticity_passed = True

    for cond, sub_df in df_subj.groupby("condition"):
        climbers = sub_df[sub_df["expertise_group"] == "climber"][dv].dropna().values
        non_climbers = sub_df[sub_df["expertise_group"] == "non_climber"][dv].dropna().values

        if len(climbers) >= 2 and len(non_climbers) >= 2:
            stat_l, p_val = stats.levene(climbers, non_climbers, center="median")
            passed = bool(p_val >= config.ALPHA_LEVEL)
            if not passed:
                homoscedasticity_passed = False
        else:
            stat_l, p_val, passed = np.nan, np.nan, True

        levene_rows.append({
            "muscle": muscle,
            "dv": dv,
            "condition": cond,
            "levene_W": float(stat_l) if not np.isnan(stat_l) else np.nan,
            "levene_p": float(p_val) if not np.isnan(p_val) else np.nan,
            "homoscedasticity_met": passed
        })

    return {
        "shapiro_df": pd.DataFrame(shapiro_rows),
        "levene_df": pd.DataFrame(levene_rows),
        "normality_passed": normality_passed,
        "homoscedasticity_passed": homoscedasticity_passed,
        "all_assumptions_met": normality_passed and homoscedasticity_passed
    }


# ==============================================================================
# 2. PARAMETRIC 2x2 MIXED FACTORIAL ANOVA
# ==============================================================================
def run_mixed_factorial_anova(df_subj: pd.DataFrame, dv: str) -> pd.DataFrame:
    """
    Computes the 2x2 Mixed Factorial ANOVA (Section 3.5):
      - Between-Subjects Factor: expertise_group (climber vs. non_climber)
      - Within-Subjects Factor : condition (climbing vs. everyday)
    """
    try:
        import pingouin as pg
        aov = pg.mixed_anova(
            data=df_subj,
            dv=dv,
            within="condition",
            between="expertise_group",
            subject="participant_id",
            effsize="np2"
        )
        aov = aov.rename(columns={"p-unc": "p_unc"})
        aov["test_type"] = "Parametric_2x2_Mixed_ANOVA"
        return aov
    except ImportError:
        pass

    # Exact Analytical 2x2 Mixed Factorial ANOVA (Pure SciPy/NumPy)
    wide = df_subj.pivot(
        index=["participant_id", "expertise_group"],
        columns="condition",
        values=dv
    ).dropna().reset_index()

    groups = wide["expertise_group"].unique()
    conds = [c for c in wide.columns if c not in ("participant_id", "expertise_group")]
    N = len(wide)
    k_cond, k_grp = len(conds), len(groups)

    if N < 3 or k_cond < 2 or k_grp < 2:
        return pd.DataFrame()

    all_vals = wide[conds].values.flatten()
    grand_mean = np.mean(all_vals)
    wide["subj_mean"] = wide[conds].mean(axis=1)

    # 1. Between-Subjects Partition
    ss_group, ss_subj_within_group = 0.0, 0.0
    for g in groups:
        sub_g = wide[wide["expertise_group"] == g]
        n_g = len(sub_g)
        g_mean = sub_g["subj_mean"].mean()
        ss_group += n_g * k_cond * ((g_mean - grand_mean) ** 2)
        ss_subj_within_group += k_cond * np.sum((sub_g["subj_mean"] - g_mean) ** 2)

    df_group = k_grp - 1
    df_error_btw = max(N - k_grp, 1)
    ms_group = ss_group / df_group
    ms_error_btw = ss_subj_within_group / df_error_btw
    f_group = ms_group / ms_error_btw if ms_error_btw > 0 else np.nan
    p_group = stats.f.sf(f_group, df_group, df_error_btw)
    np2_group = ss_group / (ss_group + ss_subj_within_group) if (ss_group + ss_subj_within_group) > 0 else 0.0

    # 2. Within-Subjects Partition
    cond_means = {c: wide[c].mean() for c in conds}
    ss_cond = sum(N * ((cond_means[c] - grand_mean) ** 2) for c in conds)

    ss_cells = 0.0
    for g in groups:
        sub_g = wide[wide["expertise_group"] == g]
        n_g = len(sub_g)
        for c in conds:
            ss_cells += n_g * ((sub_g[c].mean() - grand_mean) ** 2)

    ss_inter = ss_cells - ss_group - ss_cond
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_error_within = max(ss_total - ss_group - ss_subj_within_group - ss_cond - ss_inter, 1e-12)

    df_cond = k_cond - 1
    df_inter = df_group * df_cond
    df_error_wth = max((N - k_grp) * (k_cond - 1), 1)

    ms_cond = ss_cond / df_cond
    ms_inter = ss_inter / df_inter
    ms_error_wth = ss_error_within / df_error_wth

    f_cond = ms_cond / ms_error_wth if ms_error_wth > 0 else np.nan
    p_cond = stats.f.sf(f_cond, df_cond, df_error_wth)
    np2_cond = ss_cond / (ss_cond + ss_error_within)

    f_inter = ms_inter / ms_error_wth if ms_error_wth > 0 else np.nan
    p_inter = stats.f.sf(f_inter, df_inter, df_error_wth)
    np2_inter = ss_inter / (ss_inter + ss_error_within)

    return pd.DataFrame([
        {
            "Source": "expertise_group",
            "SS": ss_group, "DF1": df_group, "DF2": df_error_btw,
            "MS": ms_group, "F": f_group, "p_unc": p_group, "np2": np2_group,
            "test_type": "Parametric_2x2_Mixed_ANOVA"
        },
        {
            "Source": "condition",
            "SS": ss_cond, "DF1": df_cond, "DF2": df_error_wth,
            "MS": ms_cond, "F": f_cond, "p_unc": p_cond, "np2": np2_cond,
            "test_type": "Parametric_2x2_Mixed_ANOVA"
        },
        {
            "Source": "Interaction",
            "SS": ss_inter, "DF1": df_inter, "DF2": df_error_wth,
            "MS": ms_inter, "F": f_inter, "p_unc": p_inter, "np2": np2_inter,
            "test_type": "Parametric_2x2_Mixed_ANOVA"
        }
    ])


# ==============================================================================
# 3. NON-PARAMETRIC FALLBACK: ALIGNED RANK TRANSFORM (ART) 2x2 MIXED ANOVA
# ==============================================================================
def run_aligned_rank_transform_anova(df_subj: pd.DataFrame, dv: str) -> pd.DataFrame:
    """
    Aligned Rank Transform (ART) Mixed Factorial ANOVA (Wobbrock et al., 2011; Section 3.5).
    Strips nuisance main/interaction effects prior to ranking so interaction F-tests remain valid
    under skewed or high-variance sEMG distributions.
    """
    df = df_subj.copy().dropna(subset=[dv])
    grand_mean = df[dv].mean()

    group_means = df.groupby("expertise_group")[dv].transform("mean")
    cond_means = df.groupby("condition")[dv].transform("mean")
    cell_means = df.groupby(["expertise_group", "condition"])[dv].transform("mean")

    eff_A = group_means - grand_mean
    eff_B = cond_means - grand_mean
    eff_AB = cell_means - group_means - cond_means + grand_mean

    df["rank_A"] = stats.rankdata(df[dv] - eff_B - eff_AB, method="average")
    df["rank_B"] = stats.rankdata(df[dv] - eff_A - eff_AB, method="average")
    df["rank_AB"] = stats.rankdata(df[dv] - eff_A - eff_B, method="average")

    aov_A = run_mixed_factorial_anova(df, "rank_A")
    aov_B = run_mixed_factorial_anova(df, "rank_B")
    aov_AB = run_mixed_factorial_anova(df, "rank_AB")

    if aov_A.empty or aov_B.empty or aov_AB.empty:
        return pd.DataFrame()

    art_table = pd.concat([aov_A.iloc[[0]], aov_B.iloc[[1]], aov_AB.iloc[[2]]], ignore_index=True)
    art_table["test_type"] = "NonParametric_ART_2x2_Mixed_ANOVA"
    return art_table


# ==============================================================================
# 4. BONFERRONI-CORRECTED SIMPLE MAIN EFFECTS & COHEN'S d
# ==============================================================================
def run_simple_main_effects(df_subj: pd.DataFrame, dv: str, muscle: str) -> pd.DataFrame:
    """
    Decomposes the 2x2 interaction via Bonferroni-corrected simple main effects (Section 3.5):
      - Family 1 (Between-Subjects): Climbers vs. Non-Climbers within 'climbing' and within 'everyday'
      - Family 2 (Within-Subjects) : 'climbing' vs. 'everyday' within Climbers and within Non-Climbers
    """
    results = []

    # 1. Between-Subjects Simple Main Effects at each Stimulus Condition
    for cond in ["climbing", "everyday"]:
        sub = df_subj[df_subj["condition"] == cond]
        g_climb = sub[sub["expertise_group"] == "climber"][dv].dropna().values
        g_non = sub[sub["expertise_group"] == "non_climber"][dv].dropna().values

        if len(g_climb) >= 2 and len(g_non) >= 2:
            t_stat, p_param = stats.ttest_ind(g_climb, g_non, equal_var=False)
            u_stat, p_nonparam = stats.mannwhitneyu(g_climb, g_non, alternative="two-sided")
            n1, n2 = len(g_climb), len(g_non)
            s_pool = np.sqrt(((n1 - 1) * np.var(g_climb, ddof=1) + (n2 - 1) * np.var(g_non, ddof=1)) / (n1 + n2 - 2))
            cohens_d = (np.mean(g_climb) - np.mean(g_non)) / s_pool if s_pool > 0 else 0.0
        else:
            t_stat = p_param = u_stat = p_nonparam = cohens_d = np.nan

        results.append({
            "muscle": muscle,
            "dv": dv,
            "contrast_family": "Between_Groups_at_Condition",
            "comparison": f"climber vs. non_climber (in '{cond}' videos)",
            "mean_A": float(np.mean(g_climb)) if len(g_climb) else np.nan,
            "mean_B": float(np.mean(g_non)) if len(g_non) else np.nan,
            "diff_A_minus_B": float(np.mean(g_climb) - np.mean(g_non)) if (len(g_climb) and len(g_non)) else np.nan,
            "t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
            "p_unc_param": float(p_param) if not np.isnan(p_param) else np.nan,
            "p_bonf_param": min(1.0, float(p_param) * 4.0) if not np.isnan(p_param) else np.nan,
            "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else np.nan,
            "nonparam_U_or_W": float(u_stat) if not np.isnan(u_stat) else np.nan,
            "p_unc_nonparam": float(p_nonparam) if not np.isnan(p_nonparam) else np.nan,
            "p_bonf_nonparam": min(1.0, float(p_nonparam) * 4.0) if not np.isnan(p_nonparam) else np.nan
        })

    # 2. Within-Subjects Simple Main Effects across Conditions within each Group
    wide = df_subj.pivot(index=["participant_id", "expertise_group"], columns="condition", values=dv).dropna().reset_index()

    for grp in ["climber", "non_climber"]:
        sub_w = wide[wide["expertise_group"] == grp]
        v_climb = sub_w["climbing"].values if "climbing" in sub_w else np.array([])
        v_every = sub_w["everyday"].values if "everyday" in sub_w else np.array([])

        if len(v_climb) >= 2 and len(v_every) >= 2:
            t_stat, p_param = stats.ttest_rel(v_climb, v_every)
            diffs = v_climb - v_every
            cohens_dz = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0.0
            try:
                w_stat, p_nonparam = stats.wilcoxon(v_climb, v_every)
            except ValueError:
                w_stat, p_nonparam = 0.0, 1.0
        else:
            t_stat = p_param = w_stat = p_nonparam = cohens_dz = np.nan

        results.append({
            "muscle": muscle,
            "dv": dv,
            "contrast_family": "Within_Group_Across_Conditions",
            "comparison": f"climbing vs. everyday (within '{grp}')",
            "mean_A": float(np.mean(v_climb)) if len(v_climb) else np.nan,
            "mean_B": float(np.mean(v_every)) if len(v_every) else np.nan,
            "diff_A_minus_B": float(np.mean(v_climb) - np.mean(v_every)) if len(v_climb) else np.nan,
            "t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
            "p_unc_param": float(p_param) if not np.isnan(p_param) else np.nan,
            "p_bonf_param": min(1.0, float(p_param) * 4.0) if not np.isnan(p_param) else np.nan,
            "cohens_d": float(cohens_dz) if not np.isnan(cohens_dz) else np.nan,
            "nonparam_U_or_W": float(w_stat) if not np.isnan(w_stat) else np.nan,
            "p_unc_nonparam": float(p_nonparam) if not np.isnan(p_nonparam) else np.nan,
            "p_bonf_nonparam": min(1.0, float(p_nonparam) * 4.0) if not np.isnan(p_nonparam) else np.nan
        })

    return pd.DataFrame(results)


# ==============================================================================
# 5. ONE-SAMPLE TEST AGAINST RESTING BASELINE (TESTING H0 SPINAL INHIBITION)
# ==============================================================================
def run_baseline_equivalence_tests(df_subj: pd.DataFrame, dv: str, muscle: str, null_value: float = 0.0) -> pd.DataFrame:
    """
    Tests whether sEMG activation in each cell is statistically distinguishable from
    the resting baseline floor (Section 1.4 & 2.2: Spinal Inhibition Null Hypothesis).
    """
    rows = []
    for (group, cond), sub_df in df_subj.groupby(["expertise_group", "condition"]):
        vals = sub_df[dv].dropna().values
        if len(vals) >= 2:
            t_stat, p_t = stats.ttest_1samp(vals, popmean=null_value)
            try:
                w_stat, p_w = stats.wilcoxon(vals - null_value)
            except ValueError:
                w_stat, p_w = 0.0, 1.0
        else:
            t_stat = p_t = w_stat = p_w = np.nan

        rows.append({
            "muscle": muscle,
            "dv": dv,
            "expertise_group": group,
            "condition": cond,
            "null_baseline_ref": null_value,
            "cell_mean": float(np.mean(vals)) if len(vals) else np.nan,
            "cell_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "t_vs_baseline": float(t_stat) if not np.isnan(t_stat) else np.nan,
            "p_param_vs_baseline": float(p_t) if not np.isnan(p_t) else np.nan,
            "p_bonf_param": min(1.0, float(p_t) * 4.0) if not np.isnan(p_t) else np.nan,
            "wilcoxon_vs_baseline": float(w_stat) if not np.isnan(w_stat) else np.nan,
            "p_nonparam_vs_baseline": float(p_w) if not np.isnan(p_w) else np.nan
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 6. TRIAL-LEVEL LINEAR MIXED-EFFECTS MODEL (LMM) PER MUSCLE
# ==============================================================================
def run_trial_level_lmm(df_trials_muscle: pd.DataFrame, dv: str, muscle: str):
    if df_trials_muscle is None or df_trials_muscle.empty:
        return None
    try:
        formula = f"{dv} ~ C(expertise_group, Treatment(reference='non_climber')) * C(condition, Treatment(reference='everyday'))"
        model = mixedlm(formula, data=df_trials_muscle, groups=df_trials_muscle["participant_id"])
        fit = model.fit(reml=True)
        return pd.DataFrame({
            "muscle": muscle,
            "dv": dv,
            "Term": fit.params.index,
            "Estimate_Beta": fit.params.values,
            "SE": fit.bse.values,
            "z_stat": fit.tvalues.values,
            "p_value": fit.pvalues.values,
            "CI_95_low": fit.conf_int()[0].values,
            "CI_95_high": fit.conf_int()[1].values
        })
    except Exception:
        return None


# ==============================================================================
# 7. AGONIST-ANTAGONIST CO-ACTIVATION CORRELATION (FDS vs. ECR)
# ==============================================================================
def run_fds_ecr_coactivation_analysis(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Tests the biomechanical coupling between CH1 (FDS: finger flexor) and CH2 (ECR: wrist stabilizer)
    across trials within each (Group x Condition) cell.
    In rock climbing, crimping and pinching require simultaneous FDS flexion and ECR wrist extension
    co-contraction to prevent wrist collapse.
    """
    if df_trials is None or df_trials.empty or "muscle" not in df_trials.columns:
        return pd.DataFrame()

    muscles_present = set(df_trials["muscle"].unique())
    if not {"FDS", "ECR"}.issubset(muscles_present):
        return pd.DataFrame()

    piv = df_trials.pivot_table(
        index=["participant_id", "expertise_group", "trial_num", "condition", "stimulus_file"],
        columns="muscle",
        values=["norm_auc_pct", "norm_peak_pct"]
    ).dropna()

    rows = []
    for dv in ["norm_auc_pct", "norm_peak_pct"]:
        for (grp, cond), sub in piv.groupby(level=["expertise_group", "condition"]):
            fds_vals = sub[(dv, "FDS")].values
            ecr_vals = sub[(dv, "ECR")].values
            if len(fds_vals) >= 4:
                r_p, p_p = stats.pearsonr(fds_vals, ecr_vals)
                r_s, p_s = stats.spearmanr(fds_vals, ecr_vals)
            else:
                r_p = p_p = r_s = p_s = np.nan

            rows.append({
                "dv": dv,
                "expertise_group": grp,
                "condition": cond,
                "n_paired_trials": len(fds_vals),
                "pearson_r_FDS_ECR": r_p,
                "pearson_p": p_p,
                "spearman_rho_FDS_ECR": r_s,
                "spearman_p": p_s
            })

    coact_df = pd.DataFrame(rows)
    coact_df.to_csv(config.STATS_DIR / "FDS_vs_ECR_coactivation_correlations.csv", index=False)
    return coact_df


# ==============================================================================
# 8. DEMO SIMULATOR & MASTER PIPELINE RUNNER
# ==============================================================================
def simulate_demo_dataset(n_per_group: int = 32, seed: int = 42):
    """
    Generates a synthetic N=64 dual-channel dataset (32 climbers, 32 non-climbers; FDS & ECR)
    matching Section 3.4 & 3.5 of the preliminary report.
    """
    rng = np.random.default_rng(seed)
    trial_rows = []

    for ch_key, m_code in [("ch1", "FDS"), ("ch2", "ECR")]:
        for g_idx, group in enumerate(["climber", "non_climber"]):
            for s in range(1, n_per_group + 1):
                sub_id = f"sim-{group[:3].upper()}{s:02d}"
                subj_intercept = rng.normal(0.0, 7.0)

                for t_num in range(1, 31):
                    cond = "climbing" if t_num <= 15 else "everyday"
                    if cond == "everyday":
                        mu_pct = 24.0 if m_code == "FDS" else 19.0
                    else:
                        if group == "climber":
                            mu_pct = 39.0 if m_code == "FDS" else 31.0
                        else:
                            mu_pct = 4.5 if m_code == "FDS" else 3.5

                    auc_pct = mu_pct + subj_intercept + rng.normal(0.0, 15.0)
                    peak_pct = auc_pct * 1.35 + rng.normal(0.0, 11.0)

                    trial_rows.append({
                        "participant_id": sub_id,
                        "expertise_group": group,
                        "dominant_hand": "right",
                        "channel": ch_key,
                        "muscle": m_code,
                        "seed": seed + s,
                        "trial_num": t_num,
                        "condition": cond,
                        "stimulus_file": f"{cond}_{t_num:02d}.mp4",
                        "norm_auc_pct": auc_pct,
                        "norm_peak_pct": peak_pct,
                        "norm_auc_ratio": 1.0 + (auc_pct / 100.0),
                        "norm_peak_ratio": 1.0 + (peak_pct / 100.0),
                        "zscore_auc": auc_pct / 14.0,
                        "zscore_peak": peak_pct / 14.0,
                        "local_isi_norm_auc_pct": auc_pct + rng.normal(0.0, 2.0),
                        "local_isi_norm_peak_pct": peak_pct + rng.normal(0.0, 2.0)
                    })

    df_trials = pd.DataFrame(trial_rows)
    dv_cols = [
        "norm_peak_pct", "norm_auc_pct",
        "norm_peak_ratio", "norm_auc_ratio",
        "zscore_peak", "zscore_auc",
        "local_isi_norm_peak_pct", "local_isi_norm_auc_pct"
    ]
    df_subj = (
        df_trials
        .groupby(["participant_id", "expertise_group", "dominant_hand", "channel", "muscle", "condition"], as_index=False)[dv_cols]
        .mean()
    )
    return df_subj, df_trials


def run_full_statistical_analysis(df_subj: pd.DataFrame, df_trials: pd.DataFrame = None):
    if "muscle" not in df_subj.columns:
        df_subj = df_subj.copy()
        df_subj["muscle"] = "FDS"
        df_subj["channel"] = "ch1"

    muscles = sorted(df_subj["muscle"].unique(), reverse=True)  # ['FDS', 'ECR']
    dvs_to_analyze = [
        ("norm_auc_pct", "Normalized AUC (% Change vs. 30s Baseline)", 0.0),
        ("norm_peak_pct", "Normalized Peak Amplitude (% Change vs. 30s Baseline)", 0.0),
        ("zscore_auc", "Baseline Z-Scored AUC (SNR Units)", 0.0),
        ("zscore_peak", "Baseline Z-Scored Peak Amplitude (SNR Units)", 0.0),
        ("local_isi_norm_auc_pct", "Local Pre-Trial ISI Drift-Corrected AUC (%)", 0.0),
    ]

    verdict_rows = []

    for muscle in muscles:
        df_m = df_subj[df_subj["muscle"] == muscle].copy()
        df_trials_m = (
            df_trials[df_trials["muscle"] == muscle].copy()
            if (df_trials is not None and "muscle" in df_trials.columns)
            else None
        )

        for dv_col, dv_label, null_ref in dvs_to_analyze:
            if dv_col not in df_m.columns:
                continue

            prefix = f"{muscle}_{dv_col}"
            print(f"\n==========================================================================")
            print(f"MUSCLE: {muscle} | DV: {dv_col} ({dv_label})")
            print(f"==========================================================================")

            # 1. Assumption Checks (Shapiro-Wilk & Levene's Test)
            assump = check_anova_assumptions(df_m, dv_col, muscle)
            assump["shapiro_df"].to_csv(config.STATS_DIR / f"{prefix}_1_shapiro_normality.csv", index=False)
            assump["levene_df"].to_csv(config.STATS_DIR / f"{prefix}_2_levene_variance.csv", index=False)

            print(
                f"  [1] ASSUMPTIONS -> Shapiro-Wilk Normality: {'PASSED' if assump['normality_passed'] else 'VIOLATED (p < 0.05)'} | "
                f"Levene Homoscedasticity: {'PASSED' if assump['homoscedasticity_passed'] else 'VIOLATED (p < 0.05)'}"
            )

            # 2. Parametric 2x2 Mixed Factorial ANOVA
            aov_param = run_mixed_factorial_anova(df_m, dv_col)
            if not aov_param.empty:
                aov_param.insert(0, "muscle", muscle)
                aov_param.to_csv(config.STATS_DIR / f"{prefix}_3_parametric_mixed_anova.csv", index=False)
                print("\n  [2] PARAMETRIC 2x2 MIXED FACTORIAL ANOVA:")
                print(aov_param.to_string(index=False))

            # 3. Non-Parametric Aligned Rank Transform (ART) 2x2 Mixed ANOVA
            aov_art = run_aligned_rank_transform_anova(df_m, dv_col)
            if not aov_art.empty:
                aov_art.insert(0, "muscle", muscle)
                aov_art.to_csv(config.STATS_DIR / f"{prefix}_4_nonparametric_ART_anova.csv", index=False)
                print("\n  [3] NON-PARAMETRIC ALIGNED RANK TRANSFORM (ART) 2x2 ANOVA:")
                print(aov_art.to_string(index=False))

            # 4. Bonferroni-Corrected Simple Main Effects
            sme_df = run_simple_main_effects(df_m, dv_col, muscle)
            sme_df.to_csv(config.STATS_DIR / f"{prefix}_5_bonferroni_simple_main_effects.csv", index=False)
            print("\n  [4] BONFERRONI-CORRECTED SIMPLE MAIN EFFECTS:")
            print(sme_df[[
                "comparison", "mean_A", "mean_B", "diff_A_minus_B",
                "t_stat", "p_bonf_param", "cohens_d", "p_bonf_nonparam"
            ]].to_string(index=False))

            # 5. One-Sample Tests vs. Resting Baseline (Testing H0 Spinal Inhibition directly)
            base_eq_df = run_baseline_equivalence_tests(df_m, dv_col, muscle, null_value=null_ref)
            base_eq_df.to_csv(config.STATS_DIR / f"{prefix}_6_vs_resting_baseline_tests.csv", index=False)

            # 6. Trial-Level Linear Mixed-Effects Model (LMM)
            if df_trials_m is not None and dv_col in df_trials_m.columns:
                lmm_df = run_trial_level_lmm(df_trials_m, dv_col, muscle)
                if lmm_df is not None:
                    lmm_df.to_csv(config.STATS_DIR / f"{prefix}_7_trial_level_LMM.csv", index=False)

            # Record Master Verdict Row for this (Muscle, DV)
            if not aov_param.empty and not aov_art.empty:
                use_param = assump["all_assumptions_met"]
                primary_aov = aov_param if use_param else aov_art
                inter_row = primary_aov.iloc[-1]
                p_inter = float(inter_row["p_unc"])
                verdict_rows.append({
                    "muscle": muscle,
                    "dv": dv_col,
                    "normality_met": assump["normality_passed"],
                    "homoscedasticity_met": assump["homoscedasticity_passed"],
                    "recommended_primary_test": "Parametric_2x2_Mixed_ANOVA" if use_param else "NonParametric_ART_2x2_ANOVA",
                    "interaction_F": float(inter_row["F"]),
                    "interaction_p": p_inter,
                    "interaction_np2": float(inter_row["np2"]),
                    "hypothesis_supported": "H1 (Motor Resonance Interaction)" if p_inter < config.ALPHA_LEVEL else "H0 (Spinal Inhibition Null)"
                })

    # 7. Cross-Muscle Co-Activation Analysis (FDS vs. ECR)
    coact_df = run_fds_ecr_coactivation_analysis(df_trials)
    if not coact_df.empty:
        print(f"\n==========================================================================")
        print("CROSS-MUSCLE AGONIST-STABILIZER CO-ACTIVATION (FDS vs. ECR CORRELATIONS)")
        print("==========================================================================")
        print(coact_df.to_string(index=False))

    # 8. Save Master Verdict Summary Table
    if verdict_rows:
        verdict_df = pd.DataFrame(verdict_rows)
        verdict_path = config.STATS_DIR / "master_statistical_verdict_summary.csv"
        verdict_df.to_csv(verdict_path, index=False)
        print(f"\n==========================================================================")
        print("MASTER STATISTICAL VERDICT SUMMARY (FDS & ECR)")
        print("==========================================================================")
        print(verdict_df.to_string(index=False))
        print(f"\n[SAVED ALL STATISTICAL TABLES TO]: {config.STATS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dual-Channel (FDS & ECR) 2x2 Mixed Factorial Statistical Analysis")
    parser.add_argument(
        "--subj-table",
        type=str,
        default=str(config.PROCESSED_DIR / "master_subject_level_anova_table.csv"),
        help="Path to master_subject_level_anova_table.csv from postprocess_emg.py"
    )
    parser.add_argument(
        "--trial-table",
        type=str,
        default=str(config.PROCESSED_DIR / "master_trial_level_features.csv"),
        help="Path to master_trial_level_features.csv from postprocess_emg.py"
    )
    parser.add_argument(
        "--simulate-demo",
        action="store_true",
        help="Run on simulated N=64 dual-channel (FDS + ECR) dataset to verify all tests"
    )
    args = parser.parse_args()

    if args.simulate_demo:
        print("[DEMO MODE] Simulating N=64 dual-channel dataset (32 climbers, 32 non-climbers; FDS & ECR)...")
        df_subj_in, df_trials_in = simulate_demo_dataset(n_per_group=32)
    else:
        subj_path = Path(args.subj_table)
        trial_path = Path(args.trial_table)
        if not subj_path.exists():
            raise FileNotFoundError(
                f"Subject ANOVA table not found at '{subj_path}'.\n"
                f"Run 'python postprocess_emg.py' first, or test now with:\n"
                f"  python stats_analysis.py --simulate-demo"
            )
        df_subj_in = pd.read_csv(subj_path)
        df_trials_in = pd.read_csv(trial_path) if trial_path.exists() else None

    run_full_statistical_analysis(df_subj_in, df_trials_in)