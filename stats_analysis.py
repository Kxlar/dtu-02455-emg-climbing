# ./Experiment in Cognitive Science 02455/stats_analysis.py
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm

import config

warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. ASSUMPTION CHECKS: SHAPIRO-WILK (NORMALITY) & LEVENE'S (HOMOSCEDASTICITY)
# ==============================================================================
def check_anova_assumptions(df_subj: pd.DataFrame, dv: str) -> dict:
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
            passed = p_val >= config.ALPHA_LEVEL
            if not passed:
                normality_passed = False
        else:
            stat_w, p_val, passed = np.nan, np.nan, True

        shapiro_rows.append({
            "dv": dv,
            "expertise_group": group,
            "condition": cond,
            "n": len(vals),
            "mean": np.mean(vals) if len(vals) else np.nan,
            "sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "median": np.median(vals) if len(vals) else np.nan,
            "iqr": stats.iqr(vals) if len(vals) else np.nan,
            "shapiro_W": stat_w,
            "shapiro_p": p_val,
            "normality_met": passed
        })

    levene_rows = []
    homoscedasticity_passed = True

    for cond, sub_df in df_subj.groupby("condition"):
        climbers = sub_df[sub_df["expertise_group"] == "climber"][dv].dropna().values
        non_climbers = sub_df[sub_df["expertise_group"] == "non_climber"][dv].dropna().values

        if len(climbers) >= 2 and len(non_climbers) >= 2:
            stat_l, p_val = stats.levene(climbers, non_climbers, center="median")
            passed = p_val >= config.ALPHA_LEVEL
            if not passed:
                homoscedasticity_passed = False
        else:
            stat_l, p_val, passed = np.nan, np.nan, True

        levene_rows.append({
            "dv": dv,
            "condition": cond,
            "levene_W": stat_l,
            "levene_p": p_val,
            "homoscedasticity_met": passed
        })

    all_assumptions_met = normality_passed and homoscedasticity_passed
    return {
        "shapiro_df": pd.DataFrame(shapiro_rows),
        "levene_df": pd.DataFrame(levene_rows),
        "normality_passed": normality_passed,
        "homoscedasticity_passed": homoscedasticity_passed,
        "all_assumptions_met": all_assumptions_met
    }


# ==============================================================================
# 2. PARAMETRIC 2x2 MIXED FACTORIAL ANOVA
# ==============================================================================
def run_mixed_factorial_anova(df_subj: pd.DataFrame, dv: str) -> pd.DataFrame:
    """
    Computes the 2x2 Mixed Factorial ANOVA:
      - Between-Subjects Factor: expertise_group (climber vs. non_climber)
      - Within-Subjects Factor : condition (climbing vs. everyday)
    Uses pingouin.mixed_anova if installed; otherwise computes exact analytical
    Mixed ANOVA sums of squares using scipy/numpy.
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
        aov["test_type"] = "Parametric_2x2_Mixed_ANOVA"
        return aov
    except ImportError:
        pass

    # --- Exact Analytical 2x2 Mixed Factorial ANOVA (No Pingouin dependency required) ---
    # Pivot so each subject is a row with columns ['climbing', 'everyday']
    wide = df_subj.pivot(
        index=["participant_id", "expertise_group"],
        columns="condition",
        values=dv
    ).dropna().reset_index()

    groups = wide["expertise_group"].unique()
    conds = [c for c in wide.columns if c not in ("participant_id", "expertise_group")]
    N = len(wide)
    k_cond = len(conds)      # 2 conditions
    k_grp = len(groups)      # 2 groups

    all_vals = wide[conds].values.flatten()
    grand_mean = np.mean(all_vals)

    # Subject means across conditions
    wide["subj_mean"] = wide[conds].mean(axis=1)

    # 1. Between-Subjects Partition: SS_Group and SS_Subj(Group)
    ss_group = 0.0
    ss_subj_within_group = 0.0
    for g in groups:
        sub_g = wide[wide["expertise_group"] == g]
        n_g = len(sub_g)
        g_mean = sub_g["subj_mean"].mean()
        ss_group += n_g * k_cond * ((g_mean - grand_mean) ** 2)
        ss_subj_within_group += k_cond * np.sum((sub_g["subj_mean"] - g_mean) ** 2)

    df_group = k_grp - 1
    df_error_btw = N - k_grp
    ms_group = ss_group / df_group
    ms_error_btw = ss_subj_within_group / max(df_error_btw, 1)
    f_group = ms_group / ms_error_btw if ms_error_btw > 0 else np.nan
    p_group = stats.f.sf(f_group, df_group, df_error_btw)
    np2_group = ss_group / (ss_group + ss_subj_within_group) if (ss_group + ss_subj_within_group) > 0 else 0.0

    # 2. Within-Subjects Partition: SS_Condition, SS_Interaction, SS_Error_Within
    cond_means = {c: wide[c].mean() for c in conds}
    ss_cond = sum(N * ((cond_means[c] - grand_mean) ** 2) for c in conds)

    ss_cells = 0.0
    for g in groups:
        sub_g = wide[wide["expertise_group"] == g]
        n_g = len(sub_g)
        for c in conds:
            cell_mean = sub_g[c].mean()
            ss_cells += n_g * ((cell_mean - grand_mean) ** 2)

    ss_inter = ss_cells - ss_group - ss_cond

    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_error_within = ss_total - ss_group - ss_subj_within_group - ss_cond - ss_inter

    df_cond = k_cond - 1
    df_inter = df_group * df_cond
    df_error_wth = (N - k_grp) * (k_cond - 1)

    ms_cond = ss_cond / df_cond
    ms_inter = ss_inter / df_inter
    ms_error_wth = ss_error_within / max(df_error_wth, 1)

    f_cond = ms_cond / ms_error_wth if ms_error_wth > 0 else np.nan
    p_cond = stats.f.sf(f_cond, df_cond, df_error_wth)
    np2_cond = ss_cond / (ss_cond + ss_error_within) if (ss_cond + ss_error_within) > 0 else 0.0

    f_inter = ms_inter / ms_error_wth if ms_error_wth > 0 else np.nan
    p_inter = stats.f.sf(f_inter, df_inter, df_error_wth)
    np2_inter = ss_inter / (ss_inter + ss_error_within) if (ss_inter + ss_error_within) > 0 else 0.0

    return pd.DataFrame([
        {
            "Source": "expertise_group (Between)",
            "SS": ss_group, "DF1": df_group, "DF2": df_error_btw,
            "MS": ms_group, "F": f_group, "p_unc": p_group, "np2": np2_group,
            "test_type": "Parametric_2x2_Mixed_ANOVA"
        },
        {
            "Source": "condition (Within)",
            "SS": ss_cond, "DF1": df_cond, "DF2": df_error_wth,
            "MS": ms_cond, "F": f_cond, "p_unc": p_cond, "np2": np2_cond,
            "test_type": "Parametric_2x2_Mixed_ANOVA"
        },
        {
            "Source": "Interaction (expertise_group * condition)",
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
    Implements the Aligned Rank Transform (ART) procedure (Wobbrock et al., 2011; Higgins & Tashtoush, 1994)
    as specified in Section 3.5 for non-normal / high-variance sEMG data:
      1. Computes marginal means for Group (A), Condition (B), and Cell (AB)
      2. Aligns data separately for Main Effect A, Main Effect B, and Interaction AB
         by stripping out all nuisance effects from each observation:
           - Aligned_A  = Y_ijk - (CellMean_ij - GroupMean_i - CondMean_j + GrandMean) - (CondMean_j - GrandMean)
           - Aligned_B  = Y_ijk - (CellMean_ij - GroupMean_i - CondMean_j + GrandMean) - (GroupMean_i - GrandMean)
           - Aligned_AB = Y_ijk - (GroupMean_i - GrandMean) - (CondMean_j - GrandMean)
      3. Assigns mid-ranks (1..N_obs) to each aligned response column
      4. Runs a 2x2 Mixed Factorial ANOVA on each ranked column and extracts ONLY the corresponding effect row.
    """
    df = df_subj.copy().dropna(subset=[dv])
    grand_mean = df[dv].mean()

    group_means = df.groupby("expertise_group")[dv].transform("mean")
    cond_means = df.groupby("condition")[dv].transform("mean")
    cell_means = df.groupby(["expertise_group", "condition"])[dv].transform("mean")

    # Estimated effects
    eff_A = group_means - grand_mean
    eff_B = cond_means - grand_mean
    eff_AB = cell_means - group_means - cond_means + grand_mean

    # Step 1: Align observations for each effect by stripping nuisance effects
    df["aligned_A"] = df[dv] - eff_B - eff_AB
    df["aligned_B"] = df[dv] - eff_A - eff_AB
    df["aligned_AB"] = df[dv] - eff_A - eff_B

    # Step 2: Rank the aligned columns (average rank for ties)
    df["rank_A"] = stats.rankdata(df["aligned_A"], method="average")
    df["rank_B"] = stats.rankdata(df["aligned_B"], method="average")
    df["rank_AB"] = stats.rankdata(df["aligned_AB"], method="average")

    # Step 3: Run Mixed Factorial ANOVA on each aligned rank column and extract its target row
    aov_A = run_mixed_factorial_anova(df, "rank_A")
    aov_B = run_mixed_factorial_anova(df, "rank_B")
    aov_AB = run_mixed_factorial_anova(df, "rank_AB")

    row_A = aov_A.iloc[[0]].copy()
    row_B = aov_B.iloc[[1]].copy()
    row_AB = aov_AB.iloc[[2]].copy()

    art_table = pd.concat([row_A, row_B, row_AB], ignore_index=True)
    art_table["test_type"] = "NonParametric_ART_2x2_Mixed_ANOVA"
    return art_table


# ==============================================================================
# 4. BONFERRONI-CORRECTED SIMPLE MAIN EFFECTS & COHEN'S d
# ==============================================================================
def run_simple_main_effects(df_subj: pd.DataFrame, dv: str) -> pd.DataFrame:
    """
    Follow-up simple main effects analysis (Section 3.5) with Bonferroni correction:
      Family 1 (Between-Group Simple Effects at each Stimulus Condition):
        - Test 1A: Climbers vs. Non-Climbers in 'climbing' videos (Primary H1 test!)
        - Test 1B: Climbers vs. Non-Climbers in 'everyday' videos (Control check)
      Family 2 (Within-Group Simple Effects across Stimulus Conditions):
        - Test 2A: 'climbing' vs. 'everyday' videos within Climbers (Primary H1 test!)
        - Test 2B: 'climbing' vs. 'everyday' videos within Non-Climbers (Control check)
    Computes both Parametric (Welch/Student t-test + Cohen's d) and Non-Parametric
    (Mann-Whitney U / Wilcoxon Signed-Rank + Rank-Biserial correlation) statistics.
    """
    results = []

    # --- FAMILY 1: Between-Subjects Simple Main Effects (Climber vs. Non-Climber per condition) ---
    for cond in ["climbing", "everyday"]:
        sub = df_subj[df_subj["condition"] == cond]
        g_climb = sub[sub["expertise_group"] == "climber"][dv].dropna().values
        g_non = sub[sub["expertise_group"] == "non_climber"][dv].dropna().values

        if len(g_climb) >= 2 and len(g_non) >= 2:
            t_stat, p_param = stats.ttest_ind(g_climb, g_non, equal_var=False)
            u_stat, p_nonparam = stats.mannwhitneyu(g_climb, g_non, alternative="two-sided")

            # Pooled SD for Cohen's d
            n1, n2 = len(g_climb), len(g_non)
            s_pool = np.sqrt(((n1 - 1) * np.var(g_climb, ddof=1) + (n2 - 1) * np.var(g_non, ddof=1)) / (n1 + n2 - 2))
            cohens_d = (np.mean(g_climb) - np.mean(g_non)) / s_pool if s_pool > 0 else 0.0
        else:
            t_stat = p_param = u_stat = p_nonparam = cohens_d = np.nan

        results.append({
            "contrast_family": "Between_Groups_at_Condition",
            "comparison": f"climber vs. non_climber (within '{cond}' videos)",
            "mean_A": np.mean(g_climb) if len(g_climb) else np.nan,
            "mean_B": np.mean(g_non) if len(g_non) else np.nan,
            "diff_A_minus_B": (np.mean(g_climb) - np.mean(g_non)) if (len(g_climb) and len(g_non)) else np.nan,
            "t_stat": t_stat,
            "p_unc_param": p_param,
            "p_bonf_param": min(1.0, p_param * 4.0) if not np.isnan(p_param) else np.nan,
            "cohens_d": cohens_d,
            "nonparam_stat_U_or_W": u_stat,
            "p_unc_nonparam": p_nonparam,
            "p_bonf_nonparam": min(1.0, p_nonparam * 4.0) if not np.isnan(p_nonparam) else np.nan
        })

    # --- FAMILY 2: Within-Subjects Simple Main Effects (Climbing vs. Everyday per group) ---
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
            "contrast_family": "Within_Group_Across_Conditions",
            "comparison": f"climbing vs. everyday (within '{grp}' group)",
            "mean_A": np.mean(v_climb) if len(v_climb) else np.nan,
            "mean_B": np.mean(v_every) if len(v_every) else np.nan,
            "diff_A_minus_B": (np.mean(v_climb) - np.mean(v_every)) if len(v_climb) else np.nan,
            "t_stat": t_stat,
            "p_unc_param": p_param,
            "p_bonf_param": min(1.0, p_param * 4.0) if not np.isnan(p_param) else np.nan,
            "cohens_d": cohens_dz,
            "nonparam_stat_U_or_W": w_stat,
            "p_unc_nonparam": p_nonparam,
            "p_bonf_nonparam": min(1.0, p_nonparam * 4.0) if not np.isnan(p_nonparam) else np.nan
        })

    return pd.DataFrame(results)


# ==============================================================================
# 5. BONUS TEST A: ONE-SAMPLE TEST AGAINST RESTING BASELINE (SPINAL INHIBITION H0)
# ==============================================================================
def run_baseline_equivalence_tests(df_subj: pd.DataFrame, dv: str, null_value: float = 0.0) -> pd.DataFrame:
    """
    Directly tests H0 (Section 1.4 & 2.2: Spinal Inhibition Null Hypothesis):
    Checks whether each (Group x Condition) cell is significantly elevated above
    the resting baseline reference (0.0 for % change and Z-score; 1.0 for ratio).
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
            "dv": dv,
            "expertise_group": group,
            "condition": cond,
            "null_baseline_ref": null_value,
            "cell_mean": np.mean(vals) if len(vals) else np.nan,
            "cell_sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "t_vs_baseline": t_stat,
            "p_param_vs_baseline": p_t,
            "p_bonf_param": min(1.0, p_t * 4.0) if not np.isnan(p_t) else np.nan,
            "wilcoxon_vs_baseline": w_stat,
            "p_nonparam_vs_baseline": p_w
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 6. BONUS TEST B: TRIAL-LEVEL LINEAR MIXED-EFFECTS MODEL (LMM)
# ==============================================================================
def run_trial_level_lmm(df_trials: pd.DataFrame, dv: str):
    """
    Fits a Linear Mixed-Effects Model (LMM) on all N * 30 un-averaged trials:
      dv ~ expertise_group * condition + (1 | participant_id)
    Advantages over subject-averaged ANOVA:
      - Retains all 30 trials per subject (higher statistical power)
      - Naturally handles missing/dropped artifact trials
    """
    if df_trials is None or df_trials.empty:
        return None
    try:
        formula = f"{dv} ~ C(expertise_group, Treatment(reference='non_climber')) * C(condition, Treatment(reference='everyday'))"
        model = mixedlm(formula, data=df_trials, groups=df_trials["participant_id"])
        fit = model.fit(reml=True)
        summary_df = pd.DataFrame({
            "Term": fit.params.index,
            "Estimate_Beta": fit.params.values,
            "SE": fit.bse.values,
            "z_stat": fit.tvalues.values,
            "p_value": fit.pvalues.values,
            "CI_95_low": fit.conf_int()[0].values,
            "CI_95_high": fit.conf_int()[1].values
        })
        return summary_df
    except Exception as e:
        print(f"  [LMM INFO] Could not converge trial-level LMM for {dv}: {e}")
        return None


# ==============================================================================
# 7. PUBLICATION-READY INTERACTION & DISTRIBUTION PLOTS
# ==============================================================================
def plot_interaction_and_distributions(df_subj: pd.DataFrame, dv: str, dv_label: str):
    """
    Generates a 2-panel publication figure for the report:
      - Left Panel : 2x2 Interaction Plot (Mean +/- 95% CI across Stimulus Type x Expertise Group)
      - Right Panel: Individual Subject Paired Trajectories + Box/Violin Distributions
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    fig.suptitle(f"2x2 Mixed Factorial Design — {dv_label}", fontsize=13, fontweight="bold")

    colors = {"climber": "#d62728", "non_climber": "#1f77b4"}
    labels = {"climber": "Expert Climbers (Group A)", "non_climber": "Non-Climbers (Group B)"}
    cond_order = ["everyday", "climbing"]
    cond_x = [0, 1]

    # --- Panel 1: Classical 2x2 Interaction Plot (Mean +/- SEM) ---
    ax1 = axes[0]
    for grp in ["climber", "non_climber"]:
        means, sems = [], []
        for c in cond_order:
            vals = df_subj[(df_subj["expertise_group"] == grp) & (df_subj["condition"] == c)][dv].dropna().values
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(stats.sem(vals) if len(vals) > 1 else 0.0)

        offset = -0.03 if grp == "climber" else 0.03
        x_pts = [x + offset for x in cond_x]
        ax1.errorbar(
            x_pts, means, yerr=sems, marker="o", markersize=8, lw=2.2, capsize=5,
            color=colors[grp], label=labels[grp]
        )

    null_line = 1.0 if "ratio" in dv else 0.0
    ax1.axhline(null_line, color="gray", ls="--", lw=1.2, label="Resting Baseline Reference")
    ax1.set_xticks(cond_x)
    ax1.set_xticklabels(["Everyday Grasping\n(Condition B: Control)", "Climbing Grips\n(Condition A: Target)"])
    ax1.set_ylabel(dv_label)
    ax1.set_title("A. Group x Stimulus Interaction (Mean ± SEM)")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Individual Participant Spaghetti Trajectories & Boxplots ---
    ax2 = axes[1]
    wide = df_subj.pivot(index=["participant_id", "expertise_group"], columns="condition", values=dv).dropna().reset_index()

    for idx_g, grp in enumerate(["climber", "non_climber"]):
        sub_w = wide[wide["expertise_group"] == grp]
        jitter = -0.15 if grp == "climber" else 0.15
        x0, x1 = 0 + jitter, 1 + jitter

        # Plot paired lines per participant
        for _, row in sub_w.iterrows():
            ax2.plot([x0, x1], [row["everyday"], row["climbing"]], color=colors[grp], alpha=0.25, lw=1.0)
            ax2.scatter([x0, x1], [row["everyday"], row["climbing"]], color=colors[grp], alpha=0.6, s=22)

        # Overlay boxplot summaries
        if len(sub_w) > 0:
            bp = ax2.boxplot(
                [sub_w["everyday"].values, sub_w["climbing"].values],
                positions=[x0, x1],
                widths=0.18,
                patch_artist=True,
                showfliers=False
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[grp])
                patch.set_alpha(0.35)
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.8)

    ax2.axhline(null_line, color="gray", ls="--", lw=1.2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Everyday Grasping\n(Condition B: Control)", "Climbing Grips\n(Condition A: Target)"])
    ax2.set_title("B. Participant Paired Trajectories & Distributions")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = config.STATS_PLOTS_DIR / f"interaction_plot_{dv}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"  [PLOT SAVED] {fig_path.name}")


# ==============================================================================
# 8. MASTER PIPELINE RUNNER & DEMO SIMULATOR
# ==============================================================================
def simulate_demo_dataset(n_per_group: int = 32, seed: int = 42):
    """
    Generates a synthetic dataset matching Section 3.4 & 3.5 of the report
    (32 expert climbers + 32 non-climbers = 64 participants, 30 trials each)
    so you can test and verify the statistical pipeline immediately.
    """
    rng = np.random.default_rng(seed)
    trial_rows = []

    for g_idx, group in enumerate(["climber", "non_climber"]):
        for s in range(1, n_per_group + 1):
            sub_id = f"sim-{group[:3].upper()}{s:02d}"
            subj_intercept = rng.normal(0.0, 8.0)

            for t_num in range(1, 31):
                cond = "climbing" if t_num <= 15 else "everyday"
                # Expected interaction pattern per Section 3.5:
                # Everyday grasp activates both groups (~ +25% over resting baseline);
                # Climbing grips activate climbers (~ +35%) much more than non-climbers (~ +5%).
                if cond == "everyday":
                    mu_pct = 25.0
                else:
                    mu_pct = 35.0 if group == "climber" else 5.0

                auc_pct = mu_pct + subj_intercept + rng.normal(0.0, 18.0)
                peak_pct = auc_pct * 1.35 + rng.normal(0.0, 12.0)

                trial_rows.append({
                    "participant_id": sub_id,
                    "expertise_group": group,
                    "dominant_hand": "right",
                    "seed": seed + s,
                    "trial_num": t_num,
                    "condition": cond,
                    "stimulus_file": f"{cond}_{t_num:02d}.mp4",
                    "norm_auc_pct": auc_pct,
                    "norm_peak_pct": peak_pct,
                    "norm_auc_ratio": 1.0 + (auc_pct / 100.0),
                    "norm_peak_ratio": 1.0 + (peak_pct / 100.0),
                    "zscore_auc": auc_pct / 15.0,
                    "zscore_peak": peak_pct / 15.0,
                    "local_isi_norm_auc_pct": auc_pct + rng.normal(0.0, 3.0),
                    "local_isi_norm_peak_pct": peak_pct + rng.normal(0.0, 3.0)
                })

    df_trials = pd.DataFrame(trial_rows)
    dv_cols = [
        "norm_peak_pct", "norm_auc_pct",
        "norm_peak_ratio", "norm_peak_ratio",
        "zscore_peak", "zscore_auc",
        "local_isi_norm_peak_pct", "local_isi_norm_auc_pct"
    ]
    df_subj = (
        df_trials
        .groupby(["participant_id", "expertise_group", "dominant_hand", "condition"], as_index=False)[
            ["norm_peak_pct", "norm_auc_pct", "norm_peak_ratio", "norm_auc_ratio",
             "zscore_peak", "zscore_auc", "local_isi_norm_peak_pct", "local_isi_norm_auc_pct"]
        ]
        .mean()
    )
    return df_subj, df_trials


def run_full_statistical_analysis(df_subj: pd.DataFrame, df_trials: pd.DataFrame = None):
    """
    Executes the complete statistical testing workflow for both primary DVs
    (Normalized Peak Amplitude and Normalized Area Under the Curve) plus Z-score DVs.
    """
    dvs_to_analyze = [
        ("norm_auc_pct", "Normalized AUC (% Change vs. 30s Baseline)", 0.0),
        ("norm_peak_pct", "Normalized Peak Amplitude (% Change vs. 30s Baseline)", 0.0),
        ("zscore_auc", "Baseline Z-Scored AUC (SNR Units)", 0.0),
        ("zscore_peak", "Baseline Z-Scored Peak Amplitude (SNR Units)", 0.0),
    ]

    for dv_col, dv_label, null_ref in dvs_to_analyze:
        if dv_col not in df_subj.columns:
            continue

        print(f"\n======================================================================")
        print(f"STATISTICAL ANALYSIS FOR DV: {dv_col} ({dv_label})")
        print(f"======================================================================")

        # 1. Check Parametric Assumptions (Shapiro-Wilk & Levene's Test)
        assump = check_anova_assumptions(df_subj, dv_col)
        assump["shapiro_df"].to_csv(config.STATS_DIR / f"{dv_col}_1_shapiro_normality.csv", index=False)
        assump["levene_df"].to_csv(config.STATS_DIR / f"{dv_col}_2_levene_variance.csv", index=False)

        print("\n[1] ASSUMPTION CHECKS (Section 3.5):")
        print(f"  -> Normality (Shapiro-Wilk across 4 cells) : {'PASSED' if assump['normality_passed'] else 'VIOLATED (p < 0.05)'}")
        print(f"  -> Homoscedasticity (Levene's test)        : {'PASSED' if assump['homoscedasticity_passed'] else 'VIOLATED (p < 0.05)'}")

        # 2. Run Parametric 2x2 Mixed Factorial ANOVA
        aov_param = run_mixed_factorial_anova(df_subj, dv_col)
        aov_param.to_csv(config.STATS_DIR / f"{dv_col}_3_parametric_mixed_anova.csv", index=False)

        print("\n[2] PARAMETRIC 2x2 MIXED FACTORIAL ANOVA:")
        print(aov_param.to_string(index=False))

        # 3. Run Non-Parametric Aligned Rank Transform (ART) 2x2 Mixed ANOVA
        aov_art = run_aligned_rank_transform_anova(df_subj, dv_col)
        aov_art.to_csv(config.STATS_DIR / f"{dv_col}_4_nonparametric_ART_anova.csv", index=False)

        print("\n[3] NON-PARAMETRIC ALIGNED RANK TRANSFORM (ART) 2x2 ANOVA:")
        if not assump["all_assumptions_met"]:
            print("  >>> NOTE: Parametric assumptions were violated; report ART ANOVA as primary result! <<<")
        print(aov_art.to_string(index=False))

        # 4. Bonferroni-Corrected Simple Main Effects
        sme_df = run_simple_main_effects(df_subj, dv_col)
        sme_df.to_csv(config.STATS_DIR / f"{dv_col}_5_bonferroni_simple_main_effects.csv", index=False)

        print("\n[4] BONFERRONI-CORRECTED SIMPLE MAIN EFFECTS (Post-Hoc Interaction Decomposition):")
        print(sme_df[[
            "comparison", "mean_A", "mean_B", "diff_A_minus_B",
            "t_stat", "p_bonf_param", "cohens_d", "p_bonf_nonparam"
        ]].to_string(index=False))

        # 5. One-Sample Tests vs. Resting Baseline (Testing H0 Spinal Inhibition directly)
        base_eq_df = run_baseline_equivalence_tests(df_subj, dv_col, null_value=null_ref)
        base_eq_df.to_csv(config.STATS_DIR / f"{dv_col}_6_vs_resting_baseline_tests.csv", index=False)

        print("\n[5] ONE-SAMPLE TESTS VS. RESTING BASELINE (Spinal Inhibition Check):")
        print(base_eq_df[[
            "expertise_group", "condition", "cell_mean", "cell_sd",
            "t_vs_baseline", "p_bonf_param", "p_nonparam_vs_baseline"
        ]].to_string(index=False))

        # 6. Trial-Level Linear Mixed-Effects Model (Bonus)
        if df_trials is not None and dv_col in df_trials.columns:
            lmm_df = run_trial_level_lmm(df_trials, dv_col)
            if lmm_df is not None:
                lmm_df.to_csv(config.STATS_DIR / f"{dv_col}_7_trial_level_LMM.csv", index=False)
                print("\n[6] TRIAL-LEVEL LINEAR MIXED-EFFECTS MODEL (Random Subject Intercepts):")
                print(lmm_df.to_string(index=False))

        # 7. Generate Publication Plots
        plot_interaction_and_distributions(df_subj, dv_col, dv_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2x2 Mixed Factorial Statistical Analysis for 02455 sEMG")
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
        help="Run on a simulated N=64 dataset (32 climbers, 32 non-climbers) to verify tables & plots"
    )
    args = parser.parse_args()

    if args.simulate_demo:
        print("[DEMO MODE] Simulating N=64 dataset (32 climbers, 32 non-climbers) per Section 3.4...")
        df_subj_in, df_trials_in = simulate_demo_dataset(n_per_group=32)
    else:
        subj_path = Path(args.subj_table)
        trial_path = Path(args.trial_table)
        if not subj_path.exists():
            raise FileNotFoundError(
                f"Subject ANOVA table not found at '{subj_path}'.\n"
                f"Run 'python postprocess_emg.py' first, or test this script now with:\n"
                f"  python stats_analysis.py --simulate-demo"
            )
        df_subj_in = pd.read_csv(subj_path)
        df_trials_in = pd.read_csv(trial_path) if trial_path.exists() else None

    run_full_statistical_analysis(df_subj_in, df_trials_in)