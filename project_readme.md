# Automated sEMG Motor Resonance Pipeline (Rock Climbing Study)

A comprehensive, automated experimental pipeline for a surface electromyography (sEMG) motor resonance study comparing expert rock climbers and non-climbers during action observation. 

This repository contains the complete system for **PsychaPy stimulus presentation**, **dual-channel Shimmer3 EXG acquisition (CH1: FDS finger flexor, CH2: ECR wrist extensor)**, **offline SD-card sync (handling `-1` timestamps via squeeze bookends)**, **Clancy et al. (2002) signal processing cascade**, **mixed factorial statistics (Parametric ANOVA & Aligned Rank Transform)**, and **publication-ready figure generation**.

---

## Repository Structure

```text
dtu-02455-emg-climbing/
├── stimuli/                  # Video stimuli folder (climbing/ and everyday/)
├── data/
│   ├── logs/                 # PsychoPy *_events.csv session logs
│   ├── raw_shimmer/          # Raw Shimmer SD card CSV recordings
│   ├── epochs/               # Sliced dual-channel .pkl epochs & QC plots
│   ├── processed/            # Master feature tables & waveforms
│   └── stats_results/        # Statistical tables & publication figures
├── config.py                 # Central configuration (timing, channels, filters)
├── standardize_videos.py     # FFmpeg video standardization (1080p @ 30fps, 4.0s)
├── run_experiment.py         # PsychoPy continuous zero-pause experimental loop
├── preprocess_emg.py         # Slices continuous Shimmer traces using sync squeezes
├── postprocess_emg.py        # Clancy (2002) cascade, baseline norms & feature extraction
├── stats_analysis.py         # Shapiro-Wilk, Levene, Mixed ANOVA, ART, & Bonferroni tests
├── plot.py                   # Unified plotting suite & raw Shimmer file inspector
└── requirements.txt          # Python library dependencies
```

---

## Installation Guide (Lab PC Setup)

Open VS Code (or your preferred terminal) in the directory where you want to store the project, then run the following commands:

```bash
# 1. Clone the GitHub repository
git clone https://github.com/Kxlar/dtu-02455-emg-climbing.git
cd dtu-02455-emg-climbing

# 2. Create a Python virtual environment
python -m venv venv_ExpCogSc

# 3. Activate the virtual environment
# On Windows (PowerShell):
./venv_ExpCogSc/Scripts/Activate.ps1
# On Linux / macOS:
source venv_ExpCogSc/bin/activate

# 4. Install required libraries
pip install -r requirements.txt

# 5. Initialize directory structure
python config.py
```

---

## Pipeline Execution Workflow

### Step 1: Standardize Video Stimuli
Ensures all climbing and everyday action videos have identical spatial dimensions ($1920 \times 1080$), frame rate ($30\text{ fps}$), and exact duration ($4.000\text{ seconds} = 120\text{ frames}$). Supports `.mp4`, `.mov`, `.wmv`, etc.
```bash
python standardize_videos.py
```

### Step 2: Run the Experiment (PsychoPy)
Launches the continuous zero-pause experiment. Press `SPACE` on the welcome screen to start. The experiment flows automatically from the **Start Squeeze ($t=0$)** $\rightarrow$ **Settle** $\rightarrow$ **30s Baseline** $\rightarrow$ **30 Video Trials + ISI** $\rightarrow$ **End Squeeze**.
```bash
python run_experiment.py
```

### Step 3: Preprocess & Slice sEMG Recordings (`preprocess_emg.py`)
Automatically aligns offline Shimmer SD-card files (which record `-1` timestamps) using the start and end calibration squeezes, computes empirical sampling rates, and slices continuous data into epoch files (`data/epochs/`).

**Available Argparse Arguments:**
* `--events <path>`: Path to a specific PsychoPy `*_events.csv` log file.
* `--shimmer <path>`: Path to a specific raw Shimmer CSV file.
* `--fs <float>`: Override Shimmer sampling rate in Hz (default: `512.0`).
* `--manual-t0 <float>`: Manual override for Start Squeeze $t=0$ time (in seconds from start of file).
* `--export-csvs`: Export each sliced trial epoch as an individual CSV segment.
* `--simulate-demo`: Generate synthetic dual-channel Shimmer files with `-1` timestamps and preprocess them automatically.

*Example:*
```bash
python preprocess_emg.py --simulate-demo
```

### Step 4: Signal Analysis & Clancy Cascade (`postprocess_emg.py`)
Applies the 4-stage Clancy et al. (2002) filter cascade (20–500 Hz bandpass, 50 Hz notch, full-wave rectification, 8 Hz low-pass linear envelope) and extracts normalized Peak Amplitude and Area Under the Curve (AUC) for both **FDS (CH1)** and **ECR (CH2)** independently.

**Available Argparse Arguments:**
* `--file <path>`: Specific `*_epochs.pkl` participant file to process.
* `--demod <int>`: Demodulation law $d$: `1` = Rectified (MAV), `2` = RMS ($d=2$). Default is `1`.
* `--simulate-demo`: Auto-generate demo Shimmer epochs if none exist and run full postprocessing.

*Example:*
```bash
python postprocess_emg.py --demod 1
```

### Step 5: Statistical Analysis (`stats_analysis.py`)
Runs tests of normality (Shapiro-Wilk) and homoscedasticity (Levene's), followed by Parametric $2\times 2$ Mixed Factorial ANOVA, Non-Parametric Aligned Rank Transform (ART) ANOVA, Bonferroni-corrected simple main effects, and one-sample baseline equivalence tests.

**Available Argparse Arguments:**
* `--subj-table <path>`: Path to `master_subject_level_anova_table.csv`.
* `--trial-table <path>`: Path to `master_trial_level_features.csv`.
* `--simulate-demo`: Run the statistical suite on a simulated $N=64$ dual-channel dataset.

*Example:*
```bash
python stats_analysis.py --simulate-demo
```

### Step 6: Visualization & Raw Inspector (`plot.py`)

#### A. Inspect Raw Shimmer SD Files
You can directly inspect any raw Shimmer recording (`.csv`) on the lab PC to view raw waveforms, 8 Hz envelopes, and accelerometer traces:
```bash
python plot.py --raw-shimmer ./data/raw_shimmer/sub-E41CB9_Shimmer_9BF9_Calibrated_SD.csv
```

#### B. Generate Publication Figures for the Report
Generates raincloud interaction plots, agonist-stabilizer co-activation plots (`FDS` vs. `ECR`), grand average time-course envelopes, trial stability checks, and filter cascade showcases in `data/stats_results/figures/`.
```bash
python plot.py
```

**Available Argparse Arguments for `plot.py`:**
* `--raw-shimmer <path>`: Path to raw Shimmer CSV file for signal inspection.
* `--fs <float>`: Sampling rate in Hz when inspecting raw Shimmer file (default: `512.0`).
* `--no-show`: Save raw Shimmer inspection plots directly to disk without opening an interactive GUI window.
* `--subj-table <path>`: Path to subject-level ANOVA table.
* `--trial-table <path>`: Path to trial-level feature table.
* `--out-dir <path>`: Custom output directory for generated figures.
* `--simulate-demo`: Generate all publication figures using simulated data.

---

## Quick Testing (Demo Mode)
If you want to test the entire pipeline on a newly installed lab computer without hardware recordings, run:

```bash
python preprocess_emg.py --simulate-demo
python postprocess_emg.py
python stats_analysis.py
python plot.py
```