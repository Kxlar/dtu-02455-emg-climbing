# ./Experiment in Cognitive Science 02455/config.py
from pathlib import Path

# --- DIRECTORY PATHS ---
BASE_DIR = Path(__file__).resolve().parent
STIM_DIR = BASE_DIR / "stimuli"
CLIMBING_DIR = STIM_DIR / "climbing"
EVERYDAY_DIR = STIM_DIR / "everyday"

DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
RAW_SHIMMER_DIR = DATA_DIR / "raw_shimmer"
PROCESSED_DIR = DATA_DIR / "processed"

EPOCHS_DIR = DATA_DIR / "epochs"
PROCESSED_DIR = DATA_DIR / "processed"
PLOTS_DIR = PROCESSED_DIR / "qc_plots"

STATS_DIR = DATA_DIR / "stats_results"
STATS_PLOTS_DIR = STATS_DIR / "figures"

for d in [CLIMBING_DIR, EVERYDAY_DIR, LOG_DIR, RAW_SHIMMER_DIR, PROCESSED_DIR, EPOCHS_DIR, PLOTS_DIR, STATS_DIR, STATS_PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT TIMING PARAMETERS (in seconds) ---
BASELINE_DURATION = 10.0       # 10s resting baseline
ISI_MIN = 4.0                  # Minimum jittered Inter-Stimulus Interval
ISI_MAX = 7.0                  # Maximum jittered Inter-Stimulus Interval
N_CLIPS_PER_CONDITION = 4     # 15 climbing + 15 everyday = 30 trials total

# --- DISPLAY PARAMETERS ---
BG_COLOR = [0, 0, 0]           # Neutral grey in PsychoPy (-1 to 1 scale)
TEXT_COLOR = [1, 1, 1]         # White text/cross
VIDEO_SIZE = (1.2, 0.675)      # 16:9 aspect ratio in 'height' units
WINDOW_SIZE = [1920, 1200]     # Matches your actual 1920x1200 display
FULLSCREEN = True
SCREEN_NUM = 0

# --- LIVE TRIGGER CONFIGURATION ---
# Trigger modes supported: "none" (offline only), "serial" (COM/USB/BT), "lsl" (LabStreamingLayer), "parallel" (TTL)
DEFAULT_SERIAL_PORT = "COM3"   # Change to e.g. "/dev/ttyUSB0" or "/dev/rfcomm0" on Linux/macOS
SERIAL_BAUDRATE = 115200
PARALLEL_PORT_ADDRESS = 0x0378

# Standardized Trigger Codes (sent on exact frame flip)
TRIGGER_CODES = {
    "sync_onset": 1,
    "sync_offset": 2,
    "baseline_onset": 10,
    "baseline_offset": 11,
    "climbing_onset": 21,
    "everyday_onset": 22,
    "video_offset_isi_onset": 30,
    "isi_offset": 31,
    "experiment_end": 99
}

# --- SIGNAL PROCESSING PARAMETERS ---
# Filter Cascade Parameters (Clancy et al., 2002 / Protocol Section 5)
BANDPASS_LOW = 20.0            # Hz (High-pass cutoff for motion artifacts)
BANDPASS_HIGH = 500.0          # Hz (Low-pass cutoff for thermal noise)
FILTER_ORDER = 4               # 4th-order Butterworth (8th-order zero-phase via filtfilt)
NOTCH_FREQ = 50.0              # Hz (European power line interference)
NOTCH_Q = 30.0                 # Quality factor for narrow 50Hz notch (+ harmonics)
ENVELOPE_LOWPASS = 8.0         # Hz (Smooths rectified signal into linear envelope)

# --- APPEND TO config.py ---
EPOCHS_DIR = DATA_DIR / "epochs"
EPOCHS_DIR.mkdir(parents=True, exist_ok=True)

# Default Shimmer EMG settings (auto-detected from timestamps, used as fallback)
DEFAULT_SAMPLING_RATE = 1024.0  # Hz (Standard Shimmer3 EMG sampling rate: 512, 1024, or 2048 Hz)

# Common column substrings exported by Shimmer Consensys software
SHIMMER_TIME_COLS = [
    "Timestamp_Unix_CAL",
    "Timestamp_Sync_Unix_CAL",
    "System_Timestamp_Plot_Zeroed_CAL",
    "Timestamp_Unix_ms",
    "Timestamp"
]

SHIMMER_EMG_COLS = [
    "EMG_Ch1_CAL",
    "EXG1_CH1",
    "EXG_ADS1292R_1_CH1_24Bit_CAL",
    "EMG_Ch1",
    "CH1"
]

# --- STATISTICAL ANALYSIS PARAMETERS ---
ALPHA_LEVEL = 0.05