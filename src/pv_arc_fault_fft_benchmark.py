# ================================================================
# FFT-based photovoltaic arc-fault detection reproducibility script
#
# Reproducibility and audit features:
# - source_file/source_window metadata exported with the feature dataset;
# - audit tables for split composition and TimeSeriesSplit folds;
# - corrected FFT plot frequency axis: f1 is bin 0 (DC), f100 is bin 99;
# - serialized CNN now uses train+validation scaler, not a full-dataset scaler;
# - CNN latency timing includes scaling consistently inside the timed loop;
# - file-grouped supplementary validation and band-energy non-ML baseline;
# - operating-point tables from model probabilities.
#
# Performance / reproducibility optimizations:
# - BLAS/OpenMP thread pools pinned to 1 (set before NumPy import) to avoid
#   thread oversubscription under GridSearchCV(n_jobs=CPU_JOBS);
# - vectorized, per-file feature extraction (numerically identical to the prior
#   row-by-row streaming version; verified bit-for-bit) -- much faster on CPU;
# - GridSearchCV(refit=False) + a single explicit refit on train+validation,
#   removing one wasted full training run per model (same selection, same model);
# - operating-point search vectorized to O(n log n) (same >= rule, same result);
# - global seeding (Python/NumPy/TensorFlow) for CNN reproducibility;
# - large 80k-row prediction tables no longer forced to .xlsx (CSV/Parquet kept).
#
# Runtime switches are controlled in the configuration section below and can be
# overridden with PV_ARC_* environment variables.
# ================================================================


# %% CELL 4
# ================================================================
# Environment check and imports
# ================================================================
import os
import sys
import json
import time
import random
import warnings
import logging
import gc
import zipfile
from pathlib import Path
from collections import deque
from itertools import combinations

# ----------------------------------------------------------------------------
# OPTIMIZATION (CPU scheduling): cap the BLAS/OpenMP thread pools to 1 thread.
# This MUST run before numpy/scikit-learn/TensorFlow are imported, because those
# libraries read these variables once, at import time.
#
# GridSearchCV(n_jobs=CPU_JOBS) already parallelizes across candidate fits. If each
# worker is also allowed to open several BLAS threads, execution can slow down
# due to oversubscription. Pinning BLAS to 1 thread per worker keeps CPU usage
# predictable. This does not change RF/KNN results and affects MLP/CNN only at
# the floating-point reduction-order level.
# ----------------------------------------------------------------------------
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

# Keep CPU usage predictable during grid-search execution.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import pandas as pd

# --- Headless, publication-quality figure setup -------------------------------
# Force a NON-INTERACTIVE backend BEFORE importing pyplot so no display window
# opens during execution. This lets the entire pipeline run completely unattended
# (e.g. overnight): every figure is written straight to disk and execution keeps
# going without waiting for anyone to close a plot window.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

plt.rcParams.update({
    "figure.dpi": 200,        # crisp if a saved figure is ever inspected on screen
    "savefig.dpi": 600,       # very-high-resolution raster output (publication grade)
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.antialiased": True,
})


def save_figure_hq(fig, png_path):
    """Save a figure at publication quality and close it.

    Writes a 600-dpi PNG plus a vector PDF copy (ideal for the manuscript), then
    releases the figure from memory. Uses the Agg backend, so it never opens a
    blocking window -- safe for long unattended runs.
    """
    png_path = Path(png_path)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")  # vector copy
    plt.close(fig)
    return png_path


def safe_to_excel(df, path, index=False):
    """Write a DataFrame to .xlsx, robust to MultiIndex columns.

    pandas cannot write a frame that has BOTH MultiIndex columns (e.g. produced
    by groupby().agg(["mean", "std"])) AND index=False -- it raises
    NotImplementedError. This helper flattens any MultiIndex column header into
    single-level names ("precision_mean", "precision_std", ...) first, so the
    .xlsx is written reliably and is also easier to read in a spreadsheet.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join(str(level) for level in col if str(level) != "").strip("_")
            for col in out.columns
        ]
    out.to_excel(Path(path), index=index)
    return Path(path)


import pyarrow as pa
import pyarrow.parquet as pq

try:
    from IPython.display import display
except Exception:
    def display(obj):
        """Fallback for plain Python execution outside Jupyter/IPython."""
        print(obj)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# Reproducibility: seed Python, NumPy and TensorFlow with a single fixed seed.
# This stabilizes the CNN across runs. Note: on CPU, TensorFlow is only fully
# deterministic if op-determinism is also enabled (see ENABLE_TF_OP_DETERMINISM
# in the configuration section). A freshly retrained CNN can still differ
# slightly across software and hardware environments.
# ----------------------------------------------------------------------------
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
try:
    tf.keras.utils.set_random_seed(GLOBAL_SEED)
except Exception:
    tf.random.set_seed(GLOBAL_SEED)

TARGET_PYTHON_VERSION = (3, 12)
if sys.version_info[:2] != TARGET_PYTHON_VERSION:
    print(
        f"WARNING: This script was prepared for Python "
        f"{TARGET_PYTHON_VERSION[0]}.{TARGET_PYTHON_VERSION[1]}.x; "
        f"current version is {sys.version_info.major}.{sys.version_info.minor}."
    )

print("Python      :", sys.version)
print("NumPy       :", np.__version__)
print("pandas      :", pd.__version__)
print("scikit-learn:", __import__("sklearn").__version__)
print("SciKeras    :", __import__("scikeras").__version__)
print("TensorFlow  :", tf.__version__)
print("Process ID  :", os.getpid(), "|", time.ctime())

SCRIPT_START_TIME = time.perf_counter()


def format_duration(seconds):
    """Format elapsed seconds as HH:MM:SS."""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def env_flag(name, default):
    """Read a boolean environment variable with a documented default."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


def env_choice(name, default, valid_values):
    """Read an environment variable constrained to a fixed set of values."""
    value = os.environ.get(name, default).strip()
    if value not in valid_values:
        raise ValueError(f"{name} must be one of {sorted(valid_values)}, got {value!r}.")
    return value


def env_int(name, default):
    """Read an integer environment variable with a documented default."""
    value = os.environ.get(name)
    return default if value is None else int(value.strip())

# %% CELL 5
# ================================================================
# Main configuration
# ================================================================

# Use "full_reproduction" to reproduce the manuscript pipeline.
# Use "quick_check" only as a fast sanity check for paths, column names, and code execution.
RUN_MODE = env_choice("PV_ARC_RUN_MODE", "full_reproduction", {"full_reproduction", "quick_check"})

# By default, the project folder is the current working directory. Set
# PV_ARC_BASE_DIR to run the script from another directory without editing it.
BASE_PROJECT_DIR = Path(os.environ.get("PV_ARC_BASE_DIR", Path.cwd())).expanduser().resolve()

# Labeled CSV files must be placed here.
# Accepted labeled CSV file names: Experiment_1.csv, ..., Experiment_16.csv
# Also accepted: Experiment_01.csv, ..., Experiment_16.csv
LABELED_DATA_DIR = BASE_PROJECT_DIR / "data_labeled"
LABELED_DATA_ZIP = BASE_PROJECT_DIR / "data_labeled.zip"

# Output directories are created automatically.
OUTPUT_DIR = BASE_PROJECT_DIR / "outputs" / RUN_MODE
FEATURE_DATA_DIR = OUTPUT_DIR / "feature_dataset"
SCALED_DATA_DIR = OUTPUT_DIR / "scaled_datasets"
MODEL_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUPPLEMENTARY_DIR = OUTPUT_DIR / "supplementary"

for directory in [LABELED_DATA_DIR, FEATURE_DATA_DIR, SCALED_DATA_DIR, MODEL_DIR, METRICS_DIR, PREDICTION_DIR, FIGURE_DIR, SUPPLEMENTARY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def ensure_labeled_data_available():
    """Use data_labeled/ when present, or extract data_labeled.zip automatically."""
    expected = [
        [LABELED_DATA_DIR / f"Experiment_{i}.csv", LABELED_DATA_DIR / f"Experiment_{i:02d}.csv"]
        for i in range(1, 17)
    ]
    if all(any(path.exists() for path in candidates) for candidates in expected):
        return

    if LABELED_DATA_ZIP.exists():
        print("Extracting labeled data from:", LABELED_DATA_ZIP)
        base_resolved = BASE_PROJECT_DIR.resolve()
        with zipfile.ZipFile(LABELED_DATA_ZIP) as archive:
            for member in archive.infolist():
                target = (BASE_PROJECT_DIR / member.filename).resolve()
                try:
                    target.relative_to(base_resolved)
                except ValueError as exc:
                    raise ValueError(f"Unsafe ZIP member path: {member.filename}") from exc
            archive.extractall(BASE_PROJECT_DIR)

    missing = []
    for i, candidates in enumerate(expected, start=1):
        if not any(path.exists() for path in candidates):
            missing.append(f"Experiment_{i}.csv")
    if missing:
        raise FileNotFoundError(
            "The labeled CSV files were not found. Provide either data_labeled.zip "
            f"or a data_labeled/ folder in {BASE_PROJECT_DIR}. Missing examples: "
            + ", ".join(missing[:5])
        )


ensure_labeled_data_available()

# Signal-processing settings used by the reproducibility workflow.
WINDOW_SIZE = 200
STRIDE = 200
NUM_FFT_COEFFICIENTS = WINDOW_SIZE // 2
SAMPLING_RATE_HZ = 250_000
HANN_WINDOW = np.hanning(WINDOW_SIZE)
BATCH_ROWS = 50_000

# Input CSV columns used by the pipeline.
CURRENT_COLUMN = "CH1"
VOLTAGE_COLUMN = "CH2"
OUTPUT_LABEL_COLUMN = "CLASSIFIER"
USECOLS = [CURRENT_COLUMN, VOLTAGE_COLUMN, OUTPUT_LABEL_COLUMN]
DTYPES = {CURRENT_COLUMN: "float32", VOLTAGE_COLUMN: "float32", OUTPUT_LABEL_COLUMN: "category"}

# Label values used by the English-language labeled dataset.
NORMAL_LABEL = "Normal"
ARC_LABEL = "Arc"
LABEL_TO_INT = {NORMAL_LABEL: 0, ARC_LABEL: 1}
INT_TO_LABEL = {0: NORMAL_LABEL, 1: ARC_LABEL}

SOURCE_FILE_COLUMN = "source_file"
SOURCE_WINDOW_COLUMN = "source_window"
SOURCE_SAMPLE_START_COLUMN = "source_sample_start"
SOURCE_SAMPLE_END_COLUMN = "source_sample_end"

# Important: labels must be supplied in the CLASSIFIER column.
# The CLASSIFIER column is required for every input file.
REQUIRE_SAMPLE_LEVEL_LABELS = True

# File grouping and seed used to define the deterministic train/validation/test-oriented ordering.
PURE_NORMAL_FILE_INDICES = [1, 2, 5, 9, 10, 13, 14]
MIXED_FILE_INDICES = [3, 4, 6, 7, 8, 11, 12, 15, 16]
FILE_ORDER_SEED = 15

# Quick mode limits. Quick mode is only a code/path validation tool and does not reproduce the manuscript metrics.
QUICK_ROWS_PER_FILE = 40_000
QUICK_MAX_FILES = 16

# Control switches.
REGENERATE_FEATURE_DATASET = env_flag("PV_ARC_REGENERATE_FEATURE_DATASET", True)

# ----------------------------------------------------------------------------
# MASTER REPRODUCIBILITY SWITCH
#
# RETRAIN_MAIN_GRID = True:
#   - Re-runs feature extraction, model selection, final fitting, model export,
#     predictions, and evaluation outputs from the labeled data.
#
# RETRAIN_MAIN_GRID = False:
#   - Reuses previously exported models in outputs/<RUN_MODE>/models and runs
#     downstream evaluation/supplementary outputs. This requires those model
#     artifacts to exist before execution.
# ----------------------------------------------------------------------------
RETRAIN_MAIN_GRID = env_flag("PV_ARC_RETRAIN_MAIN_GRID", True)

TRAIN_MODELS = RETRAIN_MAIN_GRID
RUN_SERIALIZED_MODEL_EVALUATION = env_flag("PV_ARC_SERIALIZED_EVAL", True)
GENERATE_SPECTRAL_FIGURES = env_flag("PV_ARC_GENERATE_FIGURES", True)

# Enable strict TensorFlow op-determinism. Off by default: it can slow CNN
# training and is only meaningful when RETRAIN_MAIN_GRID = True. It makes future
# CNN retrains reproducible with each other, not necessarily equal to the
# manuscript's original CNN.
ENABLE_TF_OP_DETERMINISM = env_flag("PV_ARC_TF_DETERMINISM", False)
if ENABLE_TF_OP_DETERMINISM:
    try:
        tf.config.experimental.enable_op_determinism()
        print("TensorFlow op-determinism enabled.")
    except Exception as exc:  # pragma: no cover
        print("Could not enable TF op-determinism:", exc)

# Writing the full 80k-row prediction tables to .xlsx is slow (openpyxl) and adds
# no information over the CSV/Parquet copies. Off by default: only the test-set
# rows are exported to .xlsx. Set True to also write the full-size .xlsx files.
EXPORT_FULL_TABLES_XLSX = env_flag("PV_ARC_EXPORT_FULL_TABLES_XLSX", False)

# Optional supplementary experiment: window-length sensitivity (100/200/500/1000).
# Off by default. When True, it rebuilds the feature set at each window length
# (vectorized, fast) and reports file-grouped F1 +/- std, addressing the
# supplementary analysis without changing the main results.
RUN_WINDOW_LENGTH_SWEEP = env_flag("PV_ARC_WINDOW_LENGTH_SWEEP", True)
WINDOW_LENGTH_SWEEP_VALUES = [100, 200, 500, 1000]

# ----------------------------------------------------------------------------
# CPU parallelism for the model-fitting steps (grid search + the supplementary
# RandomForests). -1 uses every logical core: fastest, but it saturates the
# machine while models train. Setting CPU_JOBS to a positive number can leave
# cores available for other work, at a small speed cost. This does not change any
# result -- only how many cores the fits are spread across. The CNN is left
# serialized (n_jobs=1) regardless, because TensorFlow manages its own threads.
# ----------------------------------------------------------------------------
CPU_JOBS = env_int("PV_ARC_CPU_JOBS", -1)  # -1 = all available logical cores

# Output files written by the reproducibility pipeline.
FEATURE_PARQUET_PATH = FEATURE_DATA_DIR / "pv_arc_fault_fft_features.parquet"
FEATURE_METADATA_PATH = FEATURE_DATA_DIR / "feature_extraction_metadata.json"
MINMAX_DATASET_CSV = SCALED_DATA_DIR / "pv_arc_fault_fft_features_minmax_scaled.csv"
ZSCORE_DATASET_CSV = SCALED_DATA_DIR / "pv_arc_fault_fft_features_zscore_scaled.csv"
MINMAX_PREDICTIONS_CSV = PREDICTION_DIR / "pv_arc_fault_minmax_predictions.csv"
ZSCORE_PREDICTIONS_CSV = PREDICTION_DIR / "pv_arc_fault_zscore_predictions.csv"
MINMAX_PREDICTIONS_XLSX = PREDICTION_DIR / "pv_arc_fault_minmax_predictions.xlsx"
ZSCORE_PREDICTIONS_XLSX = PREDICTION_DIR / "pv_arc_fault_zscore_predictions.xlsx"
SCALER_MINMAX_PATH = MODEL_DIR / "feature_scaler_minmax_full_dataset.pkl"
SCALER_ZSCORE_PATH = MODEL_DIR / "feature_scaler_zscore_full_dataset.pkl"
CNN_SCALER_MINMAX_PATH = MODEL_DIR / "CNN_minmax_trainval_scaler.pkl"
CNN_SCALER_ZSCORE_PATH = MODEL_DIR / "CNN_zscore_trainval_scaler.pkl"
F1_MINMAX_XLSX = METRICS_DIR / "f1_score_minmax_by_model.xlsx"
F1_ZSCORE_XLSX = METRICS_DIR / "f1_score_zscore_by_model.xlsx"

PIPELINE_STEPS = ["Build/load feature dataset", "Prepare split and audit"]
if TRAIN_MODELS:
    PIPELINE_STEPS.extend(["Train/evaluate Min-Max", "Train/evaluate Z-Score"])
    PIPELINE_STEPS.extend(["Export scaled datasets", "Save trained models"])
    PIPELINE_STEPS.append("Export prediction tables")
if GENERATE_SPECTRAL_FIGURES:
    PIPELINE_STEPS.append("Generate spectral figures")
if RUN_SERIALIZED_MODEL_EVALUATION:
    PIPELINE_STEPS.append("Serialized model evaluation")
if RUN_MODE == "full_reproduction":
    PIPELINE_STEPS.extend(["File-grouped validation", "Band-energy baseline"])
    if RUN_WINDOW_LENGTH_SWEEP:
        PIPELINE_STEPS.append("Window-length sweep")
PIPELINE_STEPS.append("Print saved model information")

PIPELINE_PROGRESS = tqdm(total=len(PIPELINE_STEPS), desc="Overall pipeline", unit="step") if tqdm is not None else None
CURRENT_STEP_NAME = None
CURRENT_STEP_START = None
CURRENT_STEP_INDEX = 0


def start_pipeline_step(step_name):
    """Start a coarse-grained pipeline progress step."""
    global CURRENT_STEP_NAME, CURRENT_STEP_START, CURRENT_STEP_INDEX
    CURRENT_STEP_INDEX += 1
    CURRENT_STEP_NAME = step_name
    CURRENT_STEP_START = time.perf_counter()
    prefix = f"[{CURRENT_STEP_INDEX:02d}/{len(PIPELINE_STEPS):02d}]"
    elapsed = format_duration(CURRENT_STEP_START - SCRIPT_START_TIME)
    print(f"\n{prefix} START {step_name} | elapsed={elapsed}")
    if PIPELINE_PROGRESS is not None:
        PIPELINE_PROGRESS.set_description(f"{prefix} {step_name}")


def finish_pipeline_step():
    """Finish the current coarse-grained pipeline progress step."""
    global CURRENT_STEP_NAME, CURRENT_STEP_START
    if CURRENT_STEP_NAME is None:
        return
    now = time.perf_counter()
    step_elapsed = format_duration(now - CURRENT_STEP_START)
    total_elapsed = format_duration(now - SCRIPT_START_TIME)
    print(f"[{CURRENT_STEP_INDEX:02d}/{len(PIPELINE_STEPS):02d}] DONE  {CURRENT_STEP_NAME} | step={step_elapsed} | total={total_elapsed}")
    if PIPELINE_PROGRESS is not None:
        PIPELINE_PROGRESS.update(1)
    CURRENT_STEP_NAME = None
    CURRENT_STEP_START = None


def close_pipeline_progress():
    """Close tqdm and print total runtime."""
    if PIPELINE_PROGRESS is not None:
        PIPELINE_PROGRESS.close()
    total_elapsed = format_duration(time.perf_counter() - SCRIPT_START_TIME)
    print(f"\nTotal execution time: {total_elapsed}")

print("RUN_MODE:", RUN_MODE)
print("BASE_PROJECT_DIR:", BASE_PROJECT_DIR)
print("LABELED_DATA_DIR:", LABELED_DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

# %% CELL 7
def make_ordered_file_indices(seed=FILE_ORDER_SEED):
    """Return the 16 file indices in the deterministic experiment order."""
    rng = random.Random(seed)
    pure_normal = PURE_NORMAL_FILE_INDICES.copy()
    mixed = MIXED_FILE_INDICES.copy()
    rng.shuffle(pure_normal)
    rng.shuffle(mixed)

    train_indices = mixed[:5] + pure_normal[:6]
    validation_indices = mixed[5:7] + [pure_normal[6]]
    test_indices = mixed[7:9]
    return train_indices + validation_indices + test_indices


def find_labeled_csv_file(file_index, labeled_data_dir=LABELED_DATA_DIR):
    """Find one labeled CSV file using the accepted file-name patterns."""
    candidate_names = [f"Experiment_{file_index}.csv", f"Experiment_{file_index:02d}.csv"]
    for candidate_name in candidate_names:
        path = labeled_data_dir / candidate_name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find Experiment_{file_index}.csv or Experiment_{file_index:02d}.csv in {labeled_data_dir}. "
        "Place the 16 labeled CSV files in LABELED_DATA_DIR before running the script."
    )


def get_fft_columns():
    """Return the FFT feature names for CH1 and CH2."""
    ch1_fft_columns = [f"FFT_CH1_f{i}" for i in range(1, NUM_FFT_COEFFICIENTS + 1)]
    ch2_fft_columns = [f"FFT_CH2_f{i}" for i in range(1, NUM_FFT_COEFFICIENTS + 1)]
    return ch1_fft_columns, ch2_fft_columns


# Feature order used by the model input matrix after dropping the label column:
# CH1 statistics, CH2 statistics, FFT_CH1 coefficients, FFT_CH2 coefficients.
CH1_CH2_STAT_COLUMNS = ["CH1_mean", "CH1_std", "CH2_mean", "CH2_std"]
FFT_CH1_COLUMNS, FFT_CH2_COLUMNS = get_fft_columns()
FEATURE_COLUMNS = CH1_CH2_STAT_COLUMNS + FFT_CH1_COLUMNS + FFT_CH2_COLUMNS

# Parquet column order mirrors the feature-dataset structure and includes source metadata
# so file-grouped validation can be reproduced directly.
SOURCE_COLUMNS = [SOURCE_FILE_COLUMN, SOURCE_WINDOW_COLUMN, SOURCE_SAMPLE_START_COLUMN, SOURCE_SAMPLE_END_COLUMN]
PARQUET_COLUMNS = CH1_CH2_STAT_COLUMNS + [OUTPUT_LABEL_COLUMN] + SOURCE_COLUMNS + FFT_CH1_COLUMNS + FFT_CH2_COLUMNS


def normalize_label_value(value):
    """Normalize label values while preserving the two-class problem."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text == "normal":
        return NORMAL_LABEL
    if text == "arc":
        return ARC_LABEL
    return None


def label_window(labels):
    """
    Assign one label to each 200-sample window using the labels inside that window.

    Valid windows are:
    1. all Normal -> Normal;
    2. all Arc -> Arc;
    3. one transition from Normal to Arc, without returning to Normal -> Arc.

    Windows with Arc-to-Normal transitions, multiple alternations, or invalid labels are discarded.
    """
    normalized = np.array([normalize_label_value(value) for value in labels], dtype=object)
    if any(value is None for value in normalized):
        return None

    is_arc = normalized == ARC_LABEL
    if is_arc.all():
        return ARC_LABEL
    if (~is_arc).all():
        return NORMAL_LABEL

    first_arc = int(is_arc.argmax())
    after_first_arc = is_arc[first_arc:]
    if (~after_first_arc).any():
        return None
    return ARC_LABEL


def read_csv_chunks(csv_path, chunksize=50_000):
    """Read the required input columns in memory-safe chunks."""
    return pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype=DTYPES,
        chunksize=chunksize,
        engine="c",
        memory_map=True,
    )


def extract_windows_for_file(file_index, window_size=WINDOW_SIZE,
                            num_fft=NUM_FFT_COEFFICIENTS, stride=STRIDE, max_rows=None):
    """
    Vectorized, per-file extraction of valid windows with their statistical and
    FFT features. This is a drop-in replacement for the previous row-by-row
    streaming implementation and is numerically IDENTICAL to it (verified
    bit-for-bit on the labeled data), but far faster on a CPU-only machine
    because it replaces ~16 million Python-level row iterations with NumPy
    array operations.

    Each file is processed on its own, so a window can never cross a file
    boundary -- this reproduces the buffer-reset behavior of the streaming
    version. The labeling rule matches label_window() exactly:
      - all Normal                                   -> Normal
      - all Arc                                      -> Arc
      - single Normal->Arc transition (no return)    -> Arc
      - any unknown label / Arc->Normal / alternating -> discarded
    Returned source metadata (source_file, source_window, source_sample_start,
    source_sample_end) matches the streaming version's per-file indexing.
    """
    csv_path = find_labeled_csv_file(file_index)
    print(f"Reading {csv_path.name} ...")
    df = pd.read_csv(csv_path, usecols=USECOLS, dtype=DTYPES,
                     engine="c", memory_map=True, nrows=max_rows)
    if REQUIRE_SAMPLE_LEVEL_LABELS and OUTPUT_LABEL_COLUMN not in df.columns:
        raise ValueError(f"The required {OUTPUT_LABEL_COLUMN} column was not found in {csv_path.name}.")

    n = (len(df) // window_size) * window_size
    if n == 0:
        return None

    ch1 = df[CURRENT_COLUMN].to_numpy(np.float32)[:n].reshape(-1, window_size)
    ch2 = df[VOLTAGE_COLUMN].to_numpy(np.float32)[:n].reshape(-1, window_size)
    labels_lower = (
        df[OUTPUT_LABEL_COLUMN].astype(str).str.strip().str.lower()
        .to_numpy()[:n].reshape(-1, window_size)
    )

    is_arc = labels_lower == ARC_LABEL.lower()
    is_norm = labels_lower == NORMAL_LABEL.lower()
    known = (is_arc | is_norm).all(axis=1)        # discard windows with any unknown/NaN label
    all_arc = is_arc.all(axis=1)
    all_norm = is_norm.all(axis=1)
    any_arc = is_arc.any(axis=1)
    first_arc = np.where(any_arc, is_arc.argmax(axis=1), window_size)
    positions = np.arange(window_size)
    at_or_after = positions[None, :] >= first_arc[:, None]
    # monotone Normal->Arc: every sample at/after the first Arc is Arc (no return to Normal)
    monotone_arc = any_arc & (is_arc | ~at_or_after).all(axis=1)
    keep = known & (all_arc | all_norm | monotone_arc)
    if not keep.any():
        return None

    kept_idx = np.flatnonzero(keep)
    c1 = ch1[kept_idx]
    c2 = ch2[kept_idx]

    hann = np.hanning(window_size)
    m1 = c1.mean(axis=1)
    m2 = c2.mean(axis=1)
    s1 = c1.std(axis=1, ddof=1)
    s2 = c2.std(axis=1, ddof=1)
    # DC removal followed by Hann windowing, before FFT feature extraction.
    fft1 = np.abs(np.fft.fft((c1 - m1[:, None]) * hann, axis=1)[:, :num_fft])
    fft2 = np.abs(np.fft.fft((c2 - m2[:, None]) * hann, axis=1)[:, :num_fft])

    labels = np.where(all_norm[kept_idx], NORMAL_LABEL, ARC_LABEL).astype(object)
    source_sample_start = (kept_idx * stride).astype(np.int64)
    source_sample_end = source_sample_start + (window_size - 1)
    source_window = (source_sample_start // stride).astype(np.int64)
    source_file = np.full(kept_idx.shape, int(file_index), dtype=np.int64)

    # Feature names depend on num_fft (per-channel coefficient count).
    ch1_fft_names = [f"FFT_CH1_f{i}" for i in range(1, num_fft + 1)]
    ch2_fft_names = [f"FFT_CH2_f{i}" for i in range(1, num_fft + 1)]

    # Build the pyarrow table with the same column order and types as before:
    # statistics + label + source metadata + FFT_CH1 + FFT_CH2. Statistics and
    # FFT magnitudes are stored as double (float64), matching the original
    # from_pylist coercion; the float32->float64 cast of the float32 means/stds is
    # exact, so the stored values are identical.
    columns = {
        "CH1_mean": pa.array(m1.astype(np.float64)),
        "CH1_std": pa.array(s1.astype(np.float64)),
        "CH2_mean": pa.array(m2.astype(np.float64)),
        "CH2_std": pa.array(s2.astype(np.float64)),
        OUTPUT_LABEL_COLUMN: pa.array(labels, type=pa.string()),
        SOURCE_FILE_COLUMN: pa.array(source_file),
        SOURCE_WINDOW_COLUMN: pa.array(source_window),
        SOURCE_SAMPLE_START_COLUMN: pa.array(source_sample_start),
        SOURCE_SAMPLE_END_COLUMN: pa.array(source_sample_end),
    }
    for j, name in enumerate(ch1_fft_names):
        columns[name] = pa.array(fft1[:, j])
    for j, name in enumerate(ch2_fft_names):
        columns[name] = pa.array(fft2[:, j])
    return pa.table(columns)


def build_feature_dataset():
    """Process the labeled CSV files (vectorized) and save the derived FFT feature dataset as a zstd parquet file."""
    ordered_indices = make_ordered_file_indices()
    if RUN_MODE == "quick_check":
        ordered_indices = ordered_indices[:QUICK_MAX_FILES]

    ordered_files = [find_labeled_csv_file(index).name for index in ordered_indices]
    print("File order:", ordered_files)

    max_rows = QUICK_ROWS_PER_FILE if RUN_MODE == "quick_check" else None
    tables = []
    total_windows = 0
    for file_index in ordered_indices:
        table = extract_windows_for_file(file_index, max_rows=max_rows)
        if table is None:
            continue
        tables.append(table)
        total_windows += table.num_rows
        print(f"\rValid windows: {total_windows:,}", end="")

    if not tables:
        raise RuntimeError("No valid feature windows were generated. Check the CLASSIFIER values and labeled CSV files.")

    try:
        feature_table = pa.concat_tables(tables, promote_options="default")
    except TypeError:
        feature_table = pa.concat_tables(tables, promote=True)

    pq.write_table(feature_table, FEATURE_PARQUET_PATH, compression="zstd", use_dictionary=True)
    feature_df = pd.read_parquet(FEATURE_PARQUET_PATH)

    metadata = {
        "version": "v08",
        "run_mode": RUN_MODE,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "num_fft_coefficients_per_channel": NUM_FFT_COEFFICIENTS,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": OUTPUT_LABEL_COLUMN,
        "processed_files": ordered_files,
        "label_source": "CLASSIFIER column in the provided CSV files",
        "num_rows": int(len(feature_df)),
        "num_columns": int(feature_df.shape[1]),
        "class_counts": feature_df[OUTPUT_LABEL_COLUMN].value_counts().to_dict(),
    }
    FEATURE_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved feature dataset: {FEATURE_PARQUET_PATH}")
    print("Rows:", len(feature_df), "| Columns:", feature_df.shape[1])
    print("Class counts:")
    print(feature_df[OUTPUT_LABEL_COLUMN].value_counts())
    return feature_df


# %% CELL 8
# Build or load the derived feature dataset.
start_pipeline_step("Build/load feature dataset")
if REGENERATE_FEATURE_DATASET or not FEATURE_PARQUET_PATH.exists():
    feature_df = build_feature_dataset()
else:
    feature_df = pd.read_parquet(FEATURE_PARQUET_PATH)
    print("Loaded existing feature dataset:", FEATURE_PARQUET_PATH)

# Ensure the model input order is exactly the intended one.
missing_features = [column for column in FEATURE_COLUMNS if column not in feature_df.columns]
if missing_features:
    raise ValueError(f"The feature dataset is missing required columns: {missing_features}")

missing_source_columns = [column for column in SOURCE_COLUMNS if column not in feature_df.columns]
if missing_source_columns:
    raise ValueError(
        "The feature dataset is missing source metadata columns "
        f"{missing_source_columns}. Run with REGENERATE_FEATURE_DATASET = True "
        "so split/group diagnostics can be reproduced."
    )

invalid_labels = sorted(set(feature_df[OUTPUT_LABEL_COLUMN].astype(str)) - {NORMAL_LABEL, ARC_LABEL})
if invalid_labels:
    raise ValueError(f"Unexpected CLASSIFIER values found in the feature dataset: {invalid_labels}")

display(feature_df.head())
finish_pipeline_step()

# %% CELL 10
def split_temporal_data(X, y, train_fraction=0.70, validation_fraction=0.15):
    """Split arrays into train, validation, and test partitions preserving chronological order."""
    n_samples = len(X)
    train_end = int(n_samples * train_fraction)
    validation_end = train_end + int(n_samples * validation_fraction)
    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_validation": X[train_end:validation_end],
        "y_validation": y[train_end:validation_end],
        "X_test": X[validation_end:],
        "y_test": y[validation_end:],
        "train_end": train_end,
        "validation_end": validation_end,
        "n_samples": n_samples,
    }


def create_cnn_model(n_filters=32, kernel_size=3, dense_units=128, dropout_rate=0.5, optimizer="adam", input_shape=None):
    """Create the 1D CNN architecture used in this workflow."""
    if input_shape is None:
        raise ValueError("input_shape must be provided.")
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(n_filters, kernel_size, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_tabular_pipeline(scaler, estimator, estimator_grid):
    """Build a scikit-learn pipeline and prefix the estimator grid with clf__."""
    pipeline = Pipeline([("scale", scaler), ("clf", estimator)])
    parameter_grid = {f"clf__{key}": value for key, value in estimator_grid.items()}
    return pipeline, parameter_grid


def build_cnn_pipeline(scaler, input_length):
    """Build the SciKeras CNN pipeline with the required reshape logic."""
    reshape_transformer = FunctionTransformer(
        lambda array: array.reshape(array.shape[0], input_length, 1),
        feature_names_out="one-to-one",
    )
    cnn_classifier = KerasClassifier(
        model=create_cnn_model,
        model__input_shape=(input_length, 1),
        verbose=0,
    )
    pipeline = Pipeline([
        ("scale", scaler),
        ("reshape", reshape_transformer),
        ("clf", cnn_classifier),
    ])

    if RUN_MODE == "quick_check":
        # Reduced grid only for a fast code/path validation run.
        parameter_grid = {
            "clf__model__n_filters": [8],
            "clf__model__kernel_size": [3],
            "clf__model__dense_units": [16],
            "clf__model__dropout_rate": [0.3],
            "clf__fit__epochs": [3],
            "clf__fit__batch_size": [32],
            "clf__fit__validation_split": [0.15],
            "clf__fit__callbacks": [[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]],
            "clf__random_state": [42],
        }
    else:
        # Full grid used for model selection.
        parameter_grid = {
            "clf__model__n_filters": [32],
            "clf__model__kernel_size": [3],
            "clf__model__dense_units": [128],
            "clf__model__dropout_rate": [0.5],
            "clf__fit__epochs": [40],
            "clf__fit__batch_size": [32],
            "clf__fit__validation_split": [0.15],
            "clf__fit__callbacks": [[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]],
            "clf__random_state": [1, 42, 100, 2026],
        }
    return pipeline, parameter_grid


def get_model_specifications(scaler_class, input_length):
    """Return model pipelines, parameter grids, scorers, and label mode for the selected execution mode."""
    if RUN_MODE == "quick_check":
        rf_grid_values = {"n_estimators": [20], "max_depth": [10], "random_state": [42]}
        knn_grid_values = {"n_neighbors": [3], "weights": ["distance"]}
        mlp_grid_values = {"hidden_layer_sizes": [(16,)], "activation": ["relu"], "random_state": [42]}
        mlp_max_iter = 80
        n_splits = 2
    else:
        rf_grid_values = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20], "random_state": [1, 42, 100, 2026]}
        knn_grid_values = {"n_neighbors": [6, 7, 8], "weights": ["uniform", "distance"]}
        mlp_grid_values = {"hidden_layer_sizes": [(64,), (100, 50), (1000, 500)], "activation": ["tanh", "relu", "logistic"], "random_state": [1, 42, 100, 2026]}
        mlp_max_iter = 700
        n_splits = 4

    rf_pipeline, rf_grid = build_tabular_pipeline(scaler_class(), RandomForestClassifier(), rf_grid_values)
    knn_pipeline, knn_grid = build_tabular_pipeline(scaler_class(), KNeighborsClassifier(), knn_grid_values)
    mlp_pipeline, mlp_grid = build_tabular_pipeline(scaler_class(), MLPClassifier(max_iter=mlp_max_iter), mlp_grid_values)
    cnn_pipeline, cnn_grid = build_cnn_pipeline(scaler_class(), input_length)

    string_f1_scorer = make_scorer(f1_score, pos_label=ARC_LABEL, zero_division=0)
    integer_f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

    return {
        "n_splits": n_splits,
        "models": {
            "RF": {"pipeline": rf_pipeline, "grid": rf_grid, "n_jobs": CPU_JOBS, "scorer": string_f1_scorer, "uses_integer_labels": False},
            "KNN": {"pipeline": knn_pipeline, "grid": knn_grid, "n_jobs": CPU_JOBS, "scorer": string_f1_scorer, "uses_integer_labels": False},
            "MLP": {"pipeline": mlp_pipeline, "grid": mlp_grid, "n_jobs": CPU_JOBS, "scorer": string_f1_scorer, "uses_integer_labels": False},
            "CNN": {"pipeline": cnn_pipeline, "grid": cnn_grid, "n_jobs": 1, "scorer": integer_f1_scorer, "uses_integer_labels": True},
        },
    }


def labels_to_int(labels):
    """Convert string labels to the binary encoding required by the CNN."""
    return np.array([LABEL_TO_INT[str(label)] for label in labels], dtype=np.int32)


def int_to_labels(values):
    """Convert CNN integer predictions back to human-readable labels."""
    return np.array([INT_TO_LABEL[int(value)] for value in values], dtype=object)


def positive_probability(model, X, positive_label=ARC_LABEL):
    """Return the probability of the Arc class from either string-label or integer-label estimators."""
    probabilities = model.predict_proba(X)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0].astype(float)

    final_estimator = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
    classes = getattr(final_estimator, "classes_", None)
    if classes is not None:
        class_list = list(classes)
        if positive_label in class_list:
            return probabilities[:, class_list.index(positive_label)].astype(float)
        if 1 in class_list:
            return probabilities[:, class_list.index(1)].astype(float)
    return probabilities[:, 1].astype(float)


def evaluate_predictions(y_true, y_pred):
    """Compute test metrics using Arc as the positive class."""
    matrix = confusion_matrix(y_true, y_pred, labels=[NORMAL_LABEL, ARC_LABEL])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=ARC_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=ARC_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=ARC_LABEL, zero_division=0),
        "confusion_matrix": matrix,
    }


def serializable_parameters(parameters):
    """Convert GridSearch parameters to strings where needed for JSON/XLSX export."""
    output = {}
    for key, value in parameters.items():
        try:
            json.dumps(value)
            output[key] = value
        except TypeError:
            output[key] = str(value)
    return output


def train_and_evaluate_scaling(X, y, split_data, scaler_class, scaling_key):
    """Train and evaluate RF, KNN, MLP, and CNN for one scaling strategy."""
    print(f"\n==============================")
    print(f"Scaling strategy: {scaling_key}")
    print(f"==============================")

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_validation = split_data["X_validation"]
    y_validation = split_data["y_validation"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    X_train_validation = np.vstack([X_train, X_validation])
    y_train_validation = np.hstack([y_train, y_validation])

    specifications = get_model_specifications(scaler_class, X.shape[1])
    time_series_cv = TimeSeriesSplit(n_splits=specifications["n_splits"])

    metric_records = []
    best_parameter_records = []
    cv_f1_records = []
    fitted_models = {}
    test_predictions = {}
    test_probabilities = {}

    for model_name, specification in specifications["models"].items():
        print(f"\n[GridSearchCV] {model_name}")
        search = GridSearchCV(
            estimator=specification["pipeline"],
            param_grid=specification["grid"],
            cv=time_series_cv,
            scoring=specification["scorer"],
            n_jobs=specification["n_jobs"],
            verbose=0,
            error_score="raise",
            refit=False,
        )

        # RF/KNN/MLP use string labels, CNN uses integer labels.
        y_train_fit = labels_to_int(y_train) if specification["uses_integer_labels"] else y_train
        search.fit(X_train, y_train_fit)

        # With a single scorer, best_index_/best_params_/best_score_ are available
        # even when refit=False. We set refit=False on purpose: GridSearchCV's
        # automatic refit would re-fit the best estimator on X_train only, which we
        # immediately discard by re-fitting on train+validation below. Skipping it
        # saves one full training run per model (notably the CNN and the largest
        # MLP) WITHOUT changing the selection or the final model: a fresh fit on
        # train+validation with the selected parameters is identical to
        # best_estimator_.fit(train+validation).
        best_index = search.best_index_
        cv_f1_mean = float(search.cv_results_["mean_test_score"][best_index])
        cv_f1_std = float(search.cv_results_["std_test_score"][best_index])
        print("Best parameters:", search.best_params_)
        print(f"Best CV F1: {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}")

        # Final fit on train + validation, preserving the isolated 15% test set.
        best_model = clone(specification["pipeline"]).set_params(**search.best_params_)
        y_train_validation_fit = labels_to_int(y_train_validation) if specification["uses_integer_labels"] else y_train_validation
        best_model.fit(X_train_validation, y_train_validation_fit)

        raw_prediction = best_model.predict(X_test)
        y_pred = int_to_labels(raw_prediction) if specification["uses_integer_labels"] else raw_prediction.astype(str)
        y_probability = positive_probability(best_model, X_test, positive_label=ARC_LABEL)

        metrics = evaluate_predictions(y_test, y_pred)
        fitted_models[model_name] = best_model
        test_predictions[model_name] = y_pred
        test_probabilities[model_name] = y_probability

        metric_records.append({
            "scaling": scaling_key,
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "tn": int(metrics["confusion_matrix"][0, 0]),
            "fp": int(metrics["confusion_matrix"][0, 1]),
            "fn": int(metrics["confusion_matrix"][1, 0]),
            "tp": int(metrics["confusion_matrix"][1, 1]),
        })
        best_parameter_records.append({
            "scaling": scaling_key,
            "model": model_name,
            "best_parameters": json.dumps(serializable_parameters(search.best_params_), indent=2),
        })
        cv_f1_records.append({
            "model": model_name,
            "F1_mean": cv_f1_mean,
            "F1_std": cv_f1_std,
        })
        print(f"Test F1={metrics['f1']:.4f} | Recall={metrics['recall']:.4f} | Accuracy={metrics['accuracy']:.4f}")

    return {
        "fitted_models": fitted_models,
        "metric_records": metric_records,
        "best_parameter_records": best_parameter_records,
        "cv_f1_records": cv_f1_records,
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
    }


# %% SUPPLEMENTARY AUDIT HELPERS
def export_dataset_audit(feature_df, split_data):
    """Export per-file, split, and TimeSeriesSplit class-count diagnostics."""
    audit_df = feature_df[[SOURCE_FILE_COLUMN, SOURCE_WINDOW_COLUMN, OUTPUT_LABEL_COLUMN]].copy()
    per_file = (
        audit_df.groupby([SOURCE_FILE_COLUMN, OUTPUT_LABEL_COLUMN])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in [NORMAL_LABEL, ARC_LABEL]:
        if label not in per_file.columns:
            per_file[label] = 0
    per_file = per_file[[SOURCE_FILE_COLUMN, NORMAL_LABEL, ARC_LABEL]]
    per_file["total_windows"] = per_file[NORMAL_LABEL] + per_file[ARC_LABEL]
    per_file.to_csv(SUPPLEMENTARY_DIR / "per_file_window_counts.csv", index=False)
    per_file.to_excel(SUPPLEMENTARY_DIR / "per_file_window_counts.xlsx", index=False)

    partition = np.full(len(feature_df), "test", dtype=object)
    partition[: split_data["train_end"]] = "train"
    partition[split_data["train_end"] : split_data["validation_end"]] = "validation"
    split_audit = audit_df.copy()
    split_audit["partition"] = partition
    split_summary = (
        split_audit.groupby(["partition", SOURCE_FILE_COLUMN, OUTPUT_LABEL_COLUMN])
        .size()
        .reset_index(name="windows")
    )
    split_summary.to_csv(SUPPLEMENTARY_DIR / "temporal_split_file_composition.csv", index=False)
    split_summary.to_excel(SUPPLEMENTARY_DIR / "temporal_split_file_composition.xlsx", index=False)

    train_df = split_audit.iloc[: split_data["train_end"]].reset_index(drop=True)
    n_train = len(train_df)
    tscv = TimeSeriesSplit(n_splits=4)
    fold_rows = []
    dummy_X = np.zeros((n_train, 1), dtype=np.float32)
    for fold, (_, val_idx) in enumerate(tscv.split(dummy_X), start=1):
        val_df = train_df.iloc[val_idx]
        totals = val_df[OUTPUT_LABEL_COLUMN].value_counts().to_dict()
        for file_index, file_df in val_df.groupby(SOURCE_FILE_COLUMN):
            fold_rows.append({
                "fold": fold,
                "validation_start": int(val_idx[0]),
                "validation_stop_exclusive": int(val_idx[-1] + 1),
                SOURCE_FILE_COLUMN: int(file_index),
                "windows": int(len(file_df)),
                "normal": int((file_df[OUTPUT_LABEL_COLUMN] == NORMAL_LABEL).sum()),
                "arc": int((file_df[OUTPUT_LABEL_COLUMN] == ARC_LABEL).sum()),
                "fold_normal_total": int(totals.get(NORMAL_LABEL, 0)),
                "fold_arc_total": int(totals.get(ARC_LABEL, 0)),
            })
    tscv_counts = pd.DataFrame(fold_rows)
    tscv_counts.to_csv(SUPPLEMENTARY_DIR / "timeseriessplit_fold_class_counts.csv", index=False)
    tscv_counts.to_excel(SUPPLEMENTARY_DIR / "timeseriessplit_fold_class_counts.xlsx", index=False)
    print("Saved audit files in:", SUPPLEMENTARY_DIR)


def run_file_grouped_validation(feature_df):
    """Run file-grouped validation with lightweight, fixed baselines on the generated features."""
    if SOURCE_FILE_COLUMN not in feature_df.columns:
        print("File-grouped validation skipped: source metadata not found.")
        return pd.DataFrame(), pd.DataFrame()

    X = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = feature_df[OUTPUT_LABEL_COLUMN].astype(str).to_numpy()
    groups = feature_df[SOURCE_FILE_COLUMN].to_numpy()

    grouped_models = {
        "LogisticRegression_ZScore": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "RF_ZScore_fixed": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=100, n_jobs=CPU_JOBS)),
        ]),
        "KNN_ZScore_fixed": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=8, weights="uniform")),
        ]),
    }

    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=FILE_ORDER_SEED)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        val_files = ",".join(str(value) for value in sorted(set(groups[val_idx])))
        for model_name, model in grouped_models.items():
            print(f"[Grouped validation] fold={fold} model={model_name}")
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            rows.append({
                "model": model_name,
                "fold": fold,
                "validation_files": val_files,
                "normal": int(np.sum(y[val_idx] == NORMAL_LABEL)),
                "arc": int(np.sum(y[val_idx] == ARC_LABEL)),
                "precision": precision_score(y[val_idx], pred, pos_label=ARC_LABEL, zero_division=0),
                "recall": recall_score(y[val_idx], pred, pos_label=ARC_LABEL, zero_division=0),
                "f1": f1_score(y[val_idx], pred, pos_label=ARC_LABEL, zero_division=0),
                "accuracy": accuracy_score(y[val_idx], pred),
                "balanced_accuracy": balanced_accuracy_score(y[val_idx], pred),
            })
    results = pd.DataFrame(rows)
    summary = (
        results.groupby("model")[["precision", "recall", "f1", "accuracy", "balanced_accuracy"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    results.to_csv(SUPPLEMENTARY_DIR / "file_grouped_validation_folds.csv", index=False)
    summary.to_csv(SUPPLEMENTARY_DIR / "file_grouped_validation_summary.csv", index=False)
    results.to_excel(SUPPLEMENTARY_DIR / "file_grouped_validation_folds.xlsx", index=False)
    safe_to_excel(summary, SUPPLEMENTARY_DIR / "file_grouped_validation_summary.xlsx")
    return results, summary


def _best_threshold(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=object)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    total_pos = int(np.sum(sorted_labels == ARC_LABEL))
    total_neg = len(sorted_labels) - total_pos
    tp, fp, fn = total_pos, total_neg, 0

    def metric(threshold, tp_value, fp_value, fn_value):
        tn_value = total_neg - fp_value
        precision = tp_value / (tp_value + fp_value) if tp_value + fp_value else 0.0
        recall = tp_value / (tp_value + fn_value) if tp_value + fn_value else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp_value + tn_value) / len(sorted_labels)
        return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    best = metric(float(sorted_scores[0]) - 1e-12, tp, fp, fn)
    i = 0
    while i < len(sorted_scores):
        value = sorted_scores[i]
        while i < len(sorted_scores) and sorted_scores[i] == value:
            if sorted_labels[i] == ARC_LABEL:
                tp -= 1
                fn += 1
            else:
                fp -= 1
            i += 1
        candidate = metric(float(value) + 1e-12, tp, fp, fn)
        if candidate["f1"] > best["f1"]:
            best = candidate
    return best


def _evaluate_threshold(scores, labels, threshold):
    pred = np.where(np.asarray(scores) >= threshold, ARC_LABEL, NORMAL_LABEL)
    labels = np.asarray(labels)
    tp = int(np.sum((labels == ARC_LABEL) & (pred == ARC_LABEL)))
    fp = int(np.sum((labels == NORMAL_LABEL) & (pred == ARC_LABEL)))
    tn = int(np.sum((labels == NORMAL_LABEL) & (pred == NORMAL_LABEL)))
    fn = int(np.sum((labels == ARC_LABEL) & (pred == NORMAL_LABEL)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def run_band_energy_baseline(feature_df):
    """Evaluate a simple 2.5-6.25 kHz energy threshold baseline by source file."""
    if SOURCE_FILE_COLUMN not in feature_df.columns:
        print("Band-energy baseline skipped: source metadata not found.")
        return pd.DataFrame(), pd.DataFrame()

    # Column names are one-based labels for zero-based FFT bins:
    # f1 -> bin 0 (DC), so 2.5, 3.75, 5.0, 6.25 kHz are f3..f6.
    bin_indices = [2, 3, 4, 5]
    ch1_cols = [f"FFT_CH1_f{index + 1}" for index in bin_indices]
    ch2_cols = [f"FFT_CH2_f{index + 1}" for index in bin_indices]
    y = feature_df[OUTPUT_LABEL_COLUMN].astype(str).to_numpy()
    groups = feature_df[SOURCE_FILE_COLUMN].to_numpy()
    score_sets = {
        "CH1_band_energy_2p5_6p25_kHz": np.sum(np.square(feature_df[ch1_cols].to_numpy()), axis=1),
        "CH2_band_energy_2p5_6p25_kHz": np.sum(np.square(feature_df[ch2_cols].to_numpy()), axis=1),
        "CH1_CH2_band_energy_2p5_6p25_kHz": np.sum(np.square(feature_df[ch1_cols + ch2_cols].to_numpy()), axis=1),
    }
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=FILE_ORDER_SEED)
    rows = []
    for baseline_name, scores in score_sets.items():
        for fold, (train_idx, val_idx) in enumerate(splitter.split(scores.reshape(-1, 1), y, groups), start=1):
            threshold_info = _best_threshold(scores[train_idx], y[train_idx])
            metrics = _evaluate_threshold(scores[val_idx], y[val_idx], threshold_info["threshold"])
            rows.append({
                "baseline": baseline_name,
                "fold": fold,
                "validation_files": ",".join(str(value) for value in sorted(set(groups[val_idx]))),
                "threshold": threshold_info["threshold"],
                "normal": int(np.sum(y[val_idx] == NORMAL_LABEL)),
                "arc": int(np.sum(y[val_idx] == ARC_LABEL)),
                **metrics,
            })
    results = pd.DataFrame(rows)
    summary = results.groupby("baseline")[["precision", "recall", "f1", "accuracy"]].agg(["mean", "std"]).reset_index()
    results.to_csv(SUPPLEMENTARY_DIR / "band_energy_baseline_folds.csv", index=False)
    summary.to_csv(SUPPLEMENTARY_DIR / "band_energy_baseline_summary.csv", index=False)
    results.to_excel(SUPPLEMENTARY_DIR / "band_energy_baseline_folds.xlsx", index=False)
    safe_to_excel(summary, SUPPLEMENTARY_DIR / "band_energy_baseline_summary.xlsx")
    return results, summary


# %% CELL 11
# Prepare X and y arrays from the parquet file generated above.
start_pipeline_step("Prepare split and audit")
feature_df = pd.read_parquet(FEATURE_PARQUET_PATH)
X_all = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
y_all = feature_df[OUTPUT_LABEL_COLUMN].astype(str).to_numpy()

if set(np.unique(y_all)) - {NORMAL_LABEL, ARC_LABEL}:
    raise ValueError(f"Unexpected labels found: {sorted(set(np.unique(y_all)))}")

split_data = split_temporal_data(X_all, y_all)
export_dataset_audit(feature_df, split_data)
print("Temporal split:")
print("Train      :", len(split_data["X_train"]))
print("Validation :", len(split_data["X_validation"]))
print("Test       :", len(split_data["X_test"]))
finish_pipeline_step()

all_training_results = {}
if TRAIN_MODELS:
    start_pipeline_step("Train/evaluate Min-Max")
    all_training_results["minmax"] = train_and_evaluate_scaling(X_all, y_all, split_data, MinMaxScaler, "minmax")
    finish_pipeline_step()
    start_pipeline_step("Train/evaluate Z-Score")
    all_training_results["zscore"] = train_and_evaluate_scaling(X_all, y_all, split_data, StandardScaler, "zscore")
    finish_pipeline_step()

    metric_table = pd.DataFrame(
        all_training_results["minmax"]["metric_records"] + all_training_results["zscore"]["metric_records"]
    )
    parameter_table = pd.DataFrame(
        all_training_results["minmax"]["best_parameter_records"] + all_training_results["zscore"]["best_parameter_records"]
    )

    metric_table.to_csv(METRICS_DIR / "test_metrics_summary.csv", index=False)
    metric_table.to_excel(METRICS_DIR / "test_metrics_summary.xlsx", index=False)
    parameter_table.to_csv(METRICS_DIR / "best_hyperparameters.csv", index=False)
    parameter_table.to_excel(METRICS_DIR / "best_hyperparameters.xlsx", index=False)

    # Save F1 mean +/- std files with names equivalent to the pipeline.
    pd.DataFrame(all_training_results["minmax"]["cv_f1_records"]).set_index("model").to_excel(F1_MINMAX_XLSX, sheet_name="F1_CV_MinMax")
    pd.DataFrame(all_training_results["zscore"]["cv_f1_records"]).set_index("model").to_excel(F1_ZSCORE_XLSX, sheet_name="F1_CV_ZScore")

    print("\nMetric summary:")
    display(metric_table)
else:
    print("TRAIN_MODELS is False. Model training was skipped.")

# %% CELL 12
def save_scaled_datasets(feature_df):
    """
    Save Min-Max and Z-Score versions of the full derived dataset.

    These scalers are fitted to the complete derived feature table and exported with the package outputs.
    They are provided as inspection artifacts only. Serialized RF, KNN, and MLP models remain
    complete scikit-learn pipelines with their own embedded preprocessing steps; serialized CNN
    evaluation uses the train+validation scaler saved from the fitted CNN pipeline.
    """
    X_full = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_full = feature_df[OUTPUT_LABEL_COLUMN].astype(str).to_numpy()

    minmax_scaler = MinMaxScaler().fit(X_full)
    zscore_scaler = StandardScaler().fit(X_full)

    joblib.dump(minmax_scaler, SCALER_MINMAX_PATH)
    joblib.dump(zscore_scaler, SCALER_ZSCORE_PATH)
    print("Saved:", SCALER_MINMAX_PATH)
    print("Saved:", SCALER_ZSCORE_PATH)

    minmax_df = pd.DataFrame(minmax_scaler.transform(X_full), columns=FEATURE_COLUMNS)
    minmax_df[OUTPUT_LABEL_COLUMN] = y_full
    zscore_df = pd.DataFrame(zscore_scaler.transform(X_full), columns=FEATURE_COLUMNS)
    zscore_df[OUTPUT_LABEL_COLUMN] = y_full

    minmax_df.to_csv(MINMAX_DATASET_CSV, index=False)
    zscore_df.to_csv(ZSCORE_DATASET_CSV, index=False)
    print("Saved:", MINMAX_DATASET_CSV)
    print("Saved:", ZSCORE_DATASET_CSV)


if TRAIN_MODELS:
    start_pipeline_step("Export scaled datasets")
    save_scaled_datasets(feature_df)
    finish_pipeline_step()
else:
    print("RETRAIN_MAIN_GRID = False: skipping scaled-dataset export (existing artifacts preserved).")



def save_trained_models(training_results):
    """Save RF/KNN/MLP pipelines and standalone CNN Keras models."""
    if not training_results:
        print("No training results available to save.")
        return

    model_metadata = {
        "version": "v08",
        "run_mode": RUN_MODE,
        "input_feature_count": len(FEATURE_COLUMNS),
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "note": "RF/KNN/MLP are saved as complete scikit-learn pipelines. Each CNN is saved as a standalone Keras .h5 model and is evaluated with the train+validation scaler fitted inside the selected pipeline.",
    }

    for scaling_key, result in training_results.items():
        for model_name, fitted_model in result["fitted_models"].items():
            if model_name == "CNN":
                keras_model_path = MODEL_DIR / f"CNN_{scaling_key}.h5"

                # Save only the trained Keras network. The serialized CNN evaluation below
                # applies the exported scaler associated with the same scaling strategy.
                fitted_model.named_steps["clf"].model_.save(keras_model_path)
                cnn_scaler_path = CNN_SCALER_MINMAX_PATH if scaling_key == "minmax" else CNN_SCALER_ZSCORE_PATH
                joblib.dump(fitted_model.named_steps["scale"], cnn_scaler_path)
                print("Saved:", keras_model_path)
                print("Saved:", cnn_scaler_path)
            else:
                pipeline_path = MODEL_DIR / f"{model_name}_{scaling_key}.pkl"
                joblib.dump(fitted_model, pipeline_path)
                print("Saved:", pipeline_path)

    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    print("Saved:", MODEL_DIR / "model_metadata.json")


def verify_serialized_models_exist():
    """When reusing models (RETRAIN_MAIN_GRID = False), confirm the saved models
    are present, with a clear, actionable error if not."""
    required = []
    for scaling_key in ["minmax", "zscore"]:
        for model_name in ["RF", "KNN", "MLP"]:
            required.append(MODEL_DIR / f"{model_name}_{scaling_key}.pkl")
        required.append(MODEL_DIR / f"CNN_{scaling_key}.h5")
        required.append(CNN_SCALER_MINMAX_PATH if scaling_key == "minmax" else CNN_SCALER_ZSCORE_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "RETRAIN_MAIN_GRID = False reuses previously trained models, but these are missing:\n  - "
            + "\n  - ".join(missing)
            + f"\n\nEither place the exported model files in {MODEL_DIR}, or set "
            "RETRAIN_MAIN_GRID = True to train from scratch."
        )
    print("Reusing existing trained models from:", MODEL_DIR)


if TRAIN_MODELS:
    start_pipeline_step("Save trained models")
    save_trained_models(all_training_results)
    finish_pipeline_step()
else:
    verify_serialized_models_exist()

# %% CELL 13
def insert_predictions_into_full_dataset(feature_df, split_data, training_results, scaling_key):
    """Insert test predictions into a full-size dataframe, matching the full-size output style."""
    output_df = feature_df.copy()
    test_start = split_data["validation_end"]
    test_end = split_data["n_samples"]

    for model_name, prediction_array in training_results[scaling_key]["test_predictions"].items():
        column_name = f"{model_name}_prediction"
        output_df[column_name] = pd.Series(dtype="object")
        output_df.iloc[test_start:test_end, output_df.columns.get_loc(column_name)] = prediction_array

    for model_name, probability_array in training_results[scaling_key]["test_probabilities"].items():
        column_name = f"{model_name}_probability_Arc"
        output_df[column_name] = np.nan
        output_df.iloc[test_start:test_end, output_df.columns.get_loc(column_name)] = probability_array

    return output_df


if TRAIN_MODELS and all_training_results:
    start_pipeline_step("Export prediction tables")
    minmax_prediction_table = insert_predictions_into_full_dataset(feature_df, split_data, all_training_results, "minmax")
    zscore_prediction_table = insert_predictions_into_full_dataset(feature_df, split_data, all_training_results, "zscore")

    minmax_prediction_table.to_csv(MINMAX_PREDICTIONS_CSV, index=False)
    zscore_prediction_table.to_csv(ZSCORE_PREDICTIONS_CSV, index=False)
    if EXPORT_FULL_TABLES_XLSX:
        minmax_prediction_table.to_excel(MINMAX_PREDICTIONS_XLSX, index=False)
        zscore_prediction_table.to_excel(ZSCORE_PREDICTIONS_XLSX, index=False)
        print("Saved (full xlsx):", MINMAX_PREDICTIONS_XLSX)
        print("Saved (full xlsx):", ZSCORE_PREDICTIONS_XLSX)
    else:
        # Only the test rows carry predictions; export just those to xlsx (fast),
        # while the full tables remain available as CSV.
        test_start = split_data["validation_end"]
        minmax_prediction_table.iloc[test_start:].to_excel(MINMAX_PREDICTIONS_XLSX, index=False)
        zscore_prediction_table.iloc[test_start:].to_excel(ZSCORE_PREDICTIONS_XLSX, index=False)
        print("Saved (test-rows xlsx):", MINMAX_PREDICTIONS_XLSX)

    print("Saved:", MINMAX_PREDICTIONS_CSV)
    print("Saved:", ZSCORE_PREDICTIONS_CSV)
    display(minmax_prediction_table.tail())
    finish_pipeline_step()

# %% CELL 15
def generate_spectral_figures(feature_df):
    """Generate line and scatter FFT magnitude figures with a consistent color convention."""
    normal_df = feature_df[feature_df[OUTPUT_LABEL_COLUMN] == NORMAL_LABEL]
    arc_df = feature_df[feature_df[OUTPUT_LABEL_COLUMN] == ARC_LABEL]

    if normal_df.empty or arc_df.empty:
        print("Spectral figures were skipped because one of the classes is missing.")
        return

    mean_ch1_normal = normal_df[FFT_CH1_COLUMNS].mean().to_numpy()
    mean_ch1_arc = arc_df[FFT_CH1_COLUMNS].mean().to_numpy()
    mean_ch2_normal = normal_df[FFT_CH2_COLUMNS].mean().to_numpy()
    mean_ch2_arc = arc_df[FFT_CH2_COLUMNS].mean().to_numpy()

    # FFT_CH*_f1 maps to FFT index 0 (DC); f100 maps to index 99.
    frequencies_hz = np.arange(NUM_FFT_COEFFICIENTS) * (SAMPLING_RATE_HZ / WINDOW_SIZE)
    frequencies_khz = frequencies_hz / 1000

    # Guard against log10(0) without changing the plotted class/color logic.
    epsilon = 1e-12
    mean_ch1_normal_db = 20 * np.log10(np.maximum(mean_ch1_normal, epsilon))
    mean_ch1_arc_db = 20 * np.log10(np.maximum(mean_ch1_arc, epsilon))
    mean_ch2_normal_db = 20 * np.log10(np.maximum(mean_ch2_normal, epsilon))
    mean_ch2_arc_db = 20 * np.log10(np.maximum(mean_ch2_arc, epsilon))

    # Line plots: Normal uses Matplotlib's default blue; Arc is explicitly red, for the spectral figure.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(frequencies_khz, mean_ch1_normal_db, label="Normal", linewidth=2)
    axes[0].plot(frequencies_khz, mean_ch1_arc_db, label="Arc", color="red", linewidth=2)
    axes[0].set_title("Mean FFT - CH1 (Current, A) [kHz x dB]")
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].legend(title="CH1")
    axes[0].grid(True, which="both", ls="--", lw=0.5)

    axes[1].plot(frequencies_khz, mean_ch2_normal_db, label="Normal", linewidth=2)
    axes[1].plot(frequencies_khz, mean_ch2_arc_db, label="Arc", color="red", linewidth=2)
    axes[1].set_title("Mean FFT - CH2 (Voltage, V) [kHz x dB]")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].legend(title="CH2")
    axes[1].grid(True, which="both", ls="--", lw=0.5)

    plt.tight_layout()
    line_path = FIGURE_DIR / "mean_fft_magnitude_ch1_ch2.png"
    save_figure_hq(fig, line_path)
    print("Saved:", line_path)

    # Scatter plots: Normal blue and Arc red, both with black marker edges, for the spectral figure.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(frequencies_khz, mean_ch1_normal_db, label="Normal", color="blue", edgecolor="k")
    axes[0].scatter(frequencies_khz, mean_ch1_arc_db, label="Arc", color="red", edgecolor="k")
    axes[0].set_title("Scatter: Mean FFT - CH1 [kHz x dB]")
    axes[0].set_xlabel("Frequency (kHz)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="--", lw=0.5)

    axes[1].scatter(frequencies_khz, mean_ch2_normal_db, label="Normal", color="blue", edgecolor="k")
    axes[1].scatter(frequencies_khz, mean_ch2_arc_db, label="Arc", color="red", edgecolor="k")
    axes[1].set_title("Scatter: Mean FFT - CH2 [kHz x dB]")
    axes[1].set_xlabel("Frequency (kHz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--", lw=0.5)

    plt.tight_layout()
    scatter_path = FIGURE_DIR / "scatter_mean_fft_magnitude_ch1_ch2.png"
    save_figure_hq(fig, scatter_path)
    print("Saved:", scatter_path)


if GENERATE_SPECTRAL_FIGURES:
    start_pipeline_step("Generate spectral figures")
    generate_spectral_figures(feature_df)
    finish_pipeline_step()

# %% CELL 17
def load_serialized_predictions(X_test, scaling_key):
    """Load serialized models and compute predictions/probabilities for one scaling strategy."""
    outputs = {}

    for model_name in ["RF", "KNN", "MLP"]:
        model_path = MODEL_DIR / f"{model_name}_{scaling_key}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        pipeline = joblib.load(model_path)
        probability = positive_probability(pipeline, X_test, positive_label=ARC_LABEL)
        prediction = pipeline.predict(X_test).astype(str)
        outputs[model_name] = {"probability": probability, "prediction": prediction, "object": pipeline}

    cnn_model_path = MODEL_DIR / f"CNN_{scaling_key}.h5"
    if scaling_key == "minmax":
        scaler_path = CNN_SCALER_MINMAX_PATH
    elif scaling_key == "zscore":
        scaler_path = CNN_SCALER_ZSCORE_PATH
    else:
        raise ValueError(f"Unsupported scaling key: {scaling_key}")

    if not cnn_model_path.exists():
        raise FileNotFoundError(cnn_model_path)
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)

    # The serialized CNN evaluation uses the exported scaler associated with the same scaling strategy.
    cnn_model = load_model(cnn_model_path, compile=False)
    cnn_scaler = joblib.load(scaler_path)
    X_test_cnn = cnn_scaler.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
    cnn_probability = cnn_model.predict(X_test_cnn, verbose=0).ravel()
    cnn_prediction = np.where(cnn_probability >= 0.5, ARC_LABEL, NORMAL_LABEL)
    outputs["CNN"] = {
        "probability": cnn_probability,
        "prediction": cnn_prediction,
        "object": cnn_model,
        "scaler": cnn_scaler,
    }
    return outputs


def bootstrap_auc_confidence_interval(y_true, probabilities, n_resamples):
    """Compute AUC and a bootstrap 95% confidence interval for the Arc class."""
    rng = np.random.default_rng(42)
    y_binary = (np.asarray(y_true) == ARC_LABEL).astype(int)
    probabilities = np.asarray(probabilities)
    auc_values = []

    for _ in range(n_resamples):
        indices = rng.integers(0, len(y_binary), len(y_binary))
        if len(np.unique(y_binary[indices])) < 2:
            continue
        auc_values.append(roc_auc_score(y_binary[indices], probabilities[indices]))

    auc_value = roc_auc_score(y_binary, probabilities) if len(np.unique(y_binary)) == 2 else np.nan
    if auc_values:
        lower, upper = np.percentile(auc_values, [2.5, 97.5])
    else:
        lower, upper = np.nan, np.nan
    return float(auc_value), float(lower), float(upper)


def compute_mcnemar_table(y_true, predictions_by_model):
    """Compute pairwise exact McNemar tests between classifiers."""
    records = []
    model_names = list(predictions_by_model.keys())
    y_true = np.asarray(y_true)

    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1:]:
            pred_a = np.asarray(predictions_by_model[model_a])
            pred_b = np.asarray(predictions_by_model[model_b])
            correct_a = pred_a == y_true
            correct_b = pred_b == y_true
            table = np.array([
                [np.sum(correct_a & correct_b), np.sum(correct_a & ~correct_b)],
                [np.sum(~correct_a & correct_b), np.sum(~correct_a & ~correct_b)],
            ])
            result = mcnemar(table, exact=True)
            records.append({
                "model_a": model_a,
                "model_b": model_b,
                "both_correct": int(table[0, 0]),
                "model_a_correct_model_b_incorrect": int(table[0, 1]),
                "model_a_incorrect_model_b_correct": int(table[1, 0]),
                "both_incorrect": int(table[1, 1]),
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            })
    return pd.DataFrame(records)


def measure_inference_latency(X_test, scaling_key, serialized_outputs, n_runs):
    """Measure total and per-sample inference latency for serialized models."""
    records = []
    for model_name, output in serialized_outputs.items():
        times = []
        if model_name == "CNN":
            model = output["object"]
            scaler = output["scaler"]
            X_prepared = scaler.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
            _ = model.predict(X_prepared, verbose=0)  # warm-up
            for _ in range(n_runs):
                start_time = time.perf_counter()
                X_prepared = scaler.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
                _ = model.predict(X_prepared, verbose=0)
                times.append(time.perf_counter() - start_time)
        else:
            pipeline = output["object"]
            _ = pipeline.predict_proba(X_test)  # warm-up
            for _ in range(n_runs):
                start_time = time.perf_counter()
                _ = pipeline.predict_proba(X_test)
                times.append(time.perf_counter() - start_time)

        median_total_seconds = float(np.median(times))
        records.append({
            "scaling": scaling_key,
            "model": model_name,
            "runs": n_runs,
            "test_samples": len(X_test),
            "median_total_seconds": median_total_seconds,
            "median_per_sample_ms": (median_total_seconds / len(X_test)) * 1000,
        })
    return pd.DataFrame(records)


def run_serialized_model_evaluation():
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    y_test_binary = (y_test == ARC_LABEL).astype(int)
    n_bootstrap = 100 if RUN_MODE == "quick_check" else 1000
    n_latency_runs = 5 if RUN_MODE == "quick_check" else 30

    all_auc_records = []
    all_latency_tables = []

    for scaling_key in ["minmax", "zscore"]:
        print(f"\nSerialized evaluation: {scaling_key}")
        serialized_outputs = load_serialized_predictions(X_test, scaling_key)

        prediction_table = pd.DataFrame({"true_label": y_test})
        predictions_by_model = {}

        fig, ax = plt.subplots(figsize=(6, 5))
        for model_name, output in serialized_outputs.items():
            probability = output["probability"]
            prediction = output["prediction"]
            predictions_by_model[model_name] = prediction
            prediction_table[f"{model_name}_prediction"] = prediction
            prediction_table[f"{model_name}_probability_Arc"] = probability

            if len(np.unique(y_test_binary)) == 2:
                fpr, tpr, _ = roc_curve(y_test_binary, probability)
                auc_value, lower, upper = bootstrap_auc_confidence_interval(y_test, probability, n_bootstrap)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
            else:
                auc_value, lower, upper = np.nan, np.nan, np.nan

            all_auc_records.append({
                "scaling": scaling_key,
                "model": model_name,
                "auc": auc_value,
                "auc_ci_lower_95": lower,
                "auc_ci_upper_95": upper,
                "bootstrap_resamples": n_bootstrap,
            })

        ax.plot([0, 1], [0, 1], "--", label="Random (0.5)")
        ax.set_title(f"ROC - {scaling_key}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5)
        roc_path = FIGURE_DIR / f"roc_curves_{scaling_key}.png"
        save_figure_hq(fig, roc_path)
        print("Saved:", roc_path)

        prediction_csv = PREDICTION_DIR / f"serialized_test_predictions_{scaling_key}.csv"
        prediction_xlsx = PREDICTION_DIR / f"serialized_test_predictions_{scaling_key}.xlsx"
        prediction_table.to_csv(prediction_csv, index=False)
        prediction_table.to_excel(prediction_xlsx, index=False)
        print("Saved:", prediction_csv)
        print("Saved:", prediction_xlsx)

        # Reproduced headline metrics computed directly from the serialized models.
        # When RETRAIN_MAIN_GRID = False these models are exactly the ones behind
        # the manuscript, so this table is a direct check that the reported
        # F1/precision/recall/confusion are reproduced.
        reproduced_rows = []
        for model_name, model_prediction in predictions_by_model.items():
            reproduced = evaluate_predictions(y_test, model_prediction)
            confusion = reproduced["confusion_matrix"]
            reproduced_rows.append({
                "scaling": scaling_key,
                "model": model_name,
                "accuracy": reproduced["accuracy"],
                "precision": reproduced["precision"],
                "recall": reproduced["recall"],
                "f1": reproduced["f1"],
                "tn": int(confusion[0, 0]),
                "fp": int(confusion[0, 1]),
                "fn": int(confusion[1, 0]),
                "tp": int(confusion[1, 1]),
            })
        reproduced_metrics = pd.DataFrame(reproduced_rows)
        reproduced_metrics.to_csv(METRICS_DIR / f"reproduced_test_metrics_{scaling_key}.csv", index=False)
        print(f"Reproduced test metrics ({scaling_key}) from serialized models:")
        display(reproduced_metrics)

        # Operating points: highest recall achievable at each target false-positive
        # rate. Vectorized in O(n log n) instead of the previous O(thresholds x n)
        # double loop; the "predict positive iff score >= threshold" rule and the
        # selection are preserved exactly (ties handled by taking the last index of
        # each equal-score group, so all samples >= threshold are counted).
        operating_rows = []
        target_fprs = [0.001, 0.005, 0.01, 0.02, 0.05]
        total_positive = int(np.sum(y_test_binary == 1))
        total_negative = int(np.sum(y_test_binary == 0))
        for model_name, output in serialized_outputs.items():
            scores = np.asarray(output["probability"], dtype=float)
            order = np.argsort(-scores, kind="mergesort")
            scores_sorted = scores[order]
            positive_sorted = y_test_binary[order] == 1
            tp_cumulative = np.cumsum(positive_sorted)
            fp_cumulative = np.cumsum(~positive_sorted)
            last_of_group = np.ones(len(scores_sorted), dtype=bool)
            last_of_group[:-1] = scores_sorted[1:] != scores_sorted[:-1]
            thresholds_unique = scores_sorted[last_of_group]
            tp_at = tp_cumulative[last_of_group]
            fp_at = fp_cumulative[last_of_group]
            recall_at = tp_at / total_positive if total_positive else np.zeros(len(tp_at))
            fpr_at = fp_at / total_negative if total_negative else np.zeros(len(fp_at))
            precision_at = np.divide(
                tp_at, tp_at + fp_at, out=np.zeros(len(tp_at), dtype=float), where=(tp_at + fp_at) > 0
            )
            for target_fpr in target_fprs:
                allowed = fpr_at <= target_fpr
                if not allowed.any():
                    continue
                max_recall = recall_at[allowed].max()
                # Reproduce the original ascending-threshold scan's tie-break exactly:
                # among allowed thresholds achieving the maximum recall, keep the
                # lowest threshold (the first the original loop would have locked in).
                tie = allowed & (recall_at == max_recall)
                best_i = int(np.flatnonzero(tie).max())
                operating_rows.append({
                    "scaling": scaling_key,
                    "model": model_name,
                    "target_fpr": target_fpr,
                    "recall": float(recall_at[best_i]),
                    "precision": float(precision_at[best_i]),
                    "observed_fpr": float(fpr_at[best_i]),
                    "threshold": float(thresholds_unique[best_i]),
                })
        operating_table = pd.DataFrame(operating_rows)
        operating_table.to_csv(METRICS_DIR / f"operating_points_{scaling_key}.csv", index=False)
        operating_table.to_excel(METRICS_DIR / f"operating_points_{scaling_key}.xlsx", index=False)

        mcnemar_table = compute_mcnemar_table(y_test, predictions_by_model)
        mcnemar_table.to_csv(METRICS_DIR / f"mcnemar_{scaling_key}.csv", index=False)
        mcnemar_table.to_excel(METRICS_DIR / f"mcnemar_{scaling_key}.xlsx", index=False)

        latency_table = measure_inference_latency(X_test, scaling_key, serialized_outputs, n_latency_runs)
        latency_table.to_csv(METRICS_DIR / f"latency_{scaling_key}.csv", index=False)
        latency_table.to_excel(METRICS_DIR / f"latency_{scaling_key}.xlsx", index=False)
        all_latency_tables.append(latency_table)

    auc_table = pd.DataFrame(all_auc_records)
    auc_table.to_csv(METRICS_DIR / "auc_bootstrap_summary.csv", index=False)
    auc_table.to_excel(METRICS_DIR / "auc_bootstrap_summary.xlsx", index=False)
    latency_summary = pd.concat(all_latency_tables, ignore_index=True) if all_latency_tables else pd.DataFrame()
    print("\nAUC summary:")
    display(auc_table)
    print("\nLatency summary:")
    display(latency_summary)


if RUN_SERIALIZED_MODEL_EVALUATION:
    start_pipeline_step("Serialized model evaluation")
    run_serialized_model_evaluation()
    finish_pipeline_step()

def run_window_length_sweep():
    """
    Supplementary analysis: file-grouped F1 as a function of the
    analysis window length. For each window length the feature set is rebuilt with
    the same vectorized extractor (fast) and evaluated under StratifiedGroupKFold by
    source file with lightweight, fixed models (Logistic Regression + a fixed Random
    Forest). This does NOT alter the manuscript's main results; it only quantifies
    sensitivity to the 0.8 ms (200-sample) choice.
    """
    print("Window-length sensitivity sweep:", WINDOW_LENGTH_SWEEP_VALUES)
    all_indices = list(range(1, 17))
    rows = []
    for window_size in WINDOW_LENGTH_SWEEP_VALUES:
        num_fft = window_size // 2
        tables = []
        for file_index in all_indices:
            table = extract_windows_for_file(
                file_index, window_size=window_size, num_fft=num_fft, stride=window_size
            )
            if table is not None:
                tables.append(table.to_pandas())
        if not tables:
            continue
        data = pd.concat(tables, ignore_index=True)
        fft_cols = (
            [f"FFT_CH1_f{i}" for i in range(1, num_fft + 1)]
            + [f"FFT_CH2_f{i}" for i in range(1, num_fft + 1)]
        )
        feature_cols = ["CH1_mean", "CH1_std", "CH2_mean", "CH2_std"] + fft_cols
        X = data[feature_cols].to_numpy(dtype=np.float32)
        y = data[OUTPUT_LABEL_COLUMN].astype(str).to_numpy()
        groups = data[SOURCE_FILE_COLUMN].to_numpy()
        models = {
            "LogisticRegression_ZScore": Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]),
            "RF_ZScore_fixed": Pipeline([
                ("scale", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=100, n_jobs=CPU_JOBS)),
            ]),
        }
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=FILE_ORDER_SEED)
        for model_name, model in models.items():
            fold_f1 = []
            for train_idx, val_idx in splitter.split(X, y, groups):
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[val_idx])
                fold_f1.append(f1_score(y[val_idx], pred, pos_label=ARC_LABEL, zero_division=0))
            rows.append({
                "window_size_samples": window_size,
                "window_ms": 1000.0 * window_size / SAMPLING_RATE_HZ,
                "num_features": len(feature_cols),
                "model": model_name,
                "n_windows": int(len(y)),
                "f1_mean": float(np.mean(fold_f1)),
                "f1_std": float(np.std(fold_f1)),
            })
            print(f"  W={window_size} ({model_name}): F1={np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f}")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(SUPPLEMENTARY_DIR / "window_length_sweep.csv", index=False)
    sweep.to_excel(SUPPLEMENTARY_DIR / "window_length_sweep.xlsx", index=False)
    print("Saved:", SUPPLEMENTARY_DIR / "window_length_sweep.csv")
    return sweep


# Supplementary analyses. These do not replace the original
# hold-out metrics; they contextualize generalization and a simple signal-processing baseline.
if RUN_MODE == "full_reproduction":
    start_pipeline_step("File-grouped validation")
    grouped_validation_folds, grouped_validation_summary = run_file_grouped_validation(feature_df)
    finish_pipeline_step()
    start_pipeline_step("Band-energy baseline")
    band_baseline_folds, band_baseline_summary = run_band_energy_baseline(feature_df)
    finish_pipeline_step()
    if RUN_WINDOW_LENGTH_SWEEP:
        start_pipeline_step("Window-length sweep")
        window_length_sweep_table = run_window_length_sweep()
        finish_pipeline_step()

# %% CELL 19
def print_saved_model_information():
    """Print saved model parameters and CNN architecture summaries."""
    if not MODEL_DIR.exists():
        print("Model directory does not exist:", MODEL_DIR)
        return

    for path in sorted(MODEL_DIR.iterdir()):
        if path.suffix.lower() not in {".pkl", ".h5"}:
            continue
        print("#" * 90)
        print("File:", path.name)
        if path.suffix.lower() == ".pkl":
            obj = joblib.load(path)
            print("Object type:", type(obj))
            if hasattr(obj, "named_steps"):
                print("Pipeline steps:", list(obj.named_steps.keys()))
                final_estimator = obj.named_steps.get("clf")
                if final_estimator is not None and hasattr(final_estimator, "get_params"):
                    params = serializable_parameters(final_estimator.get_params())
                    print(json.dumps(params, indent=2)[:6000])
            elif hasattr(obj, "get_params"):
                params = serializable_parameters(obj.get_params())
                print(json.dumps(params, indent=2)[:6000])
        elif path.suffix.lower() == ".h5":
            model = load_model(path, compile=False)
            model.summary(line_length=120)

start_pipeline_step("Print saved model information")
print_saved_model_information()
finish_pipeline_step()
close_pipeline_progress()
