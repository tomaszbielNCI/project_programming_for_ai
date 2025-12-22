# train_bnn_v2.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import shutil
import os

from src.loaders.bnn_basic_loader import load_parquet_for_bnn
from src.models.bnn.model_v1 import build_bnn_showcase


# ======================================================
# CONFIGURATION (EXPERIMENT DEFINITION)
# ======================================================

WINDOW = 120                  # 120 minutes of history
TARGET_HORIZON = 60           # predict 60 minutes ahead
BATCH_SIZE = 128
EPOCHS = 10                   # easy to increase later

# --- time-based split ---
TRAIN_DAYS = 32
TEST_DAYS = 8                 # used later for walk-forward

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "parsed" / "US.100+1.parquet"

INTRADAY = True               # minutes data

# --- time-based split ---
TRAIN_END_TIMESTAMP = "2025-12-05 21:59:00"  # the last timestamp of training period


# ======================================================
# LOAD FULL DATASET
# ======================================================

X, y, timestamps = load_parquet_for_bnn(
    DATA_FILE,
    window=WINDOW,
    target_horizon=TARGET_HORIZON,
    intraday=INTRADAY,
    return_window_timestamps=True    # IMPOTRANT - for walk-forward
)

print(f"Loaded full dataset: X={X.shape}, y={y.shape}")

# ======================================================
# TIME-BASED SPLIT (NO LEAKAGE)
# ======================================================

timestamps = pd.to_datetime(timestamps)
train_end_ts = pd.Timestamp(TRAIN_END_TIMESTAMP)

train_mask = timestamps <= train_end_ts
future_mask = timestamps > train_end_ts

X_train = X[train_mask]
y_train = y[train_mask]
ts_train = timestamps[train_mask]

X_future = X[future_mask]
y_future = y[future_mask]
ts_future = timestamps[future_mask]

print(f"Train period: {ts_train[0]} → {ts_train[-1]}")
print(f"Future period (for rolling): {ts_future[0]} → {ts_future[-1]}")

# ======================================================
# BUILD MODEL (FROZEN ARCHITECTURE)
# ======================================================

feature_count = X.shape[2]

model = build_bnn_showcase(
    window=WINDOW,
    feature_count=feature_count,
    train_size=len(X_train)
)

model.summary()

# ======================================================
# TRAINING
# ======================================================

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=False,        # CRITICAL: time series
    verbose=1
)

# ======================================================
# SAVE ARTIFACTS (EXPERIMENT SNAPSHOT)
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "bnn"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"train_v2_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# --- model ---
model_dir = RUN_DIR / "model"
model.save(model_dir)

# --- history ---
joblib.dump(history.history, RUN_DIR / "history.pkl")

# --- metadata ---
metadata = {
    "run_id": RUN_ID,
    "data_file": str(DATA_FILE),
    "window": WINDOW,
    "target_horizon": TARGET_HORIZON,
    "train_days": TRAIN_DAYS,
    "test_days": TEST_DAYS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "intraday": INTRADAY,
    "train_start": str(ts_train[0]),
    "train_end": str(ts_train[-1]),
    "future_start": str(ts_future[0]),
}

joblib.dump(metadata, RUN_DIR / "metadata.pkl")

print("\n======================================")
print("TRAINING v2 COMPLETED")
print(f"Model saved to: {model_dir}")
print("This model is now FROZEN for evaluation.")
print("======================================")
