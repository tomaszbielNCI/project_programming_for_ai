# predict_rolling_bnn_v2.py
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.bnn.model_v1 import build_bnn_showcase
from src.loaders.bnn_basic_loader import load_parquet_for_bnn

# ----------------------------
# CONFIG
# ----------------------------
WINDOW = 120
TARGET_HORIZON = 60
DATA_FILE = PROJECT_ROOT / "data" / "parsed" / "US.100+1.parquet"
#DATA_FILE = PROJECT_ROOT / "data" / "parsed" / "USDJPY+1.parquet"
#DATA_FILE = PROJECT_ROOT / "data" / "parsed" / "OIL.WTI+1.parquet"
INTRADAY = True
BATCH_SIZE = 128

# ----------------------------
# FIND TRAINED MODEL
# ----------------------------
def get_latest_model_dir():
    bnn_dir = PROJECT_ROOT / "results" / "bnn"
    model_dirs = sorted(
        [d for d in bnn_dir.glob("train_*") if d.is_dir()],
        key=os.path.getmtime,
        reverse=True
    )
    if not model_dirs:
        raise FileNotFoundError("No trained model directories found in results/bnn/")
    return model_dirs[0]

LATEST = get_latest_model_dir()
MODEL_DIR = LATEST / "model"
META_FILE = LATEST / "metadata.pkl"

print(f"Using model from: {MODEL_DIR}")

# ----------------------------
# LOAD TRAINING META
# ----------------------------
metadata = joblib.load(META_FILE)
train_end_str = metadata.get("train_end")
if train_end_str is None:
    raise RuntimeError("train_end not found in metadata.pkl")

train_end = pd.to_datetime(train_end_str)

print(f"Train end timestamp: {train_end}")

# ----------------------------
# LOAD FULL DATA
# ----------------------------
print("Loading full dataset…")
X, y, timestamps = load_parquet_for_bnn(
    str(DATA_FILE),
    window=WINDOW,
    target_horizon=TARGET_HORIZON,
    intraday=INTRADAY,
    return_window_timestamps=True
)
all_df = pd.DataFrame({
    "timestamp": timestamps,
    "X_index": list(range(len(timestamps)))
})

# ----------------------------
# FILTER START INDEX FOR OOS
# ----------------------------
# We want to start walk-forward AFTER train_end
mask = all_df["timestamp"] > train_end
if not mask.any():
    raise RuntimeError("No OOS samples found after train_end")

start_index = all_df.loc[mask, "X_index"].iloc[0]
print(f"Starting OOS forecasting at index {start_index}, timestamp {timestamps[start_index]}")

# ----------------------------
# LOAD MODEL
# ----------------------------
model = build_bnn_showcase(window=WINDOW, feature_count=X.shape[2], train_size=len(X))
model.load_weights(str(MODEL_DIR / "variables" / "variables"))

# ----------------------------
# TEST DISTRIBUTION
# ----------------------------
print("\n=== MODEL DISTRIBUTION TEST ===")
test_dist = model(X[:1])
print(f"Distribution type: {test_dist.__class__.__name__}")
print(f"Distribution parameters: {test_dist.parameters}")

# Sprawdź bazową dystrybucję jeśli to Independent
if hasattr(test_dist, 'distribution'):
    base = test_dist.distribution
    print(f"Base distribution: {base.__class__.__name__}")
    print(f"Base parameters: {base.parameters}")

# ----------------------------
# WALK FORWARD OUT-OF-SAMPLE
# ----------------------------
predicted_means = []
predicted_stds = []
pred_timestamps = []

step = TARGET_HORIZON
n_samples = len(X)

print("\nStarting walk-forward prediction…")

for i in range(start_index, n_samples - WINDOW, step):
    end = i + WINDOW
    X_window = X[i:i+1]  # batch of 1

    dist = model(X_window)
    mean = dist.mean().numpy().flatten()[0]
    std = dist.stddev().numpy().flatten()[0]

    # Correct timestamp = t + TARGET_HORIZON
    t_pred = timestamps[end-1] + pd.Timedelta(minutes=TARGET_HORIZON)

    predicted_means.append(mean)
    predicted_stds.append(std)
    pred_timestamps.append(t_pred)

# Debug info for the last prediction
dist = model(X_window)
mean = dist.mean().numpy().flatten()[0]
std = dist.stddev().numpy().flatten()[0]

# Check distribution type and parameters
try:
    if hasattr(dist, 'distribution') and hasattr(dist.distribution, 'scale'):
        # For Independent distributions
        scale_param = dist.distribution.scale.numpy().flatten()[0]
        std_from_scale = scale_param * np.sqrt(2)  # For Laplace distribution
        print(f"Distribution type: {type(dist.distribution).__name__}")
        print(f"mean: {mean:.6f}, scale: {scale_param:.6f}, std_from_scale: {std_from_scale:.6f}, current_std: {std:.6f}")
        print(f"std == scale*sqrt(2)? {np.abs(std - std_from_scale) < 1e-6}")
    else:
        # For other distribution types
        print(f"Distribution type: {type(dist).__name__}")
        print(f"mean: {mean:.6f}, std: {std:.6f}")
        
        # Try to get scale if available through other means
        if hasattr(dist, 'scale'):
            scale_param = dist.scale.numpy().flatten()[0]
            print(f"scale: {scale_param:.6f}")
        
        print("Note: Direct scale parameter access not available for this distribution type")
        
except Exception as e:
    print(f"Error accessing distribution parameters: {e}")

# ----------------------------
# SAVE RESULTS
# ----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = PROJECT_ROOT / "results" / "bnn" / "predict_v2" / f"rolling_predictions_{timestamp}.pkl"

joblib.dump({
    "timestamps": pred_timestamps,
    "mean": predicted_means,
    "std": predicted_stds,
    "step_minutes": TARGET_HORIZON,
    "model_dir": str(MODEL_DIR),
    "data_file": str(DATA_FILE)
}, output_file)

print(f"\nPredictions saved to: {output_file}")
print(f"Total OOS predictions: {len(predicted_means)}")
print("Done!")