# train_bnn_v1.py
import numpy as np
from pathlib import Path
import joblib  # for saving history
from datetime import datetime
import shutil
import os
import sys
from contextlib import redirect_stdout

from src.loaders.bnn_basic_loader import load_parquet_for_bnn  # loader
from src.models.bnn.model_v1 import build_bnn_showcase  # model

# Tee class for logging to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for file_obj in self.files:
            file_obj.write(obj)
    def flush(self):
        for file_obj in self.files:
            file_obj.flush()
# ----------------------------
# CONFIGURATION
# ----------------------------
WINDOW = 120
TARGET_HORIZON = 30
BATCH_SIZE = 128
EPOCHS = 5  # short for initial test, increase later
VALIDATION_SPLIT = 0.1

DATA_DIR = Path(r"C:\python\project_programming_for_ai\data\parsed")

# --- Choose which parquet to use ---
# For HFD live data:
#data_file = DATA_DIR / "US.100+.parquet"
# For Historical 5-min:
data_file = DATA_DIR / "US.100+1.parquet"

# ----------------------------
# LOAD DATA
# ----------------------------
# For HFD live data: intraday=False
# For Historical 1-min data: intraday=True
X, y = load_parquet_for_bnn(
    data_file,
    window=WINDOW,
    target_horizon=TARGET_HORIZON,
    intraday=True  # <-- set True for intraday, False for HFD
)
print(f"Loaded data: X={X.shape}, y={y.shape}")

# ----------------------------
# OPTIONAL: train/test split
# ----------------------------
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

# ----------------------------
# BUILD MODEL
# ----------------------------
feature_count = X.shape[2]
train_size = len(X_train)
model = build_bnn_showcase(window=WINDOW, feature_count=feature_count, train_size=train_size)
model.summary()  # prints model architecture

# ----------------------------
# TRAINING
# ----------------------------
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    verbose=1
)

# ----------------------------
# SHOW TRAINING RESULTS
# ----------------------------
# Training and validation loss per epoch
for i, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss']), 1):
    print(f"Epoch {i}: loss={loss:.6f}, val_loss={val_loss:.6f}")

# ======================================================
# SAVE MODEL & RESULTS
# ======================================================
# Create results directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "bnn"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging to both console and file
original_stdout = sys.stdout
log_file = open(RUN_DIR / "training_log.txt", 'w', encoding='utf-8')
sys.stdout = Tee(original_stdout, log_file)

print(f"\nRUN ID: {RUN_ID}")
print(f"Model training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Data file: {data_file}")
print(f"Window size: {WINDOW}, Target horizon: {TARGET_HORIZON}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")

# Save model
model_dir = RUN_DIR / "model"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
model.save(model_dir)

# Save training history
joblib.dump(history.history, RUN_DIR / "history.pkl")

# Save model summary
with open(RUN_DIR / "model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# Save training parameters
params = {
    "data_file": str(data_file),
    "window_size": WINDOW,
    "target_horizon": TARGET_HORIZON,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "validation_split": VALIDATION_SPLIT,
    "run_id": RUN_ID,
    "run_timestamp": datetime.now().isoformat()
}
joblib.dump(params, RUN_DIR / "training_params.pkl")

# Print final metrics
print("\nTraining completed successfully!")
print(f"Final training loss: {history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
print(f"\nAll results saved to: {RUN_DIR}")

# Close log file and restore stdout
sys.stdout = original_stdout
log_file.close()

# ----------------------------
# NOTES:
# 1. To switch datasets, just change `data_file` path above.
# 2. Train/test split is optional; you can set split_ratio=1.0 to train on all data.
# 3. Loss shown is negative log-likelihood (BNN standard).
# 4. For semi-offline features (Hurst, volatility), compute separately and add to loader if needed.
