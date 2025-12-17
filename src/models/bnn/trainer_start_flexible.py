# trainer_start_flexible.py
import numpy as np
from pathlib import Path
import joblib  # for saving history
from src.loaders.bnn_basic_loader import load_parquet_for_bnn  # loader
from src.models.bnn.model import build_bnn_showcase  # model
# ----------------------------
# CONFIGURATION
# ----------------------------
WINDOW = 20
TARGET_HORIZON = 1
BATCH_SIZE = 128
EPOCHS = 5  # short for initial test, increase later
VALIDATION_SPLIT = 0.1

DATA_DIR = Path(r"C:\python\project_programming_for_ai\data\parsed")

# --- Choose which parquet to use ---
# For HFD live data:
data_file = DATA_DIR / "US.100+.parquet"
# For Historical 5-min:
# data_file = DATA_DIR / "US.100+5.parquet"

# ----------------------------
# LOAD DATA
# ----------------------------
X, y = load_parquet_for_bnn(data_file, window=WINDOW, target_horizon=TARGET_HORIZON)
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

# ----------------------------
# SAVE MODEL & HISTORY
# ----------------------------
model.save("bnn_laplace_model")        # Save trained model
joblib.dump(history.history, "history.pkl")  # Save training history for later plotting/analysis
print("Model and training history saved.")

# ----------------------------
# NOTES:
# 1. To switch datasets, just change `data_file` path above.
# 2. Train/test split is optional; you can set split_ratio=1.0 to train on all data.
# 3. Loss shown is negative log-likelihood (BNN standard).
# 4. For semi-offline features (Hurst, volatility), compute separately and add to loader if needed.
