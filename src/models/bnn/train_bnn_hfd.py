"""
BNN Training for Minimal HFD Aggregated Data
Trains Bayesian Neural Network on HFD aggregated 1-minute data with minimal features.
Only uses: log_return_clipped, volatility, activity, tick_count
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.bnn.model_v2 import build_bnn_showcase

# ======================================================
# CONFIGURATION - MINIMAL HFD DATA
# ======================================================

# Model parameters
WINDOW = 120  # 120 minutes lookback
TARGET_HORIZON = 60  # predict 60 minutes ahead
BATCH_SIZE = 128
EPOCHS = 10

# Data file (MINIMAL VERSION)
DATA_FILE = PROJECT_ROOT / "data" / "parsed" / "US.100_1min_minimal.parquet"

# Time-based split: last N hours for testing
TEST_HOURS = 24  # Use last 24 hours for prediction


# ======================================================
# MINIMAL LOADER WITH TIME-BASED SPLIT
# ======================================================

def load_hfd_minimal_with_split(path, window=120, target_horizon=60, test_hours=24, normalize=True):
    """
    Load minimal HFD data with time-based split.

    Args:
        path: Path to minimal aggregated Parquet file
        window: Lookback window in minutes
        target_horizon: Prediction horizon in minutes
        test_hours: Number of hours to reserve for testing
        normalize: Whether to normalize features

    Returns:
        X_train, y_train, X_test, y_test, test_timestamps, scaler
    """
    from sklearn.preprocessing import StandardScaler

    # Load data
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded minimal HFD data with columns: {list(df.columns)}")

    # Verify required columns
    required_columns = ['mid', 'tick_count', 'log_return_60min']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate 1-minute returns from mid price
    log_mid = np.log(df["mid"].values)
    lr = np.zeros_like(log_mid)
    lr[1:] = log_mid[1:] - log_mid[:-1]
    df["log_return"] = lr.astype("float32")
    df = df.iloc[1:].reset_index(drop=True)

    # Soft outlier control on 1-minute returns
    q_low, q_high = np.quantile(df["log_return"], [0.001, 0.999])
    df["log_return_clipped"] = df["log_return"].clip(q_low, q_high)

    # Calculate rolling volatility and activity from 1-minute returns
    vol_window = 20
    abs_lr = np.abs(df["log_return_clipped"].values)
    df["volatility"] = pd.Series(abs_lr).rolling(vol_window, min_periods=vol_window).std().values
    df["activity"] = pd.Series(abs_lr).rolling(vol_window, min_periods=vol_window).mean().values

    # Drop NaNs from rolling calculations
    df = df.dropna().reset_index(drop=True)

    # Time-based split
    last_timestamp = df["timestamp"].max()
    test_start = last_timestamp - timedelta(hours=test_hours)

    print(f"Full data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Test period: {test_start} to {last_timestamp}")

    # Split data
    train_mask = df["timestamp"] < test_start
    test_mask = df["timestamp"] >= test_start

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Prepare features - only 4 features
    feature_columns = ["log_return_clipped", "volatility", "activity", "tick_count"]
    train_features = train_df[feature_columns].values.astype("float32")
    test_features = test_df[feature_columns].values.astype("float32")

    # Targets
    train_target = train_df["log_return_60min"].values.astype("float32")
    test_target = test_df["log_return_60min"].values.astype("float32")
    test_timestamps = test_df["timestamp"].values

    # Normalize features
    scaler = None
    if normalize:
        print(f"\nNormalizing {len(feature_columns)} features...")
        scaler = StandardScaler()
        train_features_normalized = scaler.fit_transform(train_features)
        test_features_normalized = scaler.transform(test_features)
    else:
        train_features_normalized = train_features
        test_features_normalized = test_features

    # Create windows for train
    n_train_samples = len(train_df) - window - target_horizon + 1
    if n_train_samples <= 0:
        raise ValueError(f"Not enough training data for window {window} + horizon {target_horizon}")

    train_idx = np.arange(window)[None, :] + np.arange(n_train_samples)[:, None]
    X_train = train_features_normalized[train_idx]
    y_train = train_target[window + target_horizon - 1:window + target_horizon - 1 + n_train_samples]

    # Create windows for test
    n_test_samples = len(test_df) - window - target_horizon + 1
    if n_test_samples <= 0:
        raise ValueError(f"Not enough test data for window {window} + horizon {target_horizon}")

    test_idx = np.arange(window)[None, :] + np.arange(n_test_samples)[:, None]
    X_test = test_features_normalized[test_idx]
    y_test = test_target[window + target_horizon - 1:window + target_horizon - 1 + n_test_samples]
    test_pred_timestamps = test_timestamps[window + target_horizon - 1:window + target_horizon - 1 + n_test_samples]

    print(f"\nData shapes after windowing:")
    print(f"X_train shape: {X_train.shape} (samples × window × features)")
    print(f"X_test shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[2]}")

    return X_train, y_train, X_test, y_test, test_pred_timestamps, scaler


# ======================================================
# MAIN TRAINING FUNCTION
# ======================================================

def main():
    print("=" * 60)
    print("BNN TRAINING FOR MINIMAL HFD AGGREGATED DATA")
    print("=" * 60)

    # Check if data exists
    if not DATA_FILE.exists():
        print(f"\nERROR: Data file not found: {DATA_FILE}")
        print("Please run hfd_to_1min_minimal.py first to create minimal HFD data.")
        return None, None, None

    # Load data with time-based split and normalization
    print(f"\nLoading minimal HFD data from: {DATA_FILE}")
    X_train, y_train, X_test, y_test, test_timestamps, scaler = load_hfd_minimal_with_split(
        DATA_FILE,
        window=WINDOW,
        target_horizon=TARGET_HORIZON,
        test_hours=TEST_HOURS,
        normalize=True
    )

    # Get feature count from data
    actual_feature_count = X_train.shape[2]
    print(f"\nBuilding BNN model...")
    print(f"Window: {WINDOW}, Features: {actual_feature_count}")
    print(f"Expected features: 4 (log_return_clipped, volatility, activity, tick_count)")

    # Build model with correct feature count
    model = build_bnn_showcase(
        window=WINDOW,
        feature_count=actual_feature_count,
        train_size=len(X_train)
    )

    model.summary()

    # Train model
    print(f"\nTraining model...")
    print(f"Training samples: {len(X_train)}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")

    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=False,  # Time series data
        verbose=1,
        validation_split=0.1  # Use 10% of training for validation
    )

    # Save model and artifacts
    RESULTS_DIR = PROJECT_ROOT / "results" / "bnn_hfd"
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = RESULTS_DIR / f"hfd_minimal_train_{RUN_ID}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_dir = RUN_DIR / "model"
    model.save(model_dir)

    # Save training history
    joblib.dump(history.history, RUN_DIR / "history.pkl")

    # Save test data for prediction
    joblib.dump({
        'X_test': X_test,
        'y_test': y_test,
        'timestamps': test_timestamps
    }, RUN_DIR / "test_data.pkl")

    # Save scaler (CRITICAL for prediction)
    if scaler is not None:
        joblib.dump(scaler, RUN_DIR / "scaler.pkl")
        print(f"Scaler saved for future predictions")

    # Save metadata
    metadata = {
        'run_id': RUN_ID,
        'data_file': str(DATA_FILE),
        'window': WINDOW,
        'target_horizon': TARGET_HORIZON,
        'test_hours': TEST_HOURS,
        'features': ["log_return_clipped", "volatility", "activity", "tick_count"],
        'feature_count': actual_feature_count,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'normalized': True,
        'train_date_range': f"{pd.Timestamp(test_timestamps[0]) - timedelta(hours=TEST_HOURS + WINDOW / 60 + TARGET_HORIZON / 60)} to {pd.Timestamp(test_timestamps[0])}",
        'test_date_range': f"{test_timestamps[0]} to {test_timestamps[-1]}",
        'model_type': 'BNN_HFD_minimal',
        'notes': 'Trained on minimal HFD features (4 features) with normalization'
    }

    joblib.dump(metadata, RUN_DIR / "metadata.pkl")

    # Print summary
    print(f"\n" + "=" * 60)
    print("MINIMAL HFD TRAINING COMPLETED")
    print("=" * 60)
    print(f"Model saved to: {model_dir}")
    print(f"Test data ready for prediction: {len(test_timestamps)} samples")
    print(f"Test period: {test_timestamps[0]} to {test_timestamps[-1]}")
    print(f"Features used: 4 (log_return_clipped, volatility, activity, tick_count)")
    print(f"\nNext steps:")
    print(f"1. Run predict_rolling_bnn_hfd_minimal.py for walk-forward prediction")
    print(f"2. Compare results with baseline 1-minute model")
    print("=" * 60)

    return model, history, RUN_DIR


if __name__ == "__main__":
    model, history, run_dir = main()
    if model is None:
        sys.exit(1)