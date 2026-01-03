"""
Walk-forward prediction for Minimal HFD BNN model
"""
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.models.bnn.model_v2 import build_bnn_showcase

WINDOW = 120
TARGET_HORIZON = 60

def get_latest_hfd_model_dir():
    bnn_dir = PROJECT_ROOT / "results" / "bnn_hfd"
    model_dirs = sorted(
        [d for d in bnn_dir.glob("hfd_minimal_train_*") if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if not model_dirs:
        raise FileNotFoundError("No MINIMAL HFD trained model found")
    return model_dirs[0]

def main():
    print("=" * 60)
    print("MINIMAL HFD BNN PREDICTION WITH NORMALIZATION")
    print("=" * 60)

    # Find latest MINIMAL model
    LATEST = get_latest_hfd_model_dir()
    MODEL_DIR = LATEST / "model"
    SCALER_FILE = LATEST / "scaler.pkl"
    TEST_DATA_FILE = LATEST / "test_data.pkl"
    META_FILE = LATEST / "metadata.pkl"

    print(f"Using MINIMAL model from: {LATEST}")

    # Load metadata
    metadata = joblib.load(META_FILE)
    feature_count = metadata['feature_count']

    # Load test data (already normalized during training)
    test_data = joblib.load(TEST_DATA_FILE)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    test_timestamps = test_data['timestamps']

    print(f"\nLoaded NORMALIZED test data: {X_test.shape}")
    print(f"Features: {feature_count}, Test period: {test_timestamps[0]} to {test_timestamps[-1]}")

    # Verify normalization
    print(f"\nX_test normalized stats:")
    print(f"  Min: {X_test.min():.6f}, Max: {X_test.max():.6f}")
    print(f"  Mean: {X_test.mean():.6f}, Std: {X_test.std():.6f}")

    # Load model
    model = build_bnn_showcase(
        window=WINDOW,
        feature_count=feature_count,
        train_size=len(X_test)
    )

    # Load weights
    model.load_weights(str(MODEL_DIR / "variables" / "variables"))

    # Predict
    print(f"\nMaking predictions...")
    predicted_means, predicted_stds = [], []

    for i in range(len(X_test)):
        X_sample = X_test[i:i + 1]
        dist = model(X_sample)
        predicted_means.append(dist.mean().numpy().flatten()[0])
        predicted_stds.append(dist.stddev().numpy().flatten()[0])

    # Calculate metrics
    predicted_means = np.array(predicted_means)
    predicted_stds = np.array(predicted_stds)
    actual_returns = np.array(y_test)

    rmse = np.sqrt(np.mean((predicted_means - actual_returns) ** 2))
    z_scores = (actual_returns - predicted_means) / (predicted_stds + 1e-8)
    within_1std = np.mean(np.abs(z_scores) <= 1)
    within_2std = np.mean(np.abs(z_scores) <= 2)

    print(f"\nPrediction Metrics (MINIMAL HFD):")
    print(f"RMSE: {rmse:.6f}")
    print(f"Within ±1 std: {within_1std * 100:.1f}%")
    print(f"Within ±2 std: {within_2std * 100:.1f}%")
    print(f"Avg predicted std: {predicted_stds.mean():.6f}")
    print(f"Actual returns std: {actual_returns.std():.6f}")

    # Save predictions
    output_dir = PROJECT_ROOT / "results" / "bnn_hfd" / "predictions_minimal"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"hfd_minimal_predictions_{timestamp}.pkl"

    joblib.dump({
        'timestamps': test_timestamps,
        'predicted_means': predicted_means,
        'predicted_stds': predicted_stds,
        'actual_returns': actual_returns,
        'rmse': rmse,
        'model_dir': str(LATEST)
    }, output_file)

    print(f"\nPredictions saved to: {output_file}")
    return predicted_means, predicted_stds

if __name__ == "__main__":
    main()