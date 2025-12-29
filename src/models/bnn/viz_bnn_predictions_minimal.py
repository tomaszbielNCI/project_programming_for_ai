import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from scipy.stats import norm
# ======================================================
# CONFIG
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_DIR = PROJECT_ROOT / "results" / "bnn" / "predict_v2"
DATA_FILE = PROJECT_ROOT / "data" / "parsed" / "US.100+1.parquet"

# ======================================================
# LOAD LATEST PREDICTIONS
# ======================================================

pred_file = sorted(PREDICTIONS_DIR.glob("rolling_predictions_*.pkl"))[-1]
pred = joblib.load(pred_file)

pred_ts = pd.to_datetime(pred["timestamps"])
pred_mean = np.array(pred["mean"])
pred_std = np.array(pred["std"])

print(f"Loaded predictions: {pred_ts[0]} → {pred_ts[-1]} ({len(pred_ts)})")

# ======================================================
# LOAD MARKET DATA
# ======================================================

market = pd.read_parquet(DATA_FILE)
market["timestamp"] = pd.to_datetime(market["timestamp"])
market = market.set_index("timestamp").sort_index()

# log-returns from mid
log_mid = np.log(market["mid"].values)
market["log_return"] = np.zeros_like(log_mid)
market["log_return"][1:] = log_mid[1:] - log_mid[:-1]
market = market.iloc[1:]  # Drop first row with NaN
# align strictly to prediction timestamps
actual = (
    market["log_return"]
    .reindex(pred_ts, method="nearest")
)

# ======================================================
# SANITY CHECK
# ======================================================

assert len(actual) == len(pred_mean), "Timestamp misalignment!"

# ======================================================
# VISUALIZATION
# ======================================================

plt.figure(figsize=(14, 6))

plt.plot(pred_ts, actual, label="Actual log-return", color="black", alpha=0.6)
plt.plot(pred_ts, pred_mean, label="Predicted mean", color="blue")

plt.fill_between(
    pred_ts,
    pred_mean - pred_std,
    pred_mean + pred_std,
    color="blue",
    alpha=0.25,
    label="±1σ"
)

plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)

plt.title("BNN Walk-Forward Prediction (OOS)")
plt.xlabel("Time")
plt.ylabel("Log-return")

plt.legend()
plt.tight_layout()
plt.show()

# ======================================================
# QUICK DIAGNOSTICS (MDPI-style sanity)
# ======================================================

print("\n=== Diagnostics ===")
print(f"Mean actual:    {actual.mean():.6f}")
print(f"Mean predicted: {pred_mean.mean():.6f}")
print(f"Std actual:     {actual.std():.6f}")
print(f"Std predicted:  {pred_std.mean():.6f}")
print(f"Correlation:    {np.corrcoef(actual, pred_mean)[0,1]:.4f}")


# ======================================================
# CALCULATE CALIBRATION METRICS
# ======================================================

def calculate_calibration(actual, pred_mean, pred_std, n_bins=10):
    """Calculate calibration metrics for probabilistic predictions."""
    # Standardized residuals
    z_scores = (actual - pred_mean) / (pred_std + 1e-8)

    # Calculate observed vs expected frequencies
    bin_edges = np.linspace(0, 3, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    expected = []
    observed = []

    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        expected_frac = (norm.cdf(upper) - norm.cdf(lower)) * 2  # Two-tailed
        actual_frac = np.mean((np.abs(z_scores) >= lower) & (np.abs(z_scores) < upper))

        expected.append(expected_frac)
        observed.append(actual_frac)

    # Calculate calibration score (1 - mean absolute calibration error)
    calibration_score = 1 - np.mean(np.abs(np.array(observed) - np.array(expected)))

    # Calculate sharpness (lower is better)
    sharpness = np.mean(pred_std)

    return {
        'z_scores': z_scores,
        'bin_centers': bin_centers,
        'expected': expected,
        'observed': observed,
        'calibration_score': calibration_score,
        'sharpness': sharpness,
        'within_1std': np.mean(np.abs(z_scores) <= 1),
        'within_2std': np.mean(np.abs(z_scores) <= 2)
    }


# Calculate calibration metrics
calib_metrics = calculate_calibration(actual, pred_mean, pred_std)

# ======================================================
# ENHANCED VISUALIZATION
# ======================================================

plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

# 1. Time series plot
ax0 = plt.subplot(gs[0])
ax0.plot(pred_ts, actual, 'k-', label="Actual Returns", alpha=0.8, linewidth=1)
ax0.plot(pred_ts, pred_mean, 'b-', label="Predicted Mean", alpha=0.8)
ax0.fill_between(
    pred_ts,
    pred_mean - pred_std,
    pred_mean + pred_std,
    color='blue', alpha=0.2, label="±1 Std Dev"
)
ax0.fill_between(
    pred_ts,
    pred_mean - 2 * pred_std,
    pred_mean + 2 * pred_std,
    color='blue', alpha=0.1, label="±2 Std Dev"
)
ax0.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax0.set_title("BNN Probabilistic Predictions")
ax0.set_ylabel("Log Returns")
ax0.legend(loc='upper right')
ax0.grid(True, alpha=0.3)

# 2. Standardized residuals
ax1 = plt.subplot(gs[1], sharex=ax0)
z_scores = (actual - pred_mean) / (pred_std + 1e-8)
ax1.scatter(pred_ts, z_scores, s=10, alpha=0.6, c='r')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(1, color='gray', linestyle=':', alpha=0.5)
ax1.axhline(-1, color='gray', linestyle=':', alpha=0.5)
ax1.set_ylabel("Standardized\nResiduals")
ax1.grid(True, alpha=0.3)

# 3. Calibration plot
ax2 = plt.subplot(gs[2])
ax2.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
ax2.bar(
    calib_metrics['bin_centers'] - 0.05,
    calib_metrics['observed'],
    width=0.1,
    alpha=0.7,
    label=f"Observed (Calib Score: {calib_metrics['calibration_score']:.2f})"
)
ax2.set_xlabel("|Z-score|")
ax2.set_ylabel("Observed Frequency")
ax2.set_title("Calibration Plot")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# ======================================================
# PRINT DIAGNOSTICS
# ======================================================

print("\n" + "=" * 50)
print("BNN PREDICTION DIAGNOSTICS")
print("=" * 50)

print("\n--- Basic Statistics ---")
print(f"Mean actual return:    {actual.mean():.6f}")
print(f"Mean predicted return: {pred_mean.mean():.6f}")
print(f"Actual volatility:     {actual.std():.6f}")
print(f"Mean predicted std:    {pred_std.mean():.6f}")
print(f"Correlation:           {np.corrcoef(actual, pred_mean)[0, 1]:.4f}")

print("\n--- Calibration ---")
print(f"Calibration score:     {calib_metrics['calibration_score']:.4f} (1.0 is perfect)")
print(f"Within ±1 std:         {calib_metrics['within_1std'] * 100:.1f}% (expected ~68%)")
print(f"Within ±2 std:         {calib_metrics['within_2std'] * 100:.1f}% (expected ~95%)")
print(f"Sharpness (avg std):   {calib_metrics['sharpness']:.6f} (lower is better)")

print("\n" + "=" * 50 + "\n")

plt.show()
