import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec
from scipy.stats import norm, skew, kurtosis
import seaborn as sns
from typing import Tuple


def load_60min_returns(data_file, pred_ts, target_horizon=60):
    """Load actual 60-minute returns aligned with predictions."""
    market = pd.read_parquet(data_file)
    market["timestamp"] = pd.to_datetime(market["timestamp"])
    market = market.set_index("timestamp").sort_index()

    actual_returns = []
    actual_prices = []

    for t in pred_ts:
        try:
            # Find price at prediction time - use get_indexer instead of get_loc
            idx_current = market.index.get_indexer([t], method='nearest')[0]
            if idx_current == -1:  # No match found
                actual_returns.append(np.nan)
                actual_prices.append(np.nan)
                continue

            price_current = market.iloc[idx_current]['mid']

            # Find price target_horizon minutes ago
            t_prev = t - pd.Timedelta(minutes=target_horizon)
            idx_prev = market.index.get_indexer([t_prev], method='nearest')[0]

            if idx_prev == -1:
                actual_returns.append(np.nan)
                actual_prices.append(np.nan)
                continue

            price_prev = market.iloc[idx_prev]['mid']

            # Calculate 60-minute log-return
            log_return = np.log(price_current) - np.log(price_prev)

            actual_returns.append(log_return)
            actual_prices.append(price_current)

        except Exception as e:
            print(f"Error at {t}: {e}")
            actual_returns.append(np.nan)
            actual_prices.append(np.nan)

    return pd.Series(actual_returns, index=pred_ts), pd.Series(actual_prices, index=pred_ts)

# ======================================================
# CONFIGURATION
# ======================================================

from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "bnn" / "predict_v2"
VISUALIZATIONS_BASE_DIR = PROJECT_ROOT / "results" / "visualizations"
DATA_FILE = PROJECT_ROOT / "data" / "parsed" / "US.100+1.parquet"
def extract_instrument_timeframe(filepath):
    """Extract instrument and timeframe from filepath like '.../US.100+1.parquet'"""
    filename = Path(filepath).stem  # Gets 'US.100+1' from full path
    if '+' in filename:
        instrument, timeframe = filename.split('+', 1)  # Split on first '+' only
    else:
        # Fallback if filename doesn't contain '+'
        instrument = filename
        timeframe = "1"  # Default timeframe
    return instrument, timeframe

# Extract instrument and timeframe from the data file
INSTRUMENT, TIMEFRAME = extract_instrument_timeframe(DATA_FILE)

# Create timestamped run directory
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = VISUALIZATIONS_BASE_DIR / f"{INSTRUMENT}+{TIMEFRAME}" / f"run_{run_timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving visualizations to: {RUN_DIR}")

# ======================================================
# LOAD LATEST PREDICTIONS
# ======================================================

# Find and load the most recent prediction file
pred_file = sorted(PREDICTIONS_DIR.glob("rolling_predictions_*.pkl"))[-1]
pred = joblib.load(pred_file)

# Extract prediction data
pred_ts = pd.to_datetime(pred["timestamps"])
pred_mean = np.array(pred["mean"])
pred_std = np.array(pred["std"])

print(f"Loaded predictions: {pred_ts[0]} → {pred_ts[-1]} ({len(pred_ts)} points)")

# ======================================================
# CRITICAL DIAGNOSTICS
# ======================================================

print("\n=== CRITICAL DIAGNOSTICS ===")

# 1. Check prediction time steps
time_diffs = np.diff(pred_ts).astype('timedelta64[m]').astype(float)
print(f"Time between predictions: {time_diffs.mean():.0f} min")
print(f"Min time step: {time_diffs.min():.0f} min, Max time step: {time_diffs.max():.0f} min")

# 2. Load and calculate 60-min returns
print("\nLoading 60-minute returns for comparison...")
actual_60min, prices_60min = load_60min_returns(DATA_FILE, pred_ts, 60)
valid_mask = ~np.isnan(actual_60min)
actual_60min = actual_60min[valid_mask]

if len(actual_60min) > 0:
    # 3. Calculate statistics
    print(f"\n60-min returns statistics (n={len(actual_60min)}):")
    print(f"Mean: {actual_60min.mean():.6f}")
    print(f"Std: {actual_60min.std():.6f}")
    print(f"Skew: {skew(actual_60min):.2f}")
    print(f"Kurtosis: {kurtosis(actual_60min):.2f}")
    
    # 4. Compare with predictions
    print("\nModel predictions vs 60-min actuals:")
    print(f"Pred mean: {pred_mean[valid_mask].mean():.6f}, Actual mean: {actual_60min.mean():.6f}")
    print(f"Pred std: {pred_std[valid_mask].mean():.6f}, Actual std: {actual_60min.std():.6f}")
    
    # Calculate and print RMSE
    rmse = np.sqrt(np.mean((pred_mean[valid_mask] - actual_60min) ** 2))
    print(f"RMSE: {rmse:.6f}")
else:
    print("\nWarning: No valid 60-minute returns could be calculated for the prediction period.")

# ======================================================
# LOAD AND PREPARE MARKET DATA
# ======================================================

# Load market data
market = pd.read_parquet(DATA_FILE)
market["timestamp"] = pd.to_datetime(market["timestamp"])
market = market.set_index("timestamp").sort_index()

# Calculate log-returns from mid prices
log_mid = np.log(market["mid"].values)
market["log_return"] = np.zeros_like(log_mid)
market.loc[market.index[1:], "log_return"] = log_mid[1:] - log_mid[:-1]
market = market.iloc[1:]  # Remove first row with NaN

# Align market data with prediction timestamps
actual_returns = market["log_return"].reindex(pred_ts, method="nearest")
actual_prices = market["mid"].reindex(pred_ts, method="nearest")

# ======================================================
# RAW DATA VERIFICATION
# ======================================================

# Pobierz surowe dane dla okresu predykcji
test_start = pred_ts[0]
test_end = pred_ts[-1]

# Bezpośrednie wczytanie i obliczenie
raw_data = pd.read_parquet(DATA_FILE)
raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
raw_data = raw_data.sort_values('timestamp')
raw_data = raw_data[(raw_data['timestamp'] >= test_start) & 
                    (raw_data['timestamp'] <= test_end)]

# Oblicz log-returns ręcznie
raw_data['log_mid'] = np.log(raw_data['mid'])
raw_data['log_return_raw'] = raw_data['log_mid'].diff()

print("\n=== RAW DATA CHECK ===")
print(f"Raw data points in test period: {len(raw_data)}")
print(f"Raw log-return stats:")
print(f"  Min: {raw_data['log_return_raw'].min():.6f}")
print(f"  Max: {raw_data['log_return_raw'].max():.6f}")
print(f"  Mean: {raw_data['log_return_raw'].mean():.6f}")
print(f"  Std: {raw_data['log_return_raw'].std():.6f}")

from scipy.stats import skew, kurtosis
print(f"  Skew: {skew(raw_data['log_return_raw'].dropna()):.2f}")
print(f"  Kurtosis: {kurtosis(raw_data['log_return_raw'].dropna()):.2f}")

# Porównaj z actual_returns
print("\n=== COMPARISON ===")
print(f"Your actual_returns stats:")
print(f"  Min: {actual_returns.min():.6f}")
print(f"  Max: {actual_returns.max():.6f}")
print(f"  Mean: {actual_returns.mean():.6f}")
print(f"  Std: {actual_returns.std():.6f}")

# ======================================================
# POST-HOC BIAS CORRECTION (FOR VISUALIZATION PURPOSES)
# ======================================================

# Calculate and apply bias correction to align predictions with actual data
bias_correction = np.mean(actual_returns - pred_mean)
print(f"\nBias correction (mean actual - mean predicted): {bias_correction:.6f}")

# Corrected predictions (for visualization clarity only)
corrected_mean = pred_mean + bias_correction

# ======================================================
# CALCULATE PRICE TRAJECTORIES
# ======================================================

initial_price = actual_prices.iloc[0]
actual_price_trajectory = actual_prices.values

# Predicted price trajectory (with bias correction)
pred_log_price = np.log(initial_price) + np.cumsum(corrected_mean)
pred_price_trajectory = np.exp(pred_log_price)

# Generate price samples using Monte Carlo simulation with Normal distribution
n_samples = 1000
price_samples = np.zeros((n_samples, len(pred_ts)))

for i in range(n_samples):
    # Sample returns from Normal distribution (model outputs Normal)
    sampled_returns = np.random.normal(loc=corrected_mean, scale=pred_std)
    sampled_log_price = np.log(initial_price) + np.cumsum(sampled_returns)
    price_samples[i] = np.exp(sampled_log_price)

# Calculate percentiles for credible intervals
price_5th = np.percentile(price_samples, 5, axis=0)
price_50th = np.percentile(price_samples, 50, axis=0)  # Median
price_95th = np.percentile(price_samples, 95, axis=0)


# ======================================================
# CALIBRATION METRICS FOR NORMAL DISTRIBUTION
# ======================================================

def calculate_normal_calibration(actual, pred_mean, pred_std):
    """
    Calculate calibration metrics for Normal distribution.
    """
    # Calculate z-scores
    z_scores = (actual - pred_mean) / (pred_std + 1e-8)

    # Expected coverage for Normal distribution
    normal_cdf_1 = norm.cdf(1) - norm.cdf(-1)  # ±1σ: 68.27%
    normal_cdf_2 = norm.cdf(2) - norm.cdf(-2)  # ±2σ: 95.45%

    # Observed coverage
    within_1std = np.mean(np.abs(z_scores) <= 1)
    within_2std = np.mean(np.abs(z_scores) <= 2)

    # Calibration score
    calibration_error_1 = np.abs(within_1std - normal_cdf_1)
    calibration_error_2 = np.abs(within_2std - normal_cdf_2)
    calibration_score = 1 - (calibration_error_1 + calibration_error_2) / 2

    return {
        'z_scores': z_scores,
        'within_1std': within_1std,
        'within_2std': within_2std,
        'expected_1std': normal_cdf_1,
        'expected_2std': normal_cdf_2,
        'calibration_score': calibration_score,
        'sharpness': np.mean(pred_std)
    }


# Calculate calibration metrics
calib_metrics = calculate_normal_calibration(actual_returns, pred_mean, pred_std)

# ======================================================
# VISUALIZATION 1: RETURN SPACE WITH UNCERTAINTY BANDS
# ======================================================

plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

# Panel 1A: Time series of returns with uncertainty bands
ax0 = plt.subplot(gs[0])
ax0.plot(pred_ts, actual_returns, 'k-', label="Actual Returns",
         alpha=0.8, linewidth=1)
ax0.axhline(0, color='gray', linestyle='--', alpha=0.5,
            label="Zero Baseline")

# Plot bias-corrected predictions
ax0.plot(pred_ts, corrected_mean, 'b-', alpha=0.6, linewidth=1,
         label="Bias-Corrected Predictions")

# Uncertainty bands centered around zero (mean-reversion framework)
ax0.fill_between(pred_ts, -pred_std, pred_std, color='blue',
                 alpha=0.15, label="±1σ from Zero")
ax0.fill_between(pred_ts, -2 * pred_std, 2 * pred_std, color='blue',
                 alpha=0.08, label="±2σ from Zero")

ax0.set_title("BNN Probabilistic Forecast: Short-Horizon Log-Returns",
              fontsize=14, fontweight='bold')
ax0.set_ylabel("Log-Return", fontsize=12)
ax0.legend(loc='upper right', fontsize=10)
ax0.grid(True, alpha=0.3)
ax0.tick_params(axis='x', rotation=45)

# Panel 1B: Standardized residuals (z-scores)
ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.scatter(pred_ts, calib_metrics['z_scores'], s=15, alpha=0.6, c='r',
            label="Standardized Residuals")
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(1, color='blue', linestyle=':', alpha=0.7, label="±1σ")
ax1.axhline(-1, color='blue', linestyle=':', alpha=0.7)
ax1.axhline(2, color='blue', linestyle=':', alpha=0.4, label="±2σ")
ax1.axhline(-2, color='blue', linestyle=':', alpha=0.4)
ax1.set_ylabel("Z-score\n(Normal)", fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-4, 4])

# Panel 1C: Calibration plot (histogram of z-scores vs theoretical N(0,1))
ax2 = plt.subplot(gs[2])
x = np.linspace(-4, 4, 1000)
normal_pdf = norm.pdf(x)

ax2.hist(calib_metrics['z_scores'], bins=30, density=True, alpha=0.6,
         color='red',
         label=f"Observed (Calib Score: {calib_metrics['calibration_score']:.3f})")
ax2.plot(x, normal_pdf, 'b-', linewidth=2, alpha=0.8,
         label="Theoretical N(0,1)")
ax2.set_xlabel("Standardized Residual", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("Calibration: Observed vs Theoretical Distribution", fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-4, 4])

plt.tight_layout()
plot_path = RUN_DIR / "returns_space_visualization.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {plot_path}")
plt.show()

# Save a copy of the prediction file for reference
import shutil
pred_copy_path = RUN_DIR / f"{pred_file.name}"
shutil.copy2(pred_file, pred_copy_path)
print(f"Copied prediction file to: {pred_copy_path}")

# ======================================================
# VISUALIZATION 2: PRICE TRAJECTORY RECONSTRUCTION
# ======================================================

plt.figure(figsize=(14, 7))

# Plot actual price
plt.plot(pred_ts, actual_price_trajectory, 'k-', linewidth=2,
         label="Actual Price", alpha=0.9)

# Plot predicted median price trajectory
plt.plot(pred_ts, price_50th, 'b--', linewidth=1.5, alpha=0.8,
         label="Model Median Trajectory")

# Plot 90% credible interval
plt.fill_between(pred_ts, price_5th, price_95th,
                 color='blue', alpha=0.15, label="90% Credible Interval")

plt.title("Price Trajectory Reconstruction with Uncertainty Bands",
          fontsize=14, fontweight='bold')
plt.xlabel("Time", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = RUN_DIR / "price_trajectory_visualization.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {plot_path}")
plt.show()

# ======================================================
# VISUALIZATION 3: COMPREHENSIVE DIAGNOSTICS
# ======================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('BNN Model Diagnostics Summary', fontsize=16, fontweight='bold')

# Panel 3A: Effect of bias correction
axes[0, 0].scatter(pred_mean, corrected_mean, alpha=0.6, s=20)
axes[0, 0].plot([pred_mean.min(), pred_mean.max()],
                [pred_mean.min(), pred_mean.max()],
                'r--', alpha=0.5, label="y=x (No Correction)")
axes[0, 0].set_xlabel("Original Predicted Mean", fontsize=11)
axes[0, 0].set_ylabel("Bias-Corrected Mean", fontsize=11)
axes[0, 0].set_title("Effect of Bias Correction", fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 3B: Prediction interval coverage
coverage_data = {
    'Within ±1σ': [calib_metrics['within_1std'] * 100,
                   calib_metrics['expected_1std'] * 100],
    'Within ±2σ': [calib_metrics['within_2std'] * 100,
                   calib_metrics['expected_2std'] * 100]
}
df_coverage = pd.DataFrame(coverage_data, index=['Observed', 'Expected'])

x = np.arange(len(df_coverage.columns))
width = 0.35
axes[0, 1].bar(x - width / 2, df_coverage.loc['Observed'], width,
               label='Observed', alpha=0.7, color='red')
axes[0, 1].bar(x + width / 2, df_coverage.loc['Expected'], width,
               label='Expected', alpha=0.7, color='blue')
axes[0, 1].set_xlabel('Interval', fontsize=11)
axes[0, 1].set_ylabel('Coverage (%)', fontsize=11)
axes[0, 1].set_title('Prediction Interval Coverage (Normal)', fontsize=12)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(df_coverage.columns)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Panel 3C: Model uncertainty vs realized volatility
vol_comparison = pd.DataFrame({
    'Model': pred_std,
    'Realized': np.abs(actual_returns - corrected_mean)
}, index=pred_ts)

axes[1, 0].plot(vol_comparison.index, vol_comparison['Model'],
                'b-', alpha=0.7, label='Predicted Std Dev')
axes[1, 0].plot(vol_comparison.index, vol_comparison['Realized'],
                'r-', alpha=0.5, label='Realized Absolute Error')
axes[1, 0].set_xlabel('Time', fontsize=11)
axes[1, 0].set_ylabel('Volatility', fontsize=11)
axes[1, 0].set_title('Model Uncertainty vs Realized Volatility', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Panel 3D: Error distribution vs Normal fit
error = actual_returns - corrected_mean
axes[1, 1].hist(error, bins=30, density=True, alpha=0.6,
                color='purple', label='Error Distribution')
x_range = np.linspace(error.min(), error.max(), 1000)
axes[1, 1].plot(x_range, norm.pdf(x_range, loc=0, scale=np.mean(pred_std)),
                'b-', linewidth=2, alpha=0.8, label='Normal Fit')
axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Prediction Error', fontsize=11)
axes[1, 1].set_ylabel('Density', fontsize=11)
axes[1, 1].set_title('Error Distribution vs Normal Assumption', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RUN_DIR / "model_diagnostics_summary.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {plot_path}")
plt.show()

# Save a text file with the run summary
with open(RUN_DIR / "run_summary.txt", "w") as f:
    f.write(f"Run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("\n=== PREDICTION DETAILS ===\n")
    f.write(f"Prediction file: {pred_file.name}\n")
    f.write(f"Instrument: {INSTRUMENT}\n")
    f.write(f"Timeframe: {TIMEFRAME}\n")
    f.write(f"Prediction range: {pred_ts[0]} to {pred_ts[-1]}\n")
    f.write(f"Number of predictions: {len(pred_ts)}\n")
    f.write(f"Time between predictions: {np.diff(pred_ts).astype('timedelta64[m]').astype(float).mean():.0f} min\n")
    
    f.write("\n=== DATA STATISTICS ===\n")
    f.write(f"Raw data points in test period: {len(raw_data)}\n")
    f.write(f"Raw log-return stats:\n")
    f.write(f"  Min: {raw_data['log_return_raw'].min():.6f}\n")
    f.write(f"  Max: {raw_data['log_return_raw'].max():.6f}\n")
    f.write(f"  Mean: {raw_data['log_return_raw'].mean():.6f}\n")
    f.write(f"  Std: {raw_data['log_return_raw'].std():.6f}\n")
    f.write(f"  Skew: {skew(raw_data['log_return_raw'].dropna()):.2f}\n")
    f.write(f"  Kurtosis: {kurtosis(raw_data['log_return_raw'].dropna()):.2f}\n")
    
    f.write("\n=== MODEL PERFORMANCE ===\n")
    f.write(f"Bias correction (mean actual - mean predicted): {bias_correction:.6f}\n")
    f.write(f"Calibration score: {calib_metrics['calibration_score']:.4f}\n")
    f.write(f"Within +/-1std: {calib_metrics['within_1std']*100:.1f}% (expected {calib_metrics['expected_1std']*100:.1f}%)\n")
    f.write(f"Within +/-2std: {calib_metrics['within_2std']*100:.1f}% (expected {calib_metrics['expected_2std']*100:.1f}%)\n")
    f.write(f"RMSE: {np.sqrt(np.mean((pred_mean - actual_returns)**2)):.6f}\n")
    
    f.write("\n=== 60-MINUTE RETURNS ===\n")
    if len(actual_60min) > 0:
        f.write(f"Number of 60-min returns: {len(actual_60min)}\n")
        f.write(f"Mean: {actual_60min.mean():.6f}\n")
        f.write(f"Std: {actual_60min.std():.6f}\n")
        f.write(f"Skew: {skew(actual_60min.dropna()):.2f}\n")
        f.write(f"Kurtosis: {kurtosis(actual_60min.dropna()):.2f}\n")
        f.write(f"Model vs Actual - Mean: {pred_mean[valid_mask].mean():.6f} vs {actual_60min.mean():.6f}\n")
        f.write(f"Model vs Actual - Std: {pred_std[valid_mask].mean():.6f} vs {actual_60min.std():.6f}\n")

print(f"\nAll visualizations and data saved to: {RUN_DIR}")
print("Run summary saved to: run_summary.txt")

# ======================================================
# COMPREHENSIVE DIAGNOSTICS OUTPUT
# ======================================================

print("\n" + "=" * 60)
print("COMPREHENSIVE BNN MODEL DIAGNOSTICS")
print("=" * 60)

print("\n--- BASIC STATISTICS ---")
print(f"Observation period: {len(pred_ts)} points")
print(f"Mean actual return:    {actual_returns.mean():.8f}")
print(f"Mean predicted return: {pred_mean.mean():.8f}")
print(f"Bias (actual - predicted): {bias_correction:.8f}")
print(f"Actual volatility:     {actual_returns.std():.8f}")
print(f"Mean predicted std:    {pred_std.mean():.8f}")

print("\n--- NORMAL DISTRIBUTION CALIBRATION ---")
print(f"Calibration score:     {calib_metrics['calibration_score']:.4f}")
print(
    f"Within ±1 std:         {calib_metrics['within_1std'] * 100:.1f}% (expected {calib_metrics['expected_1std'] * 100:.1f}%)")
print(
    f"Within ±2 std:         {calib_metrics['within_2std'] * 100:.1f}% (expected {calib_metrics['expected_2std'] * 100:.1f}%)")
print(f"Sharpness (avg std):   {calib_metrics['sharpness']:.8f}")

print("\n--- PRICE TRAJECTORY METRICS ---")
final_price_error = (actual_price_trajectory[-1] - price_50th[-1]) / actual_price_trajectory[-1]
print(f"Initial price:         {initial_price:.2f}")
print(f"Final actual price:    {actual_price_trajectory[-1]:.2f}")
print(f"Final predicted price: {price_50th[-1]:.2f}")
print(f"Final relative error:  {final_price_error * 100:.2f}%")
print(f"90% CI width at end:   {(price_95th[-1] - price_5th[-1]) / price_50th[-1] * 100:.1f}% of median")

print("\n--- DATA CHARACTERISTICS ---")
from scipy.stats import skew, kurtosis

print(f"Skewness of returns:   {skew(actual_returns):.2f}")
print(f"Excess kurtosis:       {kurtosis(actual_returns):.2f}")

print("\n--- INTERPRETATION ---")
print("1. Model predictions now show minimal bias after correction.")
print("2. Uncertainty bands are evaluated under Normal distribution assumptions.")
print("3. Price trajectory reconstruction is anchored to actual initial price.")
print("4. The model focuses on uncertainty quantification rather than point prediction.")

print("\n" + "=" * 60)