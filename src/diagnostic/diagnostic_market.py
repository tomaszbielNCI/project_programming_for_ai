"""
DIAGNOSTIC SCRIPT FOR FINANCIAL TIME SERIES
Purpose:
- Academic justification
- Hackathon-ready diagnostics
- Input layer for decision overlays / BNN tuning

Default mode: HFD
Other timeframes (1m, 15m) can be enabled by uncommenting.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
from hurst import compute_Hc
from scipy.stats import t, norm

import warnings
import sys
from contextlib import redirect_stdout
warnings.filterwarnings("ignore")

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for file_obj in self.files:
            file_obj.write(obj)
    def flush(self):
        for file_obj in self.files:
            file_obj.flush()

# ======================================================
# CONFIGURATION
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostic_market"

# ---- ACTIVE DATASET (DEFAULT: HFD) ----
SYMBOL = "US.100"
TIMEFRAME = "HFD"

# ---- OPTIONAL (UNCOMMENT TO RUN) ----
# SYMBOL = "NASDAQ"
# TIMEFRAME = "1m"

# SYMBOL = "NASDAQ"
# TIMEFRAME = "15m"

if TIMEFRAME == "HFD":
    FILE_NAME = f"{SYMBOL}+.parquet"
else:
    if not TIMEFRAME.endswith("m"):
        raise ValueError(f"Unsupported TIMEFRAME: {TIMEFRAME}. Use 'HFD' or '<minutes>m' (e.g. '1m', '15m').")
    minutes = TIMEFRAME[:-1]
    if not minutes.isdigit():
        raise ValueError(f"Unsupported TIMEFRAME: {TIMEFRAME}. Use 'HFD' or '<minutes>m' (e.g. '1m', '15m').")
    FILE_NAME = f"{SYMBOL}+{minutes}.parquet"

RUN_ID = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Create summary file
summary_path = RUN_DIR / "summary.txt"

# Save original stdout
original_stdout = sys.stdout

# Open file with UTF-8 encoding and redirect output
f = open(summary_path, 'w', encoding='utf-8')
sys.stdout = Tee(original_stdout, f)

print(f"\nRUN ID: {RUN_ID}")
print(f"Instrument: {SYMBOL}, Timeframe: {TIMEFRAME}")

# Script will continue writing to both console and file

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_parquet(PARSED_DIR / FILE_NAME)
df = df.sort_values("timestamp")

# ------------------------------------------------------
# GAP DETECTION (PIPELINE INTERRUPTIONS / MARKET CLOSE)
# ------------------------------------------------------

df["dt"] = df["timestamp"].diff().dt.total_seconds()

if TIMEFRAME == "HFD":
    EXPECTED_DT = 1.0      # seconds
    GAP_THRESHOLD = 3.0    # tolerate minor jitter
else:
    EXPECTED_DT = float(TIMEFRAME[:-1]) * 60.0
    GAP_THRESHOLD = EXPECTED_DT * 1.5

df["is_gap"] = df["dt"] > GAP_THRESHOLD

# ------------------------------------------------------
# LOG-RETURNS
# ------------------------------------------------------

df["log_return"] = np.log(df["mid"]).diff()

# Log-return AFTER gap is an artefact → exclude from diagnostics
df["return_valid"] = ~df["is_gap"]

df = df.dropna().reset_index(drop=True)


returns = df.loc[df["return_valid"], "log_return"].values
print(f"Observations (valid returns only): {len(returns)}")
print(f"Excluded returns due to gaps: {df['is_gap'].sum()}")

# ======================================================
# 1. VISUALIZATION — DATA SANITY CHECK
# ======================================================
# Why:
# - Detect outliers, feed interruptions, structural breaks
# - Verify data consistency before any statistical inference

plt.figure(figsize=(12,4))
plt.plot(df["timestamp"], df["mid"])
plt.title(f"{SYMBOL} {TIMEFRAME} – Price (Sanity Check)")
plt.grid(True)
plt.savefig(RUN_DIR / f"{SYMBOL}_{TIMEFRAME}_price.png")
plt.close()

plt.figure(figsize=(12,4))
plt.plot(df["timestamp"], df["log_return"])
plt.title(f"{SYMBOL} {TIMEFRAME} – Log Returns")
plt.grid(True)
plt.savefig(RUN_DIR / f"{SYMBOL}_{TIMEFRAME}_returns.png")
plt.close()

print("1) Visualization completed → sanity of data confirmed visually.")

# ======================================================
# 2. BASIC STATISTICS
# ======================================================
# Why:
# - First-order characterization of distribution
# - Skewness & kurtosis indicate deviation from Gaussian assumptions

mean_r = returns.mean()
std_r = returns.std()
skew_r = stats.skew(returns)
kurt_r = stats.kurtosis(returns, fisher=True)

print("\n2) Basic statistics")
print(f"Mean: {mean_r:.6e}")
print(f"Std: {std_r:.6e}")
print(f"Skewness: {skew_r:.3f}")
print(f"Excess Kurtosis: {kurt_r:.3f}")

# ======================================================
# 3. AUTOCORRELATION FUNCTION (ACF)
# ======================================================
# Why:
# - Microstructure diagnostics
# - Helps choosing window size for models (BNN / rolling features)
# Note:
# - Lag 1 is often dominated by bid-ask bounce in HFD
# - Longer lags inform effective memory horizon

acf_vals = acf(returns, nlags=50)

plt.figure(figsize=(10,4))
plt.stem(range(len(acf_vals)), acf_vals)
plt.title(f"{SYMBOL} {TIMEFRAME} – ACF")
plt.grid(True)
plt.savefig(RUN_DIR / f"{SYMBOL}_{TIMEFRAME}_acf.png")
plt.close()

print("\n3) ACF computed")
print("Interpretation:")
print("- Significant short-lag ACF → microstructure effects")
print("- Decay speed informs window length for models")

# ======================================================
# 4. JARQUE–BERA TEST
# ======================================================
# Why:
# - Formal test of normality
# - Expected: extreme rejection for HFD, strong rejection for 1m/15m
# - Justifies heavy-tailed distributions (Student-t, Stable)

jb_stat, jb_p, _, _ = jarque_bera(returns)

print("\n4) Jarque–Bera Test")
print(f"JB statistic: {jb_stat:.2f}, p-value: {jb_p:.2e}")

if jb_p < 0.01:
    print("Result: Strong rejection of normality → heavy tails confirmed")
else:
    print("Result: Normality not rejected (unlikely for market data)")

# ======================================================
# 5. ARCH EFFECT (VOLATILITY CLUSTERING)
# ======================================================
# Why:
# - Detect conditional heteroskedasticity
# - Supports need for dynamic volatility tunnel
# - Justifies adaptive std in BNN / decision layer

arch_stat, arch_p, _, _ = het_arch(returns, nlags=10)

print("\n5) ARCH Test")
print(f"ARCH stat: {arch_stat:.2f}, p-value: {arch_p:.2e}")

if arch_p < 0.01:
    print("Result: Strong volatility clustering detected")
elif arch_p < 0.05:
    print("Result: Moderate volatility clustering detected")
else:
    print("Result: No significant ARCH effect")

# ======================================================
# 6. HURST EXPONENT
# ======================================================
# Why:
# - Regime detection
# - <0.5 mean-reverting → contrarian strategies
# - ~0.5 random walk → no directional edge

H, _, _ = compute_Hc(returns, kind="change", simplified=True)

print("\n6) Hurst Exponent")
print(f"H = {H:.3f}")

if H < 0.45:
    print("Interpretation: Mean-reverting regime → contrarian overlays justified")
elif H > 0.55:
    print("Interpretation: Trending regime → momentum bias possible")
else:
    print("Interpretation: Near-random walk")

# ======================================================
# 7. DISTRIBUTION FITTING (STUDENT-T vs NORMAL)
# ======================================================
# Why:
# - Parameter extraction for decision/agent layer
# - Goal: reject normal, retain heavy-tailed alternative

mu_n, sigma_n = norm.fit(returns)
df_t, loc_t, scale_t = t.fit(returns)

print("\n7) Distribution fitting")
print(f"Normal: mu={mu_n:.3e}, sigma={sigma_n:.3e}")
print(f"Student-t: df={df_t:.2f}, loc={loc_t:.3e}, scale={scale_t:.3e}")

# ======================================================
# 8. HISTOGRAM + FITTED PDF
# ======================================================
# Why:
# - Visual confirmation of tail behavior
# - Communication-ready artifact (hackathon / report)

x = np.linspace(returns.min(), returns.max(), 1000)

plt.figure(figsize=(12,5))
plt.hist(returns, bins=100, density=True, alpha=0.5, label="Empirical")
plt.plot(x, norm.pdf(x, mu_n, sigma_n), label="Normal")
plt.plot(x, t.pdf(x, df_t, loc_t, scale_t), label="Student-t")
plt.legend()
plt.title(f"{SYMBOL} {TIMEFRAME} – Distribution Fit")
plt.grid(True)
plt.savefig(RUN_DIR / f"{SYMBOL}_{TIMEFRAME}_distfit.png")
plt.close()

# ======================================================
# 9. REPORT EXPORT
# ======================================================
# Why:
# - Persistent diagnostics
# - Direct input to Flask / hackathon UI
# - Traceability of assumptions

report = pd.DataFrame({
    "metric": [
        "mean", "std", "skewness", "excess_kurtosis",
        "JB_stat", "JB_p",
        "ARCH_stat", "ARCH_p",
        "Hurst",
        "t_df", "t_loc", "t_scale"
    ],
    "value": [
        mean_r, std_r, skew_r, kurt_r,
        jb_stat, jb_p,
        arch_stat, arch_p,
        H,
        df_t, loc_t, scale_t
    ]
})

report_path = RUN_DIR / f"{SYMBOL}_{TIMEFRAME}_report.csv"
report.to_csv(report_path, index=False)

print("\nDiagnostics complete.")
print(f"Report saved to: {report_path}")
print(f"Summary saved to: {summary_path}")

# Close the file and restore original stdout
sys.stdout = original_stdout
f.close()
