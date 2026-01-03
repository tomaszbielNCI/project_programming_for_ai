"""
Minimal HFD Loader for BNN
Loads minimal HFD data (mid, tick_count) and creates features for BNN.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_hfd_minimal_for_bnn(
        path: str,
        window: int = 120,
        vol_window: int = 20,
        target_horizon: int = 60,
        return_window_timestamps: bool = False,
        normalize: bool = True
):
    """
    Load minimal HFD data (mid + tick_count) and create features for BNN.
    """
    # Load aggregated HFD data
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded HFD data with columns: {list(df.columns)}")

    # Verify we have required columns
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
    abs_lr = np.abs(df["log_return_clipped"].values)
    df["volatility"] = pd.Series(abs_lr).rolling(vol_window, min_periods=vol_window).std().values
    df["activity"] = pd.Series(abs_lr).rolling(vol_window, min_periods=vol_window).mean().values

    # Drop NaNs from rolling calculations
    df = df.dropna().reset_index(drop=True)

    # Select features - only 4: log_return_clipped, volatility, activity, tick_count
    feature_columns = [
        "log_return_clipped",
        "volatility",
        "activity",
        "tick_count"
    ]

    print(f"Using features: {feature_columns}")

    # Get features and target
    features = df[feature_columns].values.astype("float32")
    target = df["log_return_60min"].values.astype("float32")
    timestamps = df["timestamp"].values

    # NORMALIZE features (if requested)
    scaler = None
    if normalize:
        print("Normalizing features...")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    # Create windows
    n_samples = len(df) - window - target_horizon + 1
    if n_samples <= 0:
        raise ValueError(f"Not enough data for window {window} + horizon {target_horizon}")

    idx = np.arange(window)[None, :] + np.arange(n_samples)[:, None]

    X = features[idx]
    y = target[window + target_horizon - 1: window + target_horizon - 1 + n_samples]

    print(f"Created windows: X shape = {X.shape}, y shape = {y.shape}")

    if return_window_timestamps:
        ts = timestamps[window + target_horizon - 1: window + target_horizon - 1 + n_samples]
        return X, y, ts, scaler

    return X, y, scaler