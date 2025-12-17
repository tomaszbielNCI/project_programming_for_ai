import pandas as pd
import numpy as np


def load_parquet_for_bnn(
    path: str,
    window: int = 20,
    vol_window: int = 20,
    target_horizon: int = 1
):
    """
    Universal loader for Bayesian NN
    Works for:
      - HFD (live MT4)
      - Historical (1m / 5m / 15m / 60m)

    Output:
      X : [samples, window, features]
      y : [samples]
    """

    # --- 1. Load minimum required columns ---
    df = pd.read_parquet(path, columns=["timestamp", "mid"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- 2. Core signal ---
    log_mid = np.log(df["mid"].values)
    log_return = np.diff(log_mid)

    # Align dataframe
    df = df.iloc[1:].reset_index(drop=True)
    df["log_return"] = log_return.astype("float32")

    # --- 3. Volatility & Activity (same semantics everywhere) ---
    abs_lr = np.abs(df["log_return"].values)

    df["volatility"] = (
        pd.Series(abs_lr)
        .rolling(vol_window)
        .std()
        .values
    )

    df["activity"] = (
        pd.Series(abs_lr)
        .rolling(vol_window)
        .mean()
        .values
    )

    # --- 4. Drop warmup NaNs ---
    df = df.dropna().reset_index(drop=True)

    # --- 5. Feature matrix ---
    features = df[["log_return", "volatility", "activity"]].values.astype("float32")
    target = df["log_return"].values.astype("float32")

    # --- 6. NumPy sliding window (NO python loops) ---
    n_samples = len(df) - window - target_horizon + 1
    if n_samples <= 0:
        raise ValueError("Not enough data for given window and horizon")

    idx = np.arange(window)[None, :] + np.arange(n_samples)[:, None]

    X = features[idx]
    y = target[window + target_horizon - 1 : window + target_horizon - 1 + n_samples]

    return X, y
