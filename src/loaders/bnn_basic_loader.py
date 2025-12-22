import pandas as pd
import numpy as np

def load_parquet_for_bnn(
    path: str,
    window: int = 20,
    vol_window: int = 20,
    target_horizon: int = 1,
    max_gap_s: float = 5.0,  # seconds, threshold for HFD gaps
    intraday: bool = False,  # <-- NEW SWITCH
    return_window_timestamps=False # <-- NEW SWITCH
):
    """
    Universal loader for Bayesian NN
    Works for:
      - HFD (live MT4)
      - Historical (1m / 5m / 15m / 60m)

    Args:
        path (str): Path to the parquet file containing the market data
        window (int, optional): Number of lookback periods for features. Defaults to 20.
        vol_window (int, optional): Window size for volatility calculation. Defaults to 20.
        target_horizon (int, optional): Number of periods ahead to predict. Defaults to 1.
        max_gap_s (float, optional): Maximum allowed gap (seconds) between consecutive timestamps. Defaults to 5.0.
        intraday (bool, optional): If True, treats data as intraday (1m/5m/15m/60m) and handles gaps differently. Defaults to False.

    Returns:
        tuple: (X, y) where
            - X : numpy.ndarray of shape [samples, window, features]
            - y : numpy.ndarray of shape [samples] containing target values
    """

    # --- 1. Load minimum required columns ---
    df = pd.read_parquet(path, columns=["timestamp", "mid"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- 2. Core signal ---
    log_mid = np.log(df["mid"].values)
    log_return = np.diff(log_mid)
    df = df.iloc[1:].reset_index(drop=True)
    df["log_return"] = log_return.astype("float32")

    # --- 3. Detect gaps ---
    if intraday:
        # Intraday bars (1m / 5m / 15m / 60m):
        # market closures create large timestamp jumps,
        # but returns remain economically valid
        valid_return = np.ones(len(df), dtype=bool)
    else:
        # HFD: timestamp gaps break return semantics
        dt = df["timestamp"].diff().dt.total_seconds().fillna(0)
        valid_return = dt <= max_gap_s
        valid_return.iloc[0] = True  # first value is always valid

    df["valid_return"] = valid_return

    # --- 4. Volatility & Activity (rolling) ---
    abs_lr = np.abs(df["log_return"].values)
    df["volatility"] = pd.Series(abs_lr).rolling(vol_window).std().values
    df["activity"] = pd.Series(abs_lr).rolling(vol_window).mean().values

    # --- 5. Drop warmup NaNs ---
    df = df.dropna().reset_index(drop=True)

    # --- 6. Segment by gaps ---
    segment_ids = np.cumsum(~df["valid_return"].values)  # increment at each invalid return
    X_list = []
    y_list = []

    for seg_id in np.unique(segment_ids):
        seg_df = df[segment_ids == seg_id]
        if len(seg_df) < window + target_horizon:
            continue  # skip too short segments

        features = seg_df[["log_return", "volatility", "activity"]].values.astype("float32")
        target = seg_df["log_return"].values.astype("float32")

        n_samples = len(seg_df) - window - target_horizon + 1
        idx = np.arange(window)[None, :] + np.arange(n_samples)[:, None]

        X_seg = features[idx]
        y_seg = target[window + target_horizon - 1 : window + target_horizon - 1 + n_samples]

        X_list.append(X_seg)
        y_list.append(y_seg)

    if not X_list:
        raise ValueError("No segments long enough for the given window and horizon.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if return_window_timestamps:
        # timestamps dla każdego okna (ostatni timestamp w oknie + horizon)
        ts_windows = df["timestamp"].values[window + target_horizon - 1: window + target_horizon - 1 + X.shape[0]]
        return X, y, ts_windows
    else:
        return X, y

