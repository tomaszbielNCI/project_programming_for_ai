import pandas as pd
import numpy as np

def load_parquet_for_bnn(
    path: str,
    window: int = 20,
    vol_window: int = 20,
    target_horizon: int = 1,
    max_gap_s: float = 61.0,   # seconds, threshold for HFD gaps
    intraday: bool = False,   # <-- kept for compatibility, no longer disables gap logic
    return_window_timestamps=False,
    data_type: str = "intraday"  # "hfd" or "intraday"
):
    # ... (wcześniejszy kod bez zmian do sekcji 3)


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
        max_gap_s (float, optional): Maximum allowed gap (seconds) between consecutive timestamps.
                                     For intraday data this should be set to expected_freq * multiplier.
        intraday (bool, optional): Kept for backward compatibility. Gaps are always validated.
    """

    # --- 1. Load minimum required columns ---
    df = pd.read_parquet(path, columns=["timestamp", "mid"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- 2. Core signal ---
    log_mid = np.log(df["mid"].values)
    log_return = np.zeros_like(log_mid)
    log_return[1:] = log_mid[1:] - log_mid[:-1]
    df["log_return"] = log_return.astype("float32")
    df = df.iloc[1:].reset_index(drop=True)
    # --- 3. Detect gaps ---
    if data_type == "hfd":
        # Dla HFD: reset segmentu przy gap > max_gap_s
        dt = df["timestamp"].diff().dt.total_seconds().fillna(0)
        valid_return = dt <= max_gap_s
        valid_return.iloc[0] = True  # first value is always valid
    else:  # intraday
        # Dla intraday: nie resetujemy segmentu, ale możemy oznaczyć outliery po długich przerwach
        # Na razie: ustawiamy wszystkie na True (brak outlierów z powodu gapów)
        valid_return = pd.Series(True, index=df.index)
        # Ewentualnie: oznaczenie outlierów po przerwie dłuższej niż, powiedzmy, 5 minut
        # dt = df["timestamp"].diff().dt.total_seconds().fillna(0)
        # valid_return = dt <= 300  # 5 minut w sekundach
        # valid_return.iloc[0] = True

    df["valid_return"] = valid_return

    # ... (reszta kodu bez zmian do sekcji 6)

    # --- 6. Segment by gaps ---
    if data_type == "hfd":
        segment_ids = np.cumsum(~df["valid_return"].values)
    else:
        # Dla intraday: jeden segment (całość)
        segment_ids = np.zeros(len(df), dtype=int)

    # ... (reszta kodu bez zmian)
    # --- 3. Detect gaps ---
    # Unified gap detection for BOTH HFD and intraday data.
    # Any gap larger than max_gap_s breaks return continuity and starts a new segment.
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
    segment_ids = np.cumsum(~df["valid_return"].values)
    X_list = []
    y_list = []

    for seg_id in np.unique(segment_ids):
        seg_df = df[segment_ids == seg_id]
        if len(seg_df) < window + target_horizon:
            continue

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
        ts_windows = df["timestamp"].values[
            window + target_horizon - 1 :
            window + target_horizon - 1 + X.shape[0]
        ]
        return X, y, ts_windows
    else:
        return X, y
