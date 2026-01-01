import pandas as pd
import numpy as np


def load_parquet_for_bnn(
    path: str,
    window: int,
    vol_window: int = 20,
    target_horizon: int = 1,
    #data_type: str = "intraday",   # "intraday" or "hfd"
    intraday: bool = True,  # <-- NEW SWITCH- kept for compatibility, no longer disables gap logic
    return_window_timestamps: bool = False,
):
    # --------------------------------------------------
    # 1. Load & sort
    # --------------------------------------------------
    df = pd.read_parquet(path, columns=["timestamp", "mid"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------
    # 2. Log-returns
    # --------------------------------------------------
    log_mid = np.log(df["mid"].values)
    lr = np.zeros_like(log_mid)
    lr[1:] = log_mid[1:] - log_mid[:-1]
    df["log_return"] = lr.astype("float32")

    df = df.iloc[1:].reset_index(drop=True)

    # --------------------------------------------------
    # 3. Soft outlier control (CRITICAL PART)
    # --------------------------------------------------
    # Adaptive winsorization — kills close/open jumps
    q_low, q_high = np.quantile(df["log_return"], [0.001, 0.999])
    df["log_return_clipped"] = df["log_return"].clip(q_low, q_high)

    # --------------------------------------------------
    # 4. Rolling features (from CLIPPED returns)
    # --------------------------------------------------
    abs_lr = np.abs(df["log_return_clipped"].values)

    df["volatility"] = (
        pd.Series(abs_lr)
        .rolling(vol_window, min_periods=vol_window)
        .std()
        .values
    )

    df["activity"] = (
        pd.Series(abs_lr)
        .rolling(vol_window, min_periods=vol_window)
        .mean()
        .values
    )

    # --------------------------------------------------
    # 5. Drop warmup NaNs
    # --------------------------------------------------
    df = df.dropna().reset_index(drop=True)

    # --------------------------------------------------
    # 6. Windowing (NO SEGMENTS for intraday)
    # --------------------------------------------------
    features = df[["log_return_clipped", "volatility", "activity"]].values.astype("float32")
    target = df["log_return"].values.astype("float32")
    timestamps = df["timestamp"].values

    n_samples = len(df) - window - target_horizon + 1
    if n_samples <= 0:
        raise ValueError("Not enough data for given window and horizon.")

    idx = np.arange(window)[None, :] + np.arange(n_samples)[:, None]

    X = features[idx]
    y = target[window + target_horizon - 1 : window + target_horizon - 1 + n_samples]

    if return_window_timestamps:
        ts = timestamps[
            window + target_horizon - 1 :
            window + target_horizon - 1 + n_samples
        ]
        return X, y, ts

    return X, y
