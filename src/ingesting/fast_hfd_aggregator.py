"""
Minimal HFD to 1-min Aggregator for BNN
Creates 1-minute bars with only mid price and tick count.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def hfd_to_1min_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized aggregation of HFD ticks to 1-minute bars.
    Returns only: timestamp, mid (close price), tick_count, log_return_60min (target)
    """
    # Ensure timestamp is datetime and sorted
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Set timestamp as index for resampling
    df.set_index('timestamp', inplace=True)

    # Resample to 1-minute bars, using 'mid' for close price and counting ticks
    resampled = df['mid'].resample('1T').agg(
        ['last', 'count']
    ).dropna()  # Skip minutes without ticks

    # Rename columns
    resampled.columns = ['mid', 'tick_count']

    # Reset index to have timestamp as column
    resampled = resampled.reset_index()

    # Calculate 60-minute forward returns (target)
    resampled['log_mid'] = np.log(resampled['mid'])
    resampled['log_return_60min'] = resampled['log_mid'].diff(60)

    # Remove rows with NaN in target (first 60 minutes and any gaps)
    resampled = resampled.dropna(subset=['log_return_60min'])

    # Keep only necessary columns
    result = resampled[['timestamp', 'mid', 'tick_count', 'log_return_60min']].copy()

    # Optimize data types
    result['mid'] = result['mid'].astype('float32')
    result['tick_count'] = result['tick_count'].astype('int32')
    result['log_return_60min'] = result['log_return_60min'].astype('float32')

    return result.reset_index(drop=True)


if __name__ == "__main__":
    INPUT = Path(r"C:\python\project_programming_for_ai\data\parsed\US.100.parquet")
    OUTPUT = Path(r"C:\python\project_programming_for_ai\data\parsed\US.100_1min_minimal.parquet")

    print(f"Loading HFD data from: {INPUT}")
    df_hfd = pd.read_parquet(INPUT)
    print(f"Loaded {len(df_hfd):,} HFD ticks")

    print("Aggregating to minimal 1-minute bars...")
    result = hfd_to_1min_minimal(df_hfd)

    print(f"Created {len(result):,} 1-minute bars")
    print(f"Columns: {list(result.columns)}")
    print(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")

    print(f"Saving to: {OUTPUT}")
    result.to_parquet(OUTPUT, index=False)
    print("Done.")