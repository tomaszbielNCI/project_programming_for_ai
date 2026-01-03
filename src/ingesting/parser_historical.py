"""
Historical CSV Data Parser with Parquet Output
Now with MongoDB source option for academic requirements.
"""

import pandas as pd
from pathlib import Path

# --- Configuration ---
USE_MONGO = False  # Set to True to load from MongoDB instead of files

# File system paths (used when USE_MONGO = False)
HISTORICAL_DIR = Path(r"C:\python\project_programming_for_ai\data\historical\OIL.WTI").resolve()
PARQUET_DIR = Path(r"C:\python\project_programming_for_ai\data\parsed").resolve()
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def parse_historical_csv(file_path: Path, instrument_name: str, source="historical"):
    """
    Parses a single historical CSV into unified format.
    Assumes CSV columns: Date, Time, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(file_path, header=None)

    # Rename columns for clarity
    df.columns = ["date", "time", "open", "high", "low", "close", "volume"]

    # Combine date + time into timestamp
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.drop(columns=["date", "time"])
    df = df.dropna(subset=["timestamp"])

    # Map OHLC to target schema
    df["instrument"] = instrument_name
    df["bid"] = df["close"].astype("float32")  # could use "open" as alternative
    df["ask"] = df["close"].astype("float32")
    df["mid"] = ((df["bid"] + df["ask"]) / 2).astype("float32")
    df["spread"] = (df["high"] - df["low"]).astype("float32")
    df["source"] = source

    # Keep only required columns
    df = df[["timestamp", "instrument", "bid", "ask", "mid", "spread", "source"]]

    # Convert categorical columns
    df["instrument"] = df["instrument"].astype("category")
    df["source"] = df["source"].astype("category")

    return df


def load_from_mongo(instrument: str, timeframe: str) -> pd.DataFrame:
    """Load historical data from MongoDB and format it to match CSV structure."""
    try:
        from src.loaders.data_IO_v2 import DataIOv2
        io = DataIOv2()
        df = io.load_csv(instrument, timeframe)
        io.close()

        if df is None or df.empty:
            print(f"No data found in MongoDB for {instrument}+{timeframe}")
            return pd.DataFrame()

        # DataIOv2 returns DataFrame with columns already named
        # Make sure it has the 7 columns in correct order: date, time, open, high, low, close, volume
        required_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

        # If columns are missing, return empty
        if not all(col in df.columns for col in required_columns):
            print(f"MongoDB data missing required columns for {instrument}+{timeframe}")
            return pd.DataFrame()

        # Select and order columns to match CSV structure
        df = df[required_columns]

        return df
    except ImportError as e:
        print(f"Failed to import DataIOv2: {e}")
        print("Make sure data_IO_v2.py is in your Python path")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading from MongoDB: {e}")
        return pd.DataFrame()


def parse_historical_from_mongo(instrument: str, timeframe: str, source="historical"):
    """
    Load data from MongoDB and parse it using the same logic as CSV.
    Returns DataFrame in the exact same format as parse_historical_csv.
    """
    # Load raw data from MongoDB
    df = load_from_mongo(instrument, timeframe)

    if df.empty:
        return df

    # Use the same parsing logic as CSV files
    instrument_name = f"{instrument}+{timeframe}"
    return parse_historical_csv_from_dataframe(df, instrument_name, source)


def parse_historical_csv_from_dataframe(df: pd.DataFrame, instrument_name: str, source="historical"):
    """
    Same parsing logic as parse_historical_csv but accepts DataFrame instead of file.
    Assumes df has columns: date, time, open, high, low, close, volume
    """
    # Combine date + time into timestamp
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.drop(columns=["date", "time"])
    df = df.dropna(subset=["timestamp"])

    # Map OHLC to target schema
    df["instrument"] = instrument_name
    df["bid"] = df["close"].astype("float32")
    df["ask"] = df["close"].astype("float32")
    df["mid"] = ((df["bid"] + df["ask"]) / 2).astype("float32")
    df["spread"] = (df["high"] - df["low"]).astype("float32")
    df["source"] = source

    # Keep only required columns
    df = df[["timestamp", "instrument", "bid", "ask", "mid", "spread", "source"]]

    # Convert categorical columns
    df["instrument"] = df["instrument"].astype("category")
    df["source"] = df["source"].astype("category")

    return df


def process_historical_dir(historical_dir: Path, parquet_dir: Path):
    """
    Processes all CSVs in historical_dir and saves them as Parquet files
    with append mode similar to parser_arrow_0.1.py
    """
    csv_files = list(historical_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {historical_dir}")
        return

    # Create output directory if it doesn't exist
    parquet_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        # Process the CSV file
        df = parse_historical_csv(csv_file, csv_file.stem)  # Pass full filename as instrument name
        if df is None or df.empty:
            print(f"Skipping {csv_file.name} - no valid data")
            continue

        # Create output filename based on input filename
        output_file = parquet_dir / f"{csv_file.stem}.parquet"

        # If output file exists, load it and append new data
        if output_file.exists():
            try:
                existing = pd.read_parquet(output_file, engine="pyarrow")
                df = pd.concat([existing, df], ignore_index=True)
                print(f"Appending to existing {output_file.name}")
            except Exception as e:
                print(f"Error reading existing {output_file.name}: {e}")
                continue

        # Sort and remove duplicates (keep last occurrence)
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp", "instrument"], keep="last")

        # Save the combined data
        df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Saved {len(df)} records to {output_file.name} (appended: {output_file.exists()})")


def process_mongo_dir(parquet_dir: Path, instruments: list = None):
    """
    Processes data from MongoDB and saves as Parquet files.
    instruments: list of tuples [(instrument, timeframe), ...]
    Default: [("US.100", "1"), ("US.100", "5"), ("US.100", "15"), ("US.100", "60")]
    """
    if instruments is None:
        instruments = [("US.100", "1"), ("US.100", "5"), ("US.100", "15"), ("US.100", "60")
                        #("OIL.WTI", "1"), ("OIL.WTI", "5"), ("OIL.WTI", "15"), ("OIL.WTI", "60"),
                        #("USDJPY", "1"), ("USDJPY", "5"), ("USDJPY", "15"), ("USDJPY", "60")
                       ]

    parquet_dir.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in instruments:
        print(f"Processing {instrument}+{timeframe} from MongoDB...")

        # Load and parse from MongoDB
        df = parse_historical_from_mongo(instrument, timeframe)

        if df is None or df.empty:
            print(f"Skipping {instrument}+{timeframe} - no valid data")
            continue

        # Create output filename
        output_file = parquet_dir / f"{instrument}+{timeframe}.parquet"

        # If output file exists, load it and append new data
        if output_file.exists():
            try:
                existing = pd.read_parquet(output_file, engine="pyarrow")
                df = pd.concat([existing, df], ignore_index=True)
                print(f"Appending to existing {output_file.name}")
            except Exception as e:
                print(f"Error reading existing {output_file.name}: {e}")
                continue

        # Sort and remove duplicates (keep last occurrence)
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp", "instrument"], keep="last")

        # Save the combined data
        df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Saved {len(df)} records to {output_file.name} (appended: {output_file.exists()})")


def main():
    """Main function with MongoDB switch."""
    if USE_MONGO:
        print("Using MongoDB as data source...")
        process_mongo_dir(PARQUET_DIR)
    else:
        print("Using local CSV files as data source...")
        process_historical_dir(HISTORICAL_DIR, PARQUET_DIR)


if __name__ == "__main__":
    main()