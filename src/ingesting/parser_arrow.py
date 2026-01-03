"""
MT4 HFD Data Parser with Parquet Output
Now with MongoDB source option for academic requirements.
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# --- Configuration ---
USE_MONGO = False # Set to True to load from MongoDB instead of files

# File system paths (used when USE_MONGO = False)
RAW_DIR = Path(r"C:\python\project_programming_for_ai\data\hfd")
PARQUET_DIR = Path(r"C:\python\project_programming_for_ai\data\parsed")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# MongoDB settings (used when USE_MONGO = True)
MONGO_DATES = ["42025121"]  # Dates to load from MongoDB


def parse_raw_line(line, source="live"):
    """
    Parse line: '2025-12-16 03:34:14|US.100+|24862.340000|24864.190000'
    Returns dict with structural fields
    """
    try:
        timestamp_str, instrument, bid_str, ask_str = line.strip().split("|")
        bid = float(bid_str)
        ask = float(ask_str)
        if bid <= 0 or ask < bid:
            return None

        return {
            "timestamp": pd.to_datetime(timestamp_str),
            "instrument": instrument,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "spread": ask - bid,
            "source": source
        }
    except Exception:
        return None


def load_logs_from_mongo(dates: List[str]) -> List[Dict]:
    """Load HFD log data from MongoDB for specified dates."""
    try:
        from src.loaders.data_IO_v2 import DataIOv2
        io = DataIOv2()
        all_records = []

        for date in dates:
            df = io.load_logs(date)
            if df is not None and not df.empty:
                # Convert DataFrame to list of records in same format as parse_raw_line
                for _, row in df.iterrows():
                    record = {
                        "timestamp": pd.to_datetime(row['timestamp']),
                        "instrument": row['instrument'],
                        "bid": float(row['bid']),
                        "ask": float(row['ask']),
                        "mid": (float(row['bid']) + float(row['ask'])) / 2,
                        "spread": float(row['ask']) - float(row['bid']),
                        "source": "live"
                    }
                    all_records.append(record)

        io.close()
        return all_records
    except ImportError as e:
        print(f"Failed to import DataIOv2: {e}")
        return []
    except Exception as e:
        print(f"Error loading logs from MongoDB: {e}")
        return []


def process_raw_file(input_file, parquet_dir=PARQUET_DIR, source="live"):
    # Read and parse input file
    records = []
    with open(input_file, "r") as f:
        for line in f:
            if parsed := parse_raw_line(line, source):
                records.append(parsed)

    if not records:
        print(f"No valid records in {input_file}")
        return

    df = pd.DataFrame(records)

    # Validate and convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    initial_count = len(df)
    df = df.dropna(subset=["timestamp"])
    if len(df) < initial_count:
        print(f"  Dropped {initial_count - len(df)} records with invalid timestamps")

    # Optimize data types
    df["instrument"] = df["instrument"].astype("category")
    df["source"] = df["source"].astype("category")
    for col in ["bid", "ask", "mid", "spread"]:
        df[col] = df[col].astype("float32")

    # Process each instrument separately
    # NOTE: Using append mode for live data; historical data should overwrite
    # This maintains data continuity for live trading scenarios
    for instrument, group in df.groupby("instrument"):
        output_file = parquet_dir / f"{instrument}.parquet"

        # Append to existing data if file exists
        if output_file.exists():
            existing = pd.read_parquet(output_file, engine="pyarrow")
            group = pd.concat([existing, group], ignore_index=True)

        # Ensure data consistency
        group = group.sort_values("timestamp")
        group = group.drop_duplicates(subset=["timestamp", "instrument"])

        # Save optimized parquet file
        group.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Saved {len(group)} unique records for {instrument} (sorted by timestamp)")


def process_mongo_data(dates: List[str], parquet_dir=PARQUET_DIR, source="live"):
    """Process HFD data from MongoDB and save to Parquet files."""
    print(f"Loading HFD data from MongoDB for dates: {dates}")

    # Load records from MongoDB
    records = load_logs_from_mongo(dates)

    if not records:
        print("No valid records loaded from MongoDB")
        return

    df = pd.DataFrame(records)

    # Validate and convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    initial_count = len(df)
    df = df.dropna(subset=["timestamp"])
    if len(df) < initial_count:
        print(f"  Dropped {initial_count - len(df)} records with invalid timestamps")

    # Optimize data types
    df["instrument"] = df["instrument"].astype("category")
    df["source"] = df["source"].astype("category")
    for col in ["bid", "ask", "mid", "spread"]:
        df[col] = df[col].astype("float32")

    # Process each instrument separately
    # NOTE: Using append mode for live data; historical data should overwrite
    # This maintains data continuity for live trading scenarios
    for instrument, group in df.groupby("instrument"):
        output_file = parquet_dir / f"{instrument}.parquet"

        # Append to existing data if file exists
        if output_file.exists():
            existing = pd.read_parquet(output_file, engine="pyarrow")
            group = pd.concat([existing, group], ignore_index=True)

        # Ensure data consistency
        group = group.sort_values("timestamp")
        group = group.drop_duplicates(subset=["timestamp", "instrument"])

        # Save optimized parquet file
        group.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Saved {len(group)} unique records for {instrument} (sorted by timestamp)")


def main():
    if USE_MONGO:
        print("Using MongoDB as data source...")
        process_mongo_data(MONGO_DATES, PARQUET_DIR)
    else:
        print("Using local log files as data source...")
        raw_files = list(RAW_DIR.glob("*.log"))
        if not raw_files:
            print(f"No raw files found in {RAW_DIR}")
            return

        for file in raw_files:
            print(f"Processing {file.name}...")
            process_raw_file(file)
            print(f"Finished {file.name}")


if __name__ == "__main__":
    main()