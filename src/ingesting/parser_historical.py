"""
Historical CSV Data Parser with Parquet Output

Converts OHLC CSV files into unified Parquet format
matching live HFD structure for Bayesian NN / MIDAS models.
"""

import pandas as pd
from pathlib import Path

# --- Configuration ---
HISTORICAL_DIR = Path(r"C:\python\project_programming_for_ai\data\historical\US.100").resolve()
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


def process_historical_dir(historical_dir: Path, parquet_dir: Path):
    """
    Processes all CSVs in historical_dir and saves them as Parquet files
    with append mode similar to parser_arrow.py
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


def main():
    process_historical_dir(HISTORICAL_DIR, PARQUET_DIR)


if __name__ == "__main__":
    main()
