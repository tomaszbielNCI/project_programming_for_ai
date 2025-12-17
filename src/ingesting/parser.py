# scripts/structural_cleaning.py

import os
from pathlib import Path
from datetime import datetime
import pandas as pd

BUFFER_SIZE = 500  # liczba rekordów przed zapisaniem do pliku

def parse_raw_line(line, source="live"):
    """
    Parse single raw MT4 line into structured dictionary.
    Expected format: "YYYY-MM-DD HH:MM:SS|SYMBOL|BID|ASK"
    """
    try:
        timestamp_str, instrument, bid_str, ask_str = line.strip().split('|')
        bid = float(bid_str)
        ask = float(ask_str)
        # Walidacja bid/ask
        if bid <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2
        spread = ask - bid
        return {
            'timestamp': pd.to_datetime(timestamp_str),
            'instrument': instrument,
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'spread': spread,
            'source': source
        }
    except Exception as e:
        print(f"[WARN] Failed to parse line: {line.strip()} | {str(e)}")
        return None

def process_raw_file(input_file, output_dir, source="historical"):
    """
    Process raw MT4 file into structured DataFrame and save partitioned Parquet.
    Uses optimized processing by grouping data by instrument before saving.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to hold data grouped by instrument
    data = {}
    total_records = 0
    
    # Process file line by line and group by instrument
    with open(input_file, 'r') as f:
        for line in f:
            record = parse_raw_line(line, source)
            if record:
                instrument = record['instrument']
                if instrument not in data:
                    data[instrument] = []
                data[instrument].append(record)
                total_records += 1
    
    # Process and save each instrument's data
    for instrument, records in data.items():
        output_file = output_dir / f"{instrument}.parquet"
        
        # Convert records to DataFrame
        df = pd.DataFrame(records)
        
        # Ensure proper timestamp format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # If file exists, read and merge with existing data
        if output_file.exists():
            existing = pd.read_parquet(output_file)
            # Ensure existing data has proper timestamp format
            existing['timestamp'] = pd.to_datetime(existing['timestamp'])
            # Merge and remove duplicates
            df = pd.concat([existing, df]).drop_duplicates(
                subset=['timestamp', 'instrument']
            )
        
        # Sort by timestamp and save
        df = df.sort_values('timestamp')
        df.to_parquet(output_file, index=False, engine='pyarrow')
    
    print(f"[INFO] Processed {total_records} valid records from {input_file.name}")


def save_buffer(records, output_dir):
    """
    This function is kept for backward compatibility but is no longer used.
    The processing is now handled directly in process_raw_file.
    """
    pass

def main():
    # Konfiguracja katalogów
    raw_dir = r"C:\python\project_programming_for_ai\data\hfd"
    output_dir = r"C:\python\project_programming_for_ai\data\parsed"

    raw_files = list(Path(raw_dir).glob("*.log"))
    if not raw_files:
        print(f"[WARN] No raw data files found in {raw_dir}")
        return

    for file in raw_files:
        print(f"[INFO] Processing {file.name}...")
        process_raw_file(file, output_dir, source="historical")
        print(f"[INFO] Finished {file.name}")

if __name__ == "__main__":
    main()
