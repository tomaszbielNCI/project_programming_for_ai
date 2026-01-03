"""
MT4 HFD Data Parser with Parquet Output

PLANNED OPTIMIZATIONS:

1. INCREMENTAL INGESTION:
   # Track last processed position in each file
   # Use file modification time + inode to detect changes
   # Store watermarks for each data source
   # Example: {'file1.log': {'last_pos': 12345, 'last_modified': 1671234567}}

2. PERFORMANCE IMPROVEMENTS:
   # Replace pandas with Polars for faster processing
   # Use Apache Arrow streaming for memory efficiency
   # Implement chunked processing for large files
   # Add parallel processing for multiple instruments

3. LIVE DATA PROCESSING:
   # Add in-memory ring buffer for real-time inference
   # Implement async I/O for non-blocking operations
   # Add backpressure handling
   
4. DISTRIBUTED PROCESSING (Optional):
   # Kafka integration for event streaming
   # Redis for shared state and caching
   # Example: 
   #   - Kafka topics per instrument
   #   - Redis for watermark tracking
   #   - Distributed processing with Dask/Ray

5. ERROR HANDLING:
   # Dead letter queue for failed records
   # Automatic retry with exponential backoff
   # Circuit breaker pattern for external services

Current Implementation:
- Basic file parsing with pandas
- Simple append mode with deduplication
- No incremental processing
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# TODO: Add these imports when implementing optimizations
# import polars as pl
# from kafka import KafkaProducer
# import redis
# import asyncio

# --- Configuration ---
RAW_DIR = Path(r"C:\python\project_programming_for_ai\data\hfd")
PARQUET_DIR = Path(r"C:\python\project_programming_for_ai\data\parsed")
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

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

def main():
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
