"""
DataIO v2 - Ultra minimalist version.
Only sync CSV/LOGS to MongoDB and load them back.
"""

import os
from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import hashlib
# Load env
load_dotenv()


class DataIOv2:
    """Minimal MongoDB sync/load for CSV and logs."""

    def __init__(self):
        uri = os.getenv("MONGO_ATLAS_URI", "mongodb://localhost:27017")
        db = os.getenv("MONGO_ATLAS_DB", "financial_data")
        self.client = MongoClient(uri)
        self.db = self.client[db]

    # SYNC FUNCTIONS
    def sync_csv(self, csv_path: Path, instrument: str, timeframe: str):
        """Sync CSV file to MongoDB with duplicate checking."""
        # Read CSV
        df = pd.read_csv(csv_path, header=None)
        df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

        # Create unique hash for each row
        df['_hash'] = df.apply(
            lambda x: hashlib.md5(
                f"{x['date']}{x['time']}{x['open']}{x['high']}{x['low']}{x['close']}".encode()
            ).hexdigest(),
            axis=1
        )

        # Add metadata
        df['_source'] = csv_path.name
        df['instrument'] = instrument
        df['timeframe'] = timeframe
        df['ingest_time'] = datetime.utcnow()

        # Get existing hashes
        coll_name = f"csv_{instrument}_{timeframe}".replace('.', '_')
        existing_hashes = set()

        if self.db[coll_name].count_documents({}) > 0:
            existing_hashes = set(doc['_hash'] for doc in
                                  self.db[coll_name].find(
                                      {'_source': csv_path.name},
                                      {'_hash': 1, '_id': 0}
                                  ))

        # Filter out duplicates
        new_data = df[~df['_hash'].isin(existing_hashes)].to_dict('records')

        # Insert new records
        if new_data:
            self.db[coll_name].insert_many(new_data, ordered=False)

        return len(new_data)

    def sync_log(self, log_path: Path):
        """Sync log file to MongoDB with duplicate checking based on raw content."""
        from datetime import datetime
        log_path = Path(log_path) if isinstance(log_path, str) else log_path
        # Read and parse log file
        records = []
        existing_hashes = set()

        # Get existing raw hashes to avoid duplicates
        date_part = ''.join(filter(str.isdigit, log_path.stem))[:8]
        coll_name = f"log_{date_part}" if date_part else "log"

        # Get hashes of existing records for this source
        if self.db[coll_name].count_documents({}) > 0:
            existing_hashes = set(doc['_hash'] for doc in
                                  self.db[coll_name].find(
                                      {'_source': log_path.name},
                                      {'_hash': 1, '_id': 0}
                                  ))

        # Process new records
        new_records = []
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Create hash of raw line for duplicate checking
                line_hash = str(hash(line))

                if line_hash not in existing_hashes:
                    parts = line.split('|')
                    if len(parts) == 4:
                        new_records.append({
                            'raw': line,
                            '_hash': line_hash,
                            'timestamp': parts[0],
                            'instrument': parts[1].rstrip('+'),
                            'bid': float(parts[2]),
                            'ask': float(parts[3]),
                            '_source': log_path.name,
                            'ingest_time': datetime.utcnow()
                        })

        # Insert new records in batches
        if new_records:
            self.db[coll_name].insert_many(new_records, ordered=False)

        return len(new_records)

    # LOAD FUNCTIONS
    def load_csv(self, instrument: str, timeframe: str) -> pd.DataFrame:
        """Load CSV data from MongoDB."""
        coll_name = f"csv_{instrument}_{timeframe}".replace('.', '_')
        cursor = self.db[coll_name].find()
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df = df.drop(columns=['_id', '_source'], errors='ignore')
        return df

    def load_logs(self, date: str = None) -> pd.DataFrame:
        """Load logs from MongoDB."""
        colls = [c for c in self.db.list_collection_names() if c.startswith('log_')]
        if date:
            colls = [c for c in colls if date in c]

        dfs = []
        for coll in colls:
            cursor = self.db[coll].find()
            df = pd.DataFrame(list(cursor))
            if not df.empty:
                dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True).drop(columns=['_id', '_source'], errors='ignore')
        return pd.DataFrame()

    # RESULTS
    def save_results(self, data, name: str):
        """Save results to MongoDB."""
        if isinstance(data, pd.DataFrame):
            records = data.to_dict('records')
        else:
            records = [{'data': str(data)}]

        self.db[f'results_{name}'].insert_many(records)

    def close(self):
        self.client.close()


# USAGE EXAMPLE
if __name__ == "__main__":
    io = DataIOv2()

    # Sync all CSV files
    csv_folder = Path("data/historical/US.100")
    for csv_file in csv_folder.glob("*.csv"):
        stem = csv_file.stem
        if '+' in stem:
            instr, tf = stem.split('+')
            count = io.sync_csv(csv_file, instr, tf)
            print(f"Synced {csv_file.name}: {count} rows")

    # Sync all log files
    log_folder = Path("data/hfd")
    for log_file in log_folder.glob("*.log"):
        count = io.sync_log(log_file)
        print(f"Synced {log_file.name}: {count} rows")

    io.close()

"""
# Import and initialize
from pathlib import Path
import sys
sys.path.append("C:/python/project_programming_for_ai")
from src.loaders.data_IO_v2 import DataIOv2
io = DataIOv2()

# Check connection
io.client.server_info()

# Sync all CSV files
for f in Path("data/historical").rglob("*.csv"):
    if '+' in f.stem:
        instr, tf = f.stem.split('+')
        io.sync_csv(f, instr, tf)

# Sync all log files
for log in Path("data/hfd").glob("*.log"):
    io.sync_log(log)

# Check collections
io.db.list_collection_names()

# When done
io.close()
"""