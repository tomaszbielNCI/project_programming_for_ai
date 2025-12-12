import os
import pandas as pd
from pymongo import MongoClient
from typing import Optional, Dict
from dotenv import load_dotenv

# ========================
# PATHS
# ========================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMP_DATA_DIR = os.path.join(PROJECT_ROOT, 'temp_data')

dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


# ========================
# DataIO CLASS
# ========================

class DataIO:
    """
    Unified interface:
    - load()  -> CSV or Mongo (Atlas/local)
    - save()  -> CSV or Mongo
    Automatic versioning: if file/collection exists, creates indexed version.
    """

    def __init__(self):
        self.mode = None
        self.csv_path = None
        self.client = None
        self.db = None
        self.collection = None

    # ========================
    # LOAD SOURCES
    # ========================

    def from_csv(self, file_input: str):
        """Select CSV file to load."""
        if os.path.isabs(file_input):
            fp = file_input
        else:
            fp = os.path.join(TEMP_DATA_DIR, file_input)

        if not os.path.exists(fp):
            raise FileNotFoundError(f"CSV file does not exist: {fp}")

        self.mode = "csv"
        self.csv_path = fp
        return self

    def from_atlas(self, db: str, collection: str, connection_string: str):
        if not connection_string:
            raise ValueError("Missing MongoDB Atlas connection string.")

        self.client = MongoClient(connection_string)
        self.db = self.client[db]
        self.collection = collection
        self.mode = "atlas"
        return self

    def from_local(self, db: str, collection: str):
        self.client = MongoClient("mongodb://localhost:27017")
        self.db = self.client[db]
        self.collection = collection
        self.mode = "local"
        return self

    # ========================
    # LOAD
    # ========================

    def load(self, query: Dict = None, limit: Optional[int] = None) -> pd.DataFrame:
        if self.mode == "csv":
            return pd.read_csv(self.csv_path)

        if self.mode in ("atlas", "local"):
            coll = self.db[self.collection]

            if limit:
                cursor = coll.find(query or {}, limit=limit)
            else:
                cursor = coll.find(query or {})

            df = pd.DataFrame(list(cursor))

            if "_id" in df.columns:
                df["_id"] = df["_id"].astype(str)

            return df

        raise RuntimeError("DataIO not initialized with any source.")

    # ========================
    # SAVE (CSV + Mongo)
    # ========================

    def save(self, data_frame: pd.DataFrame,
             target: str = "csv",
             name: Optional[str] = None,
             db: Optional[str] = None,
             collection: Optional[str] = None,
             connection_string: Optional[str] = None):

        if target == "csv":
            return self._save_csv(data_frame, name)

        elif target == "mongo":
            return self._save_mongo(data_frame, db, collection, connection_string)

        else:
            raise ValueError("target must be 'csv' or 'mongo'")

    # ========================
    # CSV SAVE with auto-indexing
    # ========================

    @staticmethod
    def _save_csv(data_frame: pd.DataFrame, name: str):
        if not name:
            raise ValueError("CSV save requires explicit 'name'.")

        # Remove .csv extension if provided
        if name.endswith('.csv'):
            base_name = name[:-4]
        else:
            base_name = name

        # Try the original name first
        filename = f"{base_name}.csv"
        dest = os.path.join(TEMP_DATA_DIR, filename)

        # If original exists, find next available index
        if os.path.exists(dest):
            index = 1
            while True:
                filename = f"{base_name}_{index}.csv"
                dest = os.path.join(TEMP_DATA_DIR, filename)
                if not os.path.exists(dest):
                    break
                index += 1

        data_frame.to_csv(dest, index=False)
        print(f"[CSV] Saved: {dest}")
        return dest

    # ========================
    # MONGO SAVE with auto-indexing
    # ========================

    @staticmethod
    def _save_mongo(data_to_save: pd.DataFrame,
                   db: str,
                   collection: str,
                   connection_string: Optional[str] = None):

        if not db or not collection:
            raise ValueError("Mongo save requires db and collection.")

        uri = connection_string or os.getenv("MONGO_ATLAS_URI") or "mongodb://localhost:27017"

        client = MongoClient(uri)
        database = client[db]

        original_collection_name = collection
        final_collection_name = collection

        # Check if collection exists and find next available name
        if original_collection_name in database.list_collection_names():
            index = 1
            while True:
                new_name = f"{original_collection_name}_{index}"
                if new_name not in database.list_collection_names():
                    final_collection_name = new_name
                    print(
                        f"[Mongo] Collection '{original_collection_name}' exists. Using '{final_collection_name}' instead.")
                    break
                index += 1

        coll = database[final_collection_name]

        # Convert DataFrame to dict records
        records = data_to_save.to_dict(orient="records")

        if not records:
            raise ValueError("Cannot save empty DataFrame to Mongo.")

        coll.insert_many(records)
        print(f"[Mongo] Saved {len(records)} records into {db}.{final_collection_name}")
        return True


# Add this at the end of data_IO.py (before the final .)

if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

    parser = argparse.ArgumentParser(description='Load data from various sources')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['atlas', 'local', 'csv'],
                        help='Data source: atlas, local, or csv')
    parser.add_argument('--db', type=str,
                        help='Database name (required for atlas/local modes)')
    parser.add_argument('--collection', type=str,
                        help='Collection name (required for atlas/local modes)')
    parser.add_argument('--file', type=str,
                        help='CSV file path (required for csv mode)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of records (0 for no limit)')

    args = parser.parse_args()

    data_io = DataIO()

    try:
        if args.mode == 'csv':
            if not args.file:
                raise ValueError("--file argument is required for csv mode")
            df = data_io.from_csv(args.file).load()
        else:  # atlas or local
            if not args.db or not args.collection:
                raise ValueError("--db and --collection arguments are required for this mode")

            if args.mode == 'atlas':
                mongo_uri = os.getenv("MONGO_ATLAS_URI")
                if not mongo_uri:
                    raise ValueError("MONGO_ATLAS_URI not found in .env file")
                df = data_io.from_atlas(
                    db=args.db,
                    collection=args.collection,
                    connection_string=mongo_uri
                ).load(limit=args.limit if args.limit > 0 else None)
            else:  # local
                df = data_io.from_local(
                    db=args.db,
                    collection=args.collection
                ).load(limit=args.limit if args.limit > 0 else None)

        # Display results
        print(f"\nSuccessfully loaded {len(df)} records")
        print("\nFirst 5 rows:")
        pd.set_option('display.max_columns', None)
        print(df.head())

    except Exception as e:
        print(f"Error: {e}")
        exit(1)