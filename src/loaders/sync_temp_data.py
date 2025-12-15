import os
import pandas as pd
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime
import sys
from dotenv import load_dotenv

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'

# Load environment variables from .env file in project root
load_dotenv(env_path)
print(f"Loading .env from: {env_path}")
print(f"MONGO_ATLAS_URI: {'Set' if os.getenv('MONGO_ATLAS_URI') else 'Not set'}")

# Add project root to path
sys.path.append(str(project_root))


class TempDataSyncer:
    def __init__(self):
        # Initialize connections
        self.atlas_uri = os.getenv("MONGO_ATLAS_URI")
        if not self.atlas_uri:
            print("\n❌ ERROR: MONGO_ATLAS_URI not found in environment variables")
            print("Please make sure you have a .env file in the project root with MONGO_ATLAS_URI")
            
        self.local_uri = "mongodb://localhost:27017/"
        self.db_name = "development_analysis"

        # Collections that should not be overwritten
        self.protected_collections = ['file_metadata']
        
        # Debug: Print connection info
        print(f"\n=== MongoDB Connection Info ===")
        print(f"Using Atlas URI: {self.atlas_uri[:30]}..." if self.atlas_uri else "No Atlas URI found in environment")
        print(f"Local URI: {self.local_uri}")
        print(f"Database: {self.db_name}\n")

    def get_collection_names(self, client):
        """Get list of collections in the database"""
        db = client[self.db_name]
        return db.list_collection_names()

    def collection_exists(self, client, collection_name):
        """Check if collection exists and has data"""
        db = client[self.db_name]
        if collection_name in self.get_collection_names(client):
            return db[collection_name].count_documents({}) > 0
        return False

    def sync_file(self, file_path, target):
        """Sync a single file to target MongoDB"""
        try:
            filename = os.path.basename(file_path)
            collection_name = os.path.splitext(filename)[0]

            # Skip non-CSV files and protected collections
            if not file_path.endswith('.csv') or collection_name in self.protected_collections:
                return False, f"Skipped {filename}"

            # Read the file
            df = pd.read_csv(file_path)

            if df.empty:
                return False, f"Empty file: {filename}"

            # Connect to target
            client = MongoClient(self.atlas_uri if target == 'atlas' else self.local_uri)

            # Check if collection exists and has data
            if self.collection_exists(client, collection_name):
                client.close()
                return False, f"Exists in {target}: {collection_name}"

            # Save to MongoDB
            db = client[self.db_name]
            collection = db[collection_name]
            data = df.to_dict('records')
            
            # Debug: Print collection info before insert
            print(f"\n=== Inserting into {target} ===")
            print(f"Database: {self.db_name}")
            print(f"Collection: {collection_name}")
            print(f"Records to insert: {len(data)}")
            
            try:
                result = collection.insert_many(data)
                print(f"Successfully inserted {len(result.inserted_ids)} documents")
            except Exception as e:
                print(f"Error during insert: {str(e)}")
                raise

            # Update metadata
            meta_collection = db['file_metadata']
            meta_collection.update_one(
                {"filename": filename},
                {"$set": {
                    "filename": filename,
                    "collection": collection_name,
                    "target": target,
                    "last_updated": datetime.now(),
                    "record_count": len(data),
                    "columns": list(df.columns)
                }},
                upsert=True
            )

            client.close()
            return True, f"Synced to {target}: {collection_name} ({len(data)} records)"

        except Exception as e:
            return False, f"Error syncing {filename} to {target}: {str(e)}"

    def test_connections(self):
        """Test connections to MongoDB servers"""
        print("\n=== Testing Connections ===")
        for target in ['atlas', 'local']:
            try:
                uri = self.atlas_uri if target == 'atlas' else self.local_uri
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.server_info()  # Force connection
                print(f"✅ {target.upper()} connection successful")
                
                # List databases
                dbs = client.list_database_names()
                print(f"   Available databases: {', '.join(dbs)}")
                
                # List collections in our target db
                if self.db_name in dbs:
                    db = client[self.db_name]
                    collections = db.list_collection_names()
                    print(f"   Collections in {self.db_name}: {len(collections)}")
                    
            except Exception as e:
                print(f"❌ {target.upper()} connection failed: {str(e)}")
            finally:
                client.close()
    
    def sync_all(self):
        """Sync all CSV files from temp_data to both MongoDB Atlas and local"""
        temp_data_dir = project_root / "temp_data"
        if not temp_data_dir.exists():
            print(f"Directory not found: {temp_data_dir}")
            return
            
        # Test connections first
        self.test_connections()

        results = []
        for filename in os.listdir(temp_data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(temp_data_dir, filename)



                # Sync to both targets
                for target in ['atlas', 'local']:
                    success, message = self.sync_file(file_path, target)
                    results.append((filename, target, success, message))

        # Print summary
        print("\n=== Sync Summary ===")
        for filename, target, success, message in results:
            status = "✅" if success else "⚠️"
            print(f"{status} {message}")


if __name__ == "__main__":
    print("=== Starting Temp Data Sync ===")
    print("This will sync CSV files from temp_data to both MongoDB Atlas and local MongoDB")
    print("Only new collections will be created, existing ones will be skipped\n")

    syncer = TempDataSyncer()
    syncer.sync_all()