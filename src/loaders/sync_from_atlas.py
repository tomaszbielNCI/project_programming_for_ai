from src.loaders.data_IO import DataIO
import os
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import OperationFailure

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


def get_atlas_collections(connection_string, db_name):
    """Get list of collections from Atlas."""
    try:
        client = MongoClient(connection_string)
        db = client[db_name]
        return db.list_collection_names()
    except Exception as e:
        print(f"Error getting collections: {e}")
        return []
    finally:
        client.close()


def sync_collection(atlas_loader, db_name, collection_name, local_client):
    """Sync a single collection from Atlas to local MongoDB."""
    try:
        print(f"\nSyncing collection: {collection_name}")

        # Load data from Atlas
        df = atlas_loader.from_atlas(
            collection=collection_name,
            db=db_name,
            connection_string=os.getenv("MONGO_ATLAS_URI")
        ).load()

        if df is None or df.empty:
            print(f"  No data in {collection_name}")
            return False

        print(f"  Loaded {len(df)} records")

        # Save to local MongoDB
        local_db = local_client[db_name]
        collection = local_db[collection_name]

        # Clear existing data
        collection.delete_many({})

        # Insert new data
        data = df.to_dict('records')
        if data:
            collection.insert_many(data)
            print(f"  ✅ Synced {len(data)} records to local")
            return True

    except Exception as e:
        print(f"  ❌ Error syncing {collection_name}: {str(e)}")
        return False


def sync_atlas_to_local():
    """Sync all collections from Atlas to local MongoDB."""
    try:
        print("=== Starting Atlas to Local Sync ===")

        # Initialize connections
        atlas_loader = DataIO()
        local_client = MongoClient('mongodb://localhost:27017/')

        # Get all collections from Atlas
        db_name = "development_analysis"
        print(f"\nFetching collections from {db_name}...")
        collections = get_atlas_collections(os.getenv("MONGO_ATLAS_URI"), db_name)

        if not collections:
            print("No collections found in Atlas database")
            return

        print(f"Found {len(collections)} collections: {', '.join(collections)}")

        # Sync each collection
        success_count = 0
        for collection_name in collections:
            if sync_collection(atlas_loader, db_name, collection_name, local_client):
                success_count += 1

        print(f"\n✅ Sync complete! Successfully synced {success_count}/{len(collections)} collections")

    except Exception as e:
        print(f"\n❌ Error during sync: {e}")
    finally:
        local_client.close()


if __name__ == "__main__":
    sync_atlas_to_local()
    input("\nPress Enter to exit...")