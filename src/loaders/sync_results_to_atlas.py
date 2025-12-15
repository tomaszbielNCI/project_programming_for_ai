import os
import base64
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import gridfs
from bson.binary import Binary
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)


class ResultsSyncer:
    def __init__(self):
        self.atlas_uri = os.getenv("MONGO_ATLAS_URI")
        if not self.atlas_uri:
            raise ValueError("MONGO_ATLAS_URI not found in environment variables")

        self.client = MongoClient(self.atlas_uri)
        self.db = self.client["results_metadata"]
        self.results_collection = self.db["files_metadata"]
        self.fs = gridfs.GridFS(self.db, collection="results_files")

        # Create index to prevent duplicate file uploads
        self.results_collection.create_index("file_path", unique=True)

        # Results directory path
        self.results_dir = project_root / "results"

        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

    def sync_results(self):
        print(f"=== Starting Results Sync to MongoDB Atlas ===")
        print(f"Scanning directory: {self.results_dir}")

        total_files = 0
        synced_files = 0
        skipped_files = 0
        errors = 0

        # Walk through all files in results directory
        for root, _, files in os.walk(self.results_dir):
            for filename in files:
                file_path = Path(root) / filename
                relative_path = str(file_path.relative_to(self.results_dir))
                total_files += 1

                try:
                    # Skip if file already exists in database
                    if self.results_collection.find_one({"file_path": relative_path}):
                        print(f"⏩ Skipping (already exists): {relative_path}")
                        skipped_files += 1
                        continue

                    # Process file based on type
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        self._store_image(file_path, relative_path)
                    elif filename.lower().endswith(('.txt', '.csv', '.json', '.log')):
                        self._store_text_file(file_path, relative_path)
                    else:
                        print(f"⚠️  Skipping unsupported file type: {relative_path}")
                        skipped_files += 1
                        continue

                    print(f"✅ Synced: {relative_path}")
                    synced_files += 1

                except Exception as e:
                    print(f"❌ Error processing {relative_path}: {str(e)}")
                    errors += 1

        # Print summary
        print("\n=== Sync Summary ===")
        print(f"Total files found: {total_files}")
        print(f"Successfully synced: {synced_files}")
        print(f"Skipped (already exists): {skipped_files}")
        print(f"Errors: {errors}")

        if errors > 0:
            print("\n⚠️  Some files could not be synced. Check the error messages above.")
        else:
            print("\n✓ Sync completed successfully!")

    def _store_text_file(self, file_path, relative_path):
        """Store text file in MongoDB"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        document = {
            "file_path": relative_path,
            "file_name": os.path.basename(file_path),
            "file_type": "text",
            "content": content,
            "size_bytes": os.path.getsize(file_path),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        self.results_collection.insert_one(document)

    def _store_image(self, file_path, relative_path):
        """Store image file in GridFS"""
        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Store file in GridFS
        file_id = self.fs.put(
            file_data,
            filename=relative_path,
            content_type=f"image/{file_path.suffix.lower()[1:]}"
        )

        # Store metadata in collection
        document = {
            "file_path": relative_path,
            "file_name": os.path.basename(file_path),
            "file_type": "image",
            "gridfs_id": file_id,
            "size_bytes": os.path.getsize(file_path),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        self.results_collection.insert_one(document)


if __name__ == "__main__":
    try:
        syncer = ResultsSyncer()
        syncer.sync_results()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'syncer' in locals() and hasattr(syncer, 'client'):
            syncer.client.close()