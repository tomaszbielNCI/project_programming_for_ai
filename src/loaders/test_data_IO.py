# test_data_IO.py
from src.loaders.data_IO import DataIO
import pandas as pd
from dotenv import load_dotenv
import os

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 30)


def test_mongo_atlas():
    print("Testing MongoDB Atlas connection...\n")

    # Initialize DataIO
    data_io = DataIO()

    try:
        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))
        mongo_uri = os.getenv("MONGO_ATLAS_URI")

        if not mongo_uri:
            raise ValueError("MONGO_ATLAS_URI not found in .env file")

        # MongoDB connection details
        db_name = "development_analysis"
        collection_name = "combined_clean"

        print(f"Connecting to MongoDB Atlas - Database: {db_name}, Collection: {collection_name}")

        # Load data from MongoDB Atlas
        df = (data_io
              .from_atlas(
            db=db_name,
            collection=collection_name,
            connection_string=mongo_uri
        )
              .load(limit=5))  # Limit to 5 documents for testing

        # 1. Show basic info
        print("\n1. DataFrame Info:")
        print(df.info())

        # 2. Show first few rows
        print("\n2. First 5 documents:")
        print(df.head().to_string())

        # 3. Save to CSV in temp_data
        test_file = "test_mongo_export.csv"
        print(f"\n3. Saving to {test_file}...")
        try:
            saved_path = data_io._save_csv(df, "test_mongo_export")
            print(f"   ✓ Saved to: {saved_path}")

            # 4. Try saving again (should fail)
            print("\n4. Trying to save again (should fail with FileExistsError)...")
            try:
                data_io._save_csv(df, "test_mongo_export1")
            except FileExistsError as e:
                print(f"   ✓ Test passed - Got expected error: {e}")

        except Exception as e:
            print(f"   Error saving file: {e}")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    test_mongo_atlas()