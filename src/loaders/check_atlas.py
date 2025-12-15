from src.loaders.data_IO import DataIO
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables the same way as in próba.py
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


def check_connection():
    try:
        print("Connecting using the same method as próba.py...")

        # Use the same DataIO class as in próba.py
        loader = DataIO()

        # Test the Atlas connection
        print("\nTesting Atlas connection...")
        df_atlas = loader.from_atlas(
            collection="combined_clean",
            db="development_analysis",
            connection_string=os.getenv("MONGO_ATLAS_URI")
        ).load()

        print(f"✅ Successfully connected to Atlas and loaded {len(df_atlas)} records")
        print("\nFirst few records:")
        print(df_atlas.head())

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("=== MongoDB Connection Test ===")
    check_connection()
    input("\nPress Enter to exit...")