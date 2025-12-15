from src.loaders.data_IO import DataIO
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'

# Load environment variables from .env file in project root
load_dotenv(env_path)

# Create a loader instance
loader = DataIO()

# Load the data
try:
    # Load from CSV
    csv_path = project_root / "temp_data" / "combined_clean.csv"  # Adjust path as needed
    df_csv = loader.from_csv(str(csv_path)).load()

    # Load from MongoDB Atlas
    df_atlas = loader.from_atlas(
        collection="combined_clean",
        db="development_analysis",
        connection_string=os.getenv("MONGO_ATLAS_URI")
    ).load()

    # Load from local MongoDB
    df_local = loader.from_local(
        collection="combined_clean",
        db="development_analysis"
    ).load()

    # Set display options
    pd.set_option('display.max_columns', None)  # Show all columns


    # Function to display data info
    def display_data_info(df, source_name):
        print(f"\n{source_name} Data:")
        print("=" * 50)
        if df is not None and not df.empty:
            print(f"Successfully loaded {len(df)} records")
            print("\nFirst few records:")
            print(df.head())
            print("\nUnique countries/regions:")
            print(df['country'].unique())
            print("\nDataFrame info:")
            print(df.info())
        else:
            print(f"No data found in {source_name}")


    # Display info for each data source
    display_data_info(df_csv, "CSV")
    display_data_info(df_atlas, "MongoDB Atlas")
    display_data_info(df_local, "Local MongoDB")

except Exception as e:
    print(f"Error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Looking for .env at: {env_path}")
    print(f".env exists: {env_path.exists()}")
    if env_path.exists():
        print(f"Contents of .env: {open(env_path).read()[:100]}...")