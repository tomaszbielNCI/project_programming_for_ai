from src.loaders.data_IO import DataIO
import pandas as pd
import os

# Create a loader instance
loader = DataIO()

# Load the CSV file
try:
    df = loader.from_csv("combined_clean.csv").load()

    pd.set_option('display.max_columns', None)  # Show all columns
    print(f"Successfully loaded {len(df)} records")
    print("\nFirst few records:")
    print(df.head())
    
    # Display unique countries
    print("\nUnique countries/regions:")
    print(df['country'].unique())
    
    # Basic info about the DataFrame
    print("\nDataFrame info:")
    print(df.info())
except Exception as e:
    print(f"Error: {e}")