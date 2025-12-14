import pandas as pd
import numpy as np
from src.loaders.data_IO import DataIO

# Load the data
print("Loading data...")
data_io = DataIO()
df = data_io.from_csv("combined_clean.csv").load()

# Check if required columns exist
required_cols = ['country', 'mean_temp', 'temp_std', 'temp_obs', 'year']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Warning: Missing columns: {missing_cols}")
    print("Available columns:", df.columns.tolist())
    exit(1)

# 1. Show missing values for each temperature column
print("\nMissing values per column:")
print(df[['mean_temp', 'temp_std', 'temp_obs']].isna().sum())

# 2. Show statistics by country
print("\nStatistics by country:")
for country in df['country'].unique():
    print(f"\n--- {country} ---")
    country_df = df[df['country'] == country]
    print(f"Total records: {len(country_df)}")

    # Show count of non-null values and basic stats for each temp column
    for col in ['mean_temp', 'temp_std', 'temp_obs']:
        non_null = country_df[col].count()
        print(f"\n{col}:")
        print(f"  Non-null values: {non_null} ({(non_null / len(country_df)) * 100:.1f}%)")
        if non_null > 0:
            print(f"  Mean: {country_df[col].mean():.2f}")
            print(f"  Min: {country_df[col].min():.2f}")
            print(f"  Max: {country_df[col].max():.2f}")
            print(f"  Std: {country_df[col].std():.2f}")

# 3. Count NaN values per country for each temperature column
print("\nNaN counts by country and temperature column:")
nan_counts = df.groupby('country')[['mean_temp', 'temp_std', 'temp_obs']].apply(lambda x: x.isna().sum())
print(nan_counts)

# 4. Show years with data for each country
print("\nYears with data by country:")
for country in df['country'].unique():
    years = sorted(df[df['country'] == country]['year'].unique())
    print(f"{country}: {len(years)} years ({min(years)}-{max(years)})")