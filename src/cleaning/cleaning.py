# cleaning.py
"""
Sequence of commands from IPython session:
1. from pathlib import Path
2. from src.loaders.data_IO import DataIO
3. data_io = DataIO()
4. df = data_io.from_csv("combined_clean.csv").load()
5. df.describe(include='all')
6. df.count()
7. unique_countries = df[df['mean_temp'].notna()]['country'].unique()
8. len(unique_countries)
9. filtered_df = df[df['mean_temp'].notna()].copy()
10. columns_to_keep = ['country', 'year', 'population', 'energy_use', 'emissions', 'renewable_pct', 'mean_temp', 'gdp']
11. clean_df = filtered_df[columns_to_keep].copy()
!!! clean_df = data_io.from_csv("cleaned_data.csv").load()
12. data_io.save(clean_df, target='csv', name='cleaned_data')
13. temp_by_year = df[df['mean_temp'].notna()].groupby('year').size().reset_index(name='count')
14. latest_year = df[df['mean_temp'].notna()]['year'].max()
15. clean_df['country'].unique()
16. original_countries = df['country'].unique()
    cleaned_countries = clean_df['country'].unique()
    set_original = set(original_countries)
    set_cleaned = set(cleaned_countries)
    missing_countries = set_original - set_cleaned
    columns = ['population', 'energy_use', 'emissions', 'renewable_pct', 'gdp']
    numeric_df = df[columns]
    correlation_matrix = numeric_df.corr()
    correlation_matrix

"""

from src.loaders.data_IO import DataIO
import pandas as pd
from pathlib import Path


def load_country_list(file_path='countries_of_interest.txt'):
    """
    Load country list from a text file containing a Python list of countries.

    Args:
        file_path: Path to text file with a Python list of country names.

    Returns:
        List of country names or empty list if file not found or invalid.
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Safely evaluate the content as a Python list
                countries = eval(content)
                if not isinstance(countries, list):
                    print(f"Warning: File {file_path} does not contain a valid list. No filtering by country will be applied.")
                    return []
                print(f"Loaded {len(countries)} countries from {file_path}")
                return countries
        else:
            print(f"Warning: Country list file {file_path} not found. No filtering by country will be applied.")
            return []
    except Exception as e:
        print(f"Error loading country list: {e}")
        return []


def clean_data(country_list_file='countries_of_interest.txt'):
    """Function for data cleaning - selects specific columns and filters by country list"""

    # Initialize DataIO
    data_io = DataIO()

    # Load country list
    countries_to_keep = load_country_list(country_list_file)

    # Load data
    print("Loading data from combined_clean.csv...")
    df = data_io.from_csv("combined_clean.csv").load()
    print(f"Loaded {len(df)} rows of data.")

    # If we have a country list, filter the data
    if countries_to_keep:
        before = len(df)
        
        # Debug: Show some sample country names from the data
        sample_countries = df['country'].drop_duplicates().sample(min(5, len(df))).tolist()
        print(f"Sample country names in data: {sample_countries}")
        
        # Check how many countries from our list exist in the data
        matching_countries = set(countries_to_keep) & set(df['country'].unique())
        print(f"Found {len(matching_countries)} matching countries in the data")
        
        if len(matching_countries) == 0:
            print("Warning: No countries from the list match the data. First 5 countries in the list:", countries_to_keep[:5])
        
        df = df[df['country'].isin(countries_to_keep)].copy()
        print(f"Filtered to {len(df)} rows (removed {before - len(df)}) based on country list.")

    # Select columns to keep
    columns_to_keep = ['country', 'year', 'population', 'energy_use',
                       'emissions', 'renewable_pct', 'mean_temp', 'gdp']

    # Create clean DataFrame (only selected columns)
    clean_df = df[columns_to_keep].copy()

    # Information about the resulting dataset
    print(f"\nFinal dataset contains {len(clean_df)} rows and {len(clean_df.columns)} columns")
    print(f"Columns in the dataset: {list(clean_df.columns)}")

    # Information about year range
    latest_year = clean_df['year'].max()
    earliest_year = clean_df['year'].min()
    print(f"\nYear range in data: {earliest_year} - {latest_year}")

    # Number of unique countries
    unique_countries = clean_df['country'].unique()
    print(f"Number of unique countries: {len(unique_countries)}")

    # Data distribution across years (if any data left)
    if len(clean_df) > 0:
        temp_by_year = clean_df.groupby('year').size().reset_index(name='count')
        print(f"\nNumber of observations per year: {temp_by_year['count'].iloc[0]}")
    else:
        print("\nNo data left after filtering.")

    # Save to CSV file
    print("\nSaving cleaned data...")
    save_path = data_io.save(clean_df, target='csv', name='cleaned_data')
    print(f"Data saved to: {save_path}")

    return clean_df


if __name__ == "__main__":
    cleaned_data = clean_data()

    # Display sample data
    print("\nSample 5 rows of cleaned data:")
    print(cleaned_data.head())