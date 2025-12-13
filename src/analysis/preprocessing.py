# preprocessing.py
"""
Panel Data Preprocessing Module
Transforms cleaned data into panel data format with proper transformations
for time-series and cross-sectional analysis.

Sequence of commands from IPython session:
missing_by_country_year = clean_df.groupby(['country', 'year']).apply(
    lambda x: x.select_dtypes(include='number').isna().sum()
).unstack()
missing_analysis = clean_df.groupby('country').apply(
    lambda x: x.select_dtypes(include='number').isna().sum()
)
clean_df.isna().sum().sort_values(ascending=False)

"""

from src.loaders.data_IO import DataIO
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os


class PanelDataPreprocessor:
    """Preprocessor for panel data with country-year structure"""
    DEFAULT_VARIABLES = ['emissions', 'energy_use', 'gdp', 'population', 'mean_temp']

    def __init__(self, data_io: DataIO = None):
        self.data_io = data_io or DataIO()
        self.df = None

    @staticmethod
    def create_lagged_variables(df: pd.DataFrame,
                              variables: List[str] = None,
                              lags: List[int] = None) -> pd.DataFrame:
        if lags is None:
            lags = [1]  # Default to only 1 lag
        if variables is None:
            variables = PanelDataPreprocessor.DEFAULT_VARIABLES
        df = df.sort_values(by=['country', 'year']).copy()
        for var in variables:
            for lag in lags:
                df[f'{var}_lag{lag}'] = df.groupby('country')[var].shift(lag)
        total_rows = len(df)
        # Create a list of all lagged variable names to check for NAs
        lagged_vars = [f'{var}_lag{max(lags)}' for var in variables]
        non_na_rows = len(df.dropna(subset=lagged_vars))
        print(f"Created lagged variables for {len(variables)} variables with lags {lags}")
        print(f"Original rows: {total_rows}, Complete cases (no NA in max lag): {non_na_rows}")
        print(f"Observations lost due to lags: {total_rows - non_na_rows}")
        return df

    def create_differences(self, df: pd.DataFrame,
                           variables: List[str] = None) -> pd.DataFrame:
        """
        Create first differences (delta) for specified variables.
        """
        if variables is None:
            variables = PanelDataPreprocessor.DEFAULT_VARIABLES

        df = df.sort_values(by=['country', 'year']).copy()

        for var in variables:
            if var in df.columns:
                df[f'{var}_diff'] = df.groupby('country')[var].diff()
            else:
                print(f"Warning: Column '{var}' not found for differencing")

        return df

    def create_log_transformations(self, df: pd.DataFrame,
                                 variables: List[str] = None,
                                 epsilon: float = 1e-6) -> pd.DataFrame:
        """
        Apply log transformations to specified variables with handling for non-positive values.

        Args:
            df: Input DataFrame
            variables: List of variable names to transform (default: DEFAULT_VARIABLES)
            epsilon: Small value to add to handle zeros/negatives

        Returns:
            DataFrame with log-transformed variables added
        """
        if variables is None:
            variables = [var for var in self.DEFAULT_VARIABLES if var != 'mean_temp']  # Skip temperature by default
            
        df = df.copy()
        
        for var in variables:
            if var not in df.columns:
                print(f"Warning: Column '{var}' not found in DataFrame")
                continue
                
            # Check for non-positive values that would cause issues with log
            if (df[var] <= 0).any():
                print(f"Adding epsilon={epsilon} to non-positive values in '{var}' before log transform")
                df[f'log_{var}'] = np.log(df[var] + epsilon)
            else:
                df[f'log_{var}'] = np.log(df[var])
                
            print(f"Created log-transformed variable: log_{var}")
            
        return df

    def handle_missing_panel_data(self, df: pd.DataFrame,
                                  method: str = 'interpolate',
                                  group_col: str = 'country',
                                  critical_vars: List[str] = None,
                                  zero_fill_vars: List[str] = None) -> pd.DataFrame:
        """
        Handle missing data for panel analysis with variable-specific strategies.

        Args:
            df: Input DataFrame
            method: Imputation method ('interpolate' or 'drop')
            group_col: Column to group by (default: 'country')
            critical_vars: Variables where complete missing leads to country removal
            zero_fill_vars: Variables where complete missing is filled with 0

        Returns:
            DataFrame with handled missing values
        """
        if critical_vars is None:
            critical_vars = ['emissions', 'gdp', 'energy_use', 'population']

        if zero_fill_vars is None:
            zero_fill_vars = ['renewable_pct']

        df = df.copy()
        original_rows = len(df)
        original_countries = df[group_col].nunique()

        print(f"Starting missing data handling for {original_countries} countries, {original_rows} rows")

        # Remove countries with complete missing in critical variables
        countries_to_remove = set()

        for var in critical_vars:
            if var in df.columns:
                # Find countries where ALL values are NaN for this variable
                complete_missing = df.groupby(group_col)[var].apply(lambda x: x.isna().all())
                missing_countries = complete_missing[complete_missing].index.tolist()

                if missing_countries:
                    print(
                        f"  Removing {len(missing_countries)} countries with complete missing {var}: {missing_countries[:10]}{'...' if len(missing_countries) > 10 else ''}")
                    countries_to_remove.update(missing_countries)

        # Remove identified countries
        if countries_to_remove:
            df = df[~df[group_col].isin(countries_to_remove)]
            print(f"  Removed {len(countries_to_remove)} countries total")

        # Group for interpolation
        df = df.sort_values(by=[group_col, 'year']).reset_index(drop=True)

        # Process each variable group separately
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col not in df.columns:
                continue

            # Skip temperature if present
            if col == 'mean_temp':
                continue

            print(f"  Processing {col}:")

            # Check if variable needs zero-fill for complete missing
            if col in zero_fill_vars:
                # For zero-fill variables: fill complete missing with 0, partial with interpolate
                for country in df[group_col].unique():
                    country_mask = df[group_col] == country
                    country_data = df.loc[country_mask, col]

                    if country_data.isna().all():
                        # Complete missing → fill with 0
                        df.loc[country_mask, col] = 0
                        print(f"    {country}: Complete missing → filled with 0")
                    elif country_data.isna().any() and country_data.notna().sum() >= 2:
                        # Partial missing with ≥2 points → interpolate
                        df.loc[country_mask, col] = df.loc[country_mask, col].interpolate(
                            method='linear', limit_direction='both'
                        )
                        # Fill any remaining NA with 0
                        df.loc[country_mask, col] = df.loc[country_mask, col].fillna(0)
                    elif country_data.isna().any():
                        # Only 1 data point → can't interpolate, fill with that value or 0
                        non_na_value = country_data.dropna().iloc[0] if country_data.notna().any() else 0
                        df.loc[country_mask, col] = non_na_value

            elif col in critical_vars:
                # For critical variables: interpolate if possible, otherwise keep (already removed complete missing)
                for country in df[group_col].unique():
                    country_mask = df[group_col] == country
                    country_data = df.loc[country_mask, col]

                    if country_data.isna().any() and country_data.notna().sum() >= 2:
                        # Partial missing with ≥2 points → interpolate
                        df.loc[country_mask, col] = df.loc[country_mask, col].interpolate(
                            method='linear', limit_direction='both'
                        )
                    elif country_data.isna().any() and country_data.notna().sum() == 1:
                        # Only 1 data point → use that value for all years
                        non_na_value = country_data.dropna().iloc[0]
                        df.loc[country_mask, col] = non_na_value

        # For interpolate method: apply to remaining numeric columns (excluding already processed)
        if method == 'interpolate':
            remaining_numeric = [col for col in numeric_cols
                                 if col not in zero_fill_vars + critical_vars + ['mean_temp']]

            for col in remaining_numeric:
                if col in df.columns and df[col].isna().any():
                    # Groupwise interpolation
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda x: x.interpolate(method='linear', limit_direction='both')
                        if x.notna().sum() >= 2 else x
                    )

        # Final cleanup: drop rows with any remaining NA (except temperature)
        columns_to_check = [col for col in df.columns if col != 'mean_temp']
        df_clean = df.dropna(subset=columns_to_check)

        removed_rows = len(df) - len(df_clean)
        if removed_rows > 0:
            print(f"  Dropped {removed_rows} rows with remaining NA values")

        print(f"  Final: {df_clean[group_col].nunique()} countries, {len(df_clean)} rows")
        print(f"  Removed {original_countries - df_clean[group_col].nunique()} countries total")

        return df_clean


    def check_panel_balance(self, df: pd.DataFrame,
                          entity_col: str = 'country',
                          time_col: str = 'year') -> Dict:
        """
        Check if panel data is balanced and provide balance statistics.

        Args:
            df: Input DataFrame
            entity_col: Column name for entity grouping (default: 'country')
            time_col: Column name for time period (default: 'year')

        Returns:
            Dictionary with balance statistics including:
            - is_balanced: bool indicating if panel is balanced
            - entities: number of unique entities
            - min_obs: minimum observations per entity
            - max_obs: maximum observations per entity
            - time_periods: list of all time periods
            - missing_entries: DataFrame of missing entity-time combinations
        """
        # Count observations per entity
        obs_per_entity = df.groupby(entity_col).size()
        
        # Get all unique time periods
        all_periods = sorted(df[time_col].unique())
        all_entities = df[entity_col].unique()
        
        # Create a MultiIndex of all possible entity-time combinations
        full_idx = pd.MultiIndex.from_product(
            [all_entities, all_periods],
            names=[entity_col, time_col]
        )
        
        # Find missing combinations
        panel_idx = df.set_index([entity_col, time_col]).index
        missing = full_idx.difference(panel_idx)
        
        # Create DataFrame of missing entries
        missing_entries = pd.DataFrame(
            index=missing
        ).reset_index() if not missing.empty else pd.DataFrame()
        
        # Determine if panel is balanced
        is_balanced = len(set(obs_per_entity)) == 1 and missing.empty
        
        # Prepare result dictionary
        result = {
            'is_balanced': is_balanced,
            'entities': len(all_entities),
            'time_periods': len(all_periods),
            'min_obs': int(obs_per_entity.min()),
            'max_obs': int(obs_per_entity.max()),
            'avg_obs': float(obs_per_entity.mean()),
            'total_obs': len(df),
            'missing_entries_count': len(missing),
            'missing_entries': missing_entries
        }
        
        # Print summary
        print(f"Panel Balance Check:")
        print(f"- Balanced: {'Yes' if is_balanced else 'No'}")
        print(f"- Entities: {result['entities']}")
        print(f"- Time periods: {result['time_periods']} ({all_periods[0]} to {all_periods[-1]})")
        print(f"- Observations per entity: {result['min_obs']} to {result['max_obs']} (avg: {result['avg_obs']:.1f})")
        print(f"- Total observations: {result['total_obs']}")
        print(f"- Missing entries: {result['missing_entries_count']}")
        
        return result




    def save_preprocessed_data(self, df: pd.DataFrame, name: str = 'preprocessed_data') -> str:
        """
        Save preprocessed data using DataIO.

        Args:
            df: DataFrame to save
            name: Base name for saved file (without extension)

        Returns:
            str: Path to saved file
        """
        if not hasattr(self, 'data_io') or self.data_io is None:
            self.data_io = DataIO()
            
        save_path = self.data_io.save(df, target='csv', name=name)
        return save_path

    def run_full_preprocessing(self,
                             input_file: str = 'cleaned_data.csv',
                             output_file: str = 'preprocessed_data.csv',
                             config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.

        Args:
            input_file: Input data file (default: 'cleaned_data.csv')
            output_file: Output data file (default: 'preprocessed_data.csv')
            config: Configuration dictionary for preprocessing steps. If None, uses defaults.
                   Example:
                   {
                       'variables': ['gdp', 'emissions', 'energy_use', 'population'],
                       'lags': [1, 2],
                       'create_differences': True,
                       'create_logs': True,
                       'missing_data_method': 'ffill_bfill',  # 'ffill_bfill', 'interpolate', or 'drop'
                       'check_balance': True
                   }

        Returns:
            Fully preprocessed DataFrame
        """
        # Set default config if none provided
        if config is None:
            config = {
                'variables': self.DEFAULT_VARIABLES,
                'lags': [1, 2],
                'create_differences': True,
                'create_logs': True,
                'missing_data_method': 'ffill_bfill',
                'check_balance': True
            }
            
        print("Starting data preprocessing...")
        
        # 1. Load data
        print(f"Loading data from {input_file}")
        df = self.data_io.load_data(input_file)
        
        # 2. Create lagged variables
        print("\nCreating lagged variables...")
        df = self.create_lagged_variables(
            df,
            variables=config.get('variables'),
            lags=config.get('lags', [1, 2])
        )
        
        # 3. Create differences if requested
        if config.get('create_differences', True):
            print("\nCreating first differences...")
            df = self.create_differences(
                df,
                variables=config.get('variables')
            )
        
        # 4. Create log transformations if requested
        if config.get('create_logs', True):
            print("\nCreating log transformations...")
            df = self.create_log_transformations(
                df,
                variables=config.get('variables')
            )
        
        # 5. Handle missing data
        missing_method = config.get('missing_data_method', 'ffill_bfill')
        print(f"\nHandling missing data using method: {missing_method}")
        df = self.handle_missing_panel_data(df, method=missing_method)
        
        # 6. Check panel balance if requested
        if config.get('check_balance', True):
            print("\nChecking panel balance...")
            balance_info = self.check_panel_balance(df)
            
            # If panel is unbalanced, show some missing entries
            if not balance_info['is_balanced'] and not balance_info['missing_entries'].empty:
                print("\nSample of missing entries:")
                print(balance_info['missing_entries'].head())
        
        # 7. Save preprocessed data
        if output_file:
            output_path = self.save_preprocessed_data(df, output_file.replace('.csv', ''))
            print(f"\nPreprocessing complete. Data saved to: {output_path}")
        
        return df


def preprocess_data():
    """Function for data preprocessing - analogous to clean_data() in cleaning.py"""

    # Initialize DataIO
    data_io = DataIO()

    # Load data
    print("Loading data from cleaned_data.csv...")
    data_io.from_csv("cleaned_data.csv")
    df = data_io.load()
    print(f"Loaded {len(df)} rows of data.")

    # Initialize preprocessor
    preprocessor = PanelDataPreprocessor(data_io)

    # Basic preprocessing steps (to be implemented)
    print("\nPerforming preprocessing...")

    # Example transformations (to be implemented)
    # 1. Add per capita calculations
    # 2. Create lagged variables
    # 3. Handle missing data

    # For now, just copy the data
    processed_df = df.copy()

    # Information about the resulting dataset
    print(f"\nFinal preprocessed dataset contains {len(processed_df)} rows and {len(processed_df.columns)} columns")

    # Save to CSV file
    print("\nSaving preprocessed data...")
    save_path = data_io.save(processed_df, target='csv', name='preprocessed_data')
    print(f"Data saved to: {save_path}")

    return processed_df



if __name__ == "__main__":
    processed_data = preprocess_data()

    # Display sample data
    print("\nSample 5 rows of preprocessed data:")
    print(processed_data.head())