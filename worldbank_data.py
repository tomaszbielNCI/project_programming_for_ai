import pandas as pd
import os
import sys
from datetime import datetime
from pandas_datareader import wb
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

def get_worldbank_data(indicators, start_year=1960, end_year=2023):
    """
    Fetch data from World Bank using pandas-datareader
    
    Args:
        indicators (dict): Dictionary of indicator codes and their descriptions
                          Example: {'NY.GDP.MKTP.CD': 'GDP (current US$)'}
        start_year (int): Start year for data (default: 1960)
        end_year (int): End year for data (default: 2023)
        
    Returns:
        pandas.DataFrame: DataFrame containing the requested data
    """
    try:
        print(f"Fetching data for {len(indicators)} indicators from {start_year} to {end_year}")
        print("-" * 80)
        
        # Display indicators being fetched
        indicator_codes = list(indicators.keys())
        for code in indicator_codes:
            print(f"{code}: {indicators[code]}")
            
        print("\nThis might take a few minutes...")
        
        # Create an empty list to store DataFrames
        dfs = []
        
        # Fetch data for each indicator one by one
        for code in indicator_codes:
            try:
                print(f"\nFetching {code}...")
                # Fetch data for the current indicator
                df_indicator = wb.download(
                    indicator=code,
                    start=start_year,
                    end=end_year
                )
                
                if not df_indicator.empty:
                    # Reset index to make country and year columns
                    df_indicator = df_indicator.reset_index()
                    # Rename the value column to the indicator code
                    df_indicator = df_indicator.rename(columns={code: 'value'})
                    # Add indicator code and description as columns
                    df_indicator['indicator_code'] = code
                    df_indicator['indicator_name'] = indicators[code]
                    dfs.append(df_indicator)
                    print(f"Successfully fetched {code}")
                else:
                    print(f"Warning: No data returned for indicator: {code}")
                    
            except Exception as e:
                print(f"Error fetching indicator {code}: {str(e)}")
        
        if not dfs:
            print("Error: No data was fetched successfully")
            return None
            
        # Combine all DataFrames
        df = pd.concat(dfs, ignore_index=True)
        
        # Pivot the data to have indicators as columns
        df_pivoted = df.pivot_table(
            index=['country', 'year'],
            columns='indicator_name',
            values='value'
        ).reset_index()
        
        # Clean up column names
        df_pivoted.columns.name = None
        
        # Rename columns for consistency
        df_pivoted = df_pivoted.rename(columns={
            'country': 'country_name',
            'year': 'year'
        })
        
        # Add country code (using the first 3 characters of the country name for now)
        df_pivoted['country_code'] = df_pivoted['country_name'].str[:3].str.upper()
        
        # Reorder columns
        indicator_columns = [col for col in df_pivoted.columns if col not in ['country_name', 'country_code', 'year']]
        columns = ['country_code', 'country_name', 'year'] + indicator_columns
        df_final = df_pivoted[columns]
        
        print(f"\nSuccessfully fetched data with shape: {df_final.shape}")
        return df_final
        
    except Exception as e:
        import traceback
        print("\nError in get_worldbank_data:")
        traceback.print_exc()
        return None

def get_energy_climate_indicators():
    """Return a dictionary of World Bank indicators for energy and climate analysis."""
    return {
        # Energy Indicators
        'EG.USE.PCAP.KG.OE': 'Energy use (kg of oil equivalent per capita)',
        'EG.FEC.RNEW.ZS': 'Renewable energy consumption (% of total final energy consumption)',
        'EG.ELC.RNEW.ZS': 'Renewable electricity output (% of total electricity output)',
        'EG.ELC.FOSL.ZS': 'Fossil fuel energy consumption (% of total)',
        
        # CO2 and Emissions
        'EN.ATM.CO2E.PC': 'CO2 emissions (metric tons per capita)',
        'EN.ATM.CO2E.KT': 'CO2 emissions (kt)',
        'EN.ATM.GHGT.KT.CE': 'Total greenhouse gas emissions (kt of CO2 equivalent)',
        'EN.ATM.METH.KT.CE': 'Methane emissions (kt of CO2 equivalent)',
        
        # Economic Indicators
        'NY.GDP.PCAP.CD': 'GDP per capita (current US$)',
        'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',
        'NE.TRD.GNFS.ZS': 'Trade (% of GDP)',
        
        # Population and Urbanization
        'SP.POP.TOTL': 'Population, total',
        'SP.URB.TOTL.IN.ZS': 'Urban population (% of total population)',
        'SP.POP.GROW': 'Population growth (annual %)',
        
        # Climate and Environment
        'AG.LND.FRST.ZS': 'Forest area (% of land area)',
        'EN.ATM.PM25.MC.M3': 'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)',
        
        # Energy Mix
        'EG.ELC.NUCL.ZS': 'Electricity production from nuclear sources (% of total)',
        'EG.ELC.HYRO.ZS': 'Electricity production from hydroelectric sources (% of total)',
        'EG.USE.COMM.FO.ZS': 'Fossil fuel energy consumption (% of total)'
    }

def main():
    # Get all energy and climate indicators
    indicators = get_energy_climate_indicators()
    
    # Create output directory if it doesn't exist
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'worldbank_energy_climate_{timestamp}.csv')
    
    try:
        # Get data from World Bank (1960-2023)
        print("Starting data download from World Bank...")
        df = get_worldbank_data(indicators, 1960, 2023)
        
        # Display basic info
        print("\nDataFrame Info:")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nData successfully saved to: {os.path.abspath(output_file)}")
        
        # Additional information
        countries = df['country_code'].nunique()
        years = sorted(df['year'].unique())
        print(f"\nData covers {countries} countries from {min(years)} to {max(years)}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Removed incomplete file: {output_file}")
        
        # Print some basic statistics
        print("\nSummary statistics:")
        print(df.describe())
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Run the main function
    main()
