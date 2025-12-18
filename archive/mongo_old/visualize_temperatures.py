import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Create output directory for visualizations
output_dir = Path("temperature_visualizations")
output_dir.mkdir(exist_ok=True)

# Load the data
file_path = '../../temp_data/combined_clean.csv'
df = pd.read_csv(file_path)

# Get unique years and countries
years = sorted(df['year'].unique())
countries = sorted(df['country'].unique())

print(f"Found {len(countries)} countries and {len(years)} years of data")
print(f"Generating visualizations in: {output_dir.absolute()}")

# Create a color map for consistent colors across plots
colors = plt.cm.get_cmap('tab20')(range(20))
color_map = {country: colors[i % len(colors)] for i, country in enumerate(countries)}

# 1. Plot for each year, showing all countries
yearly_dir = output_dir / "by_year"
yearly_dir.mkdir(exist_ok=True)

for year in years:
    plt.figure(figsize=(15, 8))
    year_data = df[df['year'] == year].sort_values('mean_temp', ascending=False)
    
    # Only show top 30 countries to avoid clutter
    top_countries = year_data.head(30)
    
    bars = plt.barh(
        top_countries['country'],
        top_countries['mean_temp'],
        color=[color_map.get(c, 'gray') for c in top_countries['country']]
    )
    
    plt.title(f'Average Temperature by Country - {int(year)}', fontsize=14)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add temperature values on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + (0.5 if width >= 0 else -1.5), 
            bar.get_y() + bar.get_height()/2,
            f'{width:.1f}°C',
            va='center',
            ha='left' if width >= 0 else 'right'
        )
    
    plt.tight_layout()
    plt.savefig(yearly_dir / f'temperatures_{int(year)}.png', dpi=150, bbox_inches='tight')
    plt.close()

# 2. Plot temperature trends for each country
countries_dir = output_dir / "by_country"
countries_dir.mkdir(exist_ok=True)

for country in countries:
    country_data = df[df['country'] == country].sort_values('year')
    
    plt.figure(figsize=(12, 6))
    plt.plot(
        country_data['year'], 
        country_data['mean_temp'],
        'o-',
        color=color_map.get(country, 'gray'),
        label=country
    )
    
    plt.title(f'Temperature Trend: {country}', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Temperature (°C)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    
    # Filter out any infinite or NaN values before plotting
    valid_data = country_data[country_data['mean_temp'].notna() & 
                            pd.notnull(country_data['mean_temp']) & 
                            (country_data['mean_temp'] != float('inf')) & 
                            (country_data['mean_temp'] != float('-inf'))]
    
    if len(valid_data) > 0:
        # Get min and max values
        min_temp = valid_data['mean_temp'].min()
        max_temp = valid_data['mean_temp'].max()
        
        # Only add min line if we have valid data
        if pd.notna(min_temp) and min_temp != float('inf') and min_temp != float('-inf'):
            min_row = valid_data.loc[valid_data['mean_temp'].idxmin()]
            min_year = min_row['year']
            plt.axhline(y=min_temp, color='r', linestyle='--', alpha=0.5)
            plt.annotate(f'Min: {min_temp:.1f}°C ({int(min_year)})', 
                        xy=(min_year, min_temp), 
                        xytext=(10, 10), 
                        textcoords='offset points', 
                        color='r')
        
        # Only add max line if it's different from min and we have valid data
        if (pd.notna(max_temp) and max_temp != float('inf') and 
            max_temp != float('-inf') and max_temp != min_temp):
            max_row = valid_data.loc[valid_data['mean_temp'].idxmax()]
            max_year = max_row['year']
            plt.axhline(y=max_temp, color='g', linestyle='--', alpha=0.5)
            plt.annotate(f'Max: {max_temp:.1f}°C ({int(max_year)})', 
                        xy=(max_year, max_temp), 
                        xytext=(10, -20), 
                        textcoords='offset points', 
                        color='g')
    else:
        print(f"Warning: No valid temperature data for {country} - skipping annotations")
    
    plt.tight_layout()
    
    # Create a safe filename (replace special characters)
    safe_country = "".join(c if c.isalnum() else "_" for c in country)
    plt.savefig(countries_dir / f'{safe_country}_trend.png', dpi=150, bbox_inches='tight')
    plt.close()

# 3. Create a heatmap of all countries and years
plt.figure(figsize=(15, max(8, len(countries) * 0.3)))
heatmap_data = df.pivot(index='country', columns='year', values='mean_temp')

# Sort countries by average temperature
heatmap_data['avg'] = heatmap_data.mean(axis=1)
heatmap_data = heatmap_data.sort_values('avg', ascending=False).drop('avg', axis=1)

plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Heatmap by Country and Year', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.xticks(range(len(years)), [int(y) for y in years], rotation=45)
plt.yticks(range(len(countries)), heatmap_data.index)
plt.tight_layout()
plt.savefig(output_dir / 'temperature_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualizations created successfully!")
print(f"- Yearly comparisons: {yearly_dir.absolute()}")
print(f"- Country trends: {countries_dir.absolute()}")
print(f"- Heatmap: {output_dir.absolute()}/temperature_heatmap.png")
