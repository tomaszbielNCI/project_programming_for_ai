import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../../temp_data/combined_clean.csv'
df = pd.read_csv(file_path)

# Filter data for Poland
poland_data = df[df['country'] == 'Poland'].sort_values('year')

# Display basic statistics
print("Temperature data for Poland:")
print("-" * 30)
print(f"Time period: {int(poland_data['year'].min())} - {int(poland_data['year'].max())}")
print(f"Number of records: {len(poland_data)}")
print("\nTemperature statistics:")
print(poland_data['mean_temp'].describe())

# Check for suspicious values
suspicious = poland_data[poland_data['mean_temp'] < -20]  # Unrealistically low for Poland
if not suspicious.empty:
    print("\nSuspiciously low temperature records:")
    print(suspicious[['year', 'mean_temp']].to_string(index=False))

# Plot temperature trends
plt.figure(figsize=(12, 6))
plt.plot(poland_data['year'], poland_data['mean_temp'], 'b.-')
plt.axhline(y=poland_data['mean_temp'].mean(), color='r', linestyle='--', label='Average')
plt.title('Average Temperature in Poland Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
output_file = 'poland_temperature_trend.png'
plt.savefig(output_file)
print(f"\nTemperature trend plot saved as: {output_file}")

# Check for data quality issues
print("\nData quality check:")
print("-" * 30)
print(f"Missing values: {poland_data['mean_temp'].isnull().sum()}")
print(f"Zero values: {(poland_data['mean_temp'] == 0).sum()}")
print(f"Positive temperatures: {len(poland_data[poland_data['mean_temp'] > 0])}")
print(f"Negative temperatures: {len(poland_data[poland_data['mean_temp'] < 0])}")

# Check for potential data entry errors
if not poland_data.empty:
    temp_diff = poland_data['mean_temp'].diff().abs()
    large_jumps = temp_diff[temp_diff > 5]  # Check for unrealistic temperature jumps
    if not large_jumps.empty:
        print("\nPotential data entry errors (large temperature jumps):")
        years = poland_data.loc[large_jumps.index, 'year'].values
        print(f"Year: {years}, Temperature jumps: {large_jumps.values}°C")
