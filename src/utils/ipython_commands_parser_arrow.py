# Basic setup
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_parquet(r"C:\python\project_programming_for_ai\data\parsed\US.100+.parquet")

# Quick data inspection
df.head()
df.info()
df.describe()

# Check data quality
print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate timestamps:", df.duplicated(subset=["timestamp", "instrument"]).sum())

# Time difference analysis
time_diffs = df["timestamp"].diff().dropna()
print("\nTime between ticks (top 5):")
print(time_diffs.value_counts().head())

# Basic plot
plt.figure(figsize=(15, 7))
plt.plot(df['timestamp'], df['mid'], label='Mid Price', alpha=0.7)
plt.plot(df['timestamp'], df['bid'], label='Bid', alpha=0.5, linewidth=0.7)
plt.plot(df['timestamp'], df['ask'], label='Ask', alpha=0.5, linewidth=0.7)
plt.fill_between(df['timestamp'], df['bid'], df['ask'], color='gray', alpha=0.2)
plt.title(f'{df["instrument"].iloc[0]} - {df["timestamp"].dt.date.iloc[0]}')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()