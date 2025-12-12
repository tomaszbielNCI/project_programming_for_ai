import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style - using 'seaborn-v0_8' which is the modern equivalent
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")  # Using seaborn's theming system

# Rest of your code remains the same
base_path = Path(r'/mongo/data/development_analysis')
df_clean = pd.read_csv(base_path / 'temperature_clean.csv')
df_raw = pd.read_csv(base_path / 'temperature_raw.csv')

# Convert year to datetime for better x-axis
df_clean['date'] = pd.to_datetime(df_clean['year'], format='%Y')
df_raw['date'] = pd.to_datetime(df_raw['year'], format='%Y')

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Line plot of temperature anomalies
sns.lineplot(data=df_raw, x='date', y='temp_anomaly',
             label='Raw Data', ax=ax1, alpha=0.6)
sns.lineplot(data=df_clean, x='date', y='temp_anomaly',
             label='Cleaned Data', ax=ax1, linewidth=2.5)
ax1.set_title('Global Temperature Anomalies Over Time', fontsize=14, pad=15)
ax1.set_ylabel('Temperature Anomaly (°C)')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Boxplot to show distribution
sns.boxplot(data=pd.concat([
    df_raw[['temp_anomaly']].assign(Source='Raw'),
    df_clean[['temp_anomaly']].assign(Source='Clean')
], ignore_index=True),
    x='Source', y='temp_anomaly', ax=ax2)
ax2.set_title('Distribution of Temperature Anomalies', fontsize=14, pad=15)
ax2.set_ylabel('Temperature Anomaly (°C)')
ax2.set_xlabel('')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical comparison
print("\nStatistical Summary:")
print("\nRaw Data:")
print(df_raw['temp_anomaly'].describe())
print("\nCleaned Data:")
print(df_clean['temp_anomaly'].describe())

# Test for significant difference (if you have scipy installed)
try:
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(df_raw['temp_anomaly'].dropna(),
                                     df_clean['temp_anomaly'].dropna(),
                                     equal_var=False)
    print(f"\nT-test for difference in means: t = {t_stat:.3f}, p = {p_value:.3f}")
    if p_value < 0.05:
        print("The difference in means is statistically significant (p < 0.05)")
    else:
        print("No statistically significant difference found between the datasets")
except ImportError:
    print("\nInstall scipy for statistical tests: pip install scipy")