"""
diagnostic_all_countries.py
Simplified diagnostic for all countries - summary statistics only
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import warnings
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.loaders.data_IO import DataIO

warnings.filterwarnings('ignore')


class AllCountriesDiagnostic:
    def __init__(self):
        self.data_io = DataIO()
        self.data = None
        self.results_dir = None
        self.summary_results = []

    def setup_results_dir(self):
        """Create unique results directory"""
        base_path = Path(__file__).parent.parent.parent / "results" / "diagnostic_summary"
        base_path.mkdir(parents=True, exist_ok=True)

        run_number = 1
        while (base_path / f"run_{run_number:03d}").exists():
            run_number += 1

        self.results_dir = base_path / f"run_{run_number:03d}"
        self.results_dir.mkdir(exist_ok=True)

        print(f"📂 Results directory: {self.results_dir}")
        return self.results_dir

    def load_data(self):
        """Load preprocessed data"""
        print("📊 Loading data...")
        self.data = self.data_io.from_csv("preprocessed_data_5.csv").load()
        print(f"✅ Loaded {len(self.data)} rows, {self.data['country'].nunique()} countries")
        return self.data

    def analyze_country(self, country_name, target='emissions'):
        """Quick analysis for a single country"""
        country_data = self.data[self.data['country'] == country_name].copy()
        country_data = country_data.sort_values('year')

        if len(country_data) < 5:
            return None

        result = {
            'country': country_name,
            'n_obs': len(country_data),
            'min_year': country_data['year'].min(),
            'max_year': country_data['year'].max(),
            'mean_emissions': country_data[target].mean(),
            'trend': 'increasing' if country_data[target].iloc[-1] > country_data[target].iloc[0] else 'decreasing'
        }

        # ADF Test (stationarity)
        try:
            adf_result = adfuller(country_data[target].dropna())
            result['adf_pvalue'] = adf_result[1]
            result['is_stationary'] = adf_result[1] < 0.05
        except:
            result['adf_pvalue'] = np.nan
            result['is_stationary'] = False

        # Simple OLS for Durbin-Watson
        try:
            X = country_data[['energy_use', 'gdp', 'population']].copy()
            X = sm.add_constant(X)
            y = country_data[target]
            model = sm.OLS(y, X).fit()
            residuals = model.resid

            result['ols_r2'] = model.rsquared
            result['durbin_watson'] = durbin_watson(residuals)
            result['has_autocorrelation'] = result['durbin_watson'] < 1.5 or result['durbin_watson'] > 2.5
        except:
            result['ols_r2'] = np.nan
            result['durbin_watson'] = np.nan
            result['has_autocorrelation'] = False

        return result

    def analyze_all_countries(self):
        """Analyze all countries in dataset"""
        print("\n🔍 Analyzing all countries...")

        countries = self.data['country'].unique()
        print(f"Total countries to analyze: {len(countries)}")

        for i, country in enumerate(countries, 1):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(countries)} countries...")

            result = self.analyze_country(country)
            if result:
                self.summary_results.append(result)

        print(f"✅ Completed analysis for {len(self.summary_results)} countries")

        # Convert to DataFrame
        summary_df = pd.DataFrame(self.summary_results)

        return summary_df

    def create_summary_statistics(self, summary_df):
        """Create summary statistics from all countries"""
        print("\n📈 Calculating summary statistics...")

        stats = {}

        # Basic counts
        stats['total_countries'] = len(summary_df)
        stats['avg_obs_per_country'] = summary_df['n_obs'].mean()
        stats['min_obs'] = summary_df['n_obs'].min()
        stats['max_obs'] = summary_df['n_obs'].max()

        # Stationarity
        stats['stationary_countries'] = summary_df['is_stationary'].sum()
        stats['stationary_pct'] = (summary_df['is_stationary'].sum() / len(summary_df)) * 100

        # Autocorrelation
        stats['autocorrelation_countries'] = summary_df['has_autocorrelation'].sum()
        stats['autocorrelation_pct'] = (summary_df['has_autocorrelation'].sum() / len(summary_df)) * 100

        # OLS performance
        stats['avg_r2'] = summary_df['ols_r2'].mean()
        stats['median_r2'] = summary_df['ols_r2'].median()
        stats['r2_below_05'] = (summary_df['ols_r2'] < 0.5).sum()

        # Trends
        stats['increasing_trend'] = (summary_df['trend'] == 'increasing').sum()
        stats['decreasing_trend'] = (summary_df['trend'] == 'decreasing').sum()

        return stats

    def save_results(self, summary_df, stats):
        """Save all results to files"""
        print("\n💾 Saving results...")

        # 1. Save detailed results to CSV
        detailed_path = self.results_dir / "all_countries_detailed.csv"
        summary_df.to_csv(detailed_path, index=False)
        print(f"📄 Detailed results: {detailed_path}")

        # 2. Save summary statistics to CSV
        stats_df = pd.DataFrame([stats])
        stats_path = self.results_dir / "summary_statistics.csv"
        stats_df.to_csv(stats_path, index=False)

        # 3. Save summary report to text
        report_path = self.results_dir / "summary_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DIAGNOSTIC SUMMARY - ALL COUNTRIES\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"TOTAL COUNTRIES ANALYZED: {stats['total_countries']}\n")
            f.write(f"AVERAGE OBSERVATIONS PER COUNTRY: {stats['avg_obs_per_country']:.1f}\n")
            f.write(f"OBSERVATIONS RANGE: {stats['min_obs']} - {stats['max_obs']} years\n\n")

            f.write("=" * 80 + "\n")
            f.write("STATISTICAL ASSUMPTIONS VIOLATIONS\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. NON-STATIONARITY:\n")
            f.write(f"   • Stationary countries: {stats['stationary_countries']}/{stats['total_countries']}\n")
            f.write(
                f"   • Non-stationary countries: {stats['total_countries'] - stats['stationary_countries']}/{stats['total_countries']}\n")
            f.write(f"   • Percentage non-stationary: {100 - stats['stationary_pct']:.1f}%\n\n")

            f.write("2. AUTOCORRELATION (Durbin-Watson):\n")
            f.write(
                f"   • Countries with autocorrelation: {stats['autocorrelation_countries']}/{stats['total_countries']}\n")
            f.write(f"   • Percentage with autocorrelation: {stats['autocorrelation_pct']:.1f}%\n\n")

            f.write("3. MODEL FIT (Simple OLS R²):\n")
            f.write(f"   • Average R²: {stats['avg_r2']:.4f}\n")
            f.write(f"   • Median R²: {stats['median_r2']:.4f}\n")
            f.write(f"   • Countries with R² < 0.5: {stats['r2_below_05']}/{stats['total_countries']}\n\n")

            f.write("4. EMISSIONS TRENDS:\n")
            f.write(f"   • Increasing trend: {stats['increasing_trend']} countries\n")
            f.write(f"   • Decreasing trend: {stats['decreasing_trend']} countries\n\n")

            f.write("=" * 80 + "\n")
            f.write("TOP 10 COUNTRIES - WORST DATA QUALITY\n")
            f.write("=" * 80 + "\n\n")

            # Find countries with worst combination of issues
            summary_df['issue_score'] = (
                    (~summary_df['is_stationary']).astype(int) +
                    summary_df['has_autocorrelation'].astype(int) +
                    (summary_df['ols_r2'] < 0.3).astype(int)
            )

            worst_countries = summary_df.sort_values('issue_score', ascending=False).head(10)

            f.write("Country                Obs  Stationary  Autocorr  R²      Issues\n")
            f.write("-" * 60 + "\n")
            for _, row in worst_countries.iterrows():
                f.write(f"{row['country'][:20]:20} {row['n_obs']:4}  "
                        f"{'✓' if row['is_stationary'] else '✗':10}  "
                        f"{'✓' if not row['has_autocorrelation'] else '✗':9}  "
                        f"{row['ols_r2']:.3f}  {row['issue_score']}/3\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("CONCLUSIONS\n")
            f.write("=" * 80 + "\n\n")

            f.write("MAJOR FINDINGS:\n")
            f.write(f"1. {100 - stats['stationary_pct']:.0f}% of countries have NON-STATIONARY data\n")
            f.write(f"2. {stats['autocorrelation_pct']:.0f}% of countries show AUTOCORRELATION\n")
            f.write(f"3. Average OLS R² is only {stats['avg_r2']:.3f} (poor fit)\n")
            f.write(f"4. Only ~{stats['avg_obs_per_country']:.0f} observations per country\n\n")

            f.write("IMPLICATIONS FOR AI MODELS:\n")
            f.write("• Random Forest: ❌ NOT SUITABLE (autocorrelation violates independence)\n")
            f.write("• Neural Networks: ❌ NOT SUITABLE (too few observations)\n")
            f.write("• Time Series Models: ⚠️ LIMITED (need stationarity)\n")
            f.write("• Panel Data Models: ⚠️ POSSIBLE (but with limitations)\n\n")

            f.write("RECOMMENDATIONS:\n")
            f.write("1. Data is UNSUITABLE for predictive AI/ML modeling\n")
            f.write("2. Focus on DESCRIPTIVE statistics only\n")
            f.write("3. If modeling required: use first differences + panel methods\n")
            f.write("4. Request higher frequency data (>100 obs per country)\n")

        print(f"📄 Summary report: {report_path}")

        # 4. Create visualization of key metrics
        self.create_summary_visualizations(summary_df)

        return detailed_path, report_path

    def create_summary_visualizations(self, summary_df):
        """Create summary visualizations"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Distribution of observations per country
        axes[0, 0].hist(summary_df['n_obs'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(summary_df['n_obs'].mean(), color='red', linestyle='--',
                           label=f'Mean: {summary_df["n_obs"].mean():.1f}')
        axes[0, 0].set_xlabel('Observations per Country')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Observations per Country')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Stationary vs Non-stationary
        stationary_counts = summary_df['is_stationary'].value_counts()
        axes[0, 1].pie(stationary_counts.values, labels=['Non-stationary', 'Stationary'],
                       autopct='%1.1f%%', colors=['#ff9999', '#99ff99'])
        axes[0, 1].set_title('Stationarity Test Results')

        # 3. Distribution of R² values
        axes[1, 0].hist(summary_df['ols_r2'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0.5, color='red', linestyle='--', label='R² = 0.5')
        axes[1, 0].set_xlabel('OLS R² Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of OLS R² Values')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Autocorrelation presence
        autocorr_counts = summary_df['has_autocorrelation'].value_counts()
        axes[1, 1].pie(autocorr_counts.values, labels=['No Autocorrelation', 'Autocorrelation'],
                       autopct='%1.1f%%', colors=['#99ff99', '#ff9999'])
        axes[1, 1].set_title('Autocorrelation Test Results')

        plt.tight_layout()
        plt.savefig(self.results_dir / "summary_visualizations.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Summary visualizations saved")


def main():
    """Main function to run diagnostic for all countries"""
    print("=" * 70)
    print("ALL COUNTRIES DATA DIAGNOSTIC (SUMMARY)")
    print("=" * 70)

    # Initialize diagnostic
    diagnostic = AllCountriesDiagnostic()

    # Setup results directory
    diagnostic.setup_results_dir()

    # Load data
    diagnostic.load_data()

    # Analyze all countries
    summary_df = diagnostic.analyze_all_countries()

    # Create summary statistics
    stats = diagnostic.create_summary_statistics(summary_df)

    # Save results
    diagnostic.save_results(summary_df, stats)

    print("\n" + "=" * 70)
    print("✅ SUMMARY DIAGNOSTIC COMPLETE")
    print(f"📁 Results in: {diagnostic.results_dir}")
    print(f"📄 Files generated:")
    print(f"  - all_countries_detailed.csv (detailed results per country)")
    print(f"  - summary_statistics.csv (aggregate stats)")
    print(f"  - summary_report.txt (comprehensive report)")
    print(f"  - summary_visualizations.png (key metrics plots)")
    print("=" * 70)

    # Print key findings
    print("\n🔑 KEY FINDINGS:")
    print(f"   • Countries analyzed: {stats['total_countries']}")
    print(f"   • Non-stationary: {100 - stats['stationary_pct']:.1f}% of countries")
    print(f"   • With autocorrelation: {stats['autocorrelation_pct']:.1f}% of countries")
    print(f"   • Average OLS R²: {stats['avg_r2']:.3f} (poor fit)")
    print(f"   • Observations per country: {stats['avg_obs_per_country']:.1f} (too few)")


if __name__ == "__main__":
    main()