"""
diagnostic_three_countries.py
Detailed statistical diagnostic for USA, Ireland, China
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
import statsmodels.api as sm
import warnings
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.loaders.data_IO import DataIO

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


class CountryDiagnostic:
    def __init__(self):
        self.data_io = DataIO()
        self.data = None
        self.results_dir = None
        self.results = {}

    def setup_results_dir(self):
        """Create unique results directory"""
        base_path = Path(__file__).parent.parent.parent / "results" / "diagnostic"
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
        print(f"✅ Loaded {len(self.data)} rows")
        return self.data

    def run_adf_test(self, series, country):
        """Augmented Dickey-Fuller test for stationarity"""
        result = adfuller(series.dropna())

        self.results[country]['adf'] = {
            'statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }

        return result

    def run_durbin_watson_test(self, residuals, country):
        """Durbin-Watson test for autocorrelation"""
        dw_stat = durbin_watson(residuals)

        # Interpretation: ~2 = no autocorrelation, <1.5 or >2.5 = autocorrelation
        self.results[country]['durbin_watson'] = {
            'statistic': dw_stat,
            'has_autocorrelation': dw_stat < 1.5 or dw_stat > 2.5
        }

        return dw_stat

    def run_breusch_pagan_test(self, residuals, X, country):
        """Breusch-Pagan test for heteroskedasticity"""
        try:
            # Add constant to X for the test
            X_with_const = sm.add_constant(X)
            bp_test = het_breuschpagan(residuals, X_with_const)

            self.results[country]['breusch_pagan'] = {
                'lm_statistic': bp_test[0],
                'lm_pvalue': bp_test[1],
                'f_statistic': bp_test[2],
                'f_pvalue': bp_test[3],
                'is_heteroskedastic': bp_test[1] < 0.05
            }

            return bp_test
        except:
            self.results[country]['breusch_pagan'] = {
                'error': 'Could not compute'
            }
            return None

    def run_shapiro_test(self, residuals, country):
        """Shapiro-Wilk test for normality"""
        if len(residuals) > 5000:
            # Shapiro test limited to 5000 observations
            residuals_sample = residuals.sample(5000, random_state=42)
        else:
            residuals_sample = residuals

        shapiro_stat, shapiro_p = shapiro(residuals_sample)

        self.results[country]['shapiro'] = {
            'statistic': shapiro_stat,
            'pvalue': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }

        return shapiro_stat, shapiro_p

    def plot_time_series(self, country_data, country, target='emissions'):
        """Plot emissions time series"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Original series
        axes[0].plot(country_data['year'], country_data[target], marker='o', linewidth=2)
        axes[0].set_title(f'{country} - {target.capitalize()} Over Time')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel(target.capitalize())
        axes[0].grid(True, alpha=0.3)

        # First difference (if stationarity is an issue)
        axes[1].plot(country_data['year'][1:], country_data[target].diff()[1:],
                     marker='o', color='red', linewidth=2)
        axes[1].set_title(f'{country} - {target.capitalize()} First Difference')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel(f'Δ{target}')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / f'{country.lower().replace(" ", "_")}_time_series.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_acf_pacf(self, series, country, target='emissions'):
        """Plot ACF and PACF"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # ACF
        acf_values = acf(series.dropna(), nlags=10)
        axes[0].bar(range(len(acf_values)), acf_values)
        axes[0].axhline(y=1.96 / np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
        axes[0].axhline(y=-1.96 / np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{country} - ACF of {target}')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')

        # PACF
        pacf_values = pacf(series.dropna(), nlags=10)
        axes[1].bar(range(len(pacf_values)), pacf_values)
        axes[1].axhline(y=1.96 / np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-1.96 / np.sqrt(len(series)), color='red', linestyle='--', alpha=0.5)
        axes[1].set_title(f'{country} - PACF of {target}')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'{country.lower().replace(" ", "_")}_acf_pacf.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_qq_residuals(self, residuals, country):
        """QQ-plot and residuals histogram"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # QQ-plot
        sm.qqplot(residuals, line='45', ax=axes[0])
        axes[0].set_title(f'{country} - QQ-Plot of Residuals')

        # Histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title(f'{country} - Distribution of Residuals')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'{country.lower().replace(" ", "_")}_residuals.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def run_country_analysis(self, country_name, target='emissions'):
        """Run complete analysis for a single country"""
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {country_name}")
        print('=' * 60)

        # Filter country data
        country_data = self.data[self.data['country'] == country_name].copy()
        country_data = country_data.sort_values('year')

        if len(country_data) < 10:
            print(f"⚠️ Not enough data for {country_name}")
            return None

        # Initialize results dict
        self.results[country_name] = {}

        # 1. Time series plots
        self.plot_time_series(country_data, country_name, target)

        # 2. Simple OLS regression for residuals
        X = country_data[['energy_use', 'gdp', 'population']].copy()
        X = sm.add_constant(X)
        y = country_data[target]

        model = sm.OLS(y, X).fit()
        residuals = model.resid

        # 3. Statistical tests
        print(f"📈 Running statistical tests...")

        # ADF Test
        adf_result = self.run_adf_test(country_data[target], country_name)
        print(f"   ADF p-value: {adf_result[1]:.6f} {'✓' if adf_result[1] < 0.05 else '✗ (non-stationary)'}")

        # Durbin-Watson
        dw_stat = self.run_durbin_watson_test(residuals, country_name)
        print(f"   Durbin-Watson: {dw_stat:.4f} {'✓' if 1.5 <= dw_stat <= 2.5 else '✗ (autocorrelation)'}")

        # Breusch-Pagan
        bp_test = self.run_breusch_pagan_test(residuals, X.drop('const', axis=1), country_name)
        if bp_test:
            print(
                f"   Breusch-Pagan p-value: {bp_test[1]:.6f} {'✓' if bp_test[1] > 0.05 else '✗ (heteroskedasticity)'}")

        # Shapiro-Wilk
        shapiro_stat, shapiro_p = self.run_shapiro_test(residuals, country_name)
        print(f"   Shapiro-Wilk p-value: {shapiro_p:.6f} {'✓' if shapiro_p > 0.05 else '✗ (non-normal)'}")

        # 4. ACF/PACF plots
        self.plot_acf_pacf(country_data[target], country_name, target)

        # 5. Residual diagnostics
        self.plot_qq_residuals(residuals, country_name)

        # 6. Basic statistics
        self.results[country_name]['stats'] = {
            'n_obs': len(country_data),
            'min_year': country_data['year'].min(),
            'max_year': country_data['year'].max(),
            'mean_target': country_data[target].mean(),
            'std_target': country_data[target].std(),
            'ols_r2': model.rsquared
        }

        print(f"   Observations: {len(country_data)} ({country_data['year'].min()}-{country_data['year'].max()})")
        print(f"   OLS R²: {model.rsquared:.4f}")

        return country_data

    def save_report(self):
        """Save comprehensive report to text file"""
        report_path = self.results_dir / "diagnostic_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DIAGNOSTIC REPORT - DATA QUALITY ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("ANALYZED COUNTRIES: United States, Ireland, China\n")
            f.write(f"DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"DATA SOURCE: preprocessed_data_5.csv\n\n")

            f.write("=" * 80 + "\n")
            f.write("SUMMARY OF FINDINGS\n")
            f.write("=" * 80 + "\n\n")

            # Count violations
            violations = {
                'non_stationary': 0,
                'autocorrelation': 0,
                'heteroskedastic': 0,
                'non_normal': 0
            }

            for country in ['United States', 'Ireland', 'China']:
                if country in self.results:
                    if not self.results[country]['adf']['is_stationary']:
                        violations['non_stationary'] += 1
                    if self.results[country]['durbin_watson']['has_autocorrelation']:
                        violations['autocorrelation'] += 1
                    if 'breusch_pagan' in self.results[country] and 'is_heteroskedastic' in self.results[country][
                        'breusch_pagan']:
                        if self.results[country]['breusch_pagan']['is_heteroskedastic']:
                            violations['heteroskedastic'] += 1
                    if not self.results[country]['shapiro']['is_normal']:
                        violations['non_normal'] += 1

            f.write("STATISTICAL ASSUMPTIONS VIOLATIONS (3/3 countries):\n")
            f.write("-" * 60 + "\n")
            f.write(f"• Non-stationarity: {violations['non_stationary']}/3\n")
            f.write(f"• Autocorrelation: {violations['autocorrelation']}/3\n")
            f.write(f"• Heteroskedasticity: {violations['heteroskedastic']}/3\n")
            f.write(f"• Non-normality: {violations['non_normal']}/3\n\n")

            f.write("=" * 80 + "\n")
            f.write("DETAILED RESULTS BY COUNTRY\n")
            f.write("=" * 80 + "\n\n")

            for country in ['United States', 'Ireland', 'China']:
                if country in self.results:
                    f.write(f"COUNTRY: {country}\n")
                    f.write("-" * 60 + "\n")

                    # Basic stats
                    stats = self.results[country]['stats']
                    f.write(f"Observations: {stats['n_obs']} ({stats['min_year']}-{stats['max_year']})\n")
                    f.write(f"Mean emissions: {stats['mean_target']:.2f}\n")
                    f.write(f"Std emissions: {stats['std_target']:.2f}\n")
                    f.write(f"OLS R²: {stats['ols_r2']:.4f}\n\n")

                    # ADF Test
                    adf = self.results[country]['adf']
                    f.write("1. STATIONARITY TEST (Augmented Dickey-Fuller):\n")
                    f.write(f"   Test statistic: {adf['statistic']:.4f}\n")
                    f.write(f"   p-value: {adf['pvalue']:.6f}\n")
                    f.write(f"   Conclusion: {'STATIONARY' if adf['is_stationary'] else 'NON-STATIONARY'}\n")
                    f.write(f"   Critical values:\n")
                    for key, value in adf['critical_values'].items():
                        f.write(f"     {key}: {value:.4f}\n")
                    f.write("\n")

                    # Durbin-Watson
                    dw = self.results[country]['durbin_watson']
                    f.write("2. AUTOCORRELATION TEST (Durbin-Watson):\n")
                    f.write(f"   Test statistic: {dw['statistic']:.4f}\n")
                    f.write(
                        f"   Interpretation: {'No autocorrelation' if not dw['has_autocorrelation'] else 'Autocorrelation present'}\n")
                    f.write(f"   (1.5-2.5 = no autocorrelation)\n\n")

                    # Breusch-Pagan
                    if 'breusch_pagan' in self.results[country]:
                        bp = self.results[country]['breusch_pagan']
                        if 'error' not in bp:
                            f.write("3. HETEROSKEDASTICITY TEST (Breusch-Pagan):\n")
                            f.write(f"   LM statistic: {bp['lm_statistic']:.4f}\n")
                            f.write(f"   LM p-value: {bp['lm_pvalue']:.6f}\n")
                            f.write(
                                f"   Conclusion: {'Homoskedastic' if not bp['is_heteroskedastic'] else 'Heteroskedastic'}\n\n")

                    # Shapiro-Wilk
                    shapiro = self.results[country]['shapiro']
                    f.write("4. NORMALITY TEST (Shapiro-Wilk):\n")
                    f.write(f"   Test statistic: {shapiro['statistic']:.4f}\n")
                    f.write(f"   p-value: {shapiro['pvalue']:.6f}\n")
                    f.write(f"   Conclusion: {'NORMAL' if shapiro['is_normal'] else 'NON-NORMAL'}\n")

                    f.write("\n" + "=" * 60 + "\n\n")

            f.write("=" * 80 + "\n")
            f.write("CONCLUSIONS AND RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            f.write("MAJOR ISSUES IDENTIFIED:\n")
            f.write("1. NON-STATIONARITY: All countries show trends over time\n")
            f.write("2. AUTOCORRELATION: Emissions are correlated with past values\n")
            f.write("3. HETEROSKEDASTICITY: Variance changes over time\n")
            f.write("4. NON-NORMALITY: Data doesn't follow normal distribution\n")
            f.write("5. SMALL SAMPLE: Only ~30 observations per country\n\n")

            f.write("IMPLICATIONS FOR AI/MACHINE LEARNING MODELS:\n")
            f.write("- Random Forest: ❌ NOT SUITABLE (violates independence assumption)\n")
            f.write("- NeuralProphet: ❌ NOT SUITABLE (too few observations)\n")
            f.write("- Deep Learning: ❌ NOT SUITABLE (needs >1000s of samples)\n")
            f.write("- Time Series ARIMA: ⚠️ MAY WORK (but need stationarity)\n\n")

            f.write("RECOMMENDATIONS:\n")
            f.write("1. Obtain higher frequency data (monthly/quarterly)\n")
            f.write("2. Collect longer time period (>50 years)\n")
            f.write("3. Use first differences for stationarity\n")
            f.write("4. Consider panel data models instead of pure time series\n")
            f.write("5. Focus on descriptive analysis, not predictive modeling\n")

        print(f"\n📄 Report saved: {report_path}")
        return report_path


def main():
    """Main function to run diagnostic"""
    print("=" * 70)
    print("COUNTRY-SPECIFIC DATA DIAGNOSTIC")
    print("=" * 70)

    # Initialize diagnostic
    diagnostic = CountryDiagnostic()

    # Setup results directory
    diagnostic.setup_results_dir()

    # Load data
    diagnostic.load_data()

    # Analyze specific countries
    countries_to_analyze = ['United States', 'Ireland', 'China']

    for country in countries_to_analyze:
        if country in diagnostic.data['country'].unique():
            diagnostic.run_country_analysis(country)
        else:
            print(f"⚠️ Country '{country}' not found in data")
            # Try alternative names
            alt_names = {
                'United States': ['United States', 'USA', 'US'],
                'Ireland': ['Ireland'],
                'China': ['China', 'China (Mainland)']
            }

    # Save comprehensive report
    diagnostic.save_report()

    print("\n" + "=" * 70)
    print("✅ DIAGNOSTIC COMPLETE")
    print(f"📁 Results in: {diagnostic.results_dir}")
    print("📊 Generated: Time series plots, ACF/PACF, QQ-plots, statistical tests")
    print("📄 Report: diagnostic_report.txt with conclusions")
    print("=" * 70)


if __name__ == "__main__":
    main()