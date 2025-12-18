# emissions_analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys


class EmissionsAnalyzer:
    def __init__(self):
        self.data_dir = Path(r"/mongo/data/development_analysis")
        self.plots_dir = Path("plots")
        self.output_file = "emissions_analysis.txt"
        self.df = None
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        self.plots_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load emissions data from CSV"""
        try:
            file_path = self.data_dir / "emissions_raw.csv"
            self.df = pd.read_csv(file_path)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def show_menu(self):
        """Display the main menu"""
        while True:
            print("\n" + "=" * 50)
            print("EMISSIONS DATA ANALYSIS")
            print("=" * 50)
            print("1. Show dataset information")
            print("2. Analyze emissions data")
            print("3. Plot emissions distribution")
            print("4. Plot trends over time")
            print("5. Export diagnostic to file")
            print("6. Exit")
            print("=" * 50)

            choice = input("Enter your choice (1-6): ").strip()

            if choice == "1":
                self.show_dataset_info()
            elif choice == "2":
                self.analyze_emissions()
            elif choice == "3":
                self.plot_distribution()
            elif choice == "4":
                self.plot_trends()
            elif choice == "5":
                self.export_analysis()
            elif choice == "6":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    def show_dataset_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("No data loaded!")
            return

        print("\n=== DATASET INFORMATION ===")
        print(f"Shape: {self.df.shape}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nData types:")
        print(self.df.dtypes)
        print("\nMissing values:")
        print(self.df.isnull().sum())

    def analyze_emissions(self):
        """Analyze emissions data"""
        if self.df is None:
            print("No data loaded!")
            return

        print("\n=== EMISSIONS ANALYSIS ===")
        print(f"Time period: {self.df['year'].min()} - {self.df['year'].max()}")
        print(f"Number of countries: {self.df['country'].nunique()}")
        print(f"Total records: {len(self.df):,}")

        # Basic statistics
        print("\nEmissions statistics:")
        print(f"Mean: {self.df['emissions'].mean():.2f}")
        print(f"Median: {self.df['emissions'].median():.2f}")
        print(f"Min: {self.df['emissions'].min():.2f}")
        print(f"Max: {self.df['emissions'].max():.2f}")

        # Count of zero emissions
        zero_emissions = (self.df['emissions'] == 0).sum()
        print(f"\nRows with 0 emissions: {zero_emissions} ({(zero_emissions / len(self.df) * 100):.1f}%)")

        # Top emitting countries
        print("\nTop 5 countries by average emissions:")
        top_countries = self.df.groupby('country')['emissions'].mean().nlargest(5)
        print(top_countries)

    def plot_distribution(self):
        """Plot emissions distribution"""
        if self.df is None:
            print("No data loaded!")
            return

        plt.figure(figsize=(12, 5))

        # Plot 1: Distribution
        plt.subplot(1, 2, 1)
        sns.histplot(self.df['emissions'], bins=50, kde=True)
        plt.title('Distribution of Emissions')
        plt.xlabel('Emissions')

        # Plot 2: Log distribution
        plt.subplot(1, 2, 2)
        sns.histplot(self.df['emissions'], bins=50, kde=True, log_scale=True)
        plt.title('Log Distribution of Emissions')
        plt.xlabel('Emissions (log scale)')

        plt.tight_layout()

        # Save the plot
        plot_path = self.plots_dir / "emissions_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

    def plot_trends(self):
        """Plot emissions trends over time"""
        if self.df is None:
            print("No data loaded!")
            return

        # Get top 5 countries by average emissions
        top_countries = self.df.groupby('country')['emissions'].mean().nlargest(5).index

        plt.figure(figsize=(12, 6))

        for country in top_countries:
            country_data = self.df[self.df['country'] == country]
            plt.plot(country_data['year'], country_data['emissions'], label=country)

        plt.title('Emissions Trends for Top 5 Countries')
        plt.xlabel('Year')
        plt.ylabel('Emissions')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = self.plots_dir / "emissions_trends.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

    def export_analysis(self):
        """Export diagnostic to a text file"""
        if self.df is None:
            print("No data loaded!")
            return

        with open(self.output_file, 'w') as f:
            # Redirect stdout to file
            original_stdout = sys.stdout
            sys.stdout = f

            print("EMISSIONS DATA ANALYSIS REPORT")
            print("=" * 50 + "\n")

            # Dataset info
            print("DATASET INFORMATION")
            print("=" * 50)
            print(f"Shape: {self.df.shape}")
            print(f"Time period: {self.df['year'].min()} - {self.df['year'].max()}")
            print(f"Number of countries: {self.df['country'].nunique()}\n")

            # Basic statistics
            print("BASIC STATISTICS")
            print("=" * 50)
            print(f"Mean emissions: {self.df['emissions'].mean():.2f}")
            print(f"Median emissions: {self.df['emissions'].median():.2f}")
            print(f"Min emissions: {self.df['emissions'].min():.2f}")
            print(f"Max emissions: {self.df['emissions'].max():.2f}\n")

            # Top countries
            print("TOP 10 COUNTRIES BY AVERAGE EMISSIONS")
            print("=" * 50)
            top_countries = self.df.groupby('country')['emissions'].mean().nlargest(10)
            print(top_countries)

            # Reset stdout
            sys.stdout = original_stdout

        print(f"Analysis exported to {self.output_file}")


def main():
    # Create analyzer instance
    analyzer = EmissionsAnalyzer()

    # Load data
    if not analyzer.load_data():
        print("Failed to load data. Exiting...")
        return

    # Show menu
    analyzer.show_menu()


if __name__ == "__main__":
    main()