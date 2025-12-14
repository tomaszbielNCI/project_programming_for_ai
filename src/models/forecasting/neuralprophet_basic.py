"""
Minimalistyczny szablon NeuralProphet dla danych panelowych.
Współpracuje z Twoją strukturą DataIO.
"""

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Import Twojego DataIO
from src.loaders.data_IO import DataIO


class NeuralProphetForecaster:
    """
    Minimalny forecast dla jednego kraju.
    """

    def __init__(self, data_path: str = "preprocessed_data_5.csv"):
        self.data_io = DataIO()
        self.data_path = data_path
        self.df = None
        self.model = None

    def load_and_prepare(self, country: str, target_col: str = "emissions"):
        """
        Ładuje dane i przygotowuje dla jednego kraju.

        Args:
            country: Nazwa kraju (musi być w kolumnie 'country')
            target_col: Kolumna do forecastu (emissions, energy_use, etc.)

        Returns:
            df_ready: DataFrame gotowy do NeuralProphet
        """
        # Load data using your DataIO
        self.df = self.data_io.from_csv(self.data_path).load()

        # Filter for specific country
        df_country = self.df[self.df['country'] == country].copy()

        if len(df_country) == 0:
            raise ValueError(f"Country '{country}' not found in data")

        # Prepare for NeuralProphet: need 'ds' and 'y' columns
        # Convert year to datetime (first day of year)
        df_country['ds'] = pd.to_datetime(df_country['year'].astype(str) + '-01-01')

        # Target variable
        df_country['y'] = df_country[target_col]

        # Select only necessary columns
        keep_cols = ['ds', 'y']

        # Add other numeric columns as regressors (optional)
        numeric_cols = df_country.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'y', target_col]]

        # You can select specific regressors or use all
        # Example: use these common ones if they exist
        possible_regressors = ['population', 'gdp', 'energy_use', 'renewable_pct'] # 'mean_temp' - maybe letter
        regressors = [col for col in possible_regressors if col in numeric_cols]

        keep_cols.extend(regressors)

        df_ready = df_country[keep_cols].copy()

        # Drop rows where target is NaN (NeuralProphet can't handle NaN in y)
        df_ready = df_ready.dropna(subset=['y'])

        print(f"Prepared data for {country}: {len(df_ready)} rows")
        print(f"Years: {df_ready['ds'].min().year} to {df_ready['ds'].max().year}")
        print(f"Regressors used: {regressors}")

        return df_ready, regressors

    def create_model(self, regressors=None, **kwargs):
        """
        Creates NeuralProphet model with basic config.

        Args:
            regressors: List of column names to use as regressors
            **kwargs: Additional NeuralProphet parameters
        """
        # Basic config - minimal for yearly data
        model = NeuralProphet(
            n_forecasts=5,  # Forecast 5 years ahead
            n_lags=3,  # Use 3 years of history
            yearly_seasonality=True,  # Yearly patterns
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=50,  # Reduced for speed
            learning_rate=0.1,
            **kwargs
        )

        # Add regressors if specified
        if regressors:
            for reg in regressors:
                model.add_lagged_regressor(reg)

        self.model = model
        return model

    def train_and_forecast(self, df, periods=5):
        """
        Trains model and makes forecast.

        Args:
            df: Prepared DataFrame with 'ds', 'y', and regressors
            periods: Number of future periods to forecast

        Returns:
            forecast: DataFrame with predictions
            metrics: Dictionary with training metrics
        """
        if self.model is None:
            self.create_model()

        # Split data (last few years for testing)
        train_size = max(3, len(df) - periods)  # Keep at least 3 years for training
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:] if train_size < len(df) else None

        print(f"\nTraining on {len(df_train)} years")
        if df_test is not None:
            print(f"Testing on {len(df_test)} years")

        # Train
        metrics = self.model.fit(df_train, freq='Y')

        # Make future dataframe for forecast
        future = self.model.make_future_dataframe(
            df_train,
            periods=periods,
            n_historic_predictions=len(df_train)
        )

        # Forecast
        forecast = self.model.predict(future)

        return forecast, metrics

    def plot_results(self, forecast, country, target_col):
        """Simple plot of forecast results."""
        fig = self.model.plot(forecast, plotting_backend='matplotlib')
        plt.suptitle(f"NeuralProphet Forecast for {country} - {target_col}", fontsize=14)
        plt.tight_layout()

        # Save plot
        save_path = Path("results") / "analysis" / "neuralprophet"
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"np_forecast_{country}_{target_col}.png"
        plt.savefig(save_path / filename, dpi=150)
        plt.close()

        print(f"Plot saved to: {save_path / filename}")
        return fig

    def run_for_country(self, country, target_col="emissions", periods=5):
        """
        Complete pipeline for one country.
        """
        print(f"\n{'=' * 50}")
        print(f"NeuralProphet Analysis for: {country}")
        print(f"{'=' * 50}")

        # 1. Prepare data
        df_prepared, regressors = self.load_and_prepare(country, target_col)

        if len(df_prepared) < 10:
            print(f"⚠️ Warning: Only {len(df_prepared)} data points for {country}. May not be enough.")
            return None

        # 2. Create model with regressors
        self.create_model(regressors=regressors)

        # 3. Train and forecast
        forecast, metrics = self.train_and_forecast(df_prepared, periods=periods)

        # 4. Plot
        self.plot_results(forecast, country, target_col)

        # 5. Print summary
        print(f"\n📊 Summary for {country}:")
        print(f"   - Data points: {len(df_prepared)}")
        print(f"   - Last actual value: {df_prepared['y'].iloc[-1]:.2f}")
        print(f"   - Forecast for {df_prepared['ds'].max().year + 1}: {forecast[f'yhat{periods}'].iloc[-1]:.2f}")

        return forecast


def main():
    """Example usage - minimal test"""
    # Initialize forecaster
    forecaster = NeuralProphetForecaster("preprocessed_data_5.csv")

    # Test with a few countries that likely have data
    test_countries = ['Poland', 'Germany', 'France', 'China', 'United States' ]  # Change based on your data

    forecasts = {}

    for country in test_countries:
        try:
            forecast = forecaster.run_for_country(
                country=country,
                target_col="emissions",  # Change to your target
                periods=5
            )
            if forecast is not None:
                forecasts[country] = forecast
        except Exception as e:
            print(f"❌ Error for {country}: {e}")

    print(f"\n✅ Completed forecasts for {len(forecasts)} countries")


if __name__ == "__main__":
    main()