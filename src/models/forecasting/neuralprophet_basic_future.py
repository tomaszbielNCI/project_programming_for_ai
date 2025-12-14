import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

from src.loaders.data_IO import DataIO


class NeuralProphetForecasterExtended:

    def __init__(self, data_path: str = "preprocessed_data_5.csv"):
        self.data_io = DataIO()
        self.data_path = data_path
        self.df = None
        self.df_prepared = None
        self.model = None

        self.results_base = Path(__file__).parent.parent.parent.parent / "results" / "neuralprophet"
        self.results_base.mkdir(parents=True, exist_ok=True)
        print(f"📂 Results will be saved to: {self.results_base.absolute()}")

        existing_runs = []
        for item in self.results_base.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                try:
                    num = int(item.name.split("_")[1])
                    existing_runs.append(num)
                except:
                    continue

        self.run_id = max(existing_runs) + 1 if existing_runs else 1
        self.run_path = self.results_base / f"run_{self.run_id:03d}"
        self.run_path.mkdir(parents=True, exist_ok=True)

        (self.run_path / "forecasts").mkdir(exist_ok=True)
        (self.run_path / "plots").mkdir(exist_ok=True)
        (self.run_path / "models").mkdir(exist_ok=True)

        print(f"📁 Run folder: {self.run_path}")

    def load_and_prepare(self, country: str, target_col: str = "emissions"):
        """Load and prepare data for modeling"""
        self.df = self.data_io.from_csv(self.data_path).load()

        df_country = self.df[self.df['country'] == country].copy()

        if len(df_country) == 0:
            raise ValueError(f"Country '{country}' not found in data")

        df_country['ds'] = pd.to_datetime(df_country['year'].astype(str) + '-01-01')
        df_country['y'] = df_country[target_col]

        keep_cols = ['ds', 'y']
        numeric_cols = df_country.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'y', target_col]]

        possible_regressors = ['population', 'gdp', 'energy_use', 'renewable_pct']
        regressors = [col for col in possible_regressors if col in numeric_cols]
        keep_cols.extend(regressors)

        df_ready = df_country[keep_cols].copy()
        df_ready = df_ready.dropna(subset=['y'])

        print(f"Prepared data for {country}: {len(df_ready)} rows")
        print(f"Years: {df_ready['ds'].min().year} to {df_ready['ds'].max().year}")
        print(f"Regressors used: {regressors}")

        return df_ready, regressors

    def create_model(self, regressors=None, **kwargs):
        """Initialize NeuralProphet model with specified parameters"""
        model = NeuralProphet(
            n_forecasts=5,
            n_lags=3,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=50,
            learning_rate=0.1,
            **kwargs
        )

        if regressors:
            for reg in regressors:
                model.add_lagged_regressor(reg)

        self.model = model
        return model

    def train_and_forecast(self, df, periods=5):
        """Train model on all available data and generate future forecasts"""
        if self.model is None:
            self.create_model()

        print(f"\nTraining on ALL {len(df)} years of historical data")
        print(f"Generating forecast for next {periods} years...")

        # Train on full dataset
        metrics = self.model.fit(df, freq='Y')

        # Create future dataframe for forecasting
        future = self.model.make_future_dataframe(
            df,
            periods=periods,
            n_historic_predictions=len(df)
        )

        # Generate predictions
        forecast = self.model.predict(future)

        return forecast, metrics

    def plot_results(self, forecast, country, target_col):
        """Simple plot of forecast results."""
        fig = self.model.plot(forecast, plotting_backend='matplotlib')
        plt.suptitle(f"NeuralProphet Forecast for {country} - {target_col}", fontsize=14)
        plt.tight_layout()

        filename = f"forecast_{country}_{target_col}.png"
        save_path = self.run_path / "plots" / filename
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"📈 Plot saved to: {save_path}")
        return fig

    def save_forecast_data(self, forecast, country, target_col):
        """Save forecast data to CSV file"""
        filename = f"forecast_{country}_{target_col}.csv"
        save_path = self.run_path / "forecasts" / filename

        forecast['is_forecast'] = forecast['ds'] > self.df_prepared['ds'].max()

        forecast.to_csv(save_path, index=False)
        print(f"💾 Forecast data saved to: {save_path}")

    def save_model(self, country, target_col):
        """Save trained model to disk"""
        if self.model is None:
            print("⚠️ No model to save")
            return

        filename = f"model_{country}_{target_col}.np"
        save_path = self.run_path / "models" / filename
        self.model.save(save_path)
        print(f"💾 Model saved to: {save_path}")

    def save_config(self, country, params):
        """Save run configuration to JSON file"""
        config = {
            'run_id': self.run_id,
            'country': country,
            'timestamp': pd.Timestamp.now().isoformat(),
            'parameters': params
        }

        config_path = self.run_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def run_for_country(self, country, target_col="emissions", periods=5):
        """Main method to run forecasting pipeline for a country"""
        print(f"\n{'=' * 60}")
        print(f"NeuralProphet Extended Analysis for: {country}")
        print(f"Mode: Historical + {periods}-year future forecast")
        print(f"{'=' * 60}")

        self.df_prepared, regressors = self.load_and_prepare(country, target_col)

        if len(self.df_prepared) < 10:
            print(f"⚠️ Warning: Only {len(self.df_prepared)} data points for {country}")
            return None

        self.create_model(regressors=regressors)

        params = {
            'target': target_col,
            'forecast_periods': periods,
            'regressors': regressors,
            'data_points': len(self.df_prepared),
            'last_historical_year': self.df_prepared['ds'].max().year
        }
        self.save_config(country, params)

        forecast, metrics = self.train_and_forecast(self.df_prepared, periods=periods)

        self.plot_results(forecast, country, target_col)
        self.save_forecast_data(forecast, country, target_col)
        self.save_model(country, target_col)

        print(f"\n📊 Summary for {country}:")
        print(f"   - Historical data: {len(self.df_prepared)} years")
        print(f"   - Last actual year: {self.df_prepared['ds'].max().year}")
        print(f"   - Future forecast: {periods} years ({self.df_prepared['ds'].max().year + 1} to {self.df_prepared['ds'].max().year + periods})")

        future_forecast = forecast[forecast['ds'] > self.df_prepared['ds'].max()]
        for i, row in future_forecast.iterrows():
            year = row['ds'].year
            # Map future forecast rows to yhat columns
            for offset in range(1, periods+1):
                if i == future_forecast.index[offset-1]:  # to niebezpieczne, bo indexy mogą nie być kolejne
                    forecast_col = f'yhat{offset}'
                    break
            future_forecast_reset = future_forecast.reset_index(drop=True)
        future_forecast_reset = future_forecast.reset_index(drop=True)
        for offset, (idx, row) in enumerate(future_forecast_reset.iterrows(), start=1):
            year = row['ds'].year
            forecast_col = f'yhat{offset}'
            print(f"   - {year}: {row[forecast_col]:.2f}")

        return forecast


def main():
    """Example usage with predefined test countries"""
    forecaster = NeuralProphetForecasterExtended("preprocessed_data_5.csv")

    test_countries = ['Poland', 'Germany', 'France', 'China', 'United States']

    forecasts = {}

    for country in test_countries:
        try:
            forecast = forecaster.run_for_country(
                country=country,
                target_col="emissions",
                periods=5
            )
            if forecast is not None:
                forecasts[country] = forecast
        except Exception as e:
            print(f"❌ Error for {country}: {e}")

    print(f"\n✅ Completed FUTURE forecasts for {len(forecasts)} countries")
    print(f"📁 All results saved in: {forecaster.run_path}")


if __name__ == "__main__":
    main()