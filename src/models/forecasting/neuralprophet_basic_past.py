import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
import warnings

warnings.filterwarnings('ignore')

from src.loaders.data_IO import DataIO


class NeuralProphetForecaster:
    def __init__(self, data_path: str = "preprocessed_data_5.csv"):
        self.data_io = DataIO()
        self.data_path = data_path
        self.df = None
        self.model = None
        self.df_prepared = None  # Store prepared dataframe for later use

        # Set up results directory with absolute path
        self.results_base = Path(__file__).parent.parent.parent.parent / "results" / "neuralprophet"
        self.results_base.mkdir(parents=True, exist_ok=True)

        print(f"📂 Results will be saved to: {self.results_base.absolute()}")

        # Find the highest existing run number
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
        # Configure trainer for logging in the run folder
        trainer_config = {
            'default_root_dir': str(self.run_path.absolute()),
            'logger': False
        }

        model = NeuralProphet(
            n_forecasts=5,
            n_lags=3,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            epochs=50,
            learning_rate=0.1,
            trainer_config=trainer_config,
            **kwargs
        )

        if regressors:
            for reg in regressors:
                model.add_lagged_regressor(reg)

        self.model = model
        return model

    def train_and_forecast(self, df, periods=5):
        if self.model is None:
            self.create_model()

        train_size = max(3, len(df) - periods)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:] if train_size < len(df) else None

        print(f"\nTraining on {len(df_train)} years")
        if df_test is not None:
            print(f"Testing on {len(df_test)} years")

        metrics = self.model.fit(df_train, freq='Y')

        future = self.model.make_future_dataframe(
            df_train,
            periods=periods,
            n_historic_predictions=len(df_train)
        )

        forecast = self.model.predict(future)

        return forecast, metrics

    def plot_results(self, forecast, country, target_col):
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
        filename = f"forecast_{country}_{target_col}.csv"
        save_path = self.run_path / "forecasts" / filename
        forecast.to_csv(save_path, index=False)
        print(f"💾 Forecast data saved to: {save_path}")

    def save_model(self, country, target_col):
        if self.model is None:
            print("⚠️ No model to save")
            return

        filename = f"model_{country}_{target_col}.pkl"
        save_path = self.run_path / "models" / filename

        # Save model using pickle (fallback when save method not available)
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"💾 Model saved to: {save_path}")

    def save_config(self, country, params):
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
        print(f"\n{'=' * 50}")
        print(f"NeuralProphet Analysis for: {country}")
        print(f"{'=' * 50}")

        df_prepared, regressors = self.load_and_prepare(country, target_col)
        self.df_prepared = df_prepared  # Store as instance variable for later use

        if len(df_prepared) < 10:
            print(f"⚠️ Warning: Only {len(df_prepared)} data points for {country}")
            return None

        self.create_model(regressors=regressors)
        forecast, metrics = self.train_and_forecast(df_prepared, periods=periods)

        self.plot_results(forecast, country, target_col)
        self.save_forecast_data(forecast, country, target_col)
        self.save_model(country, target_col)

        params = {
            'target': target_col,
            'forecast_periods': periods,
            'regressors': regressors,
            'data_points': len(df_prepared),
            'train_test_split': True,
            'last_historical_year': df_prepared['ds'].max().year
        }
        self.save_config(country, params)

        print(f"\n📊 Summary for {country}:")
        print(f"   - Data points: {len(df_prepared)}")
        print(f"   - Last actual value: {df_prepared['y'].iloc[-1]:.2f}")
        print(f"   - Forecast for {df_prepared['ds'].max().year + 1}: {forecast[f'yhat{periods}'].iloc[-1]:.2f}")

        return forecast


def main():
    forecaster = NeuralProphetForecaster("preprocessed_data_5.csv")
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

    print(f"\n✅ Completed forecasts for {len(forecasts)} countries")
    print(f"📁 All results saved in: {forecaster.run_path}")


if __name__ == "__main__":
    main()