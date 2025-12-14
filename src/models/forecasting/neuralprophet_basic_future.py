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
        self.model = None

        # Użycie bezwzględnej ścieżki do wyników
        self.results_base = Path(__file__).parent.parent.parent.parent / "results" / "neuralprophet"
        self.results_base.mkdir(parents=True, exist_ok=True)
        print(f"📂 Results will be saved to: {self.results_base.absolute()}")

        # Znajdź najwyższy istniejący numer run
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

        # Podfoldery w run
        (self.run_path / "forecasts").mkdir(exist_ok=True)
        (self.run_path / "plots").mkdir(exist_ok=True)
        (self.run_path / "models").mkdir(exist_ok=True)

        print(f"📁 Run folder: {self.run_path}")

    def load_and_prepare(self, country: str, target_col: str = "emissions"):
        """Załaduj i przygotuj dane (bez zmian)"""
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
        """Utwórz model (bez zmian)"""
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
        """KLUCZOWA ZMIANA: Trenowanie na WSZYSTKICH danych i prognoza przyszłości"""
        if self.model is None:
            self.create_model()

        print(f"\nTraining on ALL {len(df)} years of historical data")
        print(f"Generating forecast for next {periods} years...")

        # 1. Trenowanie na wszystkich danych (bez podziału train/test)
        metrics = self.model.fit(df, freq='Y')

        # 2. Tworzenie dataframe z przyszłymi okresami
        future = self.model.make_future_dataframe(
            df,  # Używamy CAŁYCH danych
            periods=periods,  # Dodajemy przyszłe okresy
            n_historic_predictions=len(df)  # Zachowujemy historyczne dopasowanie
        )

        # 3. Prognoza (zawiera historię + przyszłość)
        forecast = self.model.predict(future)

        return forecast, metrics

    def plot_results(self, forecast, country, target_col):
        """Simple plot of forecast results."""
        fig = self.model.plot(forecast, plotting_backend='matplotlib')
        plt.suptitle(f"NeuralProphet Forecast for {country} - {target_col}", fontsize=14)
        plt.tight_layout()

        # Save plot to run folder
        filename = f"forecast_{country}_{target_col}.png"
        save_path = self.run_path / "plots" / filename
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"📈 Plot saved to: {save_path}")
        return fig

    def save_forecast_data(self, forecast, country, target_col):
        """Zapisz dane prognozy do CSV"""
        filename = f"forecast_{country}_{target_col}.csv"
        save_path = self.run_path / "forecasts" / filename

        # Dodaj kolumnę wskazującą czy to prognoza
        forecast['is_forecast'] = forecast['ds'] > self.df_prepared['ds'].max()

        forecast.to_csv(save_path, index=False)
        print(f"💾 Forecast data saved to: {save_path}")

    def save_model(self, country, target_col):
        """Zapisz wytrenowany model"""
        if self.model is None:
            print("⚠️ No model to save")
            return

        filename = f"model_{country}_{target_col}.np"
        save_path = self.run_path / "models" / filename
        self.model.save(save_path)
        print(f"💾 Model saved to: {save_path}")

    def save_config(self, country, params):
        """Zapisz konfigurację run"""
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
        """Główna metoda uruchamiająca dla kraju"""
        print(f"\n{'=' * 60}")
        print(f"NeuralProphet Extended Analysis for: {country}")
        print(f"Mode: Historical + {periods}-year future forecast")
        print(f"{'=' * 60}")

        # Załaduj dane
        self.df_prepared, regressors = self.load_and_prepare(country, target_col)

        if len(self.df_prepared) < 10:
            print(f"⚠️ Warning: Only {len(self.df_prepared)} data points for {country}")
            return None

        # Stwórz i wytrenuj model
        self.create_model(regressors=regressors)

        # Parametry do zapisu
        params = {
            'target': target_col,
            'forecast_periods': periods,
            'regressors': regressors,
            'data_points': len(self.df_prepared),
            'last_historical_year': self.df_prepared['ds'].max().year
        }
        self.save_config(country, params)

        # Trenuj i prognozuj
        forecast, metrics = self.train_and_forecast(self.df_prepared, periods=periods)

        # Zapisz wyniki
        self.plot_results(forecast, country, target_col)
        self.save_forecast_data(forecast, country, target_col)
        self.save_model(country, target_col)

        # Podsumowanie
        print(f"\n📊 Summary for {country}:")
        print(f"   - Historical data: {len(self.df_prepared)} years")
        print(f"   - Last actual year: {self.df_prepared['ds'].max().year}")
        print(f"   - Future forecast: {periods} years ({self.df_prepared['ds'].max().year + 1} to {self.df_prepared['ds'].max().year + periods})")

        # Wyświetl przyszłe prognozy
        future_forecast = forecast[forecast['ds'] > self.df_prepared['ds'].max()]
        for i, row in future_forecast.iterrows():
            year = row['ds'].year
            # Używamy kolumny yhat1, yhat2, ..., yhat5 dla kolejnych przyszłych lat
            # W forecast kolumny yhat1 do yhat5 odpowiadają kolejnym przyszłym okresom
            # Dla uproszczenia, dla każdego wiersza future_forecast, który jest i-tym wierszem w future_forecast,
            # odpowiada i+1- temu okresowi przyszłości, więc kolumna to f'yhat{i+1}'.
            # Ale to tylko jeśli future_forecast ma dokładnie periods wierszy i są w kolejności.
            # Lepiej: dla każdego wiersza future_forecast, który jest j-tym wierszem od końca forecast,
            # to kolumna yhat? W NeuralProphet, dla forecast z n_forecasts=5, mamy kolumny yhat1, yhat2, ..., yhat5.
            # Dla każdego wiersza w future, wartość yhat1 to prognoza na pierwszy przyszły okres, yhat2 na drugi, itd.
            # W naszym future dataframe, pierwszy wiersz przyszłości ma wartość yhat1, drugi yhat2, itd.
            # Zatem iterując po future_forecast z zachowaniem kolejności, możemy użyć:
            for offset in range(1, periods+1):
                if i == future_forecast.index[offset-1]:  # to niebezpieczne, bo indexy mogą nie być kolejne
                    forecast_col = f'yhat{offset}'
                    break
            # Lepsze podejście: przyszłe wiersze są w tej samej kolejności co w forecast, więc możemy użyć numeru wiersza w future_forecast.
            # Reset index, aby mieć pozycję 0,1,2...
            future_forecast_reset = future_forecast.reset_index(drop=True)
            # Ale musimy znać pozycję bieżącego wiersza w future_forecast.
            # Zrobimy to prosto: użyjemy pętli for z enumerate.
            # Zmienimy sposób wyświetlania:
        future_forecast_reset = future_forecast.reset_index(drop=True)
        for offset, (idx, row) in enumerate(future_forecast_reset.iterrows(), start=1):
            year = row['ds'].year
            forecast_col = f'yhat{offset}'
            print(f"   - {year}: {row[forecast_col]:.2f}")

        return forecast


def main():
    """Przykładowe użycie"""
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