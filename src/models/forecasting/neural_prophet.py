# src/models/forecasting/neural_prophet.py

## 📦 NeuralProphet (question 5)

import pandas as pd
from ..base_model import BaseAIModel

try:
    from neuralprophet import NeuralProphet

    NEURALPROPHET_AVAILABLE = True
except ImportError:
    NEURALPROPHET_AVAILABLE = False
    print("Warning: NeuralProphet not installed. Install with: pip install neuralprophet")


class NeuralProphetForecaster(BaseAIModel):
    """NeuralProphet for time series with seasonality - emissions AND prices"""

    def __init__(self, model_name='neural_prophet', **kwargs):
        super().__init__(model_name)

        if not NEURALPROPHET_AVAILABLE:
            raise ImportError("NeuralProphet is not installed")

        # Default config
        default_config = {
            'n_forecasts': 1,
            'n_lags': 10,
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'epochs': 50,
            'learning_rate': 0.001
        }
        default_config.update(kwargs)
        self.config.update(default_config)

        self.model = NeuralProphet(**self.config)
        self.fitted_data = None

    def prepare_data(self, df, entity_col='country', time_col='year',
                     target_col='emissions', regressor_cols=None):
        """Prepare data for NeuralProphet"""
        # NeuralProphet expects: ds (datetime), y (target), plus regressors
        data_dicts = []

        for entity in df[entity_col].unique():
            entity_data = df[df[entity_col] == entity].copy()

            # Prepare DataFrame for NeuralProphet
            np_df = pd.DataFrame({
                'ds': pd.to_datetime(entity_data[time_col], format='%Y'),
                'y': entity_data[target_col]
            })

            # Add regressors if specified
            if regressor_cols:
                for reg in regressor_cols:
                    if reg in entity_data.columns:
                        np_df[reg] = entity_data[reg].values

            np_df['entity'] = entity
            data_dicts.append(np_df)

        return {
            'data': pd.concat(data_dicts, ignore_index=True),
            'entity_col': entity_col,
            'regressor_cols': regressor_cols
        }

    def train(self, data_dict, **kwargs):
        """Train NeuralProphet model"""
        df = data_dict['data']
        regressor_cols = data_dict.get('regressor_cols', [])

        # Add regressors to model
        for reg in regressor_cols:
            self.model.add_regressor(reg)

        # Train
        metrics = self.model.fit(df, freq='Y', **kwargs)  # 'Y' for yearly, 'D' for daily in trading

        self.fitted = True
        self.fitted_data = df

        return metrics

    def predict(self, future_periods=5, regressors_future=None):
        """Make future predictions"""
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")

        # Create future dataframe
        future = self.model.make_future_dataframe(
            self.fitted_data,
            periods=future_periods,
            n_historic_predictions=len(self.fitted_data)
        )

        # Add future regressor values if provided
        if regressors_future is not None:
            for reg, values in regressors_future.items():
                if reg in future.columns:
                    future[reg] = values

        forecast = self.model.predict(future)
        return forecast

    def plot_components(self, forecast):
        """Plot trend, seasonality, etc."""
        if hasattr(self.model, 'plot_components'):
            return self.model.plot_components(forecast)
        return None

    def save(self, directory='saved_models'):
        """Save NeuralProphet model"""
        path = super().save(directory)
        # NeuralProphet has its own save method
        model_path = Path(directory) / f"{self.model_name}_neuralprophet_model.np"
        self.model.save(str(model_path))
        return path