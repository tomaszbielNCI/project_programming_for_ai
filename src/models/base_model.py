# src/models/base_model.py
import pickle
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path


class BaseAIModel(ABC):
    """Abstract base class - ALL your models inherit from this"""

    def __init__(self, model_name, config=None):
        self.model_name = model_name
        self.config = config or {}
        self.fitted = False
        self.feature_names = None
        self.target_name = None

    @abstractmethod
    def prepare_data(self, data, **kwargs):
        """Convert raw data to model format - MUST be implemented"""
        # This is what makes it portable:
        # Same method for WorldBank (country/year) and Trading (timestamp/symbol)
        pass

    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the model - MUST be implemented"""
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """Make predictions - MUST be implemented"""
        pass

    def evaluate(self, X_test, y_test, metrics=['mae', 'rmse', 'r2']):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        results = {}

        if 'mae' in metrics:
            from sklearn.metrics import mean_absolute_error
            results['mae'] = mean_absolute_error(y_test, predictions)

        if 'rmse' in metrics:
            from sklearn.metrics import mean_squared_error
            results['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))

        if 'r2' in metrics:
            from sklearn.metrics import r2_score
            results['r2'] = r2_score(y_test, predictions)

        return results

    def save(self, directory='saved_models'):
        """Save model to disk"""
        Path(directory).mkdir(exist_ok=True)
        path = Path(directory) / f"{self.model_name}_{self.__class__.__name__}.pkl"

        # Don't save data, just model parameters
        save_dict = {
            'model_state': self._get_state(),
            'config': self.config,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'fitted': self.fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        # Also save config as JSON for readability
        config_path = Path(directory) / f"{self.model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"Model saved to {path}")
        return str(path)

    def load(self, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self._set_state(save_dict['model_state'])
        self.config = save_dict['config']
        self.feature_names = save_dict['feature_names']
        self.target_name = save_dict['target_name']
        self.fitted = save_dict['fitted']

        print(f"Model loaded from {path}")
        return self

    def _get_state(self):
        """Get model's internal state - override if needed"""
        return self.__dict__

    def _set_state(self, state):
        """Set model's internal state - override if needed"""
        self.__dict__.update(state)

    def get_feature_importance(self):
        """Return feature importance if model supports it"""
        return None  # Override in subclasses

    def prepare_worldbank_data(self, df, entity_col='country', time_col='year',
                               feature_cols=None, target_col='emissions',
                               seq_length=5):
        """Prepare WorldBank panel data for time series models"""
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in [entity_col, time_col, target_col]]

        sequences = []
        targets = []
        entities = []

        for entity in df[entity_col].unique():
            entity_data = df[df[entity_col] == entity].sort_values(time_col)

            # Ensure we have enough data
            if len(entity_data) <= seq_length:
                continue

            for i in range(len(entity_data) - seq_length):
                # Get sequence of features
                seq = entity_data.iloc[i:i + seq_length][feature_cols].values
                # Get target (next value after sequence)
                target = entity_data.iloc[i + seq_length][target_col]

                sequences.append(seq)
                targets.append(target)
                entities.append(entity)

        return {
            'X': np.array(sequences),
            'y': np.array(targets),
            'entities': np.array(entities),
            'feature_names': feature_cols,
            'entity_col': entity_col,
            'time_col': time_col
        }

    def prepare_trading_data(self, df, symbol_col='symbol', time_col='timestamp',
                             feature_cols=None, target_col='close',
                             seq_length=20):
        """Prepare trading data - SAME LOGIC as WorldBank!"""
        # This is why the code is portable!
        return self.prepare_worldbank_data(
            df=df,
            entity_col=symbol_col,
            time_col=time_col,
            feature_cols=feature_cols,
            target_col=target_col,
            seq_length=seq_length
        )