# src/models/ensemble_forecaster.py
## 🚀 Step 6: Ensemble Forecaster (Question 4)

import numpy as np
from ..base_model import BaseAIModel


class EnsembleForecaster(BaseAIModel):
    """Ensemble of multiple models for better predictions"""

    def __init__(self, model_name='ensemble_forecaster', models=None, weights=None):
        super().__init__(model_name)

        self.models = models or []
        self.weights = weights or []

        if models and not weights:
            # Equal weights by default
            self.weights = [1.0 / len(models)] * len(models)

    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def prepare_data(self, data, **kwargs):
        """Prepare data - just pass through to models"""
        return data

    def train(self, data_dict, **kwargs):
        """Train all models in the ensemble"""
        if not self.models:
            raise ValueError("No models in ensemble")

        results = {}
        for i, model in enumerate(self.models):
            print(f"Training model {i + 1}/{len(self.models)}: {model.__class__.__name__}")

            if hasattr(model, 'train'):
                result = model.train(data_dict, **kwargs)
                results[f'model_{i}'] = result
            else:
                print(f"Model {i} doesn't have train method, skipping training")

        self.fitted = True
        return results

    def predict(self, X, return_individual=False):
        """Make ensemble prediction"""
        if not self.fitted:
            raise ValueError("Ensemble must be trained first")

        individual_predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                individual_predictions.append(pred)

        # Weighted average
        ensemble_pred = np.zeros_like(individual_predictions[0])
        for pred, weight in zip(individual_predictions, self.weights):
            ensemble_pred += pred * weight

        if return_individual:
            return ensemble_pred, individual_predictions
        else:
            return ensemble_pred

    def evaluate_models(self, X_test, y_test):
        """Evaluate each model individually"""
        results = {}

        for i, model in enumerate(self.models):
            if hasattr(model, 'evaluate'):
                result = model.evaluate(X_test, y_test)
                results[f'model_{i}'] = result

        # Also evaluate ensemble
        ensemble_pred = self.predict(X_test)

        from sklearn.metrics import mean_squared_error, r2_score
        results['ensemble'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'r2': r2_score(y_test, ensemble_pred)
        }

        return results
