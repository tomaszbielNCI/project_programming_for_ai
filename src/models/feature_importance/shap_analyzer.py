# src/models/feature_importance/shap_analyzer.py
import numpy as np
import pandas as pd
from ..base_model import BaseAIModel

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class SHAPAnalyzer(BaseAIModel):
    """SHAP values for feature importance - interpret any model"""

    def __init__(self, model_name='shap_analyzer', base_model=None):
        super().__init__(model_name)

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed")

        self.base_model = base_model  # Can be any model (XGBoost, RandomForest, etc.)
        self.explainer = None
        self.shap_values = None
        self.feature_names = None

    def prepare_data(self, X, feature_names=None):
        """Prepare data for SHAP analysis"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist() if feature_names is None else feature_names
            return X.values
        else:
            self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            return X

    def train(self, X, model=None, **kwargs):
        """Calculate SHAP values for a model"""
        if model is not None:
            self.base_model = model

        if self.base_model is None:
            raise ValueError("No model provided for SHAP analysis")

        X_prepared = self.prepare_data(X, kwargs.get('feature_names'))

        # Create explainer based on model type
        if hasattr(self.base_model, 'predict_proba'):
            # Classifier
            self.explainer = shap.Explainer(self.base_model, X_prepared)
            self.shap_values = self.explainer(X_prepared).values
        else:
            # Regressor
            self.explainer = shap.Explainer(self.base_model, X_prepared)
            self.shap_values = self.explainer(X_prepared).values

        self.fitted = True

        return self._calculate_importance()

    def _calculate_importance(self):
        """Calculate feature importance from SHAP values"""
        if self.shap_values is None:
            return {}

        # Mean absolute SHAP value
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'rank': np.argsort(mean_abs_shap)[::-1] + 1
        }).sort_values('mean_abs_shap', ascending=False)

        return importance_df

    def predict(self, X):
        """Return SHAP values for new data"""
        if not self.fitted:
            raise ValueError("Analyzer must be trained first")

        X_prepared = self.prepare_data(X)
        return self.explainer(X_prepared).values

    def get_summary_plot(self, X, max_display=20):
        """Generate SHAP summary plot"""
        if not self.fitted:
            raise ValueError("Analyzer must be trained first")

        X_prepared = self.prepare_data(X)

        if len(self.shap_values.shape) == 3:  # For multi-class
            shap_values_2d = self.shap_values.mean(axis=2)
        else:
            shap_values_2d = self.shap_values

        shap.summary_plot(
            shap_values_2d,
            X_prepared,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        return plt.gcf()

    def get_dependence_plot(self, feature_idx, interaction_idx=None):
        """Generate dependence plot for a specific feature"""
        if not self.fitted:
            raise ValueError("Analyzer must be trained first")

        if isinstance(feature_idx, str):
            feature_idx = self.feature_names.index(feature_idx)

        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            self.explainer.data,
            feature_names=self.feature_names,
            interaction_index=interaction_idx,
            show=False
        )

        return plt.gcf()

    def get_feature_importance(self):
        """Get feature importance DataFrame"""
        if not self.fitted:
            raise ValueError("Analyzer must be trained first")

        return self._calculate_importance()