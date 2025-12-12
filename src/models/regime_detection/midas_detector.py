# src/models/regime_detection/midas_detector.py
import numpy as np
from scipy import stats
from ..base_model import BaseAIModel

##📦 KROK 4: MIDAS Detector (Pytanie 3)

class MIDASDetector(BaseAIModel):
    """Multiple Indicator Detection of Aggregate Shifts - for regime changes"""

    def __init__(self, model_name='midas_detector', threshold=2.5,
                 window_size=10, min_periods=20):
        super().__init__(model_name)

        self.config.update({
            'threshold': threshold,
            'window_size': window_size,
            'min_periods': min_periods
        })

        self.data_buffer = []
        self.change_points = []
        self.current_regime = None

    def prepare_data(self, data, value_col=None, time_col=None):
        """Prepare streaming or batch data"""
        if value_col and time_col:
            # Batch data with timestamps
            sorted_data = data.sort_values(time_col)
            return {
                'values': sorted_data[value_col].values,
                'timestamps': sorted_data[time_col].values,
                'is_streaming': False
            }
        else:
            # Streaming data (single values)
            return {
                'values': np.array(data) if isinstance(data, list) else data,
                'is_streaming': True
            }

    def train(self, data_dict, **kwargs):
        """Detect change points in historical data"""
        values = data_dict['values']

        if len(values) < self.config['min_periods']:
            raise ValueError(f"Need at least {self.config['min_periods']} data points")

        self.change_points = self._batch_detect(values)
        self.fitted = True

        return {
            'change_points': self.change_points,
            'num_changes': len(self.change_points),
            'regimes': self._identify_regimes(values)
        }

    def _batch_detect(self, values):
        """Detect change points in batch data"""
        change_points = []

        for i in range(self.config['window_size'], len(values) - self.config['window_size']):
            # Compare two windows
            window1 = values[i - self.config['window_size']:i]
            window2 = values[i:i + self.config['window_size']]

            # Calculate divergence (can use different metrics)
            stat, p_value = stats.ttest_ind(window1, window2, equal_var=False)

            if abs(stat) > self.config['threshold'] and p_value < 0.01:
                change_points.append({
                    'index': i,
                    'statistic': stat,
                    'p_value': p_value,
                    'mean_before': np.mean(window1),
                    'mean_after': np.mean(window2)
                })

        return change_points

    def update(self, new_value):
        """Update with new data point (for streaming)"""
        self.data_buffer.append(new_value)

        if len(self.data_buffer) > 100:  # Keep buffer reasonable
            self.data_buffer.pop(0)

        # Check for change if we have enough data
        if len(self.data_buffer) >= 2 * self.config['window_size']:
            change_detected, score = self._streaming_detect()

            if change_detected:
                self.change_points.append({
                    'index': len(self.data_buffer),
                    'score': score,
                    'value': new_value
                })
                return True, score

        return False, 0

    def _streaming_detect(self):
        """Detect change in streaming data"""
        recent = self.data_buffer[-self.config['window_size']:]
        historical = self.data_buffer[-2 * self.config['window_size']:-self.config['window_size']]

        if len(historical) < self.config['window_size']:
            return False, 0

        # Calculate CUSUM-like statistic
        mean_historical = np.mean(historical)
        std_historical = np.std(historical) + 1e-10

        # Standardized deviation
        deviations = [(x - mean_historical) / std_historical for x in recent]
        score = np.sum(deviations)

        return abs(score) > self.config['threshold'], score

    def _identify_regimes(self, values):
        """Identify regimes between change points"""
        if not self.change_points:
            return [{'start': 0, 'end': len(values), 'type': 'single'}]

        regimes = []
        start_idx = 0

        for cp in sorted(self.change_points, key=lambda x: x['index']):
            regimes.append({
                'start': start_idx,
                'end': cp['index'],
                'type': self._classify_regime(values[start_idx:cp['index']])
            })
            start_idx = cp['index']

        # Last regime
        regimes.append({
            'start': start_idx,
            'end': len(values),
            'type': self._classify_regime(values[start_idx:])
        })

        return regimes

    def _classify_regime(self, values):
        """Classify regime type"""
        if len(values) < 2:
            return 'unknown'

        # Simple classification
        trend = np.polyfit(range(len(values)), values, 1)[0]

        if trend > 0.1:
            return 'increasing'
        elif trend < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def predict(self, X=None):
        """Return detected change points"""
        return self.change_points

    def get_current_regime(self):
        """Get current regime based on recent data"""
        if not self.data_buffer or len(self.data_buffer) < 10:
            return 'insufficient_data'

        recent = self.data_buffer[-10:]
        return self._classify_regime(recent)