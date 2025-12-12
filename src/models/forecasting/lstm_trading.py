import pandas as pd
from src.models.forecasting.lstm_forecaster import LSTMForecaster

# Load trading data
trading_df = pd.read_csv('EURUSD_1m.csv')

# Use the SAME model class
trading_lstm = LSTMForecaster(model_name='price_forecaster', input_dim=4)

# Prepare trading data (SAME method!)
trading_data = trading_lstm.prepare_data(
    trading_df,
    entity_col='symbol',
    time_col='timestamp',
    feature_cols=['open', 'high', 'low', 'volume'],
    target_col='close',
    seq_length=20
)

# Train or fine-tune
trading_lstm.train(trading_data, epochs=30)