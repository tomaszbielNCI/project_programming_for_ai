import pandas as pd

from src.models.forecasting.lstm_forecaster import LSTMForecaster

# Load your WorldBank data
df = pd.read_csv('worldbank_data.csv')

# Prepare data
lstm = LSTMForecaster(model_name='emissions_forecaster', input_dim=3)
data_dict = lstm.prepare_data(
    df,
    entity_col='country',
    time_col='year',
    feature_cols=['gdp_per_capita', 'population', 'energy_use'],
    target_col='emissions',
    seq_length=5
)

# Train
results = lstm.train(data_dict, epochs=50)
print(f"Training completed. Final R²: {results.get('final_r2', 'N/A')}")

# Save model
lstm.save()