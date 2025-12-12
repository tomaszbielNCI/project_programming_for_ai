# src/models/forecasting/lstm_forecaster.py

###  KROK 2: LSTM Forecaster (question 2 + 4)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..base_model import BaseAIModel


class LSTMForecaster(BaseAIModel):
    """LSTM for time series forecasting - works for emissions AND prices"""

    def __init__(self, model_name='lstm_forecaster', input_dim=3, hidden_dim=64,
                 num_layers=2, dropout=0.2, learning_rate=0.001):
        super().__init__(model_name)

        self.config.update({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate
        })

        # Model architecture
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Optional: Add attention mechanism
        self.use_attention = True
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )

        self.fc = nn.Linear(hidden_dim, 1)
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def prepare_data(self, data, **kwargs):
        """Prepare data for LSTM - handles both WorldBank and Trading"""
        if 'entity_col' in kwargs:
            # Specific columns provided
            return self.prepare_worldbank_data(data, **kwargs)
        else:
            # Try to auto-detect
            if 'country' in data.columns and 'year' in data.columns:
                return self.prepare_worldbank_data(
                    data,
                    entity_col='country',
                    time_col='year',
                    target_col=kwargs.get('target_col', 'emissions')
                )
            elif 'symbol' in data.columns and 'timestamp' in data.columns:
                return self.prepare_worldbank_data(
                    data,
                    entity_col='symbol',
                    time_col='timestamp',
                    target_col=kwargs.get('target_col', 'close')
                )
            else:
                raise ValueError("Cannot auto-detect data type. Specify columns manually.")

    def train(self, data_dict, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        X = torch.FloatTensor(data_dict['X'])
        y = torch.FloatTensor(data_dict['y']).view(-1, 1)

        # Store feature names for later
        self.feature_names = data_dict.get('feature_names', [])
        self.target_name = data_dict.get('target_col', 'target')

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'])

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            self.train()
            batch_losses = []

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                self.optimizer.zero_grad()
                predictions = self.forward(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                batch_losses.append(loss.item())

            train_loss = np.mean(batch_losses)
            train_losses.append(train_loss)

            # Validation
            self.eval()
            with torch.no_grad():
                val_predictions = self.forward(X_val)
                val_loss = self.criterion(val_predictions, y_val).item()
                val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        self.fitted = True

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }

    def forward(self, x):
        """Forward pass through the network"""
        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            # Apply attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # Use last time step
            last_step = attn_out[:, -1, :]
        else:
            # Use last time step from LSTM
            last_step = lstm_out[:, -1, :]

        output = self.fc(last_step)
        return output

    def predict(self, X, return_tensor=False):
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")

        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
            predictions = self.forward(X_tensor)

        if return_tensor:
            return predictions
        else:
            return predictions.numpy().flatten()

    def get_attention_weights(self, X):
        """Get attention weights for interpretability"""
        if not self.use_attention:
            return None

        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
            lstm_out, _ = self.lstm(X_tensor)
            _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        return attn_weights.numpy()

    def fine_tune(self, new_data_dict, epochs=10, learning_rate=0.0001):
        """Fine-tune on new data (transfer learning)"""
        if not self.fitted:
            raise ValueError("Model must be trained before fine-tuning")

        # Lower learning rate for fine-tuning
        fine_tune_optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        X = torch.FloatTensor(new_data_dict['X'])
        y = torch.FloatTensor(new_data_dict['y']).view(-1, 1)

        self.train()
        for epoch in range(epochs):
            fine_tune_optimizer.zero_grad()
            predictions = self.forward(X)
            loss = self.criterion(predictions, y)
            loss.backward()
            fine_tune_optimizer.step()

            if epoch % 5 == 0:
                print(f"Fine-tune Epoch {epoch}: Loss = {loss.item():.4f}")

        return loss.item()