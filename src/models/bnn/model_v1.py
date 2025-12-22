import os
import tensorflow as tf
import tensorflow_probability as tfp

# Force legacy Keras for compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Import layers and models from tf_keras to avoid type conflicts
try:
    from tf_keras import layers, models, optimizers
except ImportError:
    from tensorflow.keras import layers, models, optimizers

tfd = tfp.distributions


# Define custom output layer to handle KerasTensor compatibility
# Note: Using IndependentNormal instead of Laplace due to compatibility issues with newer Keras versions
# IMPORTANT: Consider implementing Laplace distribution, Keras version problem
class ProbabilisticOutput(layers.Layer):
    def __init__(self, **kwargs):
        super(ProbabilisticOutput, self).__init__(**kwargs)
        self.dist_layer = tfp.layers.IndependentNormal(1)

    def call(self, inputs):
        # This call within the layer resolves the KerasTensor issue
        return self.dist_layer(inputs)


def build_bnn_showcase(window=20, feature_count=3, train_size=30000):
    """
    Final BNN implementation compatible with Keras 2025 restrictions.
    Solves the 'A KerasTensor cannot be used as input to a TF function' error.
    """

    # 1. Input layer
    inputs = layers.Input(shape=(window, feature_count))

    # 2. Processing layers
    x = layers.Flatten()(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # 3. Output parameters (mean and standard deviation)
    # Using IndependentNormal as the output distribution (instead of Laplace) due to Keras compatibility issues
    params_size = tfp.layers.IndependentNormal.params_size(1)
    params = layers.Dense(params_size)(x)

    # 4. Using our wrapper layer
    # This works around the KerasTensor error by encapsulating IndependentNormal within a standard Layer class
    outputs = ProbabilisticOutput()(params)

    # 5. Build the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Loss function (negative log likelihood)
    def nll(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=nll
    )

    return model
