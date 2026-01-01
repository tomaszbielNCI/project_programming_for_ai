import os
import tensorflow as tf
import tensorflow_probability as tfp

# Force legacy Keras for compatibility (OK zostawić)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    from tf_keras import layers, models, optimizers
except ImportError:
    from tensorflow.keras import layers, models, optimizers

tfd = tfp.distributions


def build_bnn_showcase(window=20, feature_count=3, train_size=30000):
    """
    Bayesian Neural Network with Laplace likelihood.
    Proper DistributionLambda-based implementation.
    """

    # --------------------------------------------------
    # 1. INPUT
    # --------------------------------------------------
    inputs = layers.Input(shape=(window, feature_count))

    # --------------------------------------------------
    # 2. FEATURE EXTRACTION
    # --------------------------------------------------
    x = layers.Flatten()(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    # --------------------------------------------------
    # 3. DISTRIBUTION PARAMETERS
    # --------------------------------------------------
    # Laplace → need 2 params: μ and scale
    params = layers.Dense(2)(x)

    mu, log_scale = tf.split(params, 2, axis=-1)
    scale = tf.nn.softplus(log_scale) + 1e-6

    # --------------------------------------------------
    # 4. PROBABILISTIC OUTPUT (NO WRAPPER, NO HACKS)
    # --------------------------------------------------
    outputs = tfp.layers.DistributionLambda(
        lambda t: tfd.Independent(
            tfd.Laplace(loc=t[0], scale=t[1]),
            reinterpreted_batch_ndims=1
        ),
        convert_to_tensor_fn=tfd.Distribution.mean  # Dodano konwersję do tensora
    )([mu, scale])

    # --------------------------------------------------
    # 5. MODEL
    # --------------------------------------------------
    model = models.Model(inputs=inputs, outputs=outputs)

    # Negative log-likelihood
    def nll(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=nll
    )

    return model


if __name__ == "__main__":
    model = build_bnn_showcase(window=20, feature_count=3)
    model.summary()
