import os
import tensorflow as tf
import tensorflow_probability as tfp

# Wymuszamy legacy keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Importujemy warstwy i modele z tf_keras, aby uniknąć konfliktów typów
try:
    from tf_keras import layers, models, optimizers
except ImportError:
    from tensorflow.keras import layers, models, optimizers

tfd = tfp.distributions


# DEFINIUJEMY WŁASNĄ WARSTWĘ - tak jak prosi błąd
class ProbabilisticOutput(layers.Layer):
    def __init__(self, **kwargs):
        super(ProbabilisticOutput, self).__init__(**kwargs)
        self.dist_layer = tfp.layers.IndependentNormal(1)

    def call(self, inputs):
        # To wywołanie wewnątrz warstwy rozwiązuje problem KerasTensor
        return self.dist_layer(inputs)


def build_bnn_showcase(window=20, feature_count=3, train_size=30000):
    """
    Finalna wersja BNN zgodna z restrykcjami Keras 2025.
    Rozwiązuje błąd 'A KerasTensor cannot be used as input to a TF function'.
    """

    # 1. Wejście
    inputs = layers.Input(shape=(window, feature_count))

    # 2. Przetwarzanie
    x = layers.Flatten()(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # 3. Parametry (średnia i odchylenie)
    params_size = tfp.layers.IndependentNormal.params_size(1)
    params = layers.Dense(params_size)(x)

    # 4. UŻYCIE NASZEJ WARSTWY OPAKOWUJĄCEJ
    # To omija błąd, ponieważ IndependentNormal jest teraz 'ukryte' wewnątrz standardowej klasy Layer
    outputs = ProbabilisticOutput()(params)

    # 5. Budowa modelu
    model = models.Model(inputs=inputs, outputs=outputs)

    # Funkcja straty
    def nll(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=nll
    )

    return model
