import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import layers, ops

class SignalCompression(keras.layers.Layer):
    """
    Convolutional Encoder: Reduces (batch, 50) -> (batch, latent_dim).
    Uses Conv1D to extract temporal features.
    """
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # We define a small internal sequential model for clarity
        self.encoder_net = keras.Sequential([
            layers.Reshape((50, 1)), # Add channel dim: (50,) -> (50, 1)
            layers.Conv1D(16, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Conv1D(32, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dense(latent_dim, activation="relu")
        ])

    def call(self, inputs):
        return self.encoder_net(inputs)


class SignalExpansion(keras.layers.Layer):
    """
    Convolutional Decoder: Reconstructs (batch, latent_dim) -> (batch, 50).
    Uses Conv1DTranspose to upsample temporal resolution.
    """
    def __init__(self, output_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        
        # We need to project the latent dim back to a shape Conv1DTranspose can use
        # (13, 32) is chosen to eventually reach ~50 after two stride-2 upsamples
        self.decoder_net = keras.Sequential([
            layers.Dense(13 * 32, activation="relu"),
            layers.Reshape((13, 32)),
            layers.Conv1DTranspose(16, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Conv1DTranspose(1, kernel_size=3, strides=2, padding="same", activation="relu"),
            layers.Flatten(),
            # Final Dense ensures we hit exactly 50 timesteps regardless of padding math
            layers.Dense(output_dim, activation="relu")
        ])

    def call(self, inputs):
        return self.decoder_net(inputs)

class PhysicsAutoencoder(keras.Model):
    def __init__(self, latent_dim=8, window_size=50, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(output_dim=window_size)

    def call(self, inputs):
        z = self.encoder(inputs)
        return self.decoder(z)