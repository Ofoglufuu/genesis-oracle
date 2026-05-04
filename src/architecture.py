import os

# Set Keras backend before importing keras
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import keras
from keras import ops


def create_sliding_windows(signal_1d, window_size=50):
    """
    Slices a 1D signal into overlapping 2D matrices using a sliding window.

    Each row of the output matrix is a contiguous window of `window_size`
    timesteps. Windows overlap by (window_size - 1) timesteps (stride = 1).

    Args:
        signal_1d (numpy.ndarray): The 1D input signal.
        window_size (int): Number of timesteps per window.

    Returns:
        numpy.ndarray: 2D array of shape (num_windows, window_size).
    """
    num_windows = len(signal_1d) - window_size + 1
    windows = np.array([
        signal_1d[i : i + window_size] for i in range(num_windows)
    ])
    return windows


def prepare_train_test(data_path=None, window_size=50, split_period=60,
                       points_per_period=1000):
    """
    Loads the corrupted signal, splits it at the given period boundary,
    and returns overlapping-window matrices for training and testing.

    Args:
        data_path (str): Path to the .npy file. Defaults to data/corrupted_signal.npy.
        window_size (int): Number of timesteps per sliding window.
        split_period (int): Period index that separates training from testing data.
        points_per_period (int): Number of sample points in one period.

    Returns:
        X_train (numpy.ndarray): Sliding windows from normal data (before split_period).
        X_test  (numpy.ndarray): Sliding windows from the remaining data.
    """
    # Resolve default path relative to this file's location
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "corrupted_signal.npy"
        )

    # Load the 1D corrupted signal
    signal = np.load(data_path)
    print(f"Loaded signal shape: {signal.shape}")

    # Split at the period boundary
    split_index = split_period * points_per_period
    train_signal = signal[:split_index]   # Periods 0–59 (normal)
    test_signal  = signal[split_index:]   # Periods 60–99 (includes anomaly)

    print(f"Training signal length: {len(train_signal)}  (periods 0–{split_period - 1})")
    print(f"Testing  signal length: {len(test_signal)}  (periods {split_period}–end)")

    # Create overlapping sliding windows
    X_train = create_sliding_windows(train_signal, window_size)
    X_test  = create_sliding_windows(test_signal,  window_size)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")

    return X_train, X_test


class SignalCompression(keras.layers.Layer):
    """
    Custom Keras layer that compresses a 50-timestep input window
    to a lower-dimensional latent representation.

    Input shape:  (batch_size, window_size)   e.g. (batch_size, 50)
    Output shape: (batch_size, latent_dim)    e.g. (batch_size, 8)
    """

    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        input_dim = input_shape[-1]  # window_size (50)

        # Dense weight matrix: (window_size, latent_dim)
        self.w = self.add_weight(
            name="kernel",
            shape=(input_dim, self.latent_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Bias vector: (latent_dim,)
        self.b = self.add_weight(
            name="bias",
            shape=(self.latent_dim,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        # Linear projection: (batch_size, 50) @ (50, 8) + (8,) -> (batch_size, 8)
        z = ops.matmul(inputs, self.w) + self.b
        # ReLU activation
        return ops.relu(z)


class SignalExpansion(keras.layers.Layer):
    """
    Custom Keras layer that reconstructs a latent vector back to
    the original window dimension.

    Input shape:  (batch_size, latent_dim)    e.g. (batch_size, 8)
    Output shape: (batch_size, output_dim)    e.g. (batch_size, 50)
    """

    def __init__(self, output_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        latent_dim = input_shape[-1]  # 8

        # Dense weight matrix: (latent_dim, output_dim)
        self.w = self.add_weight(
            name="kernel",
            shape=(latent_dim, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Bias vector: (output_dim,)
        self.b = self.add_weight(
            name="bias",
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        # Linear projection: (batch_size, 8) @ (8, 50) + (50,) -> (batch_size, 50)
        z = ops.matmul(inputs, self.w) + self.b
        # ReLU activation
        return ops.relu(z)


class PhysicsAutoencoder(keras.Model):
    """
    Autoencoder that compresses 50-timestep signal windows into
    an 8-dimensional latent space and reconstructs them back.

    Architecture:
        Input (batch, 50) -> SignalCompression -> (batch, 8)
                          -> SignalExpansion   -> (batch, 50)
    """

    def __init__(self, latent_dim=8, window_size=50, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(output_dim=window_size)

    def call(self, inputs):
        # Encode: (batch_size, 50) -> (batch_size, 8)
        z = self.encoder(inputs)
        # Decode: (batch_size, 8)  -> (batch_size, 50)
        reconstructed = self.decoder(z)
        return reconstructed


if __name__ == "__main__":
    # --- Data preparation ---
    X_train, X_test = prepare_train_test()

    sample_batch = X_train[:16].astype("float32")

    # --- Encoder smoke test ---
    encoder = SignalCompression(latent_dim=8)
    encoded = encoder(sample_batch)
    print(f"\nSignalCompression smoke test:")
    print(f"  Input shape:  {sample_batch.shape}")
    print(f"  Output shape: {encoded.shape}")

    # --- Decoder smoke test ---
    decoder = SignalExpansion(output_dim=50)
    decoded = decoder(encoded)
    print(f"\nSignalExpansion smoke test:")
    print(f"  Input shape:  {encoded.shape}")
    print(f"  Output shape: {decoded.shape}")

    # --- Autoencoder smoke test ---
    autoencoder = PhysicsAutoencoder(latent_dim=8, window_size=50)
    reconstructed = autoencoder(sample_batch)
    print(f"\nPhysicsAutoencoder smoke test:")
    print(f"  Input shape:  {sample_batch.shape}")
    print(f"  Output shape: {reconstructed.shape}")
