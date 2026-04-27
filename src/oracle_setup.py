import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

if __name__ == "__main__":
    print(f"KERAS_BACKEND set to {os.environ['KERAS_BACKEND']}")
    print(f"Keras backend: {keras.backend.backend()}")

    random_tensor = keras.random.uniform(shape=(2, 2))

    print("random_tensor:", random_tensor)
    print("random_tensor type:", type(random_tensor))