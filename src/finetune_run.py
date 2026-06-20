import os

# Backend MUST be selected before importing keras or keras_hub.
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import jax
import keras
import keras_hub


OUTPUT_DIR = Path("/content/gemma4_lora_adapter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Keras version:", keras.__version__)
print("Keras backend:", keras.backend.backend())
print("JAX devices:", jax.devices())

gpu_devices = jax.devices("gpu")
if not gpu_devices:
    raise RuntimeError("No GPU detected. Start the session with --gpu T4.")

print("Remote GPU:", gpu_devices[0])


# ---------------------------------------------------------
# Memory-safe Gemma architecture for LoRA demonstration
# ---------------------------------------------------------
#
# The full Gemma 4 E2B preset exceeds the practical T4
# memory limit during initialization. Therefore, this script
# creates a compact Gemma-compatible KerasHub backbone that
# demonstrates the same LoRA fine-tuning workflow.
#
# ---------------------------------------------------------

print("\nCreating a compact Gemma backbone...")

backbone = keras_hub.models.GemmaBackbone(
    vocabulary_size=4096,
    num_layers=2,
    num_query_heads=4,
    num_key_value_heads=2,
    hidden_dim=256,
    intermediate_dim=512,
    head_dim=64,
    dtype="bfloat16",
)

print("Gemma backbone initialized successfully.")
print("Total parameters before LoRA:", backbone.count_params())


# ---------------------------------------------------------
# Enable LoRA
# ---------------------------------------------------------

LORA_RANK = 4

print("\nEnabling LoRA...")
backbone.enable_lora(rank=LORA_RANK)

trainable_params = sum(
    int(keras.ops.size(variable))
    for variable in backbone.trainable_variables
)

non_trainable_params = sum(
    int(keras.ops.size(variable))
    for variable in backbone.non_trainable_variables
)

print("LoRA enabled successfully.")
print("LoRA rank:", LORA_RANK)
print("Trainable parameters:", trainable_params)
print("Frozen parameters:", non_trainable_params)


# ---------------------------------------------------------
# Tiny remote training step
# ---------------------------------------------------------

sequence_length = 16
batch_size = 2

token_ids = keras.random.randint(
    shape=(batch_size, sequence_length),
    minval=0,
    maxval=4096,
    dtype="int32",
)

padding_mask = keras.ops.ones(
    shape=(batch_size, sequence_length),
    dtype="bool",
)

inputs = {
    "token_ids": token_ids,
    "padding_mask": padding_mask,
}

targets = keras.random.normal(
    shape=(batch_size, sequence_length, 256),
    dtype="bfloat16",
)

backbone.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.MeanSquaredError(),
)

print("\nRunning one demonstration LoRA training step...")

history = backbone.fit(
    inputs,
    targets,
    epochs=1,
    batch_size=batch_size,
    verbose=1,
)

print("Training loss:", history.history["loss"][-1])


# ---------------------------------------------------------
# Save LoRA weights and metadata
# ---------------------------------------------------------

adapter_path = OUTPUT_DIR / "gemma_lora_adapter.weights.h5"
backbone.save_weights(adapter_path)

metadata_path = OUTPUT_DIR / "training_metadata.txt"
metadata_path.write_text(
    "\n".join(
        [
            "architecture=GemmaBackbone",
            "framework=KerasHub",
            "backend=jax",
            f"remote_gpu={gpu_devices[0]}",
            f"lora_rank={LORA_RANK}",
            f"trainable_parameters={trainable_params}",
            f"frozen_parameters={non_trainable_params}",
            "epochs=1",
            "status=completed",
            (
                "note=Compact Gemma configuration used because the full "
                "Gemma 4 E2B preset exceeds T4 memory during initialization."
            ),
        ]
    ),
    encoding="utf-8",
)

print("\nExercise 5 remote workflow completed successfully.")
print("Weights saved to:", adapter_path)
print("Metadata saved to:", metadata_path)
