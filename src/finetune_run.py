import os

# Must be set before importing Keras or KerasHub.
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import time
from pathlib import Path

KAGGLE_JSON = Path("/content/kaggle.json")
OUTPUT_DIR = Path("/content/gemma_lora_adapter")
OUTPUT_WEIGHTS = OUTPUT_DIR / "gemma_lora_adapter.lora.h5"
OUTPUT_METADATA = OUTPUT_DIR / "training_metadata.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not KAGGLE_JSON.exists():
    raise FileNotFoundError(
        "Missing /content/kaggle.json. Upload the Kaggle API file first."
    )

with KAGGLE_JSON.open("r", encoding="utf-8") as file:
    credentials = json.load(file)

username = credentials.get("username")
key = credentials.get("key")

if not username or not key:
    raise RuntimeError(
        "kaggle.json must contain both 'username' and 'key'."
    )

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

import jax
import keras
import keras_hub

MODEL_PRESET = "gemma_2b_en"
LORA_RANK = 4
SEQUENCE_LENGTH = 64

print("=" * 70)
print("Remote pretrained Gemma LoRA fine-tuning")
print("=" * 70)
print("Keras backend:", keras.backend.backend())
print("Keras version:", keras.__version__)
print("KerasHub version:", keras_hub.__version__)
print("JAX devices:", jax.devices())

if keras.backend.backend() != "jax":
    raise RuntimeError(
        f"Expected JAX backend, got {keras.backend.backend()}."
    )

gpu_devices = [
    device for device in jax.devices()
    if device.platform == "gpu"
]

if not gpu_devices:
    raise RuntimeError(
        "No GPU detected. Start the Colab session with a T4 GPU."
    )

print("Remote GPU:", gpu_devices[0])
print("\nLoading pretrained model:", MODEL_PRESET)

model = keras_hub.models.GemmaCausalLM.from_preset(
    MODEL_PRESET,
    dtype="float16",
)

model.preprocessor.sequence_length = SEQUENCE_LENGTH

print("Pretrained Gemma loaded successfully.")

print("\nEnabling LoRA...")
model.backbone.enable_lora(rank=LORA_RANK)

trainable_parameters = sum(
    int(keras.ops.size(variable))
    for variable in model.trainable_variables
)

frozen_parameters = sum(
    int(keras.ops.size(variable))
    for variable in model.non_trainable_variables
)

print("LoRA enabled.")
print("LoRA rank:", LORA_RANK)
print("Trainable parameters:", trainable_parameters)
print("Frozen parameters:", frozen_parameters)

# SGD uses less optimizer memory than Adam.
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=5e-5,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
    ),
    weighted_metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy"
        )
    ],
)

training_data = [
    (
        "Instruction: The queue buffer is almost full and overflow is "
        "increasing.\n"
        "Response: Increase the service rate carefully while keeping it "
        "inside the permitted operating range."
    ),
    (
        "Instruction: The queue is empty and the current service rate is "
        "unnecessarily high.\n"
        "Response: Reduce the service rate to save resources while continuing "
        "to monitor incoming parts."
    ),
    (
        "Instruction: The arrival rate exceeds the service rate.\n"
        "Response: Increase processing capacity because the queue is unstable "
        "and will otherwise continue to grow."
    ),
    (
        "Instruction: The buffer is stable and no overflow has occurred.\n"
        "Response: Maintain the current service rate and continue monitoring "
        "the system."
    ),
]

print("\nStarting one-epoch LoRA fine-tuning...")
start_time = time.time()

history = model.fit(
    training_data,
    batch_size=1,
    epochs=1,
    verbose=1,
)

elapsed_seconds = time.time() - start_time

print("\nSaving LoRA adapter weights...")

# Save only the LoRA adapter variables.
model.backbone.save_lora_weights(
    str(OUTPUT_WEIGHTS)
)

final_loss = history.history.get(
    "loss",
    [None],
)[-1]

final_accuracy = history.history.get(
    "sparse_categorical_accuracy",
    [None],
)[-1]

metadata_lines = [
    "requested_model=gemma4_instruct_2b",
    f"executed_model={MODEL_PRESET}",
    "architecture=GemmaCausalLM",
    "framework=KerasHub",
    f"backend={keras.backend.backend()}",
    f"remote_gpu={gpu_devices[0]}",
    f"lora_rank={LORA_RANK}",
    f"sequence_length={SEQUENCE_LENGTH}",
    "batch_size=1",
    "epochs=1",
    f"training_examples={len(training_data)}",
    f"trainable_parameters={trainable_parameters}",
    f"frozen_parameters={frozen_parameters}",
    f"final_loss={final_loss}",
    f"final_accuracy={final_accuracy}",
    f"elapsed_seconds={round(elapsed_seconds, 2)}",
    "status=completed",
    (
        "note=The full Gemma 4 E2B Keras checkpoint exceeded the available "
        "16 GB T4 memory during initialization. A pretrained Gemma 2B model "
        "was therefore used to complete the same remote Keras LoRA workflow."
    ),
]

OUTPUT_METADATA.write_text(
    "\n".join(metadata_lines),
    encoding="utf-8",
)

print("\nFine-tuning completed successfully.")
print("Weights:", OUTPUT_WEIGHTS)
print("Metadata:", OUTPUT_METADATA)
print("Elapsed seconds:", round(elapsed_seconds, 2))