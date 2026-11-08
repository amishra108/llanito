from __future__ import annotations

from pathlib import Path

# Project

EXPERIMENT_NAME = "fullLoRA"
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Paths

SOURCE_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "tunewmcv"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / EXPERIMENT_NAME
MODEL_ROOT = PROJECT_ROOT / "outputs_five_fold" / "models" / EXPERIMENT_NAME
OUTPUT_ROOT = PROJECT_ROOT / "outputs_five_fold" / EXPERIMENT_NAME

# Only the two high-level summary files are also written here.
SUMMARY_ROOT = (
    PROJECT_ROOT
    / "outputs_five_fold"
    / "summary"
    / EXPERIMENT_NAME
)

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts_five_fold"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)

# Whisper

BASE_MODEL = "openai/whisper-medium"
LANGUAGE = "es"
TASK = "transcribe"

# Folds and preparation

FOLD_NUMBERS = [1, 2, 3, 4, 5]

# None = run every fold.
# [3] = run only fold 3.
# [2, 5] = run only folds 2 and 5.
RUN_FOLDS: list[int] | None = [3]

OVERWRITE_PREPARED_MANIFESTS = False

# Fold replacement

# If training from scratch, delete only the model/training outputs for
# the fold(s) selected by RUN_FOLDS before retraining them.
CLEAR_SELECTED_FOLD_BEFORE_TRAINING = True

# Evaluation caches each fold separately. After selected folds are evaluated,
# rebuild the global fixed-test aggregate from all saved fold results.
REBUILD_AGGREGATES_FROM_SAVED_FOLDS = True

# LoRA

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
]
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"

# Training

NUMBER_OF_EPOCHS = 5
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 40
TRAIN_BATCH_SIZE_PER_GPU = 1
VALIDATION_BATCH_SIZE_PER_GPU = 1
GRADIENT_ACCUMULATION_STEPS = 8
USE_FP16 = True
USE_BF16 = False
USE_GRADIENT_CHECKPOINTING = True
LOGGING_STEPS = 5
SAVE_TOTAL_LIMIT = 2
DATALOADER_WORKERS = 2
RANDOM_SEED = 2026
RESUME_FROM_LATEST_CHECKPOINT = False
MAX_TRAIN_RECORDINGS: int | None = None
MAX_VALIDATION_RECORDINGS: int | None = None

# Generation and evaluation

GENERATION_MAX_LENGTH = 225
GENERATION_NUM_BEAMS = 1

# Vanilla is evaluated once and then reused from cache while the fixed-test
# manifest is unchanged. If the fixed-test manifest changes, it is refreshed.
EVALUATE_VANILLA_BASELINE = True

MAX_TEST_RECORDINGS: int | None = None
