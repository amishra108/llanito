from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

# Project
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Data
LLANITO_PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
)

LLANITO_FIXED_TEST_MANIFEST = (
    LLANITO_PROCESSED_ROOT
    / "fixed_test.csv"
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

# Existing Miami sources
MIAMI_FULL_MODEL = (
    PROJECT_ROOT
    / "outputs_five_fold" / "models"
    / "miami_seqstack_larger"
    / "spanish_full_finetune"
)

MIAMI_FULL_LORA = (
    PROJECT_ROOT
    / "outputs_five_fold" / "models"
    / "miamifullLoRA"
    / "best_miami_validation_adapter"
)

MIAMI_ENCODER_FULL_LORA = (
    PROJECT_ROOT
    / "outputs_five_fold" / "models"
    / "miamiencoderFullLoRA"
    / "best_miami_validation_adapter"
)

# Experiments
#
# full_model_continue:
#   Miami larger full fine-tune -> continue ALL Whisper parameters on Llanito.
#
# same_lora:
#   Load the selected Miami LoRA and continue training those SAME LoRA matrices.
#   Ordinary Whisper parameters remain frozen.
#
# stacked_lora:
#   Load the selected Miami LoRA, merge it into the frozen Whisper backbone,
#   then create a fresh Llanito LoRA with the same architecture. This gives:
#       W' = W + delta_W_Miami(frozen) + delta_W_Llanito(trainable)

EXPERIMENTS = {
    "miami_full_to_llanito_full_soft": {
        "experiment_name": "miami_full_to_llanito_full_soft",
        "mode": "full_model_continue",
        "source_name": "miami_seqstack_larger",
        "source_root": MIAMI_FULL_MODEL,
    },
    "miamifullLoRA_to_llanito_sameLoRA": {
        "experiment_name": "miamifullLoRA_to_llanito_sameLoRA",
        "mode": "same_lora",
        "source_name": "miamifullLoRA_best_miami_validation",
        "source_root": MIAMI_FULL_LORA,
    },
    "miamiencoderFullLoRA_to_llanito_sameLoRA": {
        "experiment_name": "miamiencoderFullLoRA_to_llanito_sameLoRA",
        "mode": "same_lora",
        "source_name": "miamiencoderFullLoRA_best_miami_validation",
        "source_root": MIAMI_ENCODER_FULL_LORA,
    },
    "miamifullLoRA_to_llanito_stackedLoRA": {
        "experiment_name": "miamifullLoRA_to_llanito_stackedLoRA",
        "mode": "stacked_lora",
        "source_name": "miamifullLoRA_best_miami_validation",
        "source_root": MIAMI_FULL_LORA,
    },
}

RUN_EXPERIMENTS = [
    #"miami_full_to_llanito_full_soft",
    #"miamifullLoRA_to_llanito_sameLoRA",
    "miamiencoderFullLoRA_to_llanito_sameLoRA",
    #"miamifullLoRA_to_llanito_stackedLoRA",
]

DEFAULT_EXPERIMENT = RUN_EXPERIMENTS[0]

# Folds
FOLD_NUMBERS = [1, 2, 3, 4, 5]
RUN_FOLDS: list[int] | None = None

# Training
RANDOM_SEED = 2026
NUMBER_OF_EPOCHS = 5.0

# Full-model continuation is a much larger update than LoRA continuation,
# so use a conservative full-model LR. LoRA runs keep the established LR.
FULL_MODEL_LEARNING_RATE = 1e-5
LORA_LEARNING_RATE = 5e-5

WEIGHT_DECAY = 0.01
WARMUP_STEPS = 40

TRAIN_BATCH_SIZE_PER_GPU = 1
VALIDATION_BATCH_SIZE_PER_GPU = 1
GRADIENT_ACCUMULATION_STEPS = 8

USE_FP16 = True
USE_BF16 = False
USE_GRADIENT_CHECKPOINTING = False

LOGGING_STEPS = 5
SAVE_TOTAL_LIMIT = 2
DATALOADER_WORKERS = 2

GENERATION_MAX_LENGTH = 225
GENERATION_NUM_BEAMS = 1

MAX_TRAIN_RECORDINGS: int | None = None
MAX_VALIDATION_RECORDINGS: int | None = None
MAX_TEST_RECORDINGS: int | None = None

CLEAR_SELECTED_FOLD_BEFORE_TRAINING = True
RESUME_FROM_LATEST_CHECKPOINT = False


def requested_folds() -> list[int]:
    if RUN_FOLDS is None:
        return list(FOLD_NUMBERS)

    folds = sorted(set(RUN_FOLDS))
    invalid = [fold for fold in folds if fold not in FOLD_NUMBERS]

    if invalid:
        raise ValueError(f"Invalid RUN_FOLDS values: {invalid}")

    return folds


def fold_train_manifest(fold_number: int) -> Path:
    return (
        LLANITO_PROCESSED_ROOT
        / f"fold_{fold_number}"
        / "train.csv"
    )


def fold_validation_manifest(fold_number: int) -> Path:
    return (
        LLANITO_PROCESSED_ROOT
        / f"fold_{fold_number}"
        / "validation.csv"
    )


def build_runtime_config(experiment_key: str) -> SimpleNamespace:
    if experiment_key not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment_key!r}. "
            f"Valid values: {sorted(EXPERIMENTS)}"
        )

    settings = dict(EXPERIMENTS[experiment_key])
    experiment_name = settings["experiment_name"]
    mode = settings["mode"]

    learning_rate = (
        FULL_MODEL_LEARNING_RATE
        if mode == "full_model_continue"
        else LORA_LEARNING_RATE
    )

    return SimpleNamespace(
        EXPERIMENT_KEY=experiment_key,
        EXPERIMENT_NAME=experiment_name,
        MODE=mode,
        SOURCE_NAME=settings["source_name"],
        SOURCE_ROOT=Path(settings["source_root"]),
        BASE_MODEL=BASE_MODEL,
        LANGUAGE=LANGUAGE,
        TASK=TASK,
        LLANITO_PROCESSED_ROOT=LLANITO_PROCESSED_ROOT,
        LLANITO_FIXED_TEST_MANIFEST=LLANITO_FIXED_TEST_MANIFEST,
        NORMALISATION_FILE=NORMALISATION_FILE,
        MODEL_ROOT=(PROJECT_ROOT / "outputs_five_fold" / "models" / experiment_name),
        OUTPUT_ROOT=(PROJECT_ROOT / "outputs_five_fold" / experiment_name),
        SUMMARY_ROOT=(PROJECT_ROOT / "outputs_five_fold" / "summary" / experiment_name),
        RUN_FOLDS=requested_folds(),
        RANDOM_SEED=RANDOM_SEED,
        NUMBER_OF_EPOCHS=NUMBER_OF_EPOCHS,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=WEIGHT_DECAY,
        WARMUP_STEPS=WARMUP_STEPS,
        TRAIN_BATCH_SIZE_PER_GPU=TRAIN_BATCH_SIZE_PER_GPU,
        VALIDATION_BATCH_SIZE_PER_GPU=VALIDATION_BATCH_SIZE_PER_GPU,
        GRADIENT_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS,
        USE_FP16=USE_FP16,
        USE_BF16=USE_BF16,
        USE_GRADIENT_CHECKPOINTING=USE_GRADIENT_CHECKPOINTING,
        LOGGING_STEPS=LOGGING_STEPS,
        SAVE_TOTAL_LIMIT=SAVE_TOTAL_LIMIT,
        DATALOADER_WORKERS=DATALOADER_WORKERS,
        GENERATION_MAX_LENGTH=GENERATION_MAX_LENGTH,
        GENERATION_NUM_BEAMS=GENERATION_NUM_BEAMS,
        MAX_TRAIN_RECORDINGS=MAX_TRAIN_RECORDINGS,
        MAX_VALIDATION_RECORDINGS=MAX_VALIDATION_RECORDINGS,
        MAX_TEST_RECORDINGS=MAX_TEST_RECORDINGS,
        CLEAR_SELECTED_FOLD_BEFORE_TRAINING=CLEAR_SELECTED_FOLD_BEFORE_TRAINING,
        RESUME_FROM_LATEST_CHECKPOINT=RESUME_FROM_LATEST_CHECKPOINT,
    )
