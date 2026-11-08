from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

# Project
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# Whisper
BASE_MODEL = "openai/whisper-medium"
LANGUAGE = "es"
TASK = "transcribe"
ENCODER_LAYERS = 24

# Llanito data
LLANITO_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "llanito_cv"
LLANITO_FIXED_TEST_MANIFEST = LLANITO_PROCESSED_ROOT / "fixed_test.csv"

# Miami larger data
MIAMI_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "miami_seqstack_larger"
MIAMI_TRAIN_MANIFEST = MIAMI_PROCESSED_ROOT / "train_manifest.csv"
MIAMI_VALIDATION_MANIFEST = MIAMI_PROCESSED_ROOT / "validation_manifest.csv"

NORMALISATION_FILE = PROJECT_ROOT / "scripts_five_fold" / "resources" / "LLanito_normalisation_v2.txt"

# Existing source models
# Miami -> Llanito starts from the single best-Miami-validation encoder residual adapter.
MIAMI_ENCODER_SOURCE = (
    PROJECT_ROOT
    / "outputs_five_fold" / "models"
    / "miami_resadapt_encoder_all"
    / "best_miami_validation_adapter"
)

# The five Llanito fold adapters that will be parameter-averaged BEFORE either
# Llanito -> Miami transfer experiment. Each fold must have fold_X/adapter/.
LLANITO_ENCODER_SOURCE_ROOT = PROJECT_ROOT / "outputs_five_fold" / "models" / "llanito_resadapt_encoder_all"
LLANITO_SOURCE_FOLDS = [1, 2, 3, 4, 5]

# The averaged Llanito adapter is a real saved source model, shared by both
# Llanito -> Miami experiments and by evaluate_averaged_llanito.py.
AVERAGED_LLANITO_SOURCE = (
    PROJECT_ROOT
    / "outputs_five_fold" / "models"
    / "llanito_resadapt_encoder_all_averaged"
    / "adapter"
)
AVERAGED_LLANITO_OUTPUT_ROOT = PROJECT_ROOT / "outputs_five_fold" / "llanitoresadaptenc_averaged"
AVERAGED_LLANITO_SUMMARY_ROOT = (
    PROJECT_ROOT / "outputs_five_fold" / "summary" / "llanitoresadaptenc_averaged"
)

# New Miami adapter used only in stacked/hard transfer
NEW_MIAMI_ADAPTER_BOTTLENECK = 128
NEW_MIAMI_ADAPTER_DROPOUT = 0.10

# Experiments
EXPERIMENTS = {
    "miamiresadaptenc_tollanito_softtransfer": {
        "mode": "soft",
        "direction": "miami_to_llanito",
        "train_dataset": "llanito",
        "selection_metric": "normalised_wer",
        "epochs": 5.0,
        "learning_rate": 5e-5,
        "run_folds": [1, 2, 3, 4, 5],
    },
    "llanitoresadaptenc_tomiami_softtransfer": {
        "mode": "soft",
        "direction": "llanito_to_miami",
        "train_dataset": "miami",
        "selection_metric": "wer",
        "epochs": 5.0,
        "learning_rate": 5e-5,
        "run_folds": [1,2,3,4,5],
    },
    "llanitoresadaptenc_tomiamiresadaptenc": {
        "mode": "stacked",
        "direction": "llanito_to_miami",
        "train_dataset": "miami",
        "selection_metric": "wer",
        "epochs": 5.0,
        "learning_rate": 1e-4,
        "run_folds": [1,2,3,4,5],
    },
}

DEFAULT_EXPERIMENT = "llanitoresadaptenc_tomiami_softtransfer"

# Training
RANDOM_SEED = 2026
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


def build_runtime_config(experiment_name: str = DEFAULT_EXPERIMENT):
    if experiment_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment {experiment_name!r}. Valid values: {sorted(EXPERIMENTS)}"
        )

    spec = dict(EXPERIMENTS[experiment_name])
    model_root = PROJECT_ROOT / "outputs_five_fold" / "models" / experiment_name
    output_root = PROJECT_ROOT / "outputs_five_fold" / experiment_name
    summary_root = PROJECT_ROOT / "outputs_five_fold" / "summary" / experiment_name

    return SimpleNamespace(
        PROJECT_ROOT=PROJECT_ROOT,
        EXPERIMENT_NAME=experiment_name,
        BASE_MODEL=BASE_MODEL,
        LANGUAGE=LANGUAGE,
        TASK=TASK,
        ENCODER_LAYERS=ENCODER_LAYERS,
        MODE=str(spec["mode"]),
        DIRECTION=str(spec["direction"]),
        TRAIN_DATASET=str(spec["train_dataset"]),
        SELECTION_METRIC=str(spec["selection_metric"]),
        NUMBER_OF_EPOCHS=float(spec["epochs"]),
        LEARNING_RATE=float(spec["learning_rate"]),
        RUN_FOLDS=list(spec["run_folds"]),
        LLANITO_PROCESSED_ROOT=LLANITO_PROCESSED_ROOT,
        LLANITO_FIXED_TEST_MANIFEST=LLANITO_FIXED_TEST_MANIFEST,
        MIAMI_TRAIN_MANIFEST=MIAMI_TRAIN_MANIFEST,
        MIAMI_VALIDATION_MANIFEST=MIAMI_VALIDATION_MANIFEST,
        NORMALISATION_FILE=NORMALISATION_FILE,
        MIAMI_ENCODER_SOURCE=MIAMI_ENCODER_SOURCE,
        LLANITO_ENCODER_SOURCE_ROOT=LLANITO_ENCODER_SOURCE_ROOT,
        LLANITO_SOURCE_FOLDS=list(LLANITO_SOURCE_FOLDS),
        AVERAGED_LLANITO_SOURCE=AVERAGED_LLANITO_SOURCE,
        AVERAGED_LLANITO_OUTPUT_ROOT=AVERAGED_LLANITO_OUTPUT_ROOT,
        AVERAGED_LLANITO_SUMMARY_ROOT=AVERAGED_LLANITO_SUMMARY_ROOT,
        NEW_MIAMI_ADAPTER_BOTTLENECK=NEW_MIAMI_ADAPTER_BOTTLENECK,
        NEW_MIAMI_ADAPTER_DROPOUT=NEW_MIAMI_ADAPTER_DROPOUT,
        RANDOM_SEED=RANDOM_SEED,
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
        MODEL_ROOT=model_root,
        OUTPUT_ROOT=output_root,
        SUMMARY_ROOT=summary_root,
    )


def llanito_fold_source(config, fold_number: int) -> Path:
    return config.LLANITO_ENCODER_SOURCE_ROOT / f"fold_{fold_number}" / "adapter"


def source_adapter_for_fold(config, fold_number: int) -> Path:
    if config.DIRECTION == "miami_to_llanito":
        return config.MIAMI_ENCODER_SOURCE
    if config.DIRECTION == "llanito_to_miami":
        return config.AVERAGED_LLANITO_SOURCE
    raise ValueError(f"Unknown direction: {config.DIRECTION!r}")


def fold_train_manifest(config, fold_number: int) -> Path:
    if config.TRAIN_DATASET == "llanito":
        return config.LLANITO_PROCESSED_ROOT / f"fold_{fold_number}" / "train.csv"
    if config.TRAIN_DATASET == "miami":
        return config.MIAMI_TRAIN_MANIFEST
    raise ValueError(f"Unknown train dataset: {config.TRAIN_DATASET!r}")


def fold_validation_manifest(config, fold_number: int) -> Path:
    if config.TRAIN_DATASET == "llanito":
        return config.LLANITO_PROCESSED_ROOT / f"fold_{fold_number}" / "validation.csv"
    if config.TRAIN_DATASET == "miami":
        return config.MIAMI_VALIDATION_MANIFEST
    raise ValueError(f"Unknown train dataset: {config.TRAIN_DATASET!r}")
