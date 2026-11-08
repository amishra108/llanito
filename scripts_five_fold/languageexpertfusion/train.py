from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.languageexpertfusion.config import (
    BASE_MODEL,
    DATALOADER_WORKERS,
    ENGLISH_TRAIN_MANIFEST,
    ENGLISH_VALIDATION_MANIFEST,
    EXPERT_EPOCHS,
    EXPERT_LEARNING_RATE,
    FOLD_NUMBERS,
    FORCE_RETRAIN_EXPERTS,
    FORCE_RETRAIN_FUSION,
    FUSION_EPOCHS,
    FUSION_LEARNING_RATE,
    GENERATION_MAX_LENGTH,
    GENERATION_NUM_BEAMS,
    GRADIENT_ACCUMULATION_STEPS,
    LANGUAGE,
    LLANITO_PROCESSED_ROOT,
    LOGGING_STEPS,
    MAX_EXPERT_TRAIN_RECORDINGS,
    MAX_EXPERT_VALIDATION_RECORDINGS,
    MAX_LLANITO_TRAIN_RECORDINGS,
    MAX_LLANITO_VALIDATION_RECORDINGS,
    NORMALISATION_FILE,
    RANDOM_SEED,
    RUN_ARCHITECTURES,
    RUN_FOLDS,
    SAVE_TOTAL_LIMIT,
    SPANISH_TRAIN_MANIFEST,
    SPANISH_VALIDATION_MANIFEST,
    TASK,
    TRAIN_BATCH_SIZE_PER_GPU,
    USE_BF16,
    USE_FP16,
    USE_GRADIENT_CHECKPOINTING,
    VALIDATION_BATCH_SIZE_PER_GPU,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    experiment_name,
    model_root,
    output_root,
)
from scripts_five_fold.languageexpertfusion.model import (
    build_expert_model,
    build_fusion_model,
    save_adapter,
    save_fusion,
)
from scripts_five_fold.train import (
    ValidationMetricsCallback,
    WhisperDataCollator,
    WhisperManifestDataset,
    build_compute_metrics,
    build_training_arguments,
    configure_whisper_generation,
    distributed_barrier,
    is_main_process,
    load_manifest,
    load_task_mappings,
    make_trainer,
    parameter_summary,
    select_subset,
    set_seed,
    validate_cuda_environment,
    validate_manifest_disjointness,
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row.setdefault("task", "miami")

    return rows


def folds_to_run() -> list[int]:
    if RUN_FOLDS is None:
        return list(FOLD_NUMBERS)
    return sorted(set(RUN_FOLDS))


def unwrap(model: nn.Module) -> WhisperForConditionalGeneration:
    current = model
    while hasattr(current, "module"):
        current = current.module
    if not isinstance(current, WhisperForConditionalGeneration):
        raise TypeError(type(current))
    return current


def plain_wer_metrics(processor: WhisperProcessor) -> Callable[[Any], dict[str, float]]:
    def metric(prediction: Any) -> dict[str, float]:
        predictions = prediction.predictions
        labels = prediction.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        hypotheses = processor.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )
        references = processor.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        import jiwer

        wer = 100.0 * jiwer.wer(references, hypotheses)
        return {
            "wer": wer,
            "raw_wer": wer,
            "normalised_wer": wer,
        }

    return metric


def train_stage(
    *,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    checkpoint_root: Path,
    epochs: int,
    learning_rate: float,
    seed: int,
    compute_metrics: Callable[[Any], dict[str, float]],
    callback: ValidationMetricsCallback,
) -> WhisperForConditionalGeneration:
    train_dataset = WhisperManifestDataset(train_rows, processor)
    validation_dataset = WhisperManifestDataset(validation_rows, processor)

    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    args = build_training_arguments(
        checkpoint_root,
        seed,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        train_batch_size=TRAIN_BATCH_SIZE_PER_GPU,
        validation_batch_size=VALIDATION_BATCH_SIZE_PER_GPU,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        use_fp16=USE_FP16,
        use_bf16=USE_BF16,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,
        dataloader_workers=DATALOADER_WORKERS,
    )

    trainer = make_trainer(
        model=model,
        training_args=args,
        processor=processor,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )

    print(f"Train recordings: {len(train_rows)}")
    print(f"Validation recordings: {len(validation_rows)}")
    print(f"Parameters: {parameter_summary(model)}")

    trainer.train()
    trainer.evaluate(eval_dataset=validation_dataset)

    result = unwrap(trainer.model)

    del trainer
    del train_dataset
    del validation_dataset

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def expert_directory(architecture: str, expert: str) -> Path:
    return model_root(architecture) / f"{expert}_expert"


def fusion_directory(architecture: str, fold: int) -> Path:
    return model_root(architecture) / f"fold_{fold}" / "fusion"


def train_expert(
    architecture: str,
    expert: str,
    train_manifest: Path,
    validation_manifest: Path,
) -> Path:
    destination = expert_directory(architecture, expert)

    if (
        not FORCE_RETRAIN_EXPERTS
        and (destination / "adapter_state.pt").exists()
        and (destination / "adapter_config.json").exists()
    ):
        print(f"Reusing {expert} expert: {destination}")
        return destination

    seed = RANDOM_SEED + (100 if expert == "spanish" else 200)
    set_seed(seed)

    train_rows = select_subset(
        read_rows(train_manifest),
        MAX_EXPERT_TRAIN_RECORDINGS,
        seed,
    )
    validation_rows = select_subset(
        read_rows(validation_manifest),
        MAX_EXPERT_VALIDATION_RECORDINGS,
        seed + 1,
    )

    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL,
        language=LANGUAGE,
        task=TASK,
    )
    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
    )

    model = build_expert_model(architecture)

    callback = ValidationMetricsCallback(
        f"{experiment_name(architecture)}_{expert}_expert",
        0,
        output_root(architecture) / f"{expert}_expert" / "training_metrics.csv",
    )

    model = train_stage(
        model=model,
        processor=processor,
        train_rows=train_rows,
        validation_rows=validation_rows,
        checkpoint_root=model_root(architecture) / f"{expert}_expert_checkpoints",
        epochs=EXPERT_EPOCHS,
        learning_rate=EXPERT_LEARNING_RATE,
        seed=seed,
        compute_metrics=plain_wer_metrics(processor),
        callback=callback,
    )

    if is_main_process():
        save_adapter(
            model,
            destination,
            architecture,
            expert_name=expert,
        )
        processor.save_pretrained(destination)

    distributed_barrier()
    return destination


def train_fold_fusion(
    architecture: str,
    fold: int,
    spanish_directory: Path,
    english_directory: Path,
    task_mappings: dict[str, dict[str, str]],
) -> None:
    destination = fusion_directory(architecture, fold)

    if (
        not FORCE_RETRAIN_FUSION
        and (destination / "fusion_state.pt").exists()
        and (destination / "fusion_config.json").exists()
    ):
        print(f"Reusing fusion fold {fold}: {destination}")
        return

    fold_root = LLANITO_PROCESSED_ROOT / f"fold_{fold}"
    train_rows = load_manifest(fold_root / "train.csv")
    validation_rows = load_manifest(fold_root / "validation.csv")

    validate_manifest_disjointness(train_rows, validation_rows)

    seed = RANDOM_SEED + 5000 + fold
    set_seed(seed)

    train_rows = select_subset(
        train_rows,
        MAX_LLANITO_TRAIN_RECORDINGS,
        seed,
    )
    validation_rows = select_subset(
        validation_rows,
        MAX_LLANITO_VALIDATION_RECORDINGS,
        seed + 1,
    )

    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL,
        language=LANGUAGE,
        task=TASK,
    )
    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
    )

    model = build_fusion_model(
        architecture,
        spanish_directory,
        english_directory,
    )

    callback = ValidationMetricsCallback(
        f"{experiment_name(architecture)}_fusion",
        fold,
        output_root(architecture)
        / f"fold_{fold}"
        / "fusion"
        / "training_metrics.csv",
    )

    model = train_stage(
        model=model,
        processor=processor,
        train_rows=train_rows,
        validation_rows=validation_rows,
        checkpoint_root=(
            model_root(architecture)
            / f"fold_{fold}"
            / "fusion_checkpoints"
        ),
        epochs=FUSION_EPOCHS,
        learning_rate=FUSION_LEARNING_RATE,
        seed=seed,
        compute_metrics=build_compute_metrics(
            processor,
            validation_rows,
            task_mappings,
        ),
        callback=callback,
    )

    if is_main_process():
        save_fusion(
            model,
            destination,
            architecture,
            spanish_directory,
            english_directory,
        )

    distributed_barrier()


def run_architecture(architecture: str) -> None:
    print("=" * 80)
    print(f"ARCHITECTURE: {architecture}")
    print("=" * 80)

    spanish = train_expert(
        architecture,
        "spanish",
        SPANISH_TRAIN_MANIFEST,
        SPANISH_VALIDATION_MANIFEST,
    )

    english = train_expert(
        architecture,
        "english",
        ENGLISH_TRAIN_MANIFEST,
        ENGLISH_VALIDATION_MANIFEST,
    )

    task_mappings = load_task_mappings(NORMALISATION_FILE)

    for fold in folds_to_run():
        train_fold_fusion(
            architecture,
            fold,
            spanish,
            english,
            task_mappings,
        )


def main() -> None:
    validate_cuda_environment(
        use_fp16=USE_FP16,
        use_bf16=USE_BF16,
    )

    for architecture in RUN_ARCHITECTURES:
        run_architecture(architecture)


if __name__ == "__main__":
    main()
