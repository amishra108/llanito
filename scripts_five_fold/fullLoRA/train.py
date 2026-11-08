from __future__ import annotations

import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.trainer_utils import get_last_checkpoint

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.fullLoRA.config_decoderAttnLoRA import (
    BASE_MODEL,
    CLEAR_SELECTED_FOLD_BEFORE_TRAINING,
    DATALOADER_WORKERS,
    EXPERIMENT_NAME,
    FOLD_NUMBERS,
    GENERATION_MAX_LENGTH,
    GENERATION_NUM_BEAMS,
    GRADIENT_ACCUMULATION_STEPS,
    LANGUAGE,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MAX_TRAIN_RECORDINGS,
    MAX_VALIDATION_RECORDINGS,
    MODEL_ROOT,
    NORMALISATION_FILE,
    NUMBER_OF_EPOCHS,
    OUTPUT_ROOT,
    PROCESSED_ROOT,
    RANDOM_SEED,
    RESUME_FROM_LATEST_CHECKPOINT,
    RUN_FOLDS,
    SAVE_TOTAL_LIMIT,
    SUMMARY_ROOT,
    TASK,
    TRAIN_BATCH_SIZE_PER_GPU,
    USE_BF16,
    USE_FP16,
    USE_GRADIENT_CHECKPOINTING,
    VALIDATION_BATCH_SIZE_PER_GPU,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)

from scripts_five_fold.train import (
    ValidationMetricsCallback,
    WhisperDataCollator,
    WhisperManifestDataset,
    build_compute_metrics,
    build_training_arguments,
    configure_whisper_generation,
    discover_fold_directories,
    distributed_barrier,
    extract_fold_number,
    is_main_process,
    load_manifest,
    load_task_mappings,
    make_trainer,
    parameter_summary,
    save_json,
    select_subset,
    set_seed,
    validate_cuda_environment,
    validate_manifest_disjointness,
)


def build_lora_model() -> torch.nn.Module:
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    configure_whisper_generation(
        base_model,
        language=LANGUAGE,
        task=TASK,
        use_cache=False,
    )

    lora_config = LoraConfig(
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=LORA_TARGET_MODULES,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def clear_selected_fold_outputs(
    model_directory: Path,
    output_directory: Path,
) -> None:
    if not CLEAR_SELECTED_FOLD_BEFORE_TRAINING:
        return

    if RESUME_FROM_LATEST_CHECKPOINT:
        return

    if model_directory.exists():
        print(f"Removing previous fold model: {model_directory}")
        shutil.rmtree(model_directory)

    if output_directory.exists():
        print(
            "Removing previous fold training outputs: "
            f"{output_directory}"
        )
        shutil.rmtree(output_directory)


def load_saved_validation_metrics(
    fold_number: int,
) -> dict[str, Any] | None:
    path = (
        OUTPUT_ROOT
        / f"fold_{fold_number}"
        / "final_validation_metrics.json"
    )

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def rebuild_training_cross_validation_summary() -> None:
    fold_rows: list[dict[str, Any]] = []
    missing_folds: list[int] = []

    for fold_number in FOLD_NUMBERS:
        metrics = load_saved_validation_metrics(fold_number)

        if metrics is None:
            missing_folds.append(fold_number)
            continue

        fold_rows.append(
            {
                "fold": fold_number,
                "normalised_wer": metrics.get(
                    "final_validation_normalised_wer"
                ),
                "raw_wer": metrics.get(
                    "final_validation_raw_wer"
                ),
            }
        )

    normalised_wers = [
        float(row["normalised_wer"])
        for row in fold_rows
        if row["normalised_wer"] is not None
    ]

    aggregate = {
        "experiment": EXPERIMENT_NAME,
        "folds": fold_rows,
        "missing_folds": missing_folds,
        "number_of_available_folds": len(fold_rows),
        "number_of_expected_folds": len(FOLD_NUMBERS),
        "normalised_wer_macro_mean": (
            sum(normalised_wers) / len(normalised_wers)
            if normalised_wers
            else None
        ),
    }

    # Keep the normal experiment copy.
    save_json(
        OUTPUT_ROOT / "cross_validation_summary.json",
        aggregate,
    )

    # Also keep only this requested high-level summary copy.
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    save_json(
        SUMMARY_ROOT / "cross_validation_summary.json",
        aggregate,
    )

    if missing_folds:
        print(
            "Training aggregate rebuilt from available folds only. "
            f"Missing folds: {missing_folds}"
        )
    else:
        print(
            "Training aggregate rebuilt from all folds: "
            f"{FOLD_NUMBERS}"
        )


def train_one_fold(
    fold_directory: Path,
    task_mappings: dict[str, dict[str, str]],
) -> dict[str, Any]:
    fold_number = extract_fold_number(fold_directory)
    fold_seed = RANDOM_SEED + fold_number
    set_seed(fold_seed)

    train_rows = load_manifest(fold_directory / "train.csv")
    validation_rows = load_manifest(
        fold_directory / "validation.csv"
    )
    validate_manifest_disjointness(
        train_rows,
        validation_rows,
    )

    train_rows = select_subset(
        train_rows,
        MAX_TRAIN_RECORDINGS,
        fold_seed,
    )
    validation_rows = select_subset(
        validation_rows,
        MAX_VALIDATION_RECORDINGS,
        fold_seed + 1000,
    )

    model_directory = MODEL_ROOT / f"fold_{fold_number}"
    adapter_directory = model_directory / "adapter"
    checkpoint_directory = model_directory / "checkpoints"
    output_directory = OUTPUT_ROOT / f"fold_{fold_number}"
    metrics_path = output_directory / "training_metrics.csv"

    if is_main_process():
        clear_selected_fold_outputs(
            model_directory=model_directory,
            output_directory=output_directory,
        )

        checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    distributed_barrier()

    print()
    print("#" * 80)
    print(f"STARTING {EXPERIMENT_NAME} FOLD {fold_number}")
    print("#" * 80)
    print(f"Training recordings: {len(train_rows)}")
    print(f"Validation recordings: {len(validation_rows)}")

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

    model = build_lora_model()
    parameters = parameter_summary(model)

    train_dataset = WhisperManifestDataset(
        train_rows,
        processor,
    )
    validation_dataset = WhisperManifestDataset(
        validation_rows,
        processor,
    )
    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=(
            model.config.decoder_start_token_id
        ),
    )

    training_args = build_training_arguments(
        checkpoint_directory,
        fold_seed,
        epochs=NUMBER_OF_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        train_batch_size=TRAIN_BATCH_SIZE_PER_GPU,
        validation_batch_size=VALIDATION_BATCH_SIZE_PER_GPU,
        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),
        use_fp16=USE_FP16,
        use_bf16=USE_BF16,
        use_gradient_checkpointing=(
            USE_GRADIENT_CHECKPOINTING
        ),
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,
        dataloader_workers=DATALOADER_WORKERS,
    )

    trainer = make_trainer(
        model=model,
        training_args=training_args,
        processor=processor,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        collator=collator,
        compute_metrics=build_compute_metrics(
            processor,
            validation_rows,
            task_mappings,
        ),
        callbacks=[
            ValidationMetricsCallback(
                EXPERIMENT_NAME,
                fold_number,
                metrics_path,
            )
        ],
    )

    checkpoint = None

    if RESUME_FROM_LATEST_CHECKPOINT:
        checkpoint = get_last_checkpoint(
            str(checkpoint_directory)
        )
        print(f"Resume checkpoint: {checkpoint}")

    train_result = trainer.train(
        resume_from_checkpoint=checkpoint
    )

    final_validation_metrics = trainer.evaluate(
        eval_dataset=validation_dataset,
        metric_key_prefix="final_validation",
    )

    if trainer.is_world_process_zero():
        adapter_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        trainer.model.save_pretrained(
            adapter_directory,
            safe_serialization=True,
        )
        processor.save_pretrained(
            adapter_directory
        )

        summary = {
            "experiment": EXPERIMENT_NAME,
            "fold": fold_number,
            "base_model": BASE_MODEL,
            "adapter_directory": str(
                adapter_directory
            ),
            "train_manifest": str(
                fold_directory / "train.csv"
            ),
            "validation_manifest": str(
                fold_directory / "validation.csv"
            ),
            "training_recordings": len(train_rows),
            "validation_recordings": len(
                validation_rows
            ),
            "lora": {
                "target_modules": LORA_TARGET_MODULES,
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "bias": LORA_BIAS,
            },
            "parameters": parameters,
            "epochs_requested": NUMBER_OF_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "effective_batch_size": (
                TRAIN_BATCH_SIZE_PER_GPU
                * GRADIENT_ACCUMULATION_STEPS
                * trainer.args.world_size
            ),
            "best_checkpoint": (
                trainer.state.best_model_checkpoint
            ),
            "best_metric": trainer.state.best_metric,
            "train_metrics": train_result.metrics,
            "final_validation_metrics": (
                final_validation_metrics
            ),
        }

        save_json(
            output_directory / "training_summary.json",
            summary,
        )
        save_json(
            output_directory
            / "final_validation_metrics.json",
            final_validation_metrics,
        )

        print(f"Adapter saved to: {adapter_directory}")

    distributed_barrier()

    summary_for_return = {
        "fold": fold_number,
        "normalised_wer": (
            final_validation_metrics.get(
                "final_validation_normalised_wer"
            )
        ),
    }

    del trainer
    del model
    del processor
    del train_dataset
    del validation_dataset

    gc.collect()
    torch.cuda.empty_cache()

    distributed_barrier()
    return summary_for_return


def main() -> None:
    validate_cuda_environment(
        use_fp16=USE_FP16,
        use_bf16=USE_BF16,
    )

    fold_directories = discover_fold_directories(
        PROCESSED_ROOT,
        RUN_FOLDS,
    )
    task_mappings = load_task_mappings(
        NORMALISATION_FILE
    )

    if is_main_process():
        MODEL_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )
        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        selected_folds = [
            extract_fold_number(path)
            for path in fold_directories
        ]

        print(
            "Requested folds: "
            f"{RUN_FOLDS if RUN_FOLDS is not None else 'ALL'}"
        )
        print(f"Discovered folds: {selected_folds}")
        print(
            "LoRA target modules: "
            + ", ".join(LORA_TARGET_MODULES)
        )

    distributed_barrier()

    fold_summaries = [
        train_one_fold(
            fold_directory,
            task_mappings,
        )
        for fold_directory in fold_directories
    ]

    if is_main_process():
        rebuild_training_cross_validation_summary()

        print()
        print("=" * 80)
        print(f"{EXPERIMENT_NAME} TRAINING COMPLETE")
        print("=" * 80)
        print(
            "Trained folds this run: "
            f"{[item['fold'] for item in fold_summaries]}"
        )
        print(
            "Cross-validation summary also saved to: "
            f"{SUMMARY_ROOT / 'cross_validation_summary.json'}"
        )

    distributed_barrier()


if __name__ == "__main__":
    main()
