from __future__ import annotations

import argparse
import gc
import inspect
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
)
from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import (
    get_last_checkpoint,
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from scripts_five_fold.moresofttransfers.config import (
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    build_runtime_config,
    fold_train_manifest,
    fold_validation_manifest,
)

from scripts_five_fold.train import (
    WhisperDataCollator,
    WhisperManifestDataset,
    build_compute_metrics,
    build_training_arguments,
    configure_whisper_generation,
    distributed_barrier,
    is_main_process,
    load_manifest,
    load_task_mappings,
    parameter_summary,
    save_json,
    select_subset,
    set_seed,
    validate_cuda_environment,
    validate_manifest_disjointness,
)


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        choices=sorted(
            EXPERIMENTS
        ),
    )

    return parser.parse_args()


# PEFT helpers

def is_peft_adapter_directory(
    directory: Path,
) -> bool:
    return (
        (
            directory
            / "adapter_config.json"
        ).exists()
        and (
            (
                directory
                / "adapter_model.safetensors"
            ).exists()
            or (
                directory
                / "adapter_model.bin"
            ).exists()
        )
    )


def resolve_source_adapter(
    source_root: Path,
) -> Path:
    if not source_root.exists():
        raise FileNotFoundError(
            "Miami LoRA source missing:\n"
            f"{source_root}"
        )

    preferred = [
        (
            source_root
            / "best_miami_validation_adapter"
        ),
        (
            source_root
            / "best_adapter"
        ),
        (
            source_root
            / "adapter"
        ),
        source_root,
    ]

    for candidate in preferred:
        if (
            candidate.exists()
            and is_peft_adapter_directory(
                candidate
            )
        ):
            return candidate

    candidates = [
        path
        for path
        in source_root.rglob("*")
        if (
            path.is_dir()
            and is_peft_adapter_directory(
                path
            )
            and "checkpoint-"
            not in path.name
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise RuntimeError(
            "No selected PEFT adapter "
            "found under:\n"
            f"{source_root}"
        )

    raise RuntimeError(
        "Multiple PEFT adapters found; "
        "refusing to guess:\n"
        + "\n".join(
            str(path)
            for path
            in candidates[:20]
        )
    )


def lora_config_from_source(
    source_adapter: Path,
) -> LoraConfig:
    source = (
        PeftConfig
        .from_pretrained(
            source_adapter
        )
    )

    target_modules = (
        source.target_modules
    )

    if target_modules is None:
        raise RuntimeError(
            "Source LoRA config has "
            "no target_modules."
        )

    # IMPORTANT:
    #
    # Do NOT set:
    #
    # task_type=TaskType.SEQ_2_SEQ_LM
    #
    # Whisper is an audio seq2seq model,
    # and PEFT's text seq2seq wrapper can
    # pass input_ids into Whisper in a way
    # that collides with Whisper's decoder
    # arguments.
    #
    # The working miamiencdec LoRA code
    # creates a generic LoraConfig instead.
    # We follow the same pattern here.

    return LoraConfig(
        inference_mode=False,
        r=int(
            source.r
        ),
        lora_alpha=int(
            source.lora_alpha
        ),
        lora_dropout=float(
            source.lora_dropout
        ),
        bias=str(
            source.bias
        ),
        target_modules=list(
            target_modules
        ),
    )


# Model builders

def load_vanilla_whisper(
    config,
):
    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            config.BASE_MODEL
        )
    )

    configure_whisper_generation(
        model,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=False,
    )

    return model


def build_full_model_continue(
    config,
):
    if not (
        config
        .SOURCE_ROOT
        .exists()
    ):
        raise FileNotFoundError(
            "Miami full fine-tuned "
            "model missing:\n"
            f"{config.SOURCE_ROOT}"
        )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            str(
                config.SOURCE_ROOT
            )
        )
    )

    configure_whisper_generation(
        model,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=False,
    )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            True
        )

    return (
        model,
        None,
    )


def build_same_lora(
    config,
):
    source_adapter = (
        resolve_source_adapter(
            config.SOURCE_ROOT
        )
    )

    base_model = (
        load_vanilla_whisper(
            config
        )
    )

    model = (
        PeftModel
        .from_pretrained(
            base_model,
            source_adapter,
            is_trainable=True,
        )
    )

    trainable = [
        name
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if parameter.requires_grad
    ]

    unexpected = [
        name
        for name
        in trainable
        if (
            "lora_"
            not in name
            and "modules_to_save"
            not in name
        )
    ]

    if not trainable:
        raise RuntimeError(
            "No trainable LoRA "
            "parameters were loaded."
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected non-LoRA "
            "trainable parameters: "
            f"{unexpected[:20]}"
        )

    return (
        model,
        source_adapter,
    )


def build_stacked_lora(
    config,
):
    source_adapter = (
        resolve_source_adapter(
            config.SOURCE_ROOT
        )
    )

    # Reconstruct Miami LoRA
    # on vanilla Whisper.

    base_model = (
        load_vanilla_whisper(
            config
        )
    )

    miami_model = (
        PeftModel
        .from_pretrained(
            base_model,
            source_adapter,
            is_trainable=False,
        )
    )

    # Merge the Miami LoRA into
    # the ordinary Whisper weights:
    #
    # W_M = W + delta_W_Miami

    merged_miami_model = (
        miami_model
        .merge_and_unload()
    )

    configure_whisper_generation(
        merged_miami_model,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=False,
    )

    # Freeze the merged Miami
    # Whisper backbone.

    for parameter in (
        merged_miami_model
        .parameters()
    ):
        parameter.requires_grad = (
            False
        )

    # Build a fresh Llanito LoRA.
    #
    # IMPORTANT:
    # lora_config_from_source()
    # intentionally creates a generic
    # LoraConfig rather than a
    # SEQ_2_SEQ_LM PEFT wrapper.

    new_lora_config = (
        lora_config_from_source(
            source_adapter
        )
    )

    model = get_peft_model(
        merged_miami_model,
        new_lora_config,
    )

    # Verify that the fresh PEFT
    # wrapper is not the problematic
    # text seq2seq task wrapper.

    active_config = (
        model.peft_config[
            model.active_adapter
        ]
    )

    if (
        getattr(
            active_config,
            "task_type",
            None,
        )
        is not None
    ):
        raise RuntimeError(
            "Fresh stacked LoRA "
            "unexpectedly has "
            "task_type="
            f"{active_config.task_type!r}. "
            "Whisper stacked LoRA must "
            "use the generic PEFT wrapper."
        )

    trainable = [
        name
        for (
            name,
            parameter,
        ) in (
            model
            .named_parameters()
        )
        if parameter.requires_grad
    ]

    unexpected = [
        name
        for name
        in trainable
        if (
            "lora_"
            not in name
            and "modules_to_save"
            not in name
        )
    ]

    if not trainable:
        raise RuntimeError(
            "Fresh Llanito LoRA has "
            "no trainable parameters."
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected trainable "
            "parameters in stacked "
            "LoRA: "
            f"{unexpected[:20]}"
        )

    return (
        model,
        source_adapter,
    )


def build_model(
    config,
):
    if (
        config.MODE
        == "full_model_continue"
    ):
        return (
            build_full_model_continue(
                config
            )
        )

    if (
        config.MODE
        == "same_lora"
    ):
        return (
            build_same_lora(
                config
            )
        )

    if (
        config.MODE
        == "stacked_lora"
    ):
        return (
            build_stacked_lora(
                config
            )
        )

    raise ValueError(
        "Unknown transfer mode: "
        f"{config.MODE!r}"
    )


def underlying_config(
    model,
):
    if isinstance(
        model,
        PeftModel,
    ):
        return (
            model
            .get_base_model()
            .config
        )

    return model.config


def unwrap_model(
    model,
):
    current = model

    while hasattr(
        current,
        "module",
    ):
        current = (
            current.module
        )

    return current


# Best-model callback

class SaveBestTransferCallback(
    TrainerCallback
):
    def __init__(
        self,
        *,
        config,
        processor,
        destination: Path,
        fold_number: int,
        source_adapter: (
            Path | None
        ),
    ) -> None:
        self.config = config
        self.processor = processor
        self.destination = (
            destination
        )
        self.fold_number = (
            fold_number
        )
        self.source_adapter = (
            source_adapter
        )

        self.best: (
            float | None
        ) = None

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        model=None,
        **kwargs,
    ):
        if (
            not metrics
            or model is None
            or not (
                state
                .is_world_process_zero
            )
        ):
            return control

        key = (
            "eval_normalised_wer"
        )

        if key not in metrics:
            raise KeyError(
                f"Expected {key!r}; "
                "available metrics: "
                f"{sorted(metrics)}"
            )

        value = float(
            metrics[key]
        )

        if (
            self.best is not None
            and value >= self.best
        ):
            return control

        self.best = value

        if self.destination.exists():
            shutil.rmtree(
                self.destination
            )

        self.destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_model = (
            unwrap_model(
                model
            )
        )

        save_model.save_pretrained(
            self.destination
        )

        self.processor.save_pretrained(
            self.destination
        )

        save_json(
            self.destination
            / "transfer_info.json",
            {
                "experiment": (
                    self.config
                    .EXPERIMENT_NAME
                ),

                "mode": (
                    self.config
                    .MODE
                ),

                "source_name": (
                    self.config
                    .SOURCE_NAME
                ),

                "source_root": str(
                    self.config
                    .SOURCE_ROOT
                ),

                "source_adapter": (
                    str(
                        self.source_adapter
                    )
                    if (
                        self.source_adapter
                        is not None
                    )
                    else None
                ),

                "fold": (
                    self.fold_number
                ),

                "selected_by": (
                    "llanito_validation_"
                    "normalised_wer"
                ),

                "best_validation_"
                "normalised_wer": (
                    value
                ),

                "epoch": (
                    state.epoch
                ),

                "global_step": (
                    state.global_step
                ),
            },
        )

        print(
            "NEW BEST LLANITO "
            "VALIDATION NORMALISED "
            "WER: "
            f"{value:.4f}%",
            flush=True,
        )

        return control


# Trainer

def build_trainer(
    *,
    model,
    training_args,
    processor,
    train_dataset,
    validation_dataset,
    collator,
    compute_metrics,
    callbacks,
):
    kwargs: dict[
        str,
        Any,
    ] = {
        "model": (
            model
        ),

        "args": (
            training_args
        ),

        "train_dataset": (
            train_dataset
        ),

        "eval_dataset": (
            validation_dataset
        ),

        "data_collator": (
            collator
        ),

        "compute_metrics": (
            compute_metrics
        ),

        "callbacks": list(
            callbacks
        ),
    }

    parameters = (
        inspect
        .signature(
            Seq2SeqTrainer
            .__init__
        )
        .parameters
    )

    if (
        "processing_class"
        in parameters
    ):
        kwargs[
            "processing_class"
        ] = processor

    else:
        kwargs[
            "tokenizer"
        ] = (
            processor
            .feature_extractor
        )

    return Seq2SeqTrainer(
        **kwargs
    )


# Housekeeping

def clear_fold(
    model_directory: Path,
    output_directory: Path,
    config,
) -> None:
    if not (
        config
        .CLEAR_SELECTED_FOLD_BEFORE_TRAINING
    ):
        return

    if (
        config
        .RESUME_FROM_LATEST_CHECKPOINT
    ):
        return

    if model_directory.exists():
        shutil.rmtree(
            model_directory
        )

    if output_directory.exists():
        shutil.rmtree(
            output_directory
        )


# Fold training

def train_fold(
    config,
    fold_number: int,
) -> None:
    fold_seed = (
        config.RANDOM_SEED
        + fold_number
    )

    set_seed(
        fold_seed
    )

    train_manifest = (
        fold_train_manifest(
            fold_number
        )
    )

    validation_manifest = (
        fold_validation_manifest(
            fold_number
        )
    )

    train_rows = (
        load_manifest(
            train_manifest
        )
    )

    validation_rows = (
        load_manifest(
            validation_manifest
        )
    )

    validate_manifest_disjointness(
        train_rows,
        validation_rows,
    )

    train_rows = (
        select_subset(
            train_rows,
            config
            .MAX_TRAIN_RECORDINGS,
            fold_seed,
        )
    )

    validation_rows = (
        select_subset(
            validation_rows,
            config
            .MAX_VALIDATION_RECORDINGS,
            fold_seed + 1000,
        )
    )

    model_directory = (
        config.MODEL_ROOT
        / f"fold_{fold_number}"
    )

    output_directory = (
        config.OUTPUT_ROOT
        / f"fold_{fold_number}"
    )

    checkpoint_directory = (
        model_directory
        / "checkpoints"
    )

    best_directory = (
        model_directory
        / "best_model"
    )

    if is_main_process():
        clear_fold(
            model_directory,
            output_directory,
            config,
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

    # Processor

    processor = (
        WhisperProcessor
        .from_pretrained(
            config.BASE_MODEL,
            language=(
                config.LANGUAGE
            ),
            task=(
                config.TASK
            ),
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=(
            config.LANGUAGE
        ),
        task=(
            config.TASK
        ),
        predict_timestamps=False,
    )

    # Model

    (
        model,
        source_adapter,
    ) = build_model(
        config
    )

    parameters = (
        parameter_summary(
            model
        )
    )

    if is_main_process():
        print()
        print(
            "=" * 80
        )

        print(
            f"{config.EXPERIMENT_NAME} "
            f"FOLD {fold_number}"
        )

        print(
            "=" * 80
        )

        print(
            f"Mode: "
            f"{config.MODE}"
        )

        print(
            f"Miami source: "
            f"{config.SOURCE_ROOT}"
        )

        print(
            f"Llanito train: "
            f"{train_manifest}"
        )

        print(
            "Llanito validation: "
            f"{validation_manifest}"
        )

        print(
            "Train recordings: "
            f"{len(train_rows)}"
        )

        print(
            "Validation recordings: "
            f"{len(validation_rows)}"
        )

        print(
            "Learning rate: "
            f"{config.LEARNING_RATE}"
        )

        print(
            f"Parameters: "
            f"{parameters}"
        )

        if isinstance(
            model,
            PeftModel,
        ):
            active_config = (
                model.peft_config[
                    model.active_adapter
                ]
            )

            print(
                "PEFT task type: "
                f"{getattr(active_config, 'task_type', None)}"
            )

    # Datasets

    train_dataset = (
        WhisperManifestDataset(
            train_rows,
            processor,
        )
    )

    validation_dataset = (
        WhisperManifestDataset(
            validation_rows,
            processor,
        )
    )

    # Collator

    collator = (
        WhisperDataCollator(
            processor=processor,
            decoder_start_token_id=(
                underlying_config(
                    model
                )
                .decoder_start_token_id
            ),
        )
    )

    # Metrics

    mappings = (
        load_task_mappings(
            config
            .NORMALISATION_FILE
        )
    )

    compute_metrics = (
        build_compute_metrics(
            processor,
            validation_rows,
            mappings,
        )
    )

    # Training arguments

    training_args = (
        build_training_arguments(
            checkpoint_directory,
            fold_seed,

            epochs=(
                config
                .NUMBER_OF_EPOCHS
            ),

            learning_rate=(
                config
                .LEARNING_RATE
            ),

            weight_decay=(
                config
                .WEIGHT_DECAY
            ),

            warmup_steps=(
                config
                .WARMUP_STEPS
            ),

            train_batch_size=(
                config
                .TRAIN_BATCH_SIZE_PER_GPU
            ),

            validation_batch_size=(
                config
                .VALIDATION_BATCH_SIZE_PER_GPU
            ),

            gradient_accumulation_steps=(
                config
                .GRADIENT_ACCUMULATION_STEPS
            ),

            use_fp16=(
                config
                .USE_FP16
            ),

            use_bf16=(
                config
                .USE_BF16
            ),

            use_gradient_checkpointing=(
                config
                .USE_GRADIENT_CHECKPOINTING
            ),

            logging_steps=(
                config
                .LOGGING_STEPS
            ),

            save_total_limit=(
                config
                .SAVE_TOTAL_LIMIT
            ),

            generation_max_length=(
                config
                .GENERATION_MAX_LENGTH
            ),

            generation_num_beams=(
                config
                .GENERATION_NUM_BEAMS
            ),

            dataloader_workers=(
                config
                .DATALOADER_WORKERS
            ),
        )
    )

    # Best model selection is handled
    # manually by our callback.

    training_args.load_best_model_at_end = (
        False
    )

    training_args.metric_for_best_model = (
        None
    )

    training_args.greater_is_better = (
        None
    )

    callback = (
        SaveBestTransferCallback(
            config=config,
            processor=processor,
            destination=(
                best_directory
            ),
            fold_number=(
                fold_number
            ),
            source_adapter=(
                source_adapter
            ),
        )
    )

    trainer = (
        build_trainer(
            model=model,
            training_args=(
                training_args
            ),
            processor=(
                processor
            ),
            train_dataset=(
                train_dataset
            ),
            validation_dataset=(
                validation_dataset
            ),
            collator=(
                collator
            ),
            compute_metrics=(
                compute_metrics
            ),
            callbacks=[
                callback
            ],
        )
    )

    # Resume

    checkpoint = None

    if (
        config
        .RESUME_FROM_LATEST_CHECKPOINT
    ):
        checkpoint = (
            get_last_checkpoint(
                str(
                    checkpoint_directory
                )
            )
        )

        print(
            "Resume checkpoint: "
            f"{checkpoint}",
            flush=True,
        )

    # Train

    train_result = (
        trainer.train(
            resume_from_checkpoint=(
                checkpoint
            )
        )
    )

    # Save bookkeeping

    if (
        trainer
        .is_world_process_zero()
    ):
        if not (
            best_directory
            .exists()
        ):
            raise RuntimeError(
                "Best model was "
                "never saved."
            )

        save_json(
            output_directory
            / "training_summary.json",
            {
                "experiment": (
                    config
                    .EXPERIMENT_NAME
                ),

                "mode": (
                    config.MODE
                ),

                "source_name": (
                    config
                    .SOURCE_NAME
                ),

                "source_root": str(
                    config
                    .SOURCE_ROOT
                ),

                "source_adapter": (
                    str(
                        source_adapter
                    )
                    if (
                        source_adapter
                        is not None
                    )
                    else None
                ),

                "fold": (
                    fold_number
                ),

                "train_manifest": str(
                    train_manifest
                ),

                "validation_manifest": str(
                    validation_manifest
                ),

                "training_recordings": (
                    len(
                        train_rows
                    )
                ),

                "validation_recordings": (
                    len(
                        validation_rows
                    )
                ),

                "best_validation_"
                "normalised_wer": (
                    callback.best
                ),

                "best_model": str(
                    best_directory
                ),

                "parameters": (
                    parameters
                ),

                "epochs": (
                    config
                    .NUMBER_OF_EPOCHS
                ),

                "learning_rate": (
                    config
                    .LEARNING_RATE
                ),

                "train_metrics": (
                    train_result
                    .metrics
                ),
            },
        )

    distributed_barrier()

    # Cleanup

    del trainer
    del model
    del processor
    del train_dataset
    del validation_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    distributed_barrier()


# Main

def main() -> None:
    args = parse_args()

    config = (
        build_runtime_config(
            args.experiment
        )
    )

    validate_cuda_environment(
        use_fp16=(
            config.USE_FP16
        ),
        use_bf16=(
            config.USE_BF16
        ),
    )

    if not (
        config
        .LLANITO_FIXED_TEST_MANIFEST
        .exists()
    ):
        raise FileNotFoundError(
            "Authoritative Llanito "
            "fixed test is missing:\n"
            f"{config.LLANITO_FIXED_TEST_MANIFEST}"
        )

    for fold_number in (
        config.RUN_FOLDS
    ):
        train_fold(
            config,
            fold_number,
        )

    if is_main_process():
        print()
        print(
            "=" * 80
        )
        print(
            f"{config.EXPERIMENT_NAME} "
            "TRAINING COMPLETE"
        )
        print(
            "=" * 80
        )

    distributed_barrier()


if __name__ == "__main__":
    main()