from __future__ import annotations

import argparse
import csv
import gc
import getpass
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import optuna
import torch
import torch.distributed as dist
from torch import nn
from transformers import (
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


# Generic experiment specifications

@dataclass(frozen=True)
class PlacementSpec:
    # Weight-space PEFT examples:
    # region: encoder, decoder, encoder_decoder, encoder_attention, custom
    # target_modules: q_proj, k_proj, v_proj, out_proj, fc1, fc2, etc.
    #
    # Representation-space PEFT examples:
    # representation_layers: layers receiving interventions
    # temporal_policy: prefix/suffix/token/frame/time-window/routed/etc.
    name: str = "all"
    region: str = "all"
    layer_indices: tuple[int, ...] | None = None
    layer_stride: int | None = None
    target_modules: tuple[str, ...] = ()
    representation_layers: tuple[int, ...] = ()
    temporal_policy: str | None = None
    route_policy: str | None = None
    extra: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DataSpec:
    # mode:
    #   cv               -> fold train/validation templates
    #   single           -> direct train + validation manifests
    #   union_difference -> validation = union - train
    mode: str

    train_manifest: str | None = None
    validation_manifest: str | None = None

    fold_train_template: str | None = None
    fold_validation_template: str | None = None
    folds: tuple[int, ...] = ()

    union_manifest: str | None = None

    # Preferred identity columns for union-minus-train reconstruction.
    identity_columns: tuple[str, ...] = (
        "audio_path",
        "audio_filepath",
        "audio_file",
        "filepath",
        "path",
        "file",
        "wav_path",
        "wav",
        "audio",
        "recording",
        "utterance_id",
        "id",
    )


@dataclass(frozen=True)
class OptimisationSearchSpec:
    learning_rate_low: float = 2.0e-6
    learning_rate_high: float = 1.5e-5
    effective_global_batch_sizes: tuple[int, ...] = (
        16,
        32,
    )

    # Deliberate initial coverage. Architecture plugins can add their
    # own parameters to startup recipes later.
    startup_recipes: tuple[
        tuple[
            tuple[str, Any],
            ...
        ],
        ...
    ] = (
        (
            ("learning_rate", 3.0e-6),
            ("effective_global_batch_size", 16),
        ),
        (
            ("learning_rate", 3.0e-6),
            ("effective_global_batch_size", 32),
        ),
        (
            ("learning_rate", 1.0e-5),
            ("effective_global_batch_size", 16),
        ),
        (
            ("learning_rate", 1.0e-5),
            ("effective_global_batch_size", 32),
        ),
    )

    def startup_dicts(
        self,
    ) -> list[dict[str, Any]]:
        return [
            dict(
                recipe
            )
            for recipe in self.startup_recipes
        ]


@dataclass(frozen=True)
class TrainingProtocol:
    gradient_accumulation_steps: int = 8
    world_size: int = 2
    per_device_eval_batch_size: int = 1

    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    warmup_ratio: float = 0.05
    warmup_steps: int = 0

    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    dataloader_workers: int = 4
    logging_steps: int = 10

    generation_num_beams: int = 1
    generation_max_length: int = 225

    max_epochs: int = 5
    retain_all_epoch_checkpoints: bool = True

    hpo_seed: int = 2026
    final_model_seed: int = 2026
    optuna_sampler_seed: int = 2026


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    model_type: str
    domain_category: str

    project_root: str
    output_root: str

    base_model: str
    language: str
    whisper_task: str

    normalisation_file: str
    data: DataSpec
    search: OptimisationSearchSpec
    training: TrainingProtocol

    placement: PlacementSpec = PlacementSpec()

    wandb_project: str = "speech-hpo"
    study_name: str | None = None

    description: str = ""


@dataclass(frozen=True)
class ResolvedSplit:
    split_name: str
    train_manifest: Path
    validation_manifest: Path
    fold: int | None = None


# Architecture plugin API

class ArchitecturePlugin(ABC):
    # Every PEFT family gets its own plugin. The HPO engine does not
    # need to know whether the plugin edits weights, representations,
    # ranks, sparse masks, Fourier coefficients, routes, or adapters.

    method_name: str = "unknown"

    def sample_hparams(
        self,
        trial: optuna.Trial,
        experiment: ExperimentSpec,
    ) -> dict[str, Any]:
        return {}

    def startup_hparams(
        self,
        experiment: ExperimentSpec,
    ) -> list[dict[str, Any]]:
        return []

    @abstractmethod
    def build_model(
        self,
        *,
        experiment: ExperimentSpec,
        architecture_hparams: dict[str, Any],
    ) -> nn.Module:
        raise NotImplementedError

    def configure_after_build(
        self,
        *,
        model: nn.Module,
        experiment: ExperimentSpec,
        architecture_hparams: dict[str, Any],
    ) -> None:
        return None

    def build_datasets(
        self,
        *,
        default_dataset_cls,
        train_rows,
        validation_rows,
        processor,
        experiment: ExperimentSpec,
        architecture_hparams: dict[str, Any],
    ):
        # Routed/temporal methods can override this to expose
        # language IDs, frame routes, intervention masks, etc.
        return (
            default_dataset_cls(
                train_rows,
                processor,
            ),
            default_dataset_cls(
                validation_rows,
                processor,
            ),
        )

    def build_collator(
        self,
        *,
        default_collator_cls,
        processor,
        model,
        experiment: ExperimentSpec,
        architecture_hparams: dict[str, Any],
    ):
        decoder_start_token_id = (
            model.config.decoder_start_token_id
        )

        if decoder_start_token_id is None:
            raise RuntimeError(
                "Whisper decoder_start_token_id is None."
            )

        return default_collator_cls(
            processor=processor,
            decoder_start_token_id=(
                decoder_start_token_id
            ),
        )

    def create_trainer(
        self,
        *,
        default_make_trainer,
        model,
        training_args,
        processor,
        train_dataset,
        validation_dataset,
        collator,
        compute_metrics,
        experiment: ExperimentSpec,
        architecture_hparams: dict[str, Any],
    ):
        # A method with custom losses/routing can override this.
        return default_make_trainer(
            model=model,
            training_args=training_args,
            processor=processor,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            collator=collator,
            compute_metrics=compute_metrics,
        )

    def metadata(
        self,
        *,
        experiment: ExperimentSpec,
        architecture_hparams: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "placement": asdict(
                experiment.placement
            ),
            "architecture_hparams": architecture_hparams,
        }


class FullFineTunePlugin(
    ArchitecturePlugin
):
    method_name = "full_finetune"

    def build_model(
        self,
        *,
        experiment: ExperimentSpec,
        architecture_hparams: dict[str, Any],
    ) -> nn.Module:
        model = (
            WhisperForConditionalGeneration
            .from_pretrained(
                experiment.base_model,
                local_files_only=True,
            )
        )

        for parameter in model.parameters():
            parameter.requires_grad = True

        return model


# Placement helpers for future weight-space PEFT plugins

def whisper_linear_module_names(
    model: nn.Module,
    placement: PlacementSpec,
) -> list[str]:
    # Returns exact nn.Linear module names matching a placement.
    # This is useful for LoRA, RandLoRA, FourierFT, RoSA-style
    # low-rank branches, residual projections, and related methods.
    names: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(
            module,
            nn.Linear,
        ):
            continue

        if placement.region == "encoder":
            if ".encoder." not in name:
                continue

        elif placement.region == "decoder":
            if ".decoder." not in name:
                continue

        elif placement.region == "encoder_decoder":
            if (
                ".encoder." not in name
                and ".decoder." not in name
            ):
                continue

        elif placement.region == "encoder_attention":
            if (
                ".encoder." not in name
                or ".self_attn." not in name
            ):
                continue

        elif placement.region not in (
            "all",
            "custom",
        ):
            raise ValueError(
                "Unknown placement region: "
                f"{placement.region}"
            )

        if placement.layer_indices is not None:
            if not any(
                f".layers.{index}." in name
                for index in placement.layer_indices
            ):
                continue

        if placement.layer_stride is not None:
            matched_index = None

            parts = name.split(".")

            for i, token in enumerate(parts):
                if (
                    token == "layers"
                    and i + 1 < len(parts)
                ):
                    try:
                        matched_index = int(
                            parts[i + 1]
                        )
                    except ValueError:
                        matched_index = None

                    break

            if (
                matched_index is None
                or matched_index
                % placement.layer_stride
                != 0
            ):
                continue

        if placement.target_modules:
            if not any(
                name.endswith(
                    suffix
                )
                for suffix in (
                    placement.target_modules
                )
            ):
                continue

        names.append(
            name
        )

    return names


# File utilities

def save_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        handle.write("\n")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        if fieldnames is not None:
            with path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                )
                writer.writeheader()

        return

    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()

        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(
                        key
                    )
                    fieldnames.append(
                        key
                    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def append_csv_row(
    path: Path,
    row: dict[str, Any],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = (
        path.exists()
        and path.stat().st_size > 0
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            row
        )


def load_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


# Data plan resolution

def _resolve_path(
    project_root: Path,
    value: str,
) -> Path:
    path = Path(
        value
    )

    if not path.is_absolute():
        path = (
            project_root
            / path
        )

    return path.resolve()


def _derive_validation_from_union(
    *,
    train_path: Path,
    union_path: Path,
    output_path: Path,
    identity_columns: tuple[str, ...],
) -> Path:
    train_rows = load_csv_rows(
        train_path
    )

    union_rows = load_csv_rows(
        union_path
    )

    if not train_rows:
        raise RuntimeError(
            f"Training manifest is empty: {train_path}"
        )

    if not union_rows:
        raise RuntimeError(
            f"Union manifest is empty: {union_path}"
        )

    common_columns = set(
        train_rows[0]
    ).intersection(
        union_rows[0]
    )

    identity_column = next(
        (
            column
            for column in identity_columns
            if column in common_columns
        ),
        None,
    )

    if identity_column is None:
        raise RuntimeError(
            "Cannot derive validation = union - train because "
            "no stable identity column was found. "
            f"Tried {identity_columns}. "
            f"Common columns: {sorted(common_columns)}"
        )

    train_ids = {
        row.get(
            identity_column,
            ""
        ).strip()
        for row in train_rows
    }

    if "" in train_ids:
        raise RuntimeError(
            f"Blank {identity_column!r} values exist "
            f"in {train_path}."
        )

    validation_rows = [
        row
        for row in union_rows
        if row.get(
            identity_column,
            ""
        ).strip()
        not in train_ids
    ]

    if not validation_rows:
        raise RuntimeError(
            "Derived validation split is empty. "
            "Check that the union manifest actually contains "
            "train + validation and that the identity column is correct."
        )

    union_ids = [
        row.get(
            identity_column,
            ""
        ).strip()
        for row in union_rows
    ]

    if len(
        union_ids
    ) != len(
        set(
            union_ids
        )
    ):
        raise RuntimeError(
            f"Union manifest contains duplicate "
            f"{identity_column!r} values; refusing to derive "
            "a validation split ambiguously."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                union_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            validation_rows
        )

    return output_path


def resolve_splits(
    experiment: ExperimentSpec,
) -> list[ResolvedSplit]:
    project_root = Path(
        experiment.project_root
    ).resolve()

    output_root = Path(
        experiment.output_root
    ).resolve()

    data = experiment.data

    if data.mode == "cv":
        if (
            not data.fold_train_template
            or not data.fold_validation_template
            or not data.folds
        ):
            raise ValueError(
                "CV mode requires fold templates and fold IDs."
            )

        return [
            ResolvedSplit(
                split_name=(
                    f"fold_{fold}"
                ),
                fold=fold,
                train_manifest=_resolve_path(
                    project_root,
                    data.fold_train_template.format(
                        fold=fold
                    ),
                ),
                validation_manifest=_resolve_path(
                    project_root,
                    data.fold_validation_template.format(
                        fold=fold
                    ),
                ),
            )
            for fold in data.folds
        ]

    if data.mode == "single":
        if (
            not data.train_manifest
            or not data.validation_manifest
        ):
            raise ValueError(
                "Single mode requires train and validation manifests."
            )

        return [
            ResolvedSplit(
                split_name="validation",
                train_manifest=_resolve_path(
                    project_root,
                    data.train_manifest,
                ),
                validation_manifest=_resolve_path(
                    project_root,
                    data.validation_manifest,
                ),
            )
        ]

    if data.mode == "union_difference":
        if (
            not data.train_manifest
            or not data.union_manifest
        ):
            raise ValueError(
                "union_difference mode requires train_manifest "
                "and union_manifest."
            )

        train_path = _resolve_path(
            project_root,
            data.train_manifest,
        )

        union_path = _resolve_path(
            project_root,
            data.union_manifest,
        )

        derived_path = (
            output_root
            / "_derived"
            / "validation_manifest.csv"
        )

        if not derived_path.exists():
            _derive_validation_from_union(
                train_path=train_path,
                union_path=union_path,
                output_path=derived_path,
                identity_columns=data.identity_columns,
            )

        return [
            ResolvedSplit(
                split_name="validation",
                train_manifest=train_path,
                validation_manifest=derived_path,
            )
        ]

    raise ValueError(
        f"Unknown data mode: {data.mode}"
    )


# Metrics and selection

def extract_validation_history(
    log_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for entry in log_history:
        if (
            "eval_normalised_wer"
            not in entry
        ):
            continue

        rows.append(
            {
                "epoch": entry.get(
                    "epoch"
                ),
                "global_step": entry.get(
                    "step"
                ),
                "normalised_wer": entry.get(
                    "eval_normalised_wer"
                ),
                "wer": entry.get(
                    "eval_wer"
                ),
                "eval_loss": entry.get(
                    "eval_loss"
                ),
                "normalised_substitutions": entry.get(
                    "eval_normalised_substitutions"
                ),
                "normalised_deletions": entry.get(
                    "eval_normalised_deletions"
                ),
                "normalised_insertions": entry.get(
                    "eval_normalised_insertions"
                ),
                "normalised_reference_words": entry.get(
                    "eval_normalised_reference_words"
                ),
                "eval_runtime": entry.get(
                    "eval_runtime"
                ),
            }
        )

    return rows


def select_single_split_epoch(
    history: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    if not history:
        raise RuntimeError(
            "No validation WER history was recorded."
        )

    selected = min(
        history,
        key=lambda row: float(
            row[
                "normalised_wer"
            ]
        ),
    )

    return {
        "selected_epoch": int(
            round(
                float(
                    selected[
                        "epoch"
                    ]
                )
            )
        ),
        "objective_normalised_wer": float(
            selected[
                "normalised_wer"
            ]
        ),
        "validation_curve": history,
    }


def select_cv_common_epoch(
    split_results: dict[
        str,
        dict[str, Any],
    ],
    expected_epochs: int,
) -> dict[str, Any]:
    epoch_scores: dict[
        int,
        dict[str, float],
    ] = {}

    for split_name, result in (
        split_results.items()
    ):
        history = result[
            "validation_history"
        ]

        for row in history:
            epoch = int(
                round(
                    float(
                        row[
                            "epoch"
                        ]
                    )
                )
            )

            epoch_scores.setdefault(
                epoch,
                {},
            )[split_name] = float(
                row[
                    "normalised_wer"
                ]
            )

    curve: list[
        dict[str, Any]
    ] = []

    split_names = sorted(
        split_results
    )

    for epoch in range(
        1,
        expected_epochs + 1,
    ):
        if not all(
            name in epoch_scores.get(
                epoch,
                {},
            )
            for name in split_names
        ):
            raise RuntimeError(
                "Missing validation WER for one or more "
                f"splits at epoch {epoch}."
            )

        values = [
            epoch_scores[
                epoch
            ][name]
            for name in split_names
        ]

        curve.append(
            {
                "epoch": epoch,
                "mean_normalised_wer": mean(
                    values
                ),
                "std_normalised_wer_population": (
                    pstdev(
                        values
                    )
                    if len(
                        values
                    ) > 1
                    else 0.0
                ),
                "split_scores": {
                    name: epoch_scores[
                        epoch
                    ][name]
                    for name in split_names
                },
            }
        )

    selected = min(
        curve,
        key=lambda row: float(
            row[
                "mean_normalised_wer"
            ]
        ),
    )

    return {
        "selected_epoch": int(
            selected[
                "epoch"
            ]
        ),
        "objective_normalised_wer": float(
            selected[
                "mean_normalised_wer"
            ]
        ),
        "cv_curve": curve,
        "scores_at_selected_epoch": dict(
            selected[
                "split_scores"
            ]
        ),
    }


# Live W&B + CSV callback

LIVE_VALIDATION_FIELDS = [
    "session_id",
    "experiment",
    "trial_number",
    "split_name",
    "fold",
    "seed",
    "learning_rate",
    "effective_global_batch_size",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "epoch",
    "global_step",
    "normalised_wer",
    "wer",
    "eval_loss",
    "normalised_substitutions",
    "normalised_deletions",
    "normalised_insertions",
    "normalised_reference_words",
]


class LiveValidationCallback(
    TrainerCallback
):
    def __init__(
        self,
        *,
        session_id: str,
        experiment_name: str,
        trial_number: int,
        split: ResolvedSplit,
        seed: int,
        learning_rate: float,
        effective_global_batch_size: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        warmup_ratio: float,
        fold_live_csv: Path,
        global_live_csv: Path,
    ) -> None:
        self.session_id = (
            session_id
        )
        self.experiment_name = (
            experiment_name
        )
        self.trial_number = (
            trial_number
        )
        self.split = split
        self.seed = seed
        self.learning_rate = (
            learning_rate
        )
        self.effective_global_batch_size = (
            effective_global_batch_size
        )
        self.per_device_train_batch_size = (
            per_device_train_batch_size
        )
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        self.warmup_ratio = (
            warmup_ratio
        )
        self.fold_live_csv = (
            fold_live_csv
        )
        self.global_live_csv = (
            global_live_csv
        )

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):
        if not getattr(
            state,
            "is_world_process_zero",
            True,
        ):
            return control

        if not logs:
            return control

        if (
            "eval_normalised_wer"
            not in logs
        ):
            return control

        row = {
            "session_id": self.session_id,
            "experiment": self.experiment_name,
            "trial_number": self.trial_number,
            "split_name": self.split.split_name,
            "fold": self.split.fold,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "effective_global_batch_size": (
                self.effective_global_batch_size
            ),
            "per_device_train_batch_size": (
                self.per_device_train_batch_size
            ),
            "gradient_accumulation_steps": (
                self.gradient_accumulation_steps
            ),
            "warmup_ratio": (
                self.warmup_ratio
            ),
            "epoch": logs.get(
                "epoch",
                state.epoch,
            ),
            "global_step": state.global_step,
            "normalised_wer": logs.get(
                "eval_normalised_wer"
            ),
            "wer": logs.get(
                "eval_wer"
            ),
            "eval_loss": logs.get(
                "eval_loss"
            ),
            "normalised_substitutions": logs.get(
                "eval_normalised_substitutions"
            ),
            "normalised_deletions": logs.get(
                "eval_normalised_deletions"
            ),
            "normalised_insertions": logs.get(
                "eval_normalised_insertions"
            ),
            "normalised_reference_words": logs.get(
                "eval_normalised_reference_words"
            ),
        }

        append_csv_row(
            self.fold_live_csv,
            row,
            LIVE_VALIDATION_FIELDS,
        )

        append_csv_row(
            self.global_live_csv,
            row,
            LIVE_VALIDATION_FIELDS,
        )

        # Explicit clean W&B keys so WER-vs-epoch is guaranteed
        # to appear even if Transformers changes its own naming.
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "validation/normalised_wer": (
                            row[
                                "normalised_wer"
                            ]
                        ),
                        "validation/wer": (
                            row[
                                "wer"
                            ]
                        ),
                        "validation/loss": (
                            row[
                                "eval_loss"
                            ]
                        ),
                        "validation/epoch": (
                            row[
                                "epoch"
                            ]
                        ),
                    },
                    step=int(
                        state.global_step
                    ),
                )

        except Exception:
            # W&B visualisation must never make the scientific
            # training job fail.
            pass

        return control


# GPU diagnostics

def query_gpu_processes() -> list[
    dict[str, Any]
]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi is not available; refusing to assume "
            "that the GPUs are free."
        ) from exc

    if result.returncode != 0:
        message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown nvidia-smi error"
        )

        raise RuntimeError(
            "nvidia-smi failed; refusing to assume that the "
            f"GPUs are free: {message}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for line in (
        result.stdout.splitlines()
    ):
        line = line.strip()

        if not line:
            continue

        parts = [
            part.strip()
            for part in line.split(
                ","
            )
        ]

        if len(parts) != 3:
            continue

        pid = int(
            parts[0]
        )

        owner_result = subprocess.run(
            [
                "ps",
                "-o",
                "user=",
                "-p",
                str(
                    pid
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        rows.append(
            {
                "pid": pid,
                "memory_mib": int(
                    parts[1]
                ),
                "gpu_uuid": parts[2],
                "owner": (
                    owner_result.stdout.strip()
                    or "unknown"
                ),
            }
        )

    return rows


def wait_until_no_foreign_gpu_processes(
    *,
    enabled: bool,
    interval_seconds: int,
) -> None:
    if not enabled:
        return

    current_user = (
        getpass.getuser()
    )

    while True:
        foreign = [
            row
            for row in (
                query_gpu_processes()
            )
            if row[
                "owner"
            ] != current_user
        ]

        if not foreign:
            return

        print()
        print(
            "GPUs currently contain other users' compute processes:"
        )

        for row in foreign:
            print(
                "  "
                f"{row['owner']} "
                f"pid={row['pid']} "
                f"memory={row['memory_mib']} MiB"
            )

        print(
            f"Waiting {interval_seconds}s..."
        )

        time.sleep(
            interval_seconds
        )


# Internal path helpers

def _trial_dir(
    output_root: Path,
    trial_number: int,
) -> Path:
    return (
        output_root
        / "trials"
        / f"trial_{trial_number:04d}"
    )


def _split_dir(
    output_root: Path,
    trial_number: int,
    split_name: str,
) -> Path:
    return (
        _trial_dir(
            output_root,
            trial_number,
        )
        / split_name
    )


def _split_result_path(
    output_root: Path,
    trial_number: int,
    split_name: str,
) -> Path:
    return (
        _split_dir(
            output_root,
            trial_number,
            split_name,
        )
        / "result.json"
    )


def _session_manifest_path(
    output_root: Path,
) -> Path:
    return (
        output_root
        / "session.json"
    )


def _latest_checkpoint(
    checkpoint_dir: Path,
) -> Path | None:
    if not checkpoint_dir.exists():
        return None

    checkpoints = []

    for path in checkpoint_dir.glob(
        "checkpoint-*"
    ):
        if not path.is_dir():
            continue

        try:
            step = int(
                path.name.rsplit(
                    "-",
                    1,
                )[1]
            )
        except ValueError:
            continue

        checkpoints.append(
            (
                step,
                path,
            )
        )

    if not checkpoints:
        return None

    return max(
        checkpoints,
        key=lambda item: item[0],
    )[1]


# Generic runner

def _build_parser(
    experiment: ExperimentSpec,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"HPO runner for {experiment.name}"
        )
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--wandb-project",
        default=(
            experiment.wandb_project
        ),
    )

    parser.add_argument(
        "--wandb-entity",
        default=None,
    )

    parser.add_argument(
        "--wandb-mode",
        choices=(
            "online",
            "offline",
            "disabled",
        ),
        default="online",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete this experiment's HPO output directory "
            "and start a brand-new Optuna study."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue the existing Optuna study."
        ),
    )

    parser.add_argument(
        "--wait-for-free-gpus",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--gpu-check-interval",
        type=int,
        default=30,
    )

    # Worker-only arguments
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--trial-number",
        type=int,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--split-name",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--effective-global-batch-size",
        type=int,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--architecture-hparams-json",
        default="{}",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--session-id",
        help=argparse.SUPPRESS,
    )

    return parser


def _validate_output_root_safety(
    *,
    experiment: ExperimentSpec,
    launcher_path: Path,
) -> None:
    project_root = Path(
        experiment.project_root
    ).resolve()

    output_root = Path(
        experiment.output_root
    ).resolve()

    launcher = launcher_path.resolve()

    try:
        output_root.relative_to(
            project_root
        )
    except ValueError as exc:
        raise RuntimeError(
            "HPO output_root must live inside PROJECT_ROOT. "
            f"Got output_root={output_root}"
        ) from exc

    if output_root == project_root:
        raise RuntimeError(
            "HPO output_root cannot be PROJECT_ROOT itself."
        )

    try:
        launcher.relative_to(
            output_root
        )
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Unsafe HPO output_root: the launcher lives inside "
            "the directory that --reset would delete. "
            f"launcher={launcher}, output_root={output_root}"
        )


def _prepare_session(
    *,
    experiment: ExperimentSpec,
    args: argparse.Namespace,
) -> tuple[
    Path,
    str,
]:
    output_root = Path(
        experiment.output_root
    ).resolve()

    if (
        args.reset
        and args.resume
    ):
        raise ValueError(
            "Use either --reset or --resume, not both."
        )

    if args.reset:
        if output_root.exists():
            shutil.rmtree(
                output_root
            )

    study_db = (
        output_root
        / "optuna_study.sqlite3"
    )

    if (
        study_db.exists()
        and not args.resume
        and not args.worker
    ):
        raise RuntimeError(
            f"Existing HPO study found at {study_db}. "
            "Use --resume to continue it or --reset to start over."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    session_path = (
        _session_manifest_path(
            output_root
        )
    )

    if (
        args.resume
        and session_path.exists()
    ):
        session = json.loads(
            session_path.read_text(
                encoding="utf-8"
            )
        )

        session_id = str(
            session[
                "session_id"
            ]
        )

    else:
        session_id = (
            uuid.uuid4().hex[
                :10
            ]
        )

        save_json(
            session_path,
            {
                "session_id": (
                    session_id
                ),
                "experiment": (
                    experiment.name
                ),
                "created_unix": (
                    time.time()
                ),
            },
        )

    return (
        output_root,
        session_id,
    )


def _validate_experiment(
    experiment: ExperimentSpec,
    splits: list[ResolvedSplit],
) -> None:
    normalisation_file = (
        Path(
            experiment.normalisation_file
        ).resolve()
    )

    if not normalisation_file.exists():
        raise FileNotFoundError(
            "Normalisation file does not exist:\n"
            f"{normalisation_file}"
        )

    for split in splits:
        if not split.train_manifest.exists():
            raise FileNotFoundError(
                f"Missing train manifest: "
                f"{split.train_manifest}"
            )

        if not split.validation_manifest.exists():
            raise FileNotFoundError(
                f"Missing validation manifest: "
                f"{split.validation_manifest}"
            )


def _resolve_gradient_accumulation_steps(
    *,
    architecture_hparams: dict[str, Any],
    protocol: TrainingProtocol,
) -> int:
    """
    Resolve an optional per-trial GA override.

    Architecture plugins may return:
        {"gradient_accumulation_steps": N}

    If they do not, the experiment-wide TrainingProtocol value is used.
    This keeps every existing experiment unchanged while allowing Stage 3
    FCCA to realise EBS=8 on two GPUs with GA=4.
    """
    value = int(
        architecture_hparams.get(
            "gradient_accumulation_steps",
            protocol.gradient_accumulation_steps,
        )
    )

    if value < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1, "
            f"got {value}"
        )

    return value


def _per_device_batch(
    *,
    effective_global_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> int:
    denominator = (
        int(gradient_accumulation_steps)
        * int(world_size)
    )

    if (
        effective_global_batch_size
        % denominator
        != 0
    ):
        raise ValueError(
            "Effective batch is not divisible by "
            "GA * world_size: "
            f"{effective_global_batch_size} / {denominator}"
        )

    value = (
        effective_global_batch_size
        // denominator
    )

    if value < 1:
        raise ValueError(
            "Derived per-device batch is < 1."
        )

    return int(
        value
    )



def _build_training_args(
    *,
    experiment: ExperimentSpec,
    checkpoint_dir: Path,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> Seq2SeqTrainingArguments:
    p = experiment.training

    values: dict[
        str,
        Any,
    ] = {
        "output_dir": str(
            checkpoint_dir
        ),
        "num_train_epochs": float(
            p.max_epochs
        ),
        "per_device_train_batch_size": (
            per_device_train_batch_size
        ),
        "per_device_eval_batch_size": (
            p.per_device_eval_batch_size
        ),
        "gradient_accumulation_steps": (
            int(
                gradient_accumulation_steps
            )
        ),
        "learning_rate": float(
            learning_rate
        ),
        "weight_decay": (
            p.weight_decay
        ),
        "warmup_steps": (
            p.warmup_steps
        ),
        "warmup_ratio": (
            p.warmup_ratio
        ),
        "lr_scheduler_type": (
            p.lr_scheduler_type
        ),
        "max_grad_norm": (
            p.max_grad_norm
        ),
        "optim": p.optimizer,
        "fp16": p.fp16,
        "bf16": p.bf16,
        "gradient_checkpointing": (
            p.gradient_checkpointing
        ),
        "logging_strategy": "steps",
        "logging_steps": (
            p.logging_steps
        ),
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": (
            p.max_epochs
            if p.retain_all_epoch_checkpoints
            else 1
        ),
        "load_best_model_at_end": True,
        "metric_for_best_model": (
            "normalised_wer"
        ),
        "greater_is_better": False,
        "predict_with_generate": True,
        "generation_max_length": (
            p.generation_max_length
        ),
        "generation_num_beams": (
            p.generation_num_beams
        ),
        "eval_accumulation_steps": 1,
        "remove_unused_columns": False,
        "dataloader_num_workers": (
            p.dataloader_workers
        ),
        "dataloader_pin_memory": True,
        "report_to": (
            ["wandb"]
            if os.environ.get(
                "WANDB_MODE",
                "online",
            )
            != "disabled"
            else []
        ),
        "run_name": os.environ.get(
            "WANDB_NAME"
        ),
        "seed": p.hpo_seed,
        "data_seed": p.hpo_seed,
        "ddp_find_unused_parameters": False,
    }

    signature = inspect.signature(
        Seq2SeqTrainingArguments.__init__
    ).parameters

    if "eval_strategy" in signature:
        values[
            "eval_strategy"
        ] = "epoch"

    elif "evaluation_strategy" in signature:
        values[
            "evaluation_strategy"
        ] = "epoch"

    else:
        raise RuntimeError(
            "Installed Transformers exposes neither "
            "eval_strategy nor evaluation_strategy."
        )

    if (
        p.gradient_checkpointing
        and "gradient_checkpointing_kwargs"
        in signature
    ):
        values[
            "gradient_checkpointing_kwargs"
        ] = {
            "use_reentrant": False,
        }

    return Seq2SeqTrainingArguments(
        **values
    )


def _copy_epoch_checkpoints(
    *,
    trainer,
    source_root: Path,
    destination_root: Path,
    max_epochs: int,
) -> list[dict[str, Any]]:
    if not trainer.is_world_process_zero():
        return []

    if destination_root.exists():
        shutil.rmtree(
            destination_root
        )

    destination_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_by_step: dict[
        int,
        dict[str, Any],
    ] = {}

    for entry in (
        trainer.state.log_history
    ):
        if (
            "eval_normalised_wer"
            not in entry
        ):
            continue

        if (
            entry.get(
                "step"
            )
            is None
            or entry.get(
                "epoch"
            )
            is None
        ):
            continue

        eval_by_step[
            int(
                entry[
                    "step"
                ]
            )
        ] = {
            "epoch": int(
                round(
                    float(
                        entry[
                            "epoch"
                        ]
                    )
                )
            ),
            "normalised_wer": float(
                entry[
                    "eval_normalised_wer"
                ]
            ),
        }

    checkpoints = sorted(
        [
            path
            for path in (
                source_root.glob(
                    "checkpoint-*"
                )
            )
            if path.is_dir()
        ],
        key=lambda path: int(
            path.name.split(
                "-"
            )[-1]
        ),
    )

    rows: list[
        dict[str, Any]
    ] = []

    for checkpoint in checkpoints:
        step = int(
            checkpoint.name.split(
                "-"
            )[-1]
        )

        if step not in eval_by_step:
            raise RuntimeError(
                "Could not map checkpoint to validation epoch: "
                f"{checkpoint}"
            )

        info = eval_by_step[
            step
        ]

        epoch = int(
            info[
                "epoch"
            ]
        )

        destination = (
            destination_root
            / f"epoch_{epoch}"
        )

        shutil.copytree(
            checkpoint,
            destination,
        )

        rows.append(
            {
                "epoch": epoch,
                "global_step": step,
                "normalised_wer": (
                    info[
                        "normalised_wer"
                    ]
                ),
                "directory": str(
                    destination
                ),
                "is_individual_split_best": (
                    str(
                        trainer.state
                        .best_model_checkpoint
                    )
                    == str(
                        checkpoint
                    )
                ),
            }
        )

    expected = list(
        range(
            1,
            max_epochs + 1,
        )
    )

    actual = [
        int(
            row[
                "epoch"
            ]
        )
        for row in rows
    ]

    if actual != expected:
        raise RuntimeError(
            "Expected one checkpoint per epoch. "
            f"Expected {expected}, got {actual}."
        )

    return rows


def _run_worker(
    *,
    experiment: ExperimentSpec,
    plugin: ArchitecturePlugin,
    args: argparse.Namespace,
) -> None:
    from scripts.train import (
        WhisperDataCollator,
        WhisperManifestDataset,
        build_compute_metrics,
        configure_whisper_generation,
        load_manifest,
        load_task_mappings,
        make_trainer,
        parameter_summary,
        set_seed,
        validate_cuda_environment,
        validate_manifest_disjointness,
    )

    output_root = Path(
        experiment.output_root
    ).resolve()

    splits = resolve_splits(
        experiment
    )

    split = next(
        (
            candidate
            for candidate in splits
            if candidate.split_name
            == args.split_name
        ),
        None,
    )

    if split is None:
        raise ValueError(
            f"Unknown split_name {args.split_name!r}."
        )

    trial_number = int(
        args.trial_number
    )

    learning_rate = float(
        args.learning_rate
    )

    effective_batch = int(
        args.effective_global_batch_size
    )

    architecture_hparams = (
        json.loads(
            args.architecture_hparams_json
        )
    )

    protocol = (
        experiment.training
    )

    gradient_accumulation_steps = (
        _resolve_gradient_accumulation_steps(
            architecture_hparams=(
                architecture_hparams
            ),
            protocol=protocol,
        )
    )

    per_device_batch = (
        _per_device_batch(
            effective_global_batch_size=(
                effective_batch
            ),
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
            world_size=(
                protocol.world_size
            ),
        )
    )

    if int(
        os.environ.get(
            "RANK",
            "0",
        )
    ) == 0:
        print(
            "[HPO] resolved batch: "
            f"EBS={effective_batch}, "
            f"world_size={protocol.world_size}, "
            f"GA={gradient_accumulation_steps}, "
            f"per_device_train_batch_size={per_device_batch}",
            flush=True,
        )

    set_seed(
        protocol.hpo_seed
    )

    validate_cuda_environment(
        use_fp16=(
            protocol.fp16
        ),
        use_bf16=(
            protocol.bf16
        ),
    )

    train_rows = load_manifest(
        split.train_manifest
    )

    validation_rows = load_manifest(
        split.validation_manifest
    )

    validate_manifest_disjointness(
        train_rows,
        validation_rows,
    )

    split_dir = (
        _split_dir(
            output_root,
            trial_number,
            split.split_name,
        )
    )

    work_dir = (
        split_dir
        / "_work"
    )

    checkpoint_dir = (
        work_dir
        / "checkpoints"
    )

    if int(
        os.environ.get(
            "RANK",
            "0",
        )
    ) == 0:
        checkpoint = (
            _latest_checkpoint(
                checkpoint_dir
            )
        )

        if (
            work_dir.exists()
            and checkpoint is None
        ):
            shutil.rmtree(
                work_dir
            )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = (
            _latest_checkpoint(
                checkpoint_dir
            )
        )

        if checkpoint is not None:
            print(
                "[HPO] found resumable checkpoint: "
                f"{checkpoint}",
                flush=True,
            )

    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        dist.barrier()

    processor = (
        WhisperProcessor
        .from_pretrained(
            experiment.base_model,
            language=experiment.language,
            task=experiment.whisper_task,
            local_files_only=True,
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=(
            experiment.language
        ),
        task=(
            experiment.whisper_task
        ),
        predict_timestamps=False,
    )

    model = plugin.build_model(
        experiment=experiment,
        architecture_hparams=(
            architecture_hparams
        ),
    )

    configure_whisper_generation(
        model,
        language=(
            experiment.language
        ),
        task=(
            experiment.whisper_task
        ),
        use_cache=False,
    )

    plugin.configure_after_build(
        model=model,
        experiment=experiment,
        architecture_hparams=(
            architecture_hparams
        ),
    )

    (
        train_dataset,
        validation_dataset,
    ) = plugin.build_datasets(
        default_dataset_cls=(
            WhisperManifestDataset
        ),
        train_rows=train_rows,
        validation_rows=validation_rows,
        processor=processor,
        experiment=experiment,
        architecture_hparams=(
            architecture_hparams
        ),
    )

    collator = plugin.build_collator(
        default_collator_cls=(
            WhisperDataCollator
        ),
        processor=processor,
        model=model,
        experiment=experiment,
        architecture_hparams=(
            architecture_hparams
        ),
    )

    task_mappings = (
        load_task_mappings(
            Path(
                experiment.normalisation_file
            )
        )
    )

    if (
        experiment.domain_category
        == "o"
    ):
        from scripts.odia_normalisation import (
            normalise_odia_hypothesis,
            normalise_odia_reference,
        )

        compute_metrics = (
            build_compute_metrics(
                processor,
                validation_rows,
                task_mappings,
                reference_text_transform=(
                    normalise_odia_reference
                ),
                hypothesis_text_transform=(
                    normalise_odia_hypothesis
                ),
            )
        )

    else:
        compute_metrics = (
            build_compute_metrics(
                processor,
                validation_rows,
                task_mappings,
            )
        )

    training_args = (
        _build_training_args(
            experiment=experiment,
            checkpoint_dir=(
                checkpoint_dir
            ),
            learning_rate=(
                learning_rate
            ),
            per_device_train_batch_size=(
                per_device_batch
            ),
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
        )
    )

    trainer = plugin.create_trainer(
        default_make_trainer=(
            make_trainer
        ),
        model=model,
        training_args=(
            training_args
        ),
        processor=processor,
        train_dataset=(
            train_dataset
        ),
        validation_dataset=(
            validation_dataset
        ),
        collator=collator,
        compute_metrics=(
            compute_metrics
        ),
        experiment=experiment,
        architecture_hparams=(
            architecture_hparams
        ),
    )

    live_callback = (
        LiveValidationCallback(
            session_id=(
                args.session_id
            ),
            experiment_name=(
                experiment.name
            ),
            trial_number=(
                trial_number
            ),
            split=split,
            seed=(
                protocol.hpo_seed
            ),
            learning_rate=(
                learning_rate
            ),
            effective_global_batch_size=(
                effective_batch
            ),
            per_device_train_batch_size=(
                per_device_batch
            ),
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
            warmup_ratio=(
                float(training_args.warmup_ratio)
            ),
            fold_live_csv=(
                split_dir
                / "validation_live.csv"
            ),
            global_live_csv=(
                output_root
                / "validation_history_live.csv"
            ),
        )
    )

    trainer.add_callback(
        live_callback
    )

    resume_checkpoint = (
        _latest_checkpoint(
            checkpoint_dir
        )
    )

    if int(
        os.environ.get(
            "RANK",
            "0",
        )
    ) == 0:
        if resume_checkpoint is None:
            print(
                "[HPO] starting fold from scratch",
                flush=True,
            )
        else:
            print(
                "[HPO] resuming fold from "
                f"{resume_checkpoint}",
                flush=True,
            )

    train_result = (
        trainer.train(
            resume_from_checkpoint=(
                str(
                    resume_checkpoint
                )
                if resume_checkpoint
                is not None
                else None
            )
        )
    )

    history = (
        extract_validation_history(
            trainer.state.log_history
        )
    )

    epoch_checkpoints: list[
        dict[str, Any]
    ] = []

    if (
        protocol
        .retain_all_epoch_checkpoints
    ):
        epoch_checkpoints = (
            _copy_epoch_checkpoints(
                trainer=trainer,
                source_root=(
                    checkpoint_dir
                ),
                destination_root=(
                    split_dir
                    / "checkpoints"
                ),
                max_epochs=(
                    protocol.max_epochs
                ),
            )
        )

    if (
        trainer
        .is_world_process_zero()
    ):
        save_json(
            _split_result_path(
                output_root,
                trial_number,
                split.split_name,
            ),
            {
                "experiment": (
                    experiment.name
                ),
                "trial_number": (
                    trial_number
                ),
                "split_name": (
                    split.split_name
                ),
                "fold": split.fold,
                "seed": (
                    protocol.hpo_seed
                ),
                "learning_rate": (
                    learning_rate
                ),
                "effective_global_batch_size": (
                    effective_batch
                ),
                "per_device_train_batch_size": (
                    per_device_batch
                ),
                "gradient_accumulation_steps": (
                    gradient_accumulation_steps
                ),
                "warmup_ratio": (
                    float(training_args.warmup_ratio)
                ),
                "architecture": (
                    plugin.metadata(
                        experiment=experiment,
                        architecture_hparams=(
                            architecture_hparams
                        ),
                    )
                ),
                "train_manifest": str(
                    split.train_manifest
                ),
                "validation_manifest": str(
                    split.validation_manifest
                ),
                "training_examples": len(
                    train_rows
                ),
                "validation_examples": len(
                    validation_rows
                ),
                "validation_history": (
                    history
                ),
                "individual_split_best_metric": (
                    trainer.state.best_metric
                ),
                "individual_split_best_epoch": (
                    next(
                        (
                            item["epoch"]
                            for item in epoch_checkpoints
                            if item.get(
                                "is_individual_split_best"
                            )
                        ),
                        None,
                    )
                ),
                "epoch_checkpoints": (
                    epoch_checkpoints
                ),
                "train_metrics": (
                    train_result.metrics
                ),
                "parameters": (
                    parameter_summary(
                        model
                    )
                ),
            },
        )

    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        dist.barrier()

    del trainer
    del model
    del processor
    del train_dataset
    del validation_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        dist.barrier()

    if int(
        os.environ.get(
            "RANK",
            "0",
        )
    ) == 0:
        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )


def _rebuild_csv_exports(
    *,
    experiment: ExperimentSpec,
) -> None:
    output_root = Path(
        experiment.output_root
    ).resolve()

    trials_root = (
        output_root
        / "trials"
    )

    if not trials_root.exists():
        return

    long_rows: list[
        dict[str, Any]
    ] = []

    trial_rows: list[
        dict[str, Any]
    ] = []

    for trial_dir in sorted(
        trials_root.glob(
            "trial_*"
        )
    ):
        trial_number = int(
            trial_dir.name.split(
                "_"
            )[-1]
        )

        split_results: dict[
            str,
            dict[str, Any],
        ] = {}

        for result_path in sorted(
            trial_dir.glob(
                "*/result.json"
            )
        ):
            result = json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )

            split_name = str(
                result[
                    "split_name"
                ]
            )

            split_results[
                split_name
            ] = result

            for history in result[
                "validation_history"
            ]:
                long_rows.append(
                    {
                        "experiment": (
                            experiment.name
                        ),
                        "trial_number": (
                            trial_number
                        ),
                        "split_name": (
                            split_name
                        ),
                        "fold": (
                            result.get(
                                "fold"
                            )
                        ),
                        "seed": (
                            result[
                                "seed"
                            ]
                        ),
                        "learning_rate": (
                            result[
                                "learning_rate"
                            ]
                        ),
                        "effective_global_batch_size": (
                            result[
                                "effective_global_batch_size"
                            ]
                        ),
                        "per_device_train_batch_size": (
                            result[
                                "per_device_train_batch_size"
                            ]
                        ),
                        "gradient_accumulation_steps": (
                            result[
                                "gradient_accumulation_steps"
                            ]
                        ),
                        "warmup_ratio": (
                            result[
                                "warmup_ratio"
                            ]
                        ),
                        **history,
                    }
                )

        if not split_results:
            continue

        first = next(
            iter(
                split_results.values()
            )
        )

        trial_row: dict[
            str,
            Any,
        ] = {
            "experiment": (
                experiment.name
            ),
            "trial_number": (
                trial_number
            ),
            "learning_rate": (
                first[
                    "learning_rate"
                ]
            ),
            "effective_global_batch_size": (
                first[
                    "effective_global_batch_size"
                ]
            ),
            "per_device_train_batch_size": (
                first[
                    "per_device_train_batch_size"
                ]
            ),
            "gradient_accumulation_steps": (
                first[
                    "gradient_accumulation_steps"
                ]
            ),
            "warmup_ratio": (
                first[
                    "warmup_ratio"
                ]
            ),
            "completed_splits": len(
                split_results
            ),
        }

        expected_split_count = (
            len(
                resolve_splits(
                    experiment
                )
            )
        )

        for (
            split_name,
            result,
        ) in split_results.items():
            for history in result[
                "validation_history"
            ]:
                epoch = int(
                    round(
                        float(
                            history[
                                "epoch"
                            ]
                        )
                    )
                )

                trial_row[
                    f"{split_name}_wer_epoch_{epoch}"
                ] = history[
                    "normalised_wer"
                ]

        if (
            len(
                split_results
            )
            == expected_split_count
        ):
            if (
                experiment.data.mode
                == "cv"
            ):
                selection = (
                    select_cv_common_epoch(
                        split_results,
                        experiment.training.max_epochs,
                    )
                )

                trial_row[
                    "selected_epoch"
                ] = selection[
                    "selected_epoch"
                ]

                trial_row[
                    "objective_normalised_wer"
                ] = selection[
                    "objective_normalised_wer"
                ]

                for curve_row in selection[
                    "cv_curve"
                ]:
                    epoch = int(
                        curve_row[
                            "epoch"
                        ]
                    )

                    trial_row[
                        f"mean_cv_wer_epoch_{epoch}"
                    ] = curve_row[
                        "mean_normalised_wer"
                    ]

            else:
                only = next(
                    iter(
                        split_results.values()
                    )
                )

                selection = (
                    select_single_split_epoch(
                        only[
                            "validation_history"
                        ]
                    )
                )

                trial_row[
                    "selected_epoch"
                ] = selection[
                    "selected_epoch"
                ]

                trial_row[
                    "objective_normalised_wer"
                ] = selection[
                    "objective_normalised_wer"
                ]

        trial_rows.append(
            trial_row
        )

    write_csv(
        output_root
        / "validation_history.csv",
        long_rows,
    )

    write_csv(
        output_root
        / "trials.csv",
        trial_rows,
    )


def _log_trial_summary_wandb(
    *,
    experiment: ExperimentSpec,
    args: argparse.Namespace,
    session_id: str,
    trial_number: int,
    params: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    if (
        args.wandb_mode
        == "disabled"
    ):
        return

    try:
        import wandb

        run = wandb.init(
            project=(
                args.wandb_project
            ),
            entity=(
                args.wandb_entity
            ),
            name=(
                f"{experiment.name}_"
                f"trial_{trial_number:04d}_"
                "summary"
            ),
            group=(
                f"{experiment.name}_"
                f"{session_id}_"
                f"trial_{trial_number:04d}"
            ),
            job_type="hpo_summary",
            mode=(
                args.wandb_mode
            ),
            config={
                "experiment": (
                    experiment.name
                ),
                **params,
                "warmup_ratio": (
                    params.get(
                        "fcca_warmup_ratio",
                        experiment.training.warmup_ratio,
                    )
                ),
            },
            reinit=True,
        )

        if "cv_curve" in selection:
            for row in selection[
                "cv_curve"
            ]:
                wandb.log(
                    {
                        "epoch": (
                            row[
                                "epoch"
                            ]
                        ),
                        "cv/mean_normalised_wer": (
                            row[
                                "mean_normalised_wer"
                            ]
                        ),
                        "cv/std_normalised_wer": (
                            row[
                                "std_normalised_wer_population"
                            ]
                        ),
                    },
                    step=int(
                        row[
                            "epoch"
                        ]
                    ),
                )

        else:
            for row in selection[
                "validation_curve"
            ]:
                wandb.log(
                    {
                        "epoch": (
                            row[
                                "epoch"
                            ]
                        ),
                        "validation/normalised_wer": (
                            row[
                                "normalised_wer"
                            ]
                        ),
                        "validation/loss": (
                            row[
                                "eval_loss"
                            ]
                        ),
                    },
                    step=int(
                        round(
                            float(
                                row[
                                    "epoch"
                                ]
                            )
                        )
                    ),
                )

        run.summary[
            "selected_epoch"
        ] = selection[
            "selected_epoch"
        ]

        run.summary[
            "objective_normalised_wer"
        ] = selection[
            "objective_normalised_wer"
        ]

        run.finish()

    except Exception as exc:
        print(
            "W&B summary logging failed "
            f"without affecting HPO: {exc}"
        )


def _sample_common_hparams(
    *,
    trial: optuna.Trial,
    experiment: ExperimentSpec,
) -> dict[str, Any]:
    search = (
        experiment.search
    )

    return {
        "learning_rate": (
            trial.suggest_float(
                "learning_rate",
                search.learning_rate_low,
                search.learning_rate_high,
                log=True,
            )
        ),
        "effective_global_batch_size": (
            trial.suggest_categorical(
                "effective_global_batch_size",
                list(
                    search.effective_global_batch_sizes
                ),
            )
        ),
    }


def _launch_split_worker(
    *,
    experiment: ExperimentSpec,
    plugin: ArchitecturePlugin,
    launcher_path: Path,
    args: argparse.Namespace,
    session_id: str,
    trial_number: int,
    split: ResolvedSplit,
    common_hparams: dict[str, Any],
    architecture_hparams: dict[str, Any],
) -> dict[str, Any]:
    output_root = Path(
        experiment.output_root
    ).resolve()

    result_path = (
        _split_result_path(
            output_root,
            trial_number,
            split.split_name,
        )
    )

    # Same trial: completed fold.
    if result_path.exists():
        print(
            "[HPO] fold already complete; skipping "
            f"{split.split_name}",
            flush=True,
        )

        return json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

    current_trial_dir = (
        _trial_dir(
            output_root,
            trial_number,
        )
    )

    current_split_dir = (
        _split_dir(
            output_root,
            trial_number,
            split.split_name,
        )
    )

    current_checkpoint_dir = (
        current_split_dir
        / "_work"
        / "checkpoints"
    )

    # Optuna cannot continue a RUNNING trial after
    # the controlling Python process has died.
    # A replacement trial therefore recovers matching
    # completed folds/checkpoints from the old trial.
    if (
        _latest_checkpoint(
            current_checkpoint_dir
        )
        is None
    ):
        trials_root = (
            output_root
            / "trials"
        )

        if trials_root.exists():
            previous_trials = sorted(
                trials_root.glob(
                    "trial_*"
                ),
                reverse=True,
            )

            for previous_trial in previous_trials:
                if previous_trial == current_trial_dir:
                    continue

                params_path = (
                    previous_trial
                    / "trial_params.json"
                )

                if not params_path.exists():
                    continue

                try:
                    previous_params = json.loads(
                        params_path.read_text(
                            encoding="utf-8"
                        )
                    )
                except (
                    OSError,
                    json.JSONDecodeError,
                ):
                    continue

                if (
                    previous_params.get(
                        "common_hparams"
                    )
                    != common_hparams
                    or previous_params.get(
                        "architecture_hparams"
                    )
                    != architecture_hparams
                ):
                    continue

                previous_split_dir = (
                    previous_trial
                    / split.split_name
                )

                previous_result = (
                    previous_split_dir
                    / "result.json"
                )

                if previous_result.exists():
                    payload = json.loads(
                        previous_result.read_text(
                            encoding="utf-8"
                        )
                    )

                    payload[
                        "trial_number"
                    ] = trial_number

                    save_json(
                        result_path,
                        payload,
                    )

                    print(
                        "[HPO] recovered completed "
                        f"{split.split_name} from "
                        f"{previous_trial.name}",
                        flush=True,
                    )

                    _rebuild_csv_exports(
                        experiment=experiment
                    )

                    return payload

                previous_work = (
                    previous_split_dir
                    / "_work"
                )

                previous_checkpoint_dir = (
                    previous_work
                    / "checkpoints"
                )

                if (
                    _latest_checkpoint(
                        previous_checkpoint_dir
                    )
                    is not None
                ):
                    current_work = (
                        current_split_dir
                        / "_work"
                    )

                    if current_work.exists():
                        shutil.rmtree(
                            current_work
                        )

                    shutil.copytree(
                        previous_work,
                        current_work,
                    )

                    print(
                        "[HPO] recovered partial "
                        f"{split.split_name} from "
                        f"{previous_trial.name}",
                        flush=True,
                    )

                    break

    wait_until_no_foreign_gpu_processes(
        enabled=(
            args.wait_for_free_gpus
        ),
        interval_seconds=(
            args.gpu_check_interval
        ),
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        (
            f"--nproc_per_node="
            f"{experiment.training.world_size}"
        ),
        str(
            launcher_path
        ),
        "--worker",
        "--trial-number",
        str(
            trial_number
        ),
        "--split-name",
        split.split_name,
        "--learning-rate",
        repr(
            float(
                common_hparams[
                    "learning_rate"
                ]
            )
        ),
        "--effective-global-batch-size",
        str(
            int(
                common_hparams[
                    "effective_global_batch_size"
                ]
            )
        ),
        "--architecture-hparams-json",
        json.dumps(
            architecture_hparams,
            separators=(
                ",",
                ":",
            ),
        ),
        "--session-id",
        session_id,
        "--wandb-project",
        args.wandb_project,
        "--wandb-mode",
        args.wandb_mode,
    ]

    if args.wandb_entity:
        command.extend(
            [
                "--wandb-entity",
                args.wandb_entity,
            ]
        )

    env = os.environ.copy()

    env[
        "HF_HUB_OFFLINE"
    ] = "1"

    env[
        "TRANSFORMERS_OFFLINE"
    ] = "1"

    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = (
        "expandable_segments:True"
    )

    env[
        "WANDB_MODE"
    ] = args.wandb_mode

    env[
        "WANDB_PROJECT"
    ] = args.wandb_project

    env[
        "WANDB_RUN_GROUP"
    ] = (
        f"{experiment.name}_"
        f"{session_id}_"
        f"trial_{trial_number:04d}"
    )

    env[
        "WANDB_NAME"
    ] = (
        f"{experiment.name}_"
        f"trial_{trial_number:04d}_"
        f"b"
        f"{common_hparams['effective_global_batch_size']}_"
        f"{split.split_name}"
    )

    env[
        "WANDB_LOG_MODEL"
    ] = "false"

    env[
        "WANDB_WATCH"
    ] = "false"

    if args.wandb_entity:
        env[
            "WANDB_ENTITY"
        ] = (
            args.wandb_entity
        )

    subprocess.run(
        command,
        cwd=(
            experiment.project_root
        ),
        env=env,
        check=True,
    )

    if not result_path.exists():
        raise RuntimeError(
            "Worker finished without writing "
            f"{result_path}."
        )

    _rebuild_csv_exports(
        experiment=experiment
    )

    return json.loads(
        result_path.read_text(
            encoding="utf-8"
        )
    )


def _write_best_recipe(
    *,
    experiment: ExperimentSpec,
    study: optuna.Study,
) -> None:
    completed = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
            and trial.value
            is not None
        )
    ]

    if not completed:
        return

    best = (
        study.best_trial
    )

    save_json(
        Path(
            experiment.output_root
        )
        / "best_recipe.json",
        {
            "experiment": (
                experiment.name
            ),
            "best_trial": (
                best.number
            ),
            "best_objective_normalised_wer": (
                best.value
            ),
            "best_params": dict(
                best.params
            ),
            "selected_epoch": (
                best.user_attrs.get(
                    "selected_epoch"
                )
            ),
            "selection": (
                best.user_attrs.get(
                    "selection"
                )
            ),
            "warmup_ratio": (
                best.params.get(
                    "fcca_warmup_ratio",
                    experiment.training.warmup_ratio,
                )
            ),
            "hpo_seed": (
                experiment.training
                .hpo_seed
            ),
            "fixed_test_used": False,
        },
    )


def run_hpo_experiment(
    *,
    experiment: ExperimentSpec,
    plugin: ArchitecturePlugin,
    launcher_path: Path,
) -> None:
    parser = (
        _build_parser(
            experiment
        )
    )

    args = (
        parser.parse_args()
    )

    _validate_output_root_safety(
        experiment=experiment,
        launcher_path=launcher_path,
    )

    # Worker mode must not create/reset/overwrite the parent session.
    if args.worker:
        splits = resolve_splits(
            experiment
        )

        _validate_experiment(
            experiment,
            splits,
        )

        try:
            _run_worker(
                experiment=experiment,
                plugin=plugin,
                args=args,
            )

        finally:
            if (
                dist.is_available()
                and dist.is_initialized()
            ):
                dist.destroy_process_group()

        return

    output_root, session_id = (
        _prepare_session(
            experiment=experiment,
            args=args,
        )
    )

    splits = resolve_splits(
        experiment
    )

    _validate_experiment(
        experiment,
        splits,
    )

    if (
        args.wandb_mode
        != "disabled"
    ):
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "W&B is enabled but wandb is not installed. "
                "Run: pip install wandb"
            ) from exc

    # Persist the full experimental protocol before training starts.
    save_json(
        output_root
        / "search_protocol.json",
        {
            "experiment": asdict(
                experiment
            ),
            "plugin": {
                "method_name": (
                    plugin.method_name
                ),
            },
            "resolved_splits": [
                {
                    "split_name": (
                        split.split_name
                    ),
                    "fold": split.fold,
                    "train_manifest": str(
                        split.train_manifest
                    ),
                    "validation_manifest": str(
                        split.validation_manifest
                    ),
                }
                for split in splits
            ],
            "fixed_test_used": False,
        },
    )

    # Prime the model cache before offline workers are launched.
    processor = (
        WhisperProcessor
        .from_pretrained(
            experiment.base_model,
            language=experiment.language,
            task=experiment.whisper_task,
        )
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            experiment.base_model
        )
    )

    del processor
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    study_name = (
        experiment.study_name
        or f"{experiment.name}_hpo"
    )

    sampler = (
        optuna.samplers
        .TPESampler(
            seed=(
                experiment.training
                .optuna_sampler_seed
            ),
            n_startup_trials=0,
        )
    )

    study = (
        optuna.create_study(
            study_name=(
                study_name
            ),
            storage=(
                "sqlite:///"
                f"{output_root / 'optuna_study.sqlite3'}"
            ),
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
    )

    # Enqueue only startup recipes that are not already present.
    architecture_startup = (
        plugin.startup_hparams(
            experiment
        )
    )

    common_startup = (
        experiment.search
        .startup_dicts()
    )

    desired_startups: list[
        dict[str, Any]
    ] = []

    if architecture_startup:
        for common in common_startup:
            for architecture in (
                architecture_startup
            ):
                desired_startups.append(
                    {
                        **common,
                        **architecture,
                    }
                )

    else:
        desired_startups = list(
            common_startup
        )

    def trial_matches_params(
        trial: optuna.trial.FrozenTrial,
        params: dict[str, Any],
    ) -> bool:
        for key, value in (
            params.items()
        ):
            if key not in trial.params:
                return False

            if trial.params[
                key
            ] != value:
                return False

        return True

    for startup in (
        desired_startups
    ):
        already_present = any(
            trial_matches_params(
                trial,
                startup,
            )
            for trial in study.trials
            if trial.state
            in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.RUNNING,
                optuna.trial.TrialState.WAITING,
            )
        )

        if not already_present:
            study.enqueue_trial(
                startup
            )

    completed_before = sum(
        trial.state
        == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )

    remaining = max(
        0,
        int(
            args.n_trials
        )
        - completed_before,
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        common_hparams = (
            _sample_common_hparams(
                trial=trial,
                experiment=experiment,
            )
        )

        architecture_hparams = (
            plugin.sample_hparams(
                trial,
                experiment,
            )
        )

        save_json(
            _trial_dir(
                output_root,
                trial.number,
            )
            / "trial_params.json",
            {
                "common_hparams":
                    common_hparams,
                "architecture_hparams":
                    architecture_hparams,
            },
        )

        split_results: dict[
            str,
            dict[str, Any],
        ] = {}

        for split in splits:
            split_results[
                split.split_name
            ] = (
                _launch_split_worker(
                    experiment=experiment,
                    plugin=plugin,
                    launcher_path=(
                        launcher_path
                    ),
                    args=args,
                    session_id=(
                        session_id
                    ),
                    trial_number=(
                        trial.number
                    ),
                    split=split,
                    common_hparams=(
                        common_hparams
                    ),
                    architecture_hparams=(
                        architecture_hparams
                    ),
                )
            )

        if (
            experiment.data.mode
            == "cv"
        ):
            selection = (
                select_cv_common_epoch(
                    split_results,
                    experiment.training
                    .max_epochs,
                )
            )

        else:
            only_result = next(
                iter(
                    split_results.values()
                )
            )

            selection = (
                select_single_split_epoch(
                    only_result[
                        "validation_history"
                    ]
                )
            )

        trial.set_user_attr(
            "selected_epoch",
            selection[
                "selected_epoch"
            ],
        )

        trial.set_user_attr(
            "selection",
            selection,
        )

        save_json(
            _trial_dir(
                output_root,
                trial.number,
            )
            / "summary.json",
            {
                "trial_number": (
                    trial.number
                ),
                "common_hparams": (
                    common_hparams
                ),
                "architecture_hparams": (
                    architecture_hparams
                ),
                "selection": (
                    selection
                ),
            },
        )

        _log_trial_summary_wandb(
            experiment=experiment,
            args=args,
            session_id=session_id,
            trial_number=trial.number,
            params={
                **common_hparams,
                **architecture_hparams,
            },
            selection=selection,
        )

        _rebuild_csv_exports(
            experiment=experiment
        )

        return float(
            selection[
                "objective_normalised_wer"
            ]
        )

    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,

        )

    _rebuild_csv_exports(
        experiment=experiment
    )

    _write_best_recipe(
        experiment=experiment,
        study=study,
    )
