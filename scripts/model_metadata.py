from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.train import (
    make_json_safe,
    parameter_summary,
    save_json,
)


# Metadata loading

def load_metadata(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "Metadata file does not exist:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        metadata = json.load(
            input_file
        )

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "metadata.json must contain "
            "a JSON object."
        )

    return metadata


# Classification

def build_classification(
    *,
    family: str,
    transfer_type: str,
    domain_category: str | None,
) -> dict[str, Any]:
    valid_transfer_types = {
        "plain",
        "warm",
        "cold",
    }

    if transfer_type not in valid_transfer_types:
        raise ValueError(
            "transfer_type must be one of "
            f"{sorted(valid_transfer_types)}, "
            f"got {transfer_type!r}."
        )

    return {
        "family": family,
        "transfer_type": (
            transfer_type
        ),
        "domain_category": (
            domain_category
        ),
    }


# Static models

def build_static_model_metadata(
    *,
    model_name: str,
    model_type: str,
    base_model: str | None,
    classification: Mapping[
        str,
        Any,
    ],
    description: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "base_model": base_model,
        "classification": (
            make_json_safe(
                classification
            )
        ),
        "training": None,
    }

    if description is not None:
        metadata[
            "description"
        ] = description

    return metadata


def ensure_static_model_metadata(
    *,
    path: Path,
    metadata: Mapping[
        str,
        Any,
    ],
) -> dict[str, Any]:
    if path.exists():
        existing = load_metadata(
            path
        )

        expected_name = metadata.get(
            "model_name"
        )

        existing_name = existing.get(
            "model_name"
        )

        if (
            expected_name is not None
            and existing_name
            != expected_name
        ):
            raise RuntimeError(
                "Existing metadata belongs to "
                "a different model.\n"
                f"Expected: {expected_name}\n"
                f"Found:    {existing_name}"
            )

        return existing

    save_json(
        path,
        dict(
            metadata
        ),
    )

    return dict(
        metadata
    )


# Training metadata

def build_training_metadata(
    *,
    model_name: str,
    model_type: str,
    base_model: str | None,
    model: torch.nn.Module,
    classification: Mapping[
        str,
        Any,
    ],
    training: Mapping[
        str,
        Any,
    ],
    hyperparameter_selection: Mapping[
        str,
        Any,
    ] | None = None,
    architecture: Mapping[
        str,
        Any,
    ] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    parameters = parameter_summary(
        model
    )

    parameter_metadata = {
        **parameters,
        "count_basis": (
            "requires_grad_at_training_start"
        ),
    }

    metadata: dict[
        str,
        Any,
    ] = {
        "model_name": model_name,
        "model_type": model_type,
        "base_model": base_model,
        "classification": (
            make_json_safe(
                classification
            )
        ),
        "training": (
            make_json_safe(
                training
            )
        ),
        "parameters": (
            parameter_metadata
        ),
        "runtime": {
            "world_size": (
                world_size
            ),
            "visible_cuda_devices": (
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0
            ),
        },
        "status": (
            "training_started"
        ),
    }

    if (
        hyperparameter_selection
        is not None
    ):
        metadata[
            "hyperparameter_selection"
        ] = make_json_safe(
            hyperparameter_selection
        )

    if architecture is not None:
        metadata[
            "architecture"
        ] = make_json_safe(
            architecture
        )

    if description is not None:
        metadata[
            "description"
        ] = description

    return metadata


def save_training_metadata(
    *,
    path: Path,
    metadata: Mapping[
        str,
        Any,
    ],
) -> None:
    save_json(
        path,
        dict(
            metadata
        ),
    )


# Completion

def mark_training_complete(
    *,
    path: Path,
    global_step: int | None,
    epoch: float | None,
    train_metrics: Mapping[
        str,
        Any,
    ] | None,
    model_directory: Path,
) -> dict[str, Any]:
    metadata = load_metadata(
        path
    )

    metadata[
        "status"
    ] = "complete"

    metadata[
        "training_result"
    ] = {
        "global_step": (
            global_step
        ),
        "epoch": epoch,
        "train_metrics": (
            make_json_safe(
                train_metrics
                or {}
            )
        ),
    }

    metadata[
        "model_directory"
    ] = str(
        model_directory
    )

    save_json(
        path,
        metadata,
    )

    return metadata