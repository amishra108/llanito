from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.adapt_transfer import config

from scripts.adapt_transfer.train import (
    build_transfer_model,
    save_final_model,
)


# Supported recovery models

WARM_RESADAPT_MODELS = (
    "resadapt_enc_ltm_warm",
    "resadapt_encdec_ltm_warm",
    "resadapt_enc_mtl_warm",
    "resadapt_encdec_mtl_warm",
)


# Placement helpers

def placement_to_adapter_module(
    placement: str,
) -> str:
    if placement == "enc":
        return "encoder"

    if placement == "dec":
        return "decoder"

    if placement == "encdec":
        return "encoder_decoder"

    raise ValueError(
        "Unsupported ResAdapt placement: "
        f"{placement!r}"
    )


# Checkpoints

def checkpoint_number(
    path: Path,
) -> int:
    try:
        return int(
            path.name.split(
                "checkpoint-",
                1,
            )[1]
        )

    except (
        IndexError,
        ValueError,
    ):
        return -1


def find_latest_checkpoint(
    model_name: str,
) -> Path:
    checkpoint_root = (
        PROJECT_ROOT
        / "models"
        / model_name
        / "checkpoints"
    )

    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            "Checkpoint directory does not exist:\n"
            f"{checkpoint_root}"
        )

    checkpoints = [
        path
        for path in checkpoint_root.glob(
            "checkpoint-*"
        )
        if (
            path.is_dir()
            and checkpoint_number(
                path
            )
            >= 0
        )
    ]

    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints were found under:\n"
            f"{checkpoint_root}"
        )

    return max(
        checkpoints,
        key=checkpoint_number,
    )


# Checkpoint state

def load_checkpoint_state(
    checkpoint: Path,
) -> dict[
    str,
    torch.Tensor,
]:
    safetensors_path = (
        checkpoint
        / "model.safetensors"
    )

    pytorch_path = (
        checkpoint
        / "pytorch_model.bin"
    )

    if safetensors_path.is_file():
        print(
            "Loading checkpoint weights:"
        )

        print(
            f"  {safetensors_path}"
        )

        return load_file(
            str(
                safetensors_path
            ),
            device="cpu",
        )

    if pytorch_path.is_file():
        print(
            "Loading checkpoint weights:"
        )

        print(
            f"  {pytorch_path}"
        )

        state = torch.load(
            pytorch_path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(
            state,
            dict,
        ):
            raise RuntimeError(
                "pytorch_model.bin did not "
                "contain a state dictionary."
            )

        return state

    raise FileNotFoundError(
        "Could not find model weights in "
        "checkpoint:\n"
        f"{checkpoint}\n\n"
        "Expected one of:\n"
        f"  {safetensors_path}\n"
        f"  {pytorch_path}"
    )


# Trainer-state validation

def validate_final_checkpoint(
    checkpoint: Path,
) -> None:
    trainer_state_path = (
        checkpoint
        / "trainer_state.json"
    )

    if not trainer_state_path.is_file():
        print(
            "WARNING: trainer_state.json "
            "not found. Final-step validation "
            "is being skipped."
        )

        return

    with trainer_state_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        trainer_state = json.load(
            input_file
        )

    global_step = trainer_state.get(
        "global_step"
    )

    max_steps = trainer_state.get(
        "max_steps"
    )

    epoch = trainer_state.get(
        "epoch"
    )

    print()
    print(
        "Checkpoint training state:"
    )

    print(
        f"  global_step: {global_step}"
    )

    print(
        f"  max_steps: {max_steps}"
    )

    print(
        f"  epoch: {epoch}"
    )

    if (
        global_step is not None
        and max_steps is not None
        and int(global_step)
        != int(max_steps)
    ):
        raise RuntimeError(
            "Latest checkpoint is not the "
            "final training step.\n"
            f"global_step={global_step}\n"
            f"max_steps={max_steps}"
        )


# Architecture repair

def repair_architecture(
    *,
    experiment: config.ExperimentConfig,
    architecture: dict,
) -> dict:
    repaired = dict(
        architecture
    )

    adapter_module = (
        placement_to_adapter_module(
            experiment.placement
        )
    )

    repaired[
        "adapter_module"
    ] = adapter_module

    repaired[
        "placement"
    ] = experiment.placement

    repaired.setdefault(
        "adapter_layers",
        list(
            range(24)
        ),
    )

    repaired.setdefault(
        "position",
        "all",
    )

    repaired.setdefault(
        "stacked",
        False,
    )

    repaired.setdefault(
        "warm_continuation",
        True,
    )

    repaired.setdefault(
        "source_model",
        experiment.source_model_name,
    )

    repaired.setdefault(
        "source_model_root",
        str(
            experiment.source_model_root
        ),
    )

    if (
        "bottleneck"
        not in repaired
    ):
        raise RuntimeError(
            "Recovered architecture is "
            "missing bottleneck."
        )

    if (
        "dropout"
        not in repaired
    ):
        raise RuntimeError(
            "Recovered architecture is "
            "missing dropout."
        )

    return repaired


# Checkpoint loading

def load_checkpoint_into_model(
    *,
    model: torch.nn.Module,
    state: dict[
        str,
        torch.Tensor,
    ],
) -> None:
    result = model.load_state_dict(
        state,
        strict=False,
    )

    allowed_missing = {
        "proj_out.weight",
    }

    unexpected_missing = (
        set(
            result.missing_keys
        )
        - allowed_missing
    )

    if unexpected_missing:
        raise RuntimeError(
            "Checkpoint is missing unexpected "
            "model parameters.\n"
            f"Missing: "
            f"{sorted(unexpected_missing)}"
        )

    if result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint contains unexpected "
            "model parameters.\n"
            f"Unexpected: "
            f"{result.unexpected_keys}"
        )

    model.tie_weights()

    print(
        "Checkpoint loaded successfully."
    )

    if result.missing_keys:
        print(
            "Allowed tied-weight omissions:"
        )

        for key in (
            result.missing_keys
        ):
            print(
                f"  {key}"
            )


# Recovery

def recover_model(
    model_name: str,
) -> None:
    if (
        model_name
        not in WARM_RESADAPT_MODELS
    ):
        raise ValueError(
            "This recovery utility is only "
            "for the four warm ResAdapt "
            "transfer models.\n"
            f"Got: {model_name}"
        )

    experiment = (
        config.EXPERIMENTS[
            model_name
        ]
    )

    if (
        experiment.mechanism
        != "resadapt"
    ):
        raise RuntimeError(
            "Experiment is not ResAdapt."
        )

    if (
        experiment.transfer_type
        != "warm"
    ):
        raise RuntimeError(
            "Experiment is not a warm "
            "transfer."
        )

    checkpoint = (
        find_latest_checkpoint(
            model_name
        )
    )

    print()
    print("=" * 80)
    print(
        f"RECOVERING: {model_name}"
    )
    print("=" * 80)

    print()
    print(
        f"Source model: "
        f"{experiment.source_model_name}"
    )

    print(
        f"Latest checkpoint: "
        f"{checkpoint}"
    )

    validate_final_checkpoint(
        checkpoint
    )

    print()
    print(
        "Rebuilding exact warm-transfer "
        "architecture..."
    )

    (
        model,
        processor,
        architecture,
    ) = build_transfer_model(
        experiment
    )

    architecture = (
        repair_architecture(
            experiment=experiment,
            architecture=architecture,
        )
    )

    print()
    print(
        "Recovered architecture:"
    )

    print(
        f"  placement: "
        f"{architecture['placement']}"
    )

    print(
        f"  adapter_module: "
        f"{architecture['adapter_module']}"
    )

    print(
        f"  bottleneck: "
        f"{architecture['bottleneck']}"
    )

    print(
        f"  dropout: "
        f"{architecture['dropout']}"
    )

    print(
        f"  layers: "
        f"{architecture['adapter_layers']}"
    )

    state = load_checkpoint_state(
        checkpoint
    )

    print()
    print(
        "Loading final checkpoint state "
        "into reconstructed model..."
    )

    load_checkpoint_into_model(
        model=model,
        state=state,
    )

    print()
    print(
        "Exporting standard final "
        "ResAdapt payload..."
    )

    save_final_model(
        model=model,
        processor=processor,
        experiment=experiment,
        architecture=architecture,
    )

    model_directory = (
        experiment.model_directory
    )

    adapter_state_path = (
        model_directory
        / "adapter_state.pt"
    )

    adapter_config_path = (
        model_directory
        / "adapter_config.json"
    )

    if not adapter_state_path.is_file():
        raise RuntimeError(
            "Recovery did not create:\n"
            f"{adapter_state_path}"
        )

    if not adapter_config_path.is_file():
        raise RuntimeError(
            "Recovery did not create:\n"
            f"{adapter_config_path}"
        )

    print()
    print("=" * 80)
    print(
        f"RECOVERY COMPLETE: "
        f"{model_name}"
    )
    print("=" * 80)

    print()
    print(
        "Recovered final payload:"
    )

    print(
        f"  {adapter_state_path}"
    )

    print(
        f"  {adapter_config_path}"
    )

    print()
    print(
        "No optimizer step or additional "
        "training was performed."
    )


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=(
            WARM_RESADAPT_MODELS
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    recover_model(
        args.model
    )


if __name__ == "__main__":
    main()