from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import WhisperProcessor

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from scripts_five_fold.stackingtransfers.config import (
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    build_runtime_config,
    source_adapter_for_fold,
)

from scripts_five_fold.stackingtransfers.train import (
    build_transfer_model,
    new_miami_adapter_modules,
    source_adapter_modules,
)

from scripts_five_fold.evaluate import (
    evaluate_loaded_model,
    load_manifest,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        choices=sorted(EXPERIMENTS),
    )

    return parser.parse_args()


def load_saved_transfer_state(
    adapter_directory: Path,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor],
]:
    config_path = (
        adapter_directory
        / "adapter_config.json"
    )

    state_path = (
        adapter_directory
        / "adapter_state.pt"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Saved transfer config missing:\n"
            f"{config_path}"
        )

    if not state_path.exists():
        raise FileNotFoundError(
            f"Saved transfer state missing:\n"
            f"{state_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    try:
        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            state_path,
            map_location="cpu",
        )

    if not isinstance(state, dict):
        raise TypeError(
            "Saved transfer state must "
            "be a dictionary."
        )

    return (
        metadata,
        state,
    )


def load_transfer_into_model(
    model,
    adapter_directory: Path,
    config,
    fold_number: int,
) -> None:
    metadata, state = (
        load_saved_transfer_state(
            adapter_directory
        )
    )

    checks = {
        "experiment": (
            config.EXPERIMENT_NAME
        ),
        "fold": fold_number,
        "mode": config.MODE,
        "direction": config.DIRECTION,
    }

    for key, expected in checks.items():
        actual = metadata.get(key)

        if actual != expected:
            raise ValueError(
                "Saved transfer metadata mismatch. "
                f"{key}: saved={actual!r}, "
                f"expected={expected!r}"
            )

    modules = (
        source_adapter_modules(
            model,
            config,
        )
        if config.MODE == "soft"
        else new_miami_adapter_modules(
            model,
            config,
        )
    )

    for layer_index, adapter in modules:
        prefix = (
            f"encoder.layer_{layer_index}."
        )

        layer_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }

        if not layer_state:
            raise RuntimeError(
                "No saved transfer weights "
                f"for encoder layer {layer_index}."
            )

        adapter.load_state_dict(
            layer_state,
            strict=True,
        )


def flatten_summary(
    summary: dict[str, Any],
    *,
    fold_number: int,
) -> dict[str, Any]:
    row = {
        "model_name": (
            summary["model_name"]
        ),
        "fold": fold_number,
        "evaluation_set": (
            "llanito_fixed_test"
        ),
        "recordings": (
            summary["recordings"]
        ),
    }

    for key, value in (
        summary["overall"].items()
    ):
        row[
            f"overall_{key}"
        ] = value

    for (
        task_name,
        task_summary,
    ) in (
        summary["by_task"].items()
    ):
        for key, value in (
            task_summary.items()
        ):
            row[
                f"{task_name}_{key}"
            ] = value

    return row


def numeric_value(
    row: dict[str, Any],
    key: str,
) -> float | None:
    value = row.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def aggregate_fold_results(
    experiment_name: str,
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = {
        "model_name": (
            experiment_name
            + "_fold_mean"
        ),
        "fold": "mean",
        "evaluation_set": (
            "llanito_fixed_test"
        ),
        "recordings": (
            fold_rows[0].get(
                "recordings"
            )
            if fold_rows
            else None
        ),
    }

    if not fold_rows:
        return aggregate

    keys = sorted({
        key
        for row in fold_rows
        for key in row
        if key.startswith(
            (
                "overall_",
                "sentence_",
                "passage_",
            )
        )
    })

    for key in keys:
        values = [
            value
            for row in fold_rows
            if (
                value := numeric_value(
                    row,
                    key,
                )
            )
            is not None
        ]

        if values:
            aggregate[key] = (
                sum(values)
                / len(values)
            )

    return aggregate


def evaluate_fold(
    config,
    fold_number: int,
    rows,
):
    source_directory = (
        source_adapter_for_fold(
            config,
            fold_number,
        )
    )

    adapter_directory = (
        config.MODEL_ROOT
        / f"fold_{fold_number}"
        / "adapter"
    )

    if not adapter_directory.exists():
        raise FileNotFoundError(
            f"Trained transfer adapter missing:\n"
            f"{adapter_directory}"
        )

    processor = (
        WhisperProcessor
        .from_pretrained(
            config.BASE_MODEL,
            language=config.LANGUAGE,
            task=config.TASK,
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=config.LANGUAGE,
        task=config.TASK,
        predict_timestamps=False,
    )

    model = build_transfer_model(
        config,
        source_directory,
    )

    load_transfer_into_model(
        model,
        adapter_directory,
        config,
        fold_number,
    )

    dtype = (
        torch.float16
        if (
            torch.cuda.is_available()
            and config.USE_FP16
        )
        else torch.float32
    )

    model = model.to(
        dtype=dtype
    )

    model.eval()

    output_root = (
        config.OUTPUT_ROOT
        / f"fold_{fold_number}"
        / "fixed_test"
    )

    predictions, summary = (
        evaluate_loaded_model(
            model_name=(
                f"{config.EXPERIMENT_NAME}"
                f"_fold_{fold_number}"
            ),
            processor=processor,
            model=model,
            rows=rows,
            output_root=(
                output_root
            ),
            normalisation_file=(
                config.NORMALISATION_FILE
            ),
            language=config.LANGUAGE,
            task=config.TASK,
            use_fp16=config.USE_FP16,
            generation_max_length=(
                config
                .GENERATION_MAX_LENGTH
            ),
            generation_num_beams=(
                config
                .GENERATION_NUM_BEAMS
            ),
        )
    )

    for row in predictions:
        row["fold"] = fold_number
        row[
            "evaluation_set"
        ] = "llanito_fixed_test"

    write_csv(
        output_root
        / "all_model_predictions.csv",
        predictions,
    )

    result = flatten_summary(
        summary,
        fold_number=fold_number,
    )

    write_csv(
        output_root
        / "all_model_results.csv",
        [
            result
        ],
    )

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        result,
        predictions,
    )


def main() -> None:
    print(
        "STACKING TRANSFER EVALUATOR STARTED",
        flush=True,
    )

    args = parse_args()

    config = build_runtime_config(
        args.experiment
    )

    rows = load_manifest(
        config
        .LLANITO_FIXED_TEST_MANIFEST
    )

    if (
        config.MAX_TEST_RECORDINGS
        is not None
    ):
        rows = rows[
            :config.MAX_TEST_RECORDINGS
        ]

    all_results = []
    all_predictions = []

    for fold_number in (
        config.RUN_FOLDS
    ):
        print()
        print("=" * 80)
        print(
            f"EVALUATING "
            f"{config.EXPERIMENT_NAME} "
            f"FOLD {fold_number}"
        )
        print("=" * 80)

        result, predictions = (
            evaluate_fold(
                config,
                fold_number,
                rows,
            )
        )

        all_results.append(
            result
        )

        all_predictions.extend(
            predictions
        )

    if len(all_results) > 1:
        mean_row = aggregate_fold_results(
            config.EXPERIMENT_NAME,
            all_results,
        )
        results_with_mean = [*all_results, mean_row]
    else:
        results_with_mean = list(all_results)

    config.OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.SUMMARY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for root in (
        config.OUTPUT_ROOT,
        config.SUMMARY_ROOT,
    ):
        write_csv(
            root
            / "all_model_results.csv",
            results_with_mean,
        )

        write_csv(
            root
            / "all_model_predictions.csv",
            all_predictions,
        )

    print()
    print("=" * 80)
    print(
        f"{config.EXPERIMENT_NAME} "
        "EVALUATION COMPLETE"
    )
    print("=" * 80)
    print(
        "Summary results: "
        f"{config.SUMMARY_ROOT / 'all_model_results.csv'}"
    )
    print(
        "Summary predictions: "
        f"{config.SUMMARY_ROOT / 'all_model_predictions.csv'}"
    )


if __name__ == "__main__":
    main()
