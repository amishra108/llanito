from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from scripts_five_fold.mlpinteractionfusion.config import (
    BASE_MODEL,
    BASELINE_MODEL_NAME,
    EVALUATE_BASELINE,
    EXPERIMENT_NAME,
    RUN_EXPERIMENTS,
    resolve_experiment,
    FIXED_TEST_MANIFEST,
    FOLD_NUMBERS,
    GENERATION_MAX_LENGTH,
    GENERATION_NUM_BEAMS,
    LANGUAGE,
    MAX_TEST_RECORDINGS,
    MODEL_ROOT,
    NORMALISATION_FILE,
    OUTPUT_ROOT as EXPERIMENT_OUTPUT_ROOT,
    REBUILD_AGGREGATES_FROM_SAVED_FOLDS,
    RUN_FOLDS,
    SUMMARY_ROOT,
    TASK,
    USE_FP16,
)

from scripts_five_fold.mlpinteractionfusion.train import (
    build_mlp_interaction_model,
    fusion_directory,
    fusion_exists,
    llanito_adapter_directory,
    load_fusion_into_model,
    miami_adapter_directory,
    adapter_exists,
    apply_experiment_variant as apply_training_experiment_variant,
)

from scripts_five_fold.evaluate import (
    evaluate_loaded_model,
    load_manifest,
    save_json,
    summarise_model_results,
    write_csv,
)


OUTPUT_ROOT = (
    EXPERIMENT_OUTPUT_ROOT
    / "fixed_test"
)


# Experiment switching

def apply_experiment_variant(
    experiment_key: str,
) -> None:
    global BASE_MODEL
    global BASELINE_MODEL_NAME
    global EXPERIMENT_NAME
    global EXPERIMENT_OUTPUT_ROOT
    global MODEL_ROOT
    global OUTPUT_ROOT
    global SUMMARY_ROOT

    settings = resolve_experiment(
        experiment_key
    )

    BASE_MODEL = settings[
        "base_model"
    ]

    BASELINE_MODEL_NAME = settings[
        "baseline_model_name"
    ]

    EXPERIMENT_NAME = settings[
        "experiment_name"
    ]

    MODEL_ROOT = (
        PROJECT_ROOT
        / "outputs_five_fold" / "models"
        / EXPERIMENT_NAME
    )

    EXPERIMENT_OUTPUT_ROOT = (
        PROJECT_ROOT
        / "outputs_five_fold"
        / EXPERIMENT_NAME
    )

    OUTPUT_ROOT = (
        EXPERIMENT_OUTPUT_ROOT
        / "fixed_test"
    )

    SUMMARY_ROOT = (
        PROJECT_ROOT
        / "outputs_five_fold"
        / "summary"
        / EXPERIMENT_NAME
    )

    # build_mlp_interaction_model and all directory helpers live in train.py,
    # so its module globals must be switched to the same base experiment.
    apply_training_experiment_variant(
        experiment_key
    )



# Helpers

def requested_folds() -> list[int]:
    if RUN_FOLDS is None:
        return list(
            FOLD_NUMBERS
        )

    folds = sorted(
        set(
            RUN_FOLDS
        )
    )

    invalid = [
        fold
        for fold in folds
        if fold not in FOLD_NUMBERS
    ]

    if invalid:
        raise ValueError(
            f"Invalid RUN_FOLDS values: {invalid}"
        )

    return folds


def file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as input_file:
        for chunk in iter(
            lambda: input_file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def combined_sha256(
    paths: list[Path],
) -> str:
    digest = hashlib.sha256()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Fingerprint input is missing:\n{path}"
            )

        digest.update(
            str(path).encode(
                "utf-8"
            )
        )

        with path.open(
            "rb"
        ) as input_file:
            for chunk in iter(
                lambda: input_file.read(
                    1024 * 1024
                ),
                b"",
            ):
                digest.update(
                    chunk
                )

    return digest.hexdigest()


def fold_model_sha256(
    fold_number: int,
) -> str:
    return combined_sha256(
        [
            miami_adapter_directory()
            / "adapter_state.pt",
            llanito_adapter_directory(
                fold_number
            )
            / "adapter_state.pt",
            fusion_directory(
                fold_number
            )
            / "fusion_state.pt",
        ]
    )


def read_csv(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        return list(
            csv.DictReader(
                input_file
            )
        )


def load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        value = json.load(
            input_file
        )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object in {path}"
        )

    return value


def fold_cache_directory(
    fold_number: int,
) -> Path:
    return (
        OUTPUT_ROOT
        / f"fold_{fold_number}"
    )


def fold_predictions_path(
    fold_number: int,
) -> Path:
    return (
        fold_cache_directory(
            fold_number
        )
        / "cached_predictions.csv"
    )


def fold_summary_path(
    fold_number: int,
) -> Path:
    return (
        fold_cache_directory(
            fold_number
        )
        / "cached_summary.json"
    )


def vanilla_directory() -> Path:
    return (
        OUTPUT_ROOT
        / BASELINE_MODEL_NAME
    )


def vanilla_predictions_path() -> Path:
    return (
        vanilla_directory()
        / "cached_predictions.csv"
    )


def vanilla_summary_path() -> Path:
    return (
        vanilla_directory()
        / "cached_summary.json"
    )


def cache_is_current(
    summary_path: Path,
    manifest_hash: str,
    model_hash: str | None = None,
) -> bool:
    if not summary_path.exists():
        return False

    try:
        summary = load_json(
            summary_path
        )
    except Exception:
        return False

    if (
        summary.get(
            "_test_manifest_sha256"
        )
        != manifest_hash
    ):
        return False

    if model_hash is None:
        return True

    return (
        summary.get(
            "_model_sha256"
        )
        == model_hash
    )


def validate_models(
    folds: list[int],
) -> None:
    miami_directory = (
        miami_adapter_directory()
    )

    if not adapter_exists(
        miami_directory
    ):
        raise FileNotFoundError(
            "Miami adapter is missing:\n"
            f"{miami_directory}"
        )

    problems: list[str] = []

    for fold_number in folds:
        llanito_directory = (
            llanito_adapter_directory(
                fold_number
            )
        )

        fusion_dir = fusion_directory(
            fold_number
        )

        if not adapter_exists(
            llanito_directory
        ):
            problems.append(
                f"fold {fold_number}: missing Llanito adapter "
                f"{llanito_directory}"
            )

        if not fusion_exists(
            fusion_dir
        ):
            problems.append(
                f"fold {fold_number}: missing fusion "
                f"{fusion_dir}"
            )

    if problems:
        raise RuntimeError(
            "MLP interaction fusion preflight failed:\n\n"
            + "\n".join(
                problems
            )
        )


# Model loading

def load_mlp_fusion_model(
    fold_number: int,
) -> tuple[
    WhisperProcessor,
    WhisperForConditionalGeneration,
]:
    miami_directory = (
        miami_adapter_directory()
    )

    llanito_directory = (
        llanito_adapter_directory(
            fold_number
        )
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            BASE_MODEL,
            language=LANGUAGE,
            task=TASK,
        )
    )

    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
    )

    model = build_mlp_interaction_model(
        miami_directory,
        llanito_directory,
    )

    load_fusion_into_model(
        model,
        fusion_directory(
            fold_number
        ),
    )

    # Force a clean FP32 GPU evaluation first
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device=device,
        dtype=torch.float32,
    )

    # Inference settings
    model.config.use_cache = True

    if hasattr(
        model,
        "generation_config",
    ):
        model.generation_config.use_cache = True
        model.generation_config.language = LANGUAGE
        model.generation_config.task = TASK
        model.generation_config.forced_decoder_ids = None

        if hasattr(
            model.generation_config,
            "condition_on_prev_tokens",
        ):
            model.generation_config.condition_on_prev_tokens = False

        if hasattr(
            model.generation_config,
            "return_timestamps",
        ):
            model.generation_config.return_timestamps = False

    model.eval()

    print(
        f"MLP fusion model device: "
        f"{next(model.parameters()).device}"
    )
    print(
        f"MLP fusion model dtype: "
        f"{next(model.parameters()).dtype}"
    )
    print(
        f"MLP fusion model use_cache: "
        f"{model.config.use_cache}"
    )

    return (
        processor,
        model,
    )

def load_vanilla(
) -> tuple[
    WhisperProcessor,
    WhisperForConditionalGeneration,
]:
    dtype = (
        torch.float16
        if (
            torch.cuda.is_available()
            and USE_FP16
        )
        else torch.float32
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            BASE_MODEL,
            language=LANGUAGE,
            task=TASK,
        )
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
        )
    )

    model.eval()

    return (
        processor,
        model,
    )


# Formatting

def flatten_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_name": (
            summary["model_name"]
        ),
        "recordings": (
            summary["recordings"]
        ),
    }

    if "fold" in summary:
        row["fold"] = (
            summary["fold"]
        )

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


# Evaluation

def evaluate_base_model(
    test_rows: list[dict[str, Any]],
    manifest_hash: str,
) -> None:
    if not EVALUATE_BASELINE:
        return

    if (
        vanilla_predictions_path().exists()
        and cache_is_current(
            vanilla_summary_path(),
            manifest_hash,
        )
    ):
        print(
            f"Reusing cached base-model evaluation: {BASELINE_MODEL_NAME}"
        )
        return

    processor, model = load_vanilla()

    predictions, summary = (
        evaluate_loaded_model(
            model_name=(
                BASELINE_MODEL_NAME
            ),
            processor=processor,
            model=model,
            rows=test_rows,
            output_root=(
                vanilla_directory()
            ),
            normalisation_file=(
                NORMALISATION_FILE
            ),
            language=LANGUAGE,
            task=TASK,
            use_fp16=USE_FP16,
            generation_max_length=(
                GENERATION_MAX_LENGTH
            ),
            generation_num_beams=(
                GENERATION_NUM_BEAMS
            ),
        )
    )

    summary[
        "_test_manifest_sha256"
    ] = manifest_hash

    write_csv(
        vanilla_predictions_path(),
        predictions,
    )

    save_json(
        vanilla_summary_path(),
        summary,
    )

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_fold(
    fold_number: int,
    test_rows: list[dict[str, Any]],
    manifest_hash: str,
) -> None:
    model_hash = fold_model_sha256(
        fold_number
    )

    if (
        fold_predictions_path(
            fold_number
        ).exists()
        and cache_is_current(
            fold_summary_path(
                fold_number
            ),
            manifest_hash,
            model_hash,
        )
    ):
        print(
            f"Reusing cached fold {fold_number} evaluation."
        )
        return

    processor, model = (
        load_mlp_fusion_model(
            fold_number
        )
    )

    predictions, summary = (
        evaluate_loaded_model(
            model_name=(
                f"{EXPERIMENT_NAME}_fold_{fold_number}"
            ),
            processor=processor,
            model=model,
            rows=test_rows,
            output_root=(
                fold_cache_directory(
                    fold_number
                )
            ),
            normalisation_file=(
                NORMALISATION_FILE
            ),
            language=LANGUAGE,
            task=TASK,
            use_fp16=USE_FP16,
            generation_max_length=(
                GENERATION_MAX_LENGTH
            ),
            generation_num_beams=(
                GENERATION_NUM_BEAMS
            ),
        )
    )

    summary["fold"] = (
        fold_number
    )

    summary[
        "_test_manifest_sha256"
    ] = manifest_hash

    summary[
        "_model_sha256"
    ] = model_hash

    for row in predictions:
        row["fold"] = (
            fold_number
        )

    write_csv(
        fold_predictions_path(
            fold_number
        ),
        predictions,
    )

    save_json(
        fold_summary_path(
            fold_number
        ),
        summary,
    )

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def folds_needing_inference(
    manifest_hash: str,
) -> list[int]:
    selected = requested_folds()

    if not REBUILD_AGGREGATES_FROM_SAVED_FOLDS:
        return selected

    stale_or_missing = [
        fold
        for fold in FOLD_NUMBERS
        if (
            not fold_predictions_path(
                fold
            ).exists()
            or not cache_is_current(
                fold_summary_path(
                    fold
                ),
                manifest_hash,
                fold_model_sha256(
                    fold
                ),
            )
        )
    ]

    return sorted(
        set(selected)
        | set(stale_or_missing)
    )


def load_fold_results(
    folds: list[int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    predictions: list[
        dict[str, Any]
    ] = []

    summaries: list[
        dict[str, Any]
    ] = []

    for fold_number in folds:
        predictions.extend(
            read_csv(
                fold_predictions_path(
                    fold_number
                )
            )
        )

        summaries.append(
            load_json(
                fold_summary_path(
                    fold_number
                )
            )
        )

    return (
        predictions,
        summaries,
    )


def evaluate_current_experiment() -> None:
    if not FIXED_TEST_MANIFEST.exists():
        raise FileNotFoundError(
            "Authoritative fixed test is missing:\n"
            f"{FIXED_TEST_MANIFEST}"
        )

    test_rows = load_manifest(
        FIXED_TEST_MANIFEST
    )

    if MAX_TEST_RECORDINGS is not None:
        test_rows = (
            test_rows[
                :MAX_TEST_RECORDINGS
            ]
        )

    manifest_hash = file_sha256(
        FIXED_TEST_MANIFEST
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_to_evaluate = (
        folds_needing_inference(
            manifest_hash
        )
    )

    validate_models(
        folds_to_evaluate
    )

    print("=" * 80)
    print(
        "MLP INTERACTION FUSION FIXED-TEST EVALUATION"
    )
    print("=" * 80)
    print(
        f"Fixed test: {FIXED_TEST_MANIFEST}"
    )
    print(
        f"Folds evaluated this run: {folds_to_evaluate}"
    )

    evaluate_base_model(
        test_rows,
        manifest_hash,
    )

    for fold_number in folds_to_evaluate:
        evaluate_fold(
            fold_number,
            test_rows,
            manifest_hash,
        )

    aggregate_folds = (
        list(FOLD_NUMBERS)
        if REBUILD_AGGREGATES_FROM_SAVED_FOLDS
        else requested_folds()
    )

    (
        fold_predictions,
        fold_summaries,
    ) = load_fold_results(
        aggregate_folds
    )

    vanilla_predictions: list[
        dict[str, Any]
    ] = []

    vanilla_summaries: list[
        dict[str, Any]
    ] = []

    if (
        vanilla_predictions_path().exists()
        and vanilla_summary_path().exists()
    ):
        vanilla_predictions = read_csv(
            vanilla_predictions_path()
        )

        vanilla_summaries = [
            load_json(
                vanilla_summary_path()
            )
        ]

    combined_predictions = (
        vanilla_predictions
        + fold_predictions
    )

    combined_summaries = (
        vanilla_summaries
        + fold_summaries
    )

    aggregate = summarise_model_results(
        fold_summaries
    )

    all_model_results = [
        flatten_summary(
            summary
        )
        for summary in combined_summaries
    ]

    write_csv(
        OUTPUT_ROOT
        / "all_model_results.csv",
        all_model_results,
    )

    write_csv(
        OUTPUT_ROOT
        / "all_model_predictions.csv",
        combined_predictions,
    )

    write_csv(
        OUTPUT_ROOT
        / "all_recording_predictions.csv",
        combined_predictions,
    )

    save_json(
        OUTPUT_ROOT
        / "fold_model_aggregate.json",
        aggregate,
    )

    SUMMARY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        SUMMARY_ROOT
        / "all_model_results.csv",
        all_model_results,
    )

    write_csv(
        SUMMARY_ROOT
        / "all_model_predictions.csv",
        combined_predictions,
    )

    write_csv(
        SUMMARY_ROOT
        / "all_recording_predictions.csv",
        combined_predictions,
    )

    save_json(
        SUMMARY_ROOT
        / "fold_model_aggregate.json",
        aggregate,
    )

    print()
    print("=" * 80)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 80)
    print(
        "Normalised WER across folds: "
        f"{aggregate['normalised_wer_mean']:.4f}% ± "
        f"{aggregate['normalised_wer_sample_sd']:.4f}"
    )
    print(
        f"Summary root: {SUMMARY_ROOT}"
    )


def main() -> None:
    for experiment_index, experiment_key in enumerate(
        RUN_EXPERIMENTS,
        start=1,
    ):
        apply_experiment_variant(
            experiment_key
        )

        print()
        print("=" * 88)
        print(
            f"EVALUATING BASE-MODEL EXPERIMENT "
            f"{experiment_index}/{len(RUN_EXPERIMENTS)}"
        )
        print("=" * 88)
        print(
            f"Experiment key: {experiment_key}"
        )
        print(
            f"Experiment name: {EXPERIMENT_NAME}"
        )
        print(
            f"Starting/base model: {BASE_MODEL}"
        )
        print(
            f"Baseline label: {BASELINE_MODEL_NAME}"
        )

        evaluate_current_experiment()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 88)
    print("ALL MLP INTERACTION FUSION EVALUATIONS COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()
