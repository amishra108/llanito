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

from scripts_five_fold.llanito_resadapt.config import (
    ADAPTER_LAYERS,
    ADAPTER_MODULE,
    ADAPTER_POSITION,
    BASE_MODEL,
    EVALUATE_VANILLA_BASELINE,
    EXPERIMENT_NAME,
    FOLD_NUMBERS,
    GENERATION_MAX_LENGTH,
    GENERATION_NUM_BEAMS,
    LANGUAGE,
    MAX_TEST_RECORDINGS,
    MODEL_ROOT,
    NORMALISATION_FILE,
    OUTPUT_ROOT as EXPERIMENT_OUTPUT_ROOT,
    PROCESSED_ROOT,
    REBUILD_AGGREGATES_FROM_SAVED_FOLDS,
    RUN_FOLDS,
    SUMMARY_ROOT,
    TASK,
    USE_FP16,
)

from scripts_five_fold.llanito_resadapt.train import (
    build_residual_adapter_model,
    canonical_saved_adapter_module,
    load_residual_adapter,
    load_saved_adapter_config,
)

from scripts_five_fold.evaluate import (
    evaluate_loaded_model,
    load_manifest,
    save_json,
    summarise_model_results,
    write_csv,
)


# Llanito fixed-test output

OUTPUT_ROOT = (
    EXPERIMENT_OUTPUT_ROOT
    / "fixed_test"
)

FIXED_TEST_MANIFEST = (
    PROCESSED_ROOT
    / "fixed_test.csv"
)


# General helpers

def requested_fold_numbers(
) -> list[int]:
    if RUN_FOLDS is None:
        return list(
            FOLD_NUMBERS
        )

    requested = sorted(
        set(
            RUN_FOLDS
        )
    )

    invalid = [
        fold
        for fold in requested
        if fold not in FOLD_NUMBERS
    ]

    if invalid:
        raise ValueError(
            "Invalid RUN_FOLDS "
            f"values: {invalid}. "
            f"Valid folds: "
            f"{FOLD_NUMBERS}"
        )

    return requested


def adapter_directory_for_fold(
    fold_number: int,
) -> Path:
    return (
        MODEL_ROOT
        / f"fold_{fold_number}"
        / "adapter"
    )


def validate_adapter_directory(
    fold_number: int,
) -> Path:
    directory = (
        adapter_directory_for_fold(
            fold_number
        )
    )

    if not directory.exists():
        raise FileNotFoundError(
            f"Fold {fold_number} "
            "adapter directory "
            "is missing:\n"
            f"{directory}"
        )

    required = [
        directory
        / "adapter_config.json",
        directory
        / "adapter_state.pt",
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Fold {fold_number} "
            "adapter is incomplete:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    saved_config = (
        load_saved_adapter_config(
            directory
        )
    )

    saved_module = (
        canonical_saved_adapter_module(
            saved_config
        )
    )

    if saved_module != ADAPTER_MODULE:
        raise ValueError(
            f"Fold {fold_number} "
            f"is a {saved_module!r} "
            "adapter but current "
            "config expects "
            f"{ADAPTER_MODULE!r}."
        )

    saved_training_data = (
        saved_config.get(
            "training_data"
        )
    )

    if (
        saved_training_data
        is not None
        and saved_training_data
        != "joint_miami_llanito"
    ):
        raise ValueError(
            f"Fold {fold_number} "
            "was not saved as a "
            "joint Miami-Llanito "
            "adapter. Saved value: "
            f"{saved_training_data!r}"
        )

    return directory


def preflight_validate_all_requested_models(
    fold_numbers: list[int],
) -> None:
    problems: list[
        str
    ] = []

    for fold_number in fold_numbers:
        try:
            validate_adapter_directory(
                fold_number
            )

        except Exception as error:
            problems.append(
                f"fold {fold_number}: "
                f"{error}"
            )

    if problems:
        raise RuntimeError(
            "Adapter preflight failed "
            "before inference:\n\n"
            + "\n\n".join(
                problems
            )
        )


def load_json(
    path: Path,
) -> dict[
    str,
    Any,
]:
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
            "Expected JSON object "
            f"in {path}"
        )

    return value


def read_csv(
    path: Path,
) -> list[
    dict[
        str,
        Any,
    ]
]:
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


def file_sha256(
    path: Path,
) -> str:
    digest = (
        hashlib.sha256()
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


# Cache paths

def fold_cache_directory(
    fold_number: int,
) -> Path:
    return (
        OUTPUT_ROOT
        / f"fold_{fold_number}"
    )


def fold_predictions_cache_path(
    fold_number: int,
) -> Path:
    return (
        fold_cache_directory(
            fold_number
        )
        / "cached_predictions.csv"
    )


def fold_summary_cache_path(
    fold_number: int,
) -> Path:
    return (
        fold_cache_directory(
            fold_number
        )
        / "cached_summary.json"
    )


def vanilla_cache_directory(
) -> Path:
    return (
        OUTPUT_ROOT
        / "vanilla_whisper_medium"
    )


def vanilla_predictions_cache_path(
) -> Path:
    return (
        vanilla_cache_directory()
        / "cached_predictions.csv"
    )


def vanilla_summary_cache_path(
) -> Path:
    return (
        vanilla_cache_directory()
        / "cached_summary.json"
    )


def cache_exists_for_fold(
    fold_number: int,
    test_manifest_sha256: str,
) -> bool:
    predictions = (
        fold_predictions_cache_path(
            fold_number
        )
    )

    summary_path = (
        fold_summary_cache_path(
            fold_number
        )
    )

    if (
        not predictions.exists()
        or not summary_path.exists()
    ):
        return False

    try:
        summary = load_json(
            summary_path
        )

    except Exception:
        return False

    return (
        summary.get(
            "_test_manifest_sha256"
        )
        == test_manifest_sha256
    )


# Model loading

def processor_from_adapter_or_base(
    adapter_directory: Path,
) -> WhisperProcessor:
    try:
        processor = (
            WhisperProcessor
            .from_pretrained(
                adapter_directory,
                language=LANGUAGE,
                task=TASK,
            )
        )

    except Exception as error:
        print(
            "Could not load processor "
            "from adapter directory; "
            "falling back to base "
            "Whisper processor."
        )

        print(
            "Processor load error: "
            f"{error}"
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

    return processor


def load_resadapt(
    adapter_directory: Path,
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
        processor_from_adapter_or_base(
            adapter_directory
        )
    )

    model = (
        build_residual_adapter_model()
    )

    load_residual_adapter(
        model,
        adapter_directory,
    )

    model = model.to(
        dtype=dtype
    )

    model.eval()

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

    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
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


# Summary formatting

def flatten_summary(
    summary: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    row: dict[
        str,
        Any,
    ] = {
        "model_name": (
            summary[
                "model_name"
            ]
        ),
        "recordings": (
            summary[
                "recordings"
            ]
        ),
    }

    if "fold" in summary:
        row["fold"] = (
            summary[
                "fold"
            ]
        )

    for (
        key,
        value,
    ) in (
        summary[
            "overall"
        ].items()
    ):
        row[
            f"overall_{key}"
        ] = value

    for (
        task_name,
        task_summary,
    ) in (
        summary[
            "by_task"
        ].items()
    ):
        for (
            key,
            value,
        ) in (
            task_summary.items()
        ):
            row[
                f"{task_name}_{key}"
            ] = value

    return row


# Vanilla evaluation

def evaluate_vanilla_if_needed(
    test_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    test_manifest_sha256: str,
) -> None:
    if not EVALUATE_VANILLA_BASELINE:
        return

    if (
        vanilla_predictions_cache_path()
        .exists()
        and vanilla_summary_cache_path()
        .exists()
    ):
        try:
            cached_summary = (
                load_json(
                    vanilla_summary_cache_path()
                )
            )

            if (
                cached_summary.get(
                    "_test_manifest_sha256"
                )
                == test_manifest_sha256
            ):
                print(
                    "Reusing cached "
                    "vanilla fixed-test "
                    "evaluation."
                )
                return

        except Exception:
            print(
                "Vanilla cache exists "
                "but is unreadable; "
                "rebuilding it."
            )

    processor, model = (
        load_vanilla()
    )

    (
        predictions,
        summary,
    ) = (
        evaluate_loaded_model(
            model_name=(
                "vanilla_whisper_medium"
            ),
            processor=processor,
            model=model,
            rows=test_rows,
            output_root=(
                vanilla_cache_directory()
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
    ] = (
        test_manifest_sha256
    )

    write_csv(
        vanilla_predictions_cache_path(),
        predictions,
    )

    save_json(
        vanilla_summary_cache_path(),
        summary,
    )

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Fold evaluation

def folds_that_need_inference(
    test_manifest_sha256: str,
) -> list[int]:
    selected = (
        requested_fold_numbers()
    )

    if not (
        REBUILD_AGGREGATES_FROM_SAVED_FOLDS
    ):
        return selected

    stale_or_missing = [
        fold
        for fold in FOLD_NUMBERS
        if not cache_exists_for_fold(
            fold,
            test_manifest_sha256,
        )
    ]

    folds = sorted(
        set(
            selected
        )
        | set(
            stale_or_missing
        )
    )

    if stale_or_missing:
        print(
            "Missing or stale "
            "fixed-test cache for "
            f"folds {stale_or_missing}; "
            "they will be evaluated."
        )

    return folds


def evaluate_fold(
    fold_number: int,
    test_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    test_manifest_sha256: str,
) -> None:
    adapter_directory = (
        validate_adapter_directory(
            fold_number
        )
    )

    processor, model = (
        load_resadapt(
            adapter_directory
        )
    )

    (
        predictions,
        summary,
    ) = (
        evaluate_loaded_model(
            model_name=(
                f"{EXPERIMENT_NAME}_"
                f"fold_{fold_number}"
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

    summary[
        "fold"
    ] = (
        fold_number
    )

    summary[
        "training_data"
    ] = (
        "joint_miami_llanito"
    )

    summary[
        "evaluation_data"
    ] = (
        "llanito_fixed_test"
    )

    summary[
        "_test_manifest_sha256"
    ] = (
        test_manifest_sha256
    )

    for row in predictions:
        row[
            "fold"
        ] = (
            fold_number
        )

        row[
            "training_data"
        ] = (
            "joint_miami_llanito"
        )

        row[
            "evaluation_set"
        ] = (
            "llanito_fixed_test"
        )

    write_csv(
        fold_predictions_cache_path(
            fold_number
        ),
        predictions,
    )

    save_json(
        fold_summary_cache_path(
            fold_number
        ),
        summary,
    )

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Aggregate loading

def load_fold_results(
    fold_numbers: list[int],
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    predictions: list[
        dict[
            str,
            Any,
        ]
    ] = []

    summaries: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for fold_number in fold_numbers:
        prediction_path = (
            fold_predictions_cache_path(
                fold_number
            )
        )

        summary_path = (
            fold_summary_cache_path(
                fold_number
            )
        )

        if not prediction_path.exists():
            raise FileNotFoundError(
                "Missing cached "
                "predictions for "
                f"fold {fold_number}:\n"
                f"{prediction_path}"
            )

        if not summary_path.exists():
            raise FileNotFoundError(
                "Missing cached summary "
                f"for fold {fold_number}:\n"
                f"{summary_path}"
            )

        predictions.extend(
            read_csv(
                prediction_path
            )
        )

        summaries.append(
            load_json(
                summary_path
            )
        )

    return (
        predictions,
        summaries,
    )


def load_cached_vanilla_results(
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    if not (
        vanilla_predictions_cache_path()
        .exists()
        and vanilla_summary_cache_path()
        .exists()
    ):
        return (
            [],
            [],
        )

    return (
        read_csv(
            vanilla_predictions_cache_path()
        ),
        [
            load_json(
                vanilla_summary_cache_path()
            )
        ],
    )


# Main

def main() -> None:
    if not (
        FIXED_TEST_MANIFEST.exists()
    ):
        raise FileNotFoundError(
            "Authoritative Llanito "
            "fixed test is missing:\n"
            f"{FIXED_TEST_MANIFEST}"
        )

    test_manifest_sha256 = (
        file_sha256(
            FIXED_TEST_MANIFEST
        )
    )

    test_rows = (
        load_manifest(
            FIXED_TEST_MANIFEST
        )
    )

    if MAX_TEST_RECORDINGS is not None:
        test_rows = (
            test_rows[
                :MAX_TEST_RECORDINGS
            ]
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested = (
        requested_fold_numbers()
    )

    folds_to_evaluate = (
        folds_that_need_inference(
            test_manifest_sha256
        )
    )

    preflight_validate_all_requested_models(
        folds_to_evaluate
    )

    print()
    print("=" * 80)
    print(
        "JOINT MIAMI-LLANITO "
        "RESIDUAL ADAPTER "
        "FIXED-TEST EVALUATION"
    )
    print("=" * 80)

    print(
        f"Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Model root: "
        f"{MODEL_ROOT}"
    )

    print(
        f"Adapter module: "
        f"{ADAPTER_MODULE}"
    )

    print(
        f"Adapter position: "
        f"{ADAPTER_POSITION}"
    )

    print(
        f"Adapter layers: "
        f"{ADAPTER_LAYERS}"
    )

    print(
        "Training data: "
        "JOINT MIAMI + LLANITO"
    )

    print(
        "Evaluation data: "
        "LLANITO FIXED TEST ONLY"
    )

    print(
        f"Fixed test: "
        f"{FIXED_TEST_MANIFEST}"
    )

    print(
        f"Test recordings: "
        f"{len(test_rows)}"
    )

    print(
        f"Requested folds: "
        f"{requested}"
    )

    print(
        "Folds evaluated "
        "this run: "
        f"{folds_to_evaluate}"
    )

    evaluate_vanilla_if_needed(
        test_rows,
        test_manifest_sha256,
    )

    for fold_number in (
        folds_to_evaluate
    ):
        evaluate_fold(
            fold_number,
            test_rows,
            test_manifest_sha256,
        )

    aggregate_folds = (
        list(
            FOLD_NUMBERS
        )
        if (
            REBUILD_AGGREGATES_FROM_SAVED_FOLDS
        )
        else requested
    )

    (
        fold_predictions,
        fold_summaries,
    ) = (
        load_fold_results(
            aggregate_folds
        )
    )

    (
        vanilla_predictions,
        vanilla_summaries,
    ) = (
        load_cached_vanilla_results()
    )

    combined_predictions = (
        vanilla_predictions
        + fold_predictions
    )

    combined_summaries = (
        vanilla_summaries
        + fold_summaries
    )

    aggregate = (
        summarise_model_results(
            fold_summaries
        )
    )

    aggregate[
        "experiment"
    ] = (
        EXPERIMENT_NAME
    )

    aggregate[
        "training_data"
    ] = (
        "joint_miami_llanito"
    )

    aggregate[
        "evaluation_data"
    ] = (
        "llanito_fixed_test"
    )

    aggregate[
        "adapter_module"
    ] = (
        ADAPTER_MODULE
    )

    aggregate[
        "adapter_position"
    ] = (
        ADAPTER_POSITION
    )

    aggregate[
        "adapter_layers"
    ] = (
        ADAPTER_LAYERS
    )

    all_model_results = [
        flatten_summary(
            summary
        )
        for summary in (
            combined_summaries
        )
    ]

    # Detailed outputs.

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

    # Mirror into summary root.

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
        f"{aggregate['normalised_wer_mean']:.4f}% "
        "± "
        f"{aggregate['normalised_wer_sample_sd']:.4f}"
    )

    print(
        f"Summary root: "
        f"{SUMMARY_ROOT}"
    )


if __name__ == "__main__":
    main()