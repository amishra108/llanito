from __future__ import annotations

import csv
import gc
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.fullLoRA.config_decoderAttnLoRA import (
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


def extract_fold_number(path: Path) -> int:
    match = re.fullmatch(
        r"fold_(\d+)",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"Invalid fold directory: {path}"
        )

    return int(match.group(1))


def requested_fold_numbers() -> list[int]:
    if RUN_FOLDS is None:
        return list(FOLD_NUMBERS)

    requested = sorted(set(RUN_FOLDS))

    invalid = [
        fold_number
        for fold_number in requested
        if fold_number not in FOLD_NUMBERS
    ]

    if invalid:
        raise ValueError(
            f"RUN_FOLDS contains invalid folds: {invalid}. "
            f"Valid folds: {FOLD_NUMBERS}"
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


def validate_adapter(
    fold_number: int,
) -> Path:
    adapter_directory = (
        adapter_directory_for_fold(
            fold_number
        )
    )

    if not (
        adapter_directory
        / "adapter_config.json"
    ).exists():
        raise FileNotFoundError(
            f"Fold {fold_number} adapter is missing "
            "or incomplete:\n"
            f"{adapter_directory}"
        )

    return adapter_directory


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
        fold_cache_directory(fold_number)
        / "cached_predictions.csv"
    )


def fold_summary_cache_path(
    fold_number: int,
) -> Path:
    return (
        fold_cache_directory(fold_number)
        / "cached_summary.json"
    )


def vanilla_cache_directory() -> Path:
    return (
        OUTPUT_ROOT
        / "vanilla_whisper_medium"
    )


def vanilla_predictions_cache_path() -> Path:
    return (
        vanilla_cache_directory()
        / "cached_predictions.csv"
    )


def vanilla_summary_cache_path() -> Path:
    return (
        vanilla_cache_directory()
        / "cached_summary.json"
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        return json.load(input_file)


def read_csv(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        return list(csv.DictReader(input_file))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def cache_exists_for_fold(
    fold_number: int,
    test_manifest_sha256: str,
) -> bool:
    predictions_path = (
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
        not predictions_path.exists()
        or not summary_path.exists()
    ):
        return False

    summary = load_json(summary_path)

    return (
        summary.get(
            "_test_manifest_sha256"
        )
        == test_manifest_sha256
    )


def load_lora_adapter(
    adapter_directory: Path,
) -> tuple[
    WhisperProcessor,
    PeftModel,
]:
    dtype = (
        torch.float16
        if torch.cuda.is_available()
        and USE_FP16
        else torch.float32
    )

    processor = (
        WhisperProcessor.from_pretrained(
            adapter_directory,
            language=LANGUAGE,
            task=TASK,
        )
    )

    base_model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
        )
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_directory,
        is_trainable=False,
    )

    return processor, model


def load_vanilla() -> tuple[
    WhisperProcessor,
    WhisperForConditionalGeneration,
]:
    dtype = (
        torch.float16
        if torch.cuda.is_available()
        and USE_FP16
        else torch.float32
    )

    processor = (
        WhisperProcessor.from_pretrained(
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

    return processor, model


def flatten_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "model_name": summary["model_name"],
        "recordings": summary["recordings"],
    }

    if "fold" in summary:
        row["fold"] = summary["fold"]

    for key, value in (
        summary["overall"].items()
    ):
        row[f"overall_{key}"] = value

    for task_name, task_summary in (
        summary["by_task"].items()
    ):
        for key, value in (
            task_summary.items()
        ):
            row[
                f"{task_name}_{key}"
            ] = value

    return row


def evaluate_vanilla_if_needed(
    test_rows: list[dict[str, Any]],
    test_manifest_sha256: str,
) -> None:
    if not EVALUATE_VANILLA_BASELINE:
        return

    if (
        vanilla_predictions_cache_path().exists()
        and vanilla_summary_cache_path().exists()
    ):
        cached_summary = load_json(
            vanilla_summary_cache_path()
        )

        if (
            cached_summary.get(
                "_test_manifest_sha256"
            )
            == test_manifest_sha256
        ):
            print(
                "Reusing cached vanilla "
                "fixed-test evaluation."
            )
            return

        print(
            "Fixed-test manifest changed; "
            "vanilla cache will be refreshed."
        )

    processor, model = load_vanilla()

    predictions, summary = (
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

    summary["_test_manifest_sha256"] = (
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


def folds_that_need_inference(
    test_manifest_sha256: str,
) -> list[int]:
    selected = requested_fold_numbers()

    if not (
        REBUILD_AGGREGATES_FROM_SAVED_FOLDS
    ):
        return selected

    stale_or_missing_folds = [
        fold_number
        for fold_number in FOLD_NUMBERS
        if not cache_exists_for_fold(
            fold_number,
            test_manifest_sha256,
        )
    ]

    folds_to_evaluate = sorted(
        set(selected)
        | set(stale_or_missing_folds)
    )

    if stale_or_missing_folds:
        print(
            "Missing or stale fixed-test cache "
            f"for folds {stale_or_missing_folds}; "
            "they will be evaluated automatically "
            "so the complete aggregate is valid."
        )

    return folds_to_evaluate


def evaluate_fold(
    fold_number: int,
    test_rows: list[dict[str, Any]],
    test_manifest_sha256: str,
) -> None:
    adapter_directory = (
        validate_adapter(fold_number)
    )

    processor, model = (
        load_lora_adapter(
            adapter_directory
        )
    )

    predictions, summary = (
        evaluate_loaded_model(
            model_name=(
                f"fullLoRA_fold_{fold_number}"
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

    summary["fold"] = fold_number
    summary["_test_manifest_sha256"] = (
        test_manifest_sha256
    )

    for row in predictions:
        row["fold"] = fold_number

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


def load_complete_fold_results() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    all_predictions: list[
        dict[str, Any]
    ] = []
    all_summaries: list[
        dict[str, Any]
    ] = []

    for fold_number in FOLD_NUMBERS:
        predictions_path = (
            fold_predictions_cache_path(
                fold_number
            )
        )
        summary_path = (
            fold_summary_cache_path(
                fold_number
            )
        )

        if not predictions_path.exists():
            raise FileNotFoundError(
                "Cannot build full aggregate "
                "because cached predictions are "
                f"missing for fold {fold_number}:\n"
                f"{predictions_path}"
            )

        if not summary_path.exists():
            raise FileNotFoundError(
                "Cannot build full aggregate "
                "because cached summary is "
                f"missing for fold {fold_number}:\n"
                f"{summary_path}"
            )

        all_predictions.extend(
            read_csv(predictions_path)
        )
        all_summaries.append(
            load_json(summary_path)
        )

    return (
        all_predictions,
        all_summaries,
    )


def load_selected_fold_results() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    all_predictions: list[
        dict[str, Any]
    ] = []
    all_summaries: list[
        dict[str, Any]
    ] = []

    for fold_number in (
        requested_fold_numbers()
    ):
        all_predictions.extend(
            read_csv(
                fold_predictions_cache_path(
                    fold_number
                )
            )
        )
        all_summaries.append(
            load_json(
                fold_summary_cache_path(
                    fold_number
                )
            )
        )

    return (
        all_predictions,
        all_summaries,
    )


def load_cached_vanilla_results() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not (
        vanilla_predictions_cache_path().exists()
        and vanilla_summary_cache_path().exists()
    ):
        return [], []

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


def main() -> None:
    test_manifest_path = (
        PROCESSED_ROOT
        / "fixed_test.csv"
    )

    test_manifest_sha256 = (
        file_sha256(test_manifest_path)
    )

    test_rows = load_manifest(
        test_manifest_path
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

    selected_folds = (
        requested_fold_numbers()
    )

    print("=" * 80)
    print("FULL LORA FIXED-TEST EVALUATION")
    print("=" * 80)
    print(
        "Requested folds: "
        f"{RUN_FOLDS if RUN_FOLDS is not None else 'ALL'}"
    )
    print(
        f"Selected folds: {selected_folds}"
    )

    evaluate_vanilla_if_needed(
        test_rows,
        test_manifest_sha256,
    )

    folds_to_evaluate = (
        folds_that_need_inference(
            test_manifest_sha256
        )
    )

    print(
        "Folds evaluated this run: "
        f"{folds_to_evaluate}"
    )

    for fold_number in folds_to_evaluate:
        evaluate_fold(
            fold_number=fold_number,
            test_rows=test_rows,
            test_manifest_sha256=(
                test_manifest_sha256
            ),
        )

    if (
        REBUILD_AGGREGATES_FROM_SAVED_FOLDS
    ):
        (
            fold_predictions,
            fold_summaries,
        ) = load_complete_fold_results()
    else:
        (
            fold_predictions,
            fold_summaries,
        ) = load_selected_fold_results()

    (
        vanilla_predictions,
        vanilla_summaries,
    ) = load_cached_vanilla_results()

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
        flatten_summary(summary)
        for summary in combined_summaries
    ]

    # Normal detailed experiment outputs stay where they already live.
    write_csv(
        OUTPUT_ROOT
        / "all_model_predictions.csv",
        combined_predictions,
    )
    write_csv(
        OUTPUT_ROOT
        / "all_model_results.csv",
        all_model_results,
    )
    save_json(
        OUTPUT_ROOT
        / "fold_model_aggregate.json",
        aggregate,
    )

    # ALSO write only all_model_results.csv to outputs/summary/fullLoRA.
    SUMMARY_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    write_csv(
        SUMMARY_ROOT
        / "all_model_results.csv",
        all_model_results,
    )

    print()
    print("=" * 80)
    print(
        "FULL LORA FIXED-TEST "
        "EVALUATION COMPLETE"
    )
    print("=" * 80)
    print(
        "Fold models included in aggregate: "
        f"{[summary['fold'] for summary in fold_summaries]}"
    )
    print(
        "Normalised WER across LoRA folds: "
        f"{aggregate['normalised_wer_mean']:.4f}% ± "
        f"{aggregate['normalised_wer_sample_sd']:.4f}"
    )
    print(f"Results: {OUTPUT_ROOT}")
    print(
        "Summary CSV also saved to: "
        f"{SUMMARY_ROOT / 'all_model_results.csv'}"
    )


if __name__ == "__main__":
    main()
