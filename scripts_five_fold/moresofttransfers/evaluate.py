from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.moresofttransfers.config import (
    DEFAULT_EXPERIMENT,
    EXPERIMENTS,
    build_runtime_config,
)

from scripts_five_fold.evaluate import (
    evaluate_loaded_model,
    load_manifest,
    write_csv,
)

from scripts_five_fold.moresofttransfers.train import (
    resolve_source_adapter,
)

from scripts_five_fold.train import configure_whisper_generation


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        choices=sorted(EXPERIMENTS),
    )
    return parser.parse_args()


# Loading

def load_full_model(config, best_directory: Path):
    model = WhisperForConditionalGeneration.from_pretrained(
        str(best_directory)
    )
    configure_whisper_generation(
        model,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=True,
    )
    return model


def load_same_lora_model(config, best_directory: Path):
    base = WhisperForConditionalGeneration.from_pretrained(config.BASE_MODEL)
    configure_whisper_generation(
        base,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=True,
    )

    return PeftModel.from_pretrained(
        base,
        best_directory,
        is_trainable=False,
    )


def load_stacked_lora_model(config, best_directory: Path):
    source_adapter = resolve_source_adapter(config.SOURCE_ROOT)

    base = WhisperForConditionalGeneration.from_pretrained(config.BASE_MODEL)
    configure_whisper_generation(
        base,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=True,
    )

    miami_model = PeftModel.from_pretrained(
        base,
        source_adapter,
        is_trainable=False,
    )

    merged_miami = miami_model.merge_and_unload()
    configure_whisper_generation(
        merged_miami,
        language=config.LANGUAGE,
        task=config.TASK,
        use_cache=True,
    )

    return PeftModel.from_pretrained(
        merged_miami,
        best_directory,
        is_trainable=False,
    )


def build_fold_model(config, best_directory: Path):
    if not best_directory.exists():
        raise FileNotFoundError(
            f"Best fold model missing:\n{best_directory}"
        )

    if config.MODE == "full_model_continue":
        return load_full_model(config, best_directory)

    if config.MODE == "same_lora":
        return load_same_lora_model(config, best_directory)

    if config.MODE == "stacked_lora":
        return load_stacked_lora_model(config, best_directory)

    raise ValueError(f"Unknown mode: {config.MODE!r}")


# Summary

def flatten_summary(
    experiment_name: str,
    fold_number: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "model_name": f"{experiment_name}_fold_{fold_number}",
        "fold": fold_number,
        "evaluation_set": "llanito_fixed_test",
        "recordings": summary["recordings"],
    }

    for key, value in summary["overall"].items():
        row[f"overall_{key}"] = value

    for task_name, task_summary in summary["by_task"].items():
        for key, value in task_summary.items():
            row[f"{task_name}_{key}"] = value

    return row


def numeric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_mean_row(
    experiment_name: str,
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not fold_rows:
        raise ValueError("No fold rows to average.")

    row = {
        "model_name": f"{experiment_name}_fold_mean",
        "fold": "mean",
        "evaluation_set": "llanito_fixed_test",
        "recordings": fold_rows[0].get("recordings"),
    }

    metric_keys = sorted({
        key
        for fold_row in fold_rows
        for key in fold_row
        if key.startswith(("overall_", "sentence_", "passage_"))
    })

    for key in metric_keys:
        values = [
            value
            for fold_row in fold_rows
            if (value := numeric_value(fold_row, key)) is not None
        ]

        if values:
            row[key] = sum(values) / len(values)

    return row


# Fold evaluation

def evaluate_fold(config, fold_number: int, fixed_test_rows):
    best_directory = (
        config.MODEL_ROOT
        / f"fold_{fold_number}"
        / "best_model"
    )

    processor = WhisperProcessor.from_pretrained(
        config.BASE_MODEL,
        language=config.LANGUAGE,
        task=config.TASK,
    )
    processor.tokenizer.set_prefix_tokens(
        language=config.LANGUAGE,
        task=config.TASK,
        predict_timestamps=False,
    )

    model = build_fold_model(config, best_directory)

    dtype = (
        torch.float16
        if torch.cuda.is_available() and config.USE_FP16
        else torch.float32
    )
    model = model.to(dtype=dtype)
    model.eval()

    output_root = (
        config.OUTPUT_ROOT
        / f"fold_{fold_number}"
        / "fixed_test"
    )

    predictions, summary = evaluate_loaded_model(
        model_name=f"{config.EXPERIMENT_NAME}_fold_{fold_number}",
        processor=processor,
        model=model,
        rows=fixed_test_rows,
        output_root=output_root,
        normalisation_file=config.NORMALISATION_FILE,
        language=config.LANGUAGE,
        task=config.TASK,
        use_fp16=config.USE_FP16,
        generation_max_length=config.GENERATION_MAX_LENGTH,
        generation_num_beams=config.GENERATION_NUM_BEAMS,
    )

    for prediction in predictions:
        prediction["fold"] = fold_number
        prediction["evaluation_set"] = "llanito_fixed_test"
        prediction["transfer_type"] = config.MODE
        prediction["source_miami_model"] = config.SOURCE_NAME

    result = flatten_summary(
        config.EXPERIMENT_NAME,
        fold_number,
        summary,
    )
    result["transfer_type"] = config.MODE
    result["source_miami_model"] = config.SOURCE_NAME

    write_csv(output_root / "all_model_results.csv", [result])
    write_csv(output_root / "all_model_predictions.csv", predictions)

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, predictions


# Main

def main() -> None:
    args = parse_args()
    config = build_runtime_config(args.experiment)

    fixed_test_rows = load_manifest(config.LLANITO_FIXED_TEST_MANIFEST)
    if config.MAX_TEST_RECORDINGS is not None:
        fixed_test_rows = fixed_test_rows[: config.MAX_TEST_RECORDINGS]

    all_results = []
    all_predictions = []

    for fold_number in config.RUN_FOLDS:
        print()
        print("=" * 80)
        print(f"EVALUATING {config.EXPERIMENT_NAME} FOLD {fold_number}")
        print("=" * 80)

        result, predictions = evaluate_fold(
            config,
            fold_number,
            fixed_test_rows,
        )
        all_results.append(result)
        all_predictions.extend(predictions)

    mean_row = build_mean_row(config.EXPERIMENT_NAME, all_results)
    mean_row["transfer_type"] = config.MODE
    mean_row["source_miami_model"] = config.SOURCE_NAME

    results_with_mean = [*all_results, mean_row]

    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    for root in (config.OUTPUT_ROOT, config.SUMMARY_ROOT):
        write_csv(root / "all_model_results.csv", results_with_mean)
        write_csv(root / "all_model_predictions.csv", all_predictions)

    print()
    print("=" * 80)
    print(f"{config.EXPERIMENT_NAME} EVALUATION COMPLETE")
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
