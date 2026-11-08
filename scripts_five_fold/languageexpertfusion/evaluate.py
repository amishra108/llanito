from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.evaluate import (
    evaluate_loaded_model,
    load_manifest,
    save_json,
    summarise_model_results,
    write_csv,
)
from scripts_five_fold.languageexpertfusion.config import (
    BASE_MODEL,
    BASELINE_MODEL_NAME,
    EVALUATE_BASELINE,
    FIXED_TEST_MANIFEST,
    FOLD_NUMBERS,
    GENERATION_MAX_LENGTH,
    GENERATION_NUM_BEAMS,
    LANGUAGE,
    MAX_TEST_RECORDINGS,
    NORMALISATION_FILE,
    RUN_ARCHITECTURES,
    RUN_FOLDS,
    TASK,
    USE_FP16,
    experiment_name,
    output_root,
    summary_root,
)
from scripts_five_fold.languageexpertfusion.model import (
    build_fusion_model,
    load_fusion,
)
from scripts_five_fold.languageexpertfusion.train import (
    fusion_directory,
    expert_directory,
)


def folds_to_run() -> list[int]:
    if RUN_FOLDS is None:
        return list(FOLD_NUMBERS)
    return sorted(set(RUN_FOLDS))


def flatten_summary(
    model_name: str,
    fold: int | str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_name": model_name,
        "fold": fold,
        "recordings": summary["recordings"],
    }

    for key, value in summary["overall"].items():
        row[f"overall_{key}"] = value

    for task_name, task_summary in summary["by_task"].items():
        for key, value in task_summary.items():
            row[f"{task_name}_{key}"] = value

    return row


def load_fused(
    architecture: str,
    fold: int,
) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL,
        language=LANGUAGE,
        task=TASK,
    )

    model = build_fusion_model(
        architecture,
        expert_directory(architecture, "spanish"),
        expert_directory(architecture, "english"),
    )

    load_fusion(
        model,
        fusion_directory(architecture, fold),
        architecture,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() and USE_FP16 else torch.float32

    model = model.to(device=device, dtype=dtype)
    model.eval()

    return processor, model


def evaluate_architecture(architecture: str) -> None:
    rows = load_manifest(FIXED_TEST_MANIFEST)

    if MAX_TEST_RECORDINGS is not None:
        rows = rows[:MAX_TEST_RECORDINGS]

    predictions_all: list[dict[str, Any]] = []
    results_all: list[dict[str, Any]] = []

    if EVALUATE_BASELINE:
        processor = WhisperProcessor.from_pretrained(
            BASE_MODEL,
            language=LANGUAGE,
            task=TASK,
        )

        dtype = (
            torch.float16
            if torch.cuda.is_available() and USE_FP16
            else torch.float32
        )

        baseline = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
        )

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        baseline = baseline.to(device)
        baseline.eval()

        baseline_predictions, baseline_summary = evaluate_loaded_model(
            model_name=BASELINE_MODEL_NAME,
            processor=processor,
            model=baseline,
            rows=rows,
            output_root=output_root(architecture) / "fixed_test" / BASELINE_MODEL_NAME,
            normalisation_file=NORMALISATION_FILE,
            language=LANGUAGE,
            task=TASK,
            use_fp16=USE_FP16,
            generation_max_length=GENERATION_MAX_LENGTH,
            generation_num_beams=GENERATION_NUM_BEAMS,
        )

        for row in baseline_predictions:
            row["fold"] = "baseline"

        predictions_all.extend(baseline_predictions)
        results_all.append(
            flatten_summary(
                BASELINE_MODEL_NAME,
                "baseline",
                baseline_summary,
            )
        )

        del baseline
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for fold in folds_to_run():
        processor, model = load_fused(architecture, fold)

        predictions, summary = evaluate_loaded_model(
            model_name=f"{experiment_name(architecture)}_fold_{fold}",
            processor=processor,
            model=model,
            rows=rows,
            output_root=output_root(architecture) / "fixed_test" / f"fold_{fold}",
            normalisation_file=NORMALISATION_FILE,
            language=LANGUAGE,
            task=TASK,
            use_fp16=USE_FP16,
            generation_max_length=GENERATION_MAX_LENGTH,
            generation_num_beams=GENERATION_NUM_BEAMS,
        )

        for row in predictions:
            row["fold"] = fold

        predictions_all.extend(predictions)
        results_all.append(
            flatten_summary(
                f"{experiment_name(architecture)}_fold_{fold}",
                fold,
                summary,
            )
        )

        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    destination = summary_root(architecture)
    destination.mkdir(parents=True, exist_ok=True)

    write_csv(
        destination / "all_model_predictions.csv",
        predictions_all,
    )
    write_csv(
        destination / "all_model_results.csv",
        results_all,
    )

    print(f"Saved: {destination}")


def main() -> None:
    for architecture in RUN_ARCHITECTURES:
        evaluate_architecture(architecture)


if __name__ == "__main__":
    main()
