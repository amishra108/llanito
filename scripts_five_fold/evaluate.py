from __future__ import annotations

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from scripts_five_fold.data_utils import TARGET_SAMPLE_RATE, load_wav_for_whisper
from scripts_five_fold.train import (
    calculate_corpus_metrics,
    canonicalise_tokens,
    configure_whisper_generation,
    load_task_mappings,
    tokenise_transcript,
)


def load_manifest(
    path: Path,
    required_columns: Sequence[str] = (
        "task",
        "speaker",
        "audio_path",
        "transcript",
    ),
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist:\n{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header:\n{path}")

        missing = set(required_columns) - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        rows = [dict(row) for row in reader]

    if not rows:
        raise RuntimeError(f"Manifest is empty:\n{path}")

    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def transcribe_one(
    row: dict[str, str],
    processor: WhisperProcessor,
    model: torch.nn.Module,
    device: torch.device,
    use_fp16: bool,
    generation_max_length: int,
    generation_num_beams: int,
) -> str:
    audio = load_wav_for_whisper(
        Path(row["audio_path"]),
        target_sample_rate=TARGET_SAMPLE_RATE,
        print_audio_check=False,
    )
    inputs = processor.feature_extractor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    if device.type == "cuda" and use_fp16:
        input_features = input_features.to(torch.float16)

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features=input_features,
            max_length=generation_max_length,
            num_beams=generation_num_beams,
            do_sample=False,
        )

    return processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def evaluate_loaded_model(
    *,
    model_name: str,
    processor: WhisperProcessor,
    model: torch.nn.Module,
    rows: Sequence[dict[str, str]],
    output_root: Path,
    normalisation_file: Path | None,
    language: str,
    task: str,
    use_fp16: bool,
    generation_max_length: int,
    generation_num_beams: int,
    device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    task_mappings = load_task_mappings(normalisation_file)
    processor.tokenizer.set_prefix_tokens(
        language=language,
        task=task,
        predict_timestamps=False,
    )
    configure_whisper_generation(
        model,
        language=language,
        task=task,
        use_cache=True,
    )
    model.to(device)
    model.eval()

    prediction_rows: list[dict[str, Any]] = []
    references: list[str] = []
    hypotheses: list[str] = []
    tasks: list[str] = []

    for index, row in enumerate(rows, start=1):
        task_name = str(row["task"]).strip().casefold()
        reference = str(row["transcript"])
        hypothesis = transcribe_one(
            row,
            processor,
            model,
            device,
            use_fp16,
            generation_max_length,
            generation_num_beams,
        )
        mapping = task_mappings.get(task_name, {})
        raw_reference = tokenise_transcript(reference)
        raw_hypothesis = tokenise_transcript(hypothesis)
        normalised_reference = canonicalise_tokens(raw_reference, mapping)
        normalised_hypothesis = canonicalise_tokens(raw_hypothesis, mapping)

        row_metrics = calculate_corpus_metrics(
            [reference],
            [hypothesis],
            [task_name],
            task_mappings,
        )
        output_row: dict[str, Any] = {
            "model_name": model_name,
            "task": task_name,
            "speaker": row.get("speaker", ""),
            "prompt_id": row.get("prompt_id", ""),
            "prompt_key": row.get("prompt_key", ""),
            "repetition": row.get("repetition", ""),
            "utterance_id": row.get("utterance_id", ""),
            "recording": row.get("recording", ""),
            "audio_path": row["audio_path"],
            "reference": reference,
            "prediction": hypothesis,
            "normalised_reference": " ".join(normalised_reference),
            "normalised_prediction": " ".join(normalised_hypothesis),
            **row_metrics,
        }
        prediction_rows.append(output_row)
        references.append(reference)
        hypotheses.append(hypothesis)
        tasks.append(task_name)

        print(
            f"[{index:>3}/{len(rows)}] "
            f"{task_name:<8} | {row.get('speaker', ''):<24} | "
            f"{row.get('recording', row.get('utterance_id', ''))}"
        )

    overall = calculate_corpus_metrics(
        references,
        hypotheses,
        tasks,
        task_mappings,
    )
    by_task: dict[str, Any] = {}

    for task_name in sorted(set(tasks)):
        indices = [
            index
            for index, value in enumerate(tasks)
            if value == task_name
        ]
        by_task[task_name] = calculate_corpus_metrics(
            [references[index] for index in indices],
            [hypotheses[index] for index in indices],
            [tasks[index] for index in indices],
            task_mappings,
        )
        by_task[task_name]["recordings"] = len(indices)

    summary = {
        "model_name": model_name,
        "recordings": len(rows),
        "overall": overall,
        "by_task": by_task,
    }

    write_csv(output_root / "predictions.csv", prediction_rows)
    save_json(output_root / "summary.json", summary)

    print()
    print(
        f"{model_name} raw WER: {overall['wer']:.4f}%"
    )
    print(
        f"{model_name} normalised WER: "
        f"{overall['normalised_wer']:.4f}%"
    )

    return prediction_rows, summary


def load_standard_whisper(
    model_reference: str | Path,
    language: str,
    task: str,
    use_fp16: bool,
) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    dtype = (
        torch.float16
        if torch.cuda.is_available() and use_fp16
        else torch.float32
    )
    processor = WhisperProcessor.from_pretrained(
        model_reference,
        language=language,
        task=task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_reference,
        torch_dtype=dtype,
    )
    return processor, model


def cleanup_model(*objects: Any) -> None:
    for value in objects:
        del value

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarise_model_results(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw_wers = np.array(
        [summary["overall"]["wer"] for summary in summaries],
        dtype=np.float64,
    )
    normalised_wers = np.array(
        [
            summary["overall"]["normalised_wer"]
            for summary in summaries
        ],
        dtype=np.float64,
    )

    return {
        "models": len(summaries),
        "raw_wer_mean": float(raw_wers.mean()),
        "raw_wer_sample_sd": (
            float(raw_wers.std(ddof=1))
            if len(raw_wers) > 1
            else 0.0
        ),
        "normalised_wer_mean": float(normalised_wers.mean()),
        "normalised_wer_sample_sd": (
            float(normalised_wers.std(ddof=1))
            if len(normalised_wers) > 1
            else 0.0
        ),
    }
