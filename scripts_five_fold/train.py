from __future__ import annotations

import csv
import inspect
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from scripts_five_fold.data_utils import TARGET_SAMPLE_RATE, load_wav_for_whisper


# Reproducibility and process helpers

def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_cuda_environment(
    use_fp16: bool,
    use_bf16: bool,
    require_torchrun_for_multiple_gpus: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    if use_fp16 and use_bf16:
        raise ValueError("USE_FP16 and USE_BF16 cannot both be enabled.")

    gpu_count = torch.cuda.device_count()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "-1"))
    launched_with_torchrun = world_size > 1 and local_rank >= 0 and rank >= 0

    print(f"PyTorch version: {torch.__version__}")
    print(f"Visible CUDA devices: {gpu_count}")

    for index in range(gpu_count):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

    if (
        require_torchrun_for_multiple_gpus
        and gpu_count > 1
        and not launched_with_torchrun
    ):
        raise RuntimeError(
            "Multiple GPUs are visible, but the script was launched with "
            "ordinary Python. Use torchrun."
        )


# File and fold helpers

def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_manifest(
    path: Path,
    required_columns: Iterable[str] = (
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


def extract_fold_number(path: Path) -> int:
    match = re.fullmatch(r"fold_(\d+)", path.name)

    if match is None:
        raise ValueError(f"Invalid fold directory: {path}")

    return int(match.group(1))


def discover_fold_directories(
    processed_root: Path,
    run_folds: Sequence[int] | None = None,
) -> list[Path]:
    fold_directories = [
        path
        for path in processed_root.glob("fold_*")
        if path.is_dir() and re.fullmatch(r"fold_\d+", path.name)
    ]
    fold_directories.sort(key=extract_fold_number)

    if run_folds is not None:
        selected = set(run_folds)
        fold_directories = [
            path
            for path in fold_directories
            if extract_fold_number(path) in selected
        ]

    if not fold_directories:
        raise RuntimeError(f"No folds found under:\n{processed_root}")

    return fold_directories


def select_subset(
    rows: Sequence[dict[str, Any]],
    maximum_rows: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows]
    random.Random(seed).shuffle(selected)

    if maximum_rows is not None:
        selected = selected[:maximum_rows]

    return selected


def validate_manifest_disjointness(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
) -> None:
    train_audio = {str(row["audio_path"]) for row in train_rows}
    validation_audio = {
        str(row["audio_path"])
        for row in validation_rows
    }
    train_speakers = {
        str(row["speaker"]).strip()
        for row in train_rows
    }
    validation_speakers = {
        str(row["speaker"]).strip()
        for row in validation_rows
    }

    if train_audio & validation_audio:
        raise RuntimeError("Train/validation audio leakage detected.")

    if train_speakers & validation_speakers:
        raise RuntimeError(
            "Train/validation speaker leakage detected: "
            f"{sorted(train_speakers & validation_speakers)}"
        )

    for task in ("sentence", "passage"):
        train_prompts = {
            str(row["prompt_key"]).strip()
            for row in train_rows
            if str(row["task"]).strip().casefold() == task
        }
        validation_prompts = {
            str(row["prompt_key"]).strip()
            for row in validation_rows
            if str(row["task"]).strip().casefold() == task
        }

        if train_prompts & validation_prompts:
            raise RuntimeError(
                f"Train/validation {task} prompt leakage detected: "
                f"{sorted(train_prompts & validation_prompts)}"
            )


# Normalisation and WER

class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: str) -> str:
        self.add(item)

        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])

        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)

        if first_root != second_root:
            self.parent[second_root] = first_root


def normalise_mapping_token(token: str) -> str:
    return unicodedata.normalize("NFKC", str(token)).strip().casefold()


def tokenise_transcript(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("_", " ")
    characters = [
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    ]
    cleaned = " ".join("".join(characters).split())
    return cleaned.split() if cleaned else []


def parse_equivalence_mapping(section_text: str) -> dict[str, str]:
    pattern = re.compile(
        r"^\s*\[([^\]]+)\]\s*=>\s*\[\{([^}]+)\}\]\s*$",
        flags=re.MULTILINE,
    )
    union_find = UnionFind()
    preferred_forms: list[str] = []
    matches = list(pattern.finditer(section_text))

    if not matches:
        raise ValueError("No mappings found in a normalisation section.")

    for match in matches:
        source = normalise_mapping_token(match.group(1))
        alternatives = [
            normalise_mapping_token(value)
            for value in match.group(2).split("/")
            if normalise_mapping_token(value)
        ]

        if not alternatives:
            continue

        preferred = alternatives[0]
        preferred_forms.append(preferred)
        union_find.add(source)

        for alternative in alternatives:
            union_find.union(preferred, alternative)
            union_find.union(preferred, source)

    preferred_by_root: dict[str, str] = {}

    for preferred in preferred_forms:
        preferred_by_root.setdefault(union_find.find(preferred), preferred)

    return {
        token: preferred_by_root.get(
            union_find.find(token),
            union_find.find(token),
        )
        for token in union_find.parent
    }


def load_task_mappings(
    normalisation_file: Path | None,
) -> dict[str, dict[str, str]]:
    if normalisation_file is None:
        return {"sentence": {}, "passage": {}}

    if not normalisation_file.exists():
        raise FileNotFoundError(
            f"Normalisation file not found:\n{normalisation_file}"
        )

    text = normalisation_file.read_text(encoding="utf-8-sig")
    passage_marker = "SECTION 1 - PASSAGE RECORDINGS ONLY"
    sentence_marker = "SECTION 2 - SENTENCE-TASK RECORDINGS ONLY"
    passage_position = text.find(passage_marker)
    sentence_position = text.find(sentence_marker)

    if passage_position == -1 or sentence_position == -1:
        raise ValueError(
            "Normalisation file is missing its sentence/passage markers."
        )

    return {
        "passage": parse_equivalence_mapping(
            text[passage_position:sentence_position]
        ),
        "sentence": parse_equivalence_mapping(text[sentence_position:]),
    }


def canonicalise_tokens(
    tokens: Iterable[str],
    mapping: dict[str, str],
) -> list[str]:
    return [
        mapping.get(normalise_mapping_token(token), normalise_mapping_token(token))
        for token in tokens
    ]


def calculate_edit_counts(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> tuple[int, int, int]:
    reference_length = len(reference)
    hypothesis_length = len(hypothesis)
    costs = [
        [0] * (hypothesis_length + 1)
        for _ in range(reference_length + 1)
    ]
    operations = [
        [""] * (hypothesis_length + 1)
        for _ in range(reference_length + 1)
    ]

    for index in range(1, reference_length + 1):
        costs[index][0] = index
        operations[index][0] = "D"

    for index in range(1, hypothesis_length + 1):
        costs[0][index] = index
        operations[0][index] = "I"

    for reference_index in range(1, reference_length + 1):
        for hypothesis_index in range(1, hypothesis_length + 1):
            equal = (
                reference[reference_index - 1]
                == hypothesis[hypothesis_index - 1]
            )
            substitution = (
                costs[reference_index - 1][hypothesis_index - 1]
                + (0 if equal else 1)
            )
            deletion = costs[reference_index - 1][hypothesis_index] + 1
            insertion = costs[reference_index][hypothesis_index - 1] + 1
            best = min(substitution, deletion, insertion)
            costs[reference_index][hypothesis_index] = best

            if substitution == best:
                operations[reference_index][hypothesis_index] = (
                    "C" if equal else "S"
                )
            elif deletion == best:
                operations[reference_index][hypothesis_index] = "D"
            else:
                operations[reference_index][hypothesis_index] = "I"

    substitutions = deletions = insertions = 0
    reference_index = reference_length
    hypothesis_index = hypothesis_length

    while reference_index > 0 or hypothesis_index > 0:
        operation = operations[reference_index][hypothesis_index]

        if operation == "C":
            reference_index -= 1
            hypothesis_index -= 1
        elif operation == "S":
            substitutions += 1
            reference_index -= 1
            hypothesis_index -= 1
        elif operation == "D":
            deletions += 1
            reference_index -= 1
        elif operation == "I":
            insertions += 1
            hypothesis_index -= 1
        elif reference_index > 0:
            deletions += 1
            reference_index -= 1
        else:
            insertions += 1
            hypothesis_index -= 1

    return substitutions, deletions, insertions


def calculate_corpus_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    task_names: Sequence[str],
    task_mappings: dict[str, dict[str, str]],
) -> dict[str, float | int]:
    totals = {
        "raw_substitutions": 0,
        "raw_deletions": 0,
        "raw_insertions": 0,
        "raw_reference_words": 0,
        "normalised_substitutions": 0,
        "normalised_deletions": 0,
        "normalised_insertions": 0,
        "normalised_reference_words": 0,
    }

    for reference, hypothesis, task in zip(
        references,
        hypotheses,
        task_names,
        strict=True,
    ):
        raw_reference = tokenise_transcript(reference)
        raw_hypothesis = tokenise_transcript(hypothesis)
        substitutions, deletions, insertions = calculate_edit_counts(
            raw_reference,
            raw_hypothesis,
        )
        totals["raw_substitutions"] += substitutions
        totals["raw_deletions"] += deletions
        totals["raw_insertions"] += insertions
        totals["raw_reference_words"] += len(raw_reference)

        mapping = task_mappings.get(task, {})
        normalised_reference = canonicalise_tokens(raw_reference, mapping)
        normalised_hypothesis = canonicalise_tokens(raw_hypothesis, mapping)
        substitutions, deletions, insertions = calculate_edit_counts(
            normalised_reference,
            normalised_hypothesis,
        )
        totals["normalised_substitutions"] += substitutions
        totals["normalised_deletions"] += deletions
        totals["normalised_insertions"] += insertions
        totals["normalised_reference_words"] += len(normalised_reference)

    raw_errors = (
        totals["raw_substitutions"]
        + totals["raw_deletions"]
        + totals["raw_insertions"]
    )
    normalised_errors = (
        totals["normalised_substitutions"]
        + totals["normalised_deletions"]
        + totals["normalised_insertions"]
    )

    return {
        "wer": (
            raw_errors / totals["raw_reference_words"] * 100.0
            if totals["raw_reference_words"]
            else 0.0
        ),
        "substitutions": totals["raw_substitutions"],
        "deletions": totals["raw_deletions"],
        "insertions": totals["raw_insertions"],
        "reference_words": totals["raw_reference_words"],
        "normalised_wer": (
            normalised_errors
            / totals["normalised_reference_words"]
            * 100.0
            if totals["normalised_reference_words"]
            else 0.0
        ),
        "normalised_substitutions": totals["normalised_substitutions"],
        "normalised_deletions": totals["normalised_deletions"],
        "normalised_insertions": totals["normalised_insertions"],
        "normalised_reference_words": totals["normalised_reference_words"],
        "errors_removed_by_normalisation": raw_errors - normalised_errors,
    }


# Dataset and collator

class WhisperManifestDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        processor: WhisperProcessor,
    ) -> None:
        self.rows = list(rows)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio = load_wav_for_whisper(
            Path(str(row["audio_path"])),
            target_sample_rate=TARGET_SAMPLE_RATE,
            print_audio_check=False,
        )
        processed = self.processor.feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_attention_mask=True,
            return_tensors="np",
        )
        labels = self.processor.tokenizer(
            str(row["transcript"]),
            add_special_tokens=True,
        ).input_ids

        return {
            "input_features": processed.input_features[0],
            "attention_mask": processed.attention_mask[0],
            "labels": labels,
        }


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        audio_features = [
            {
                "input_features": feature["input_features"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            audio_features,
            return_tensors="pt",
        )
        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        if (
            labels.shape[1] > 0
            and torch.all(
                labels[:, 0] == self.decoder_start_token_id
            ).item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_compute_metrics(
    processor: WhisperProcessor,
    validation_rows: Sequence[dict[str, Any]],
    task_mappings: dict[str, dict[str, str]],
) -> Callable[[Any], dict[str, float | int]]:
    """Build validation metrics without assuming a particular CV fold count.

    In distributed evaluation, PyTorch/Hugging Face may pad the validation
    sampler so every rank receives the same number of examples. For example,
    111 validation recordings on 2 GPUs can temporarily produce 112 gathered
    predictions. The extra item is sampler padding, not a real validation
    recording.

    The manifest remains the source of truth. We therefore trim gathered
    predictions/references back to the manifest length before pairing them
    with task labels.
    """
    tasks = [
        str(row["task"]).strip().casefold()
        for row in validation_rows
    ]
    expected_examples = len(tasks)

    if expected_examples == 0:
        raise ValueError("validation_rows must not be empty.")

    def compute_metrics(output: Any) -> dict[str, float | int]:
        predictions = output.predictions
        labels = output.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = np.where(
            labels == -100,
            processor.tokenizer.pad_token_id,
            labels,
        )

        hypotheses = processor.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        references = processor.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if len(hypotheses) != len(references):
            raise RuntimeError(
                "Validation prediction/reference count mismatch: "
                f"{len(hypotheses)} hypotheses vs "
                f"{len(references)} references."
            )

        if len(hypotheses) < expected_examples:
            raise RuntimeError(
                "Distributed validation returned fewer examples than "
                "the manifest contains: "
                f"{len(hypotheses)} predictions for "
                f"{expected_examples} manifest rows."
            )

        if len(hypotheses) > expected_examples:
            extra = len(hypotheses) - expected_examples
            print(
                "Validation gather contains "
                f"{extra} padded distributed sample(s); "
                "trimming back to manifest length."
            )
            hypotheses = hypotheses[:expected_examples]
            references = references[:expected_examples]

        return calculate_corpus_metrics(
            references,
            hypotheses,
            tasks,
            task_mappings,
        )

    return compute_metrics


class ValidationMetricsCallback(TrainerCallback):
    def __init__(
        self,
        experiment_name: str,
        fold_number: int,
        metrics_path: Path,
    ) -> None:
        self.experiment_name = experiment_name
        self.fold_number = fold_number
        self.metrics_path = metrics_path

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs,
    ):
        if (
            not metrics
            or not state.is_world_process_zero
            or "eval_normalised_wer" not in metrics
        ):
            return control

        row = {
            "experiment": self.experiment_name,
            "fold": self.fold_number,
            "epoch": state.epoch,
            "global_step": state.global_step,
            "validation_loss": metrics.get("eval_loss"),
            "raw_wer_percent": metrics.get("eval_wer"),
            "normalised_wer_percent": metrics.get("eval_normalised_wer"),
            "normalised_substitutions": metrics.get(
                "eval_normalised_substitutions"
            ),
            "normalised_deletions": metrics.get(
                "eval_normalised_deletions"
            ),
            "normalised_insertions": metrics.get(
                "eval_normalised_insertions"
            ),
            "normalised_reference_words": metrics.get(
                "eval_normalised_reference_words"
            ),
        }
        append_csv(self.metrics_path, row)
        return control


# Model, trainer, and reporting helpers

def configure_whisper_generation(
    model: WhisperForConditionalGeneration,
    language: str,
    task: str,
    use_cache: bool,
) -> None:
    model.config.use_cache = use_cache
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    if hasattr(model.generation_config, "return_timestamps"):
        model.generation_config.return_timestamps = False


def build_training_arguments(
    checkpoint_directory: Path,
    seed: int,
    *,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    train_batch_size: int,
    validation_batch_size: int,
    gradient_accumulation_steps: int,
    use_fp16: bool,
    use_bf16: bool,
    use_gradient_checkpointing: bool,
    logging_steps: int,
    save_total_limit: int,
    generation_max_length: int,
    generation_num_beams: int,
    dataloader_workers: int,
    lr_scheduler_type: str = "linear",
    max_grad_norm: float = 1.0,
    eval_accumulation_steps: int = 1,
) -> Seq2SeqTrainingArguments:
    values: dict[str, Any] = {
        "output_dir": str(checkpoint_directory),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": validation_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": lr_scheduler_type,
        "max_grad_norm": max_grad_norm,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": use_gradient_checkpointing,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "normalised_wer",
        "greater_is_better": False,
        "predict_with_generate": True,
        "generation_max_length": generation_max_length,
        "generation_num_beams": generation_num_beams,
        "eval_accumulation_steps": eval_accumulation_steps,
        "remove_unused_columns": False,
        "dataloader_num_workers": dataloader_workers,
        "dataloader_pin_memory": True,
        "optim": "adamw_torch",
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "ddp_find_unused_parameters": False,
    }

    parameters = inspect.signature(
        Seq2SeqTrainingArguments.__init__
    ).parameters

    if "gradient_checkpointing_kwargs" in parameters:
        values["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False
        }

    if "eval_strategy" in parameters:
        values["eval_strategy"] = "epoch"
    else:
        values["evaluation_strategy"] = "epoch"

    return Seq2SeqTrainingArguments(**values)


def make_trainer(
    *,
    model: torch.nn.Module,
    training_args: Seq2SeqTrainingArguments,
    processor: WhisperProcessor,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    collator: WhisperDataCollator,
    compute_metrics: Callable[[Any], dict[str, float | int]],
    callbacks: Sequence[TrainerCallback] = (),
) -> Seq2SeqTrainer:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": list(callbacks),
    }

    parameters = inspect.signature(Seq2SeqTrainer.__init__).parameters

    if "processing_class" in parameters:
        kwargs["processing_class"] = processor
    else:
        kwargs["tokenizer"] = processor.feature_extractor

    return Seq2SeqTrainer(**kwargs)


def parameter_summary(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": trainable / total * 100.0 if total else 0.0,
    }
