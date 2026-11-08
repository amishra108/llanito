from __future__ import annotations

import csv
import inspect
import json
import os
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainerCallback,
    WhisperProcessor,
)

from scripts.data_utils import (
    TARGET_SAMPLE_RATE,
    load_wav_for_whisper,
)


# Process helpers

def is_main_process() -> bool:
    return int(
        os.environ.get(
            "RANK",
            "0",
        )
    ) == 0


def distributed_barrier() -> None:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()


def set_seed(
    seed: int,
) -> None:
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
        raise RuntimeError(
            "CUDA is not available to PyTorch."
        )

    if use_fp16 and use_bf16:
        raise ValueError(
            "FP16 and BF16 cannot both be enabled."
        )

    gpu_count = torch.cuda.device_count()

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "-1",
        )
    )

    rank = int(
        os.environ.get(
            "RANK",
            "-1",
        )
    )

    launched_with_torchrun = (
        world_size > 1
        and local_rank >= 0
        and rank >= 0
    )

    if is_main_process():
        print(
            f"PyTorch version: "
            f"{torch.__version__}"
        )

        print(
            f"Visible CUDA devices: "
            f"{gpu_count}"
        )

        print(
            f"Distributed world size: "
            f"{world_size}"
        )

        for index in range(gpu_count):
            print(
                f"GPU {index}: "
                f"{torch.cuda.get_device_name(index)}"
            )

    if (
        require_torchrun_for_multiple_gpus
        and gpu_count > 1
        and not launched_with_torchrun
    ):
        raise RuntimeError(
            "Multiple GPUs are visible, but the script "
            "was launched with ordinary Python. "
            "Use torchrun."
        )


# Generic serialisation helpers

def make_json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if is_dataclass(value):
        return make_json_safe(
            asdict(value)
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        torch.dtype,
    ):
        return str(value)

    return value


def save_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            make_json_safe(value),
            output_file,
            ensure_ascii=False,
            indent=2,
        )

        output_file.write("\n")


def append_csv(
    path: Path,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(row),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            make_json_safe(row)
        )


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = list(rows)

    if fieldnames is None:
        if not rows:
            raise ValueError(
                "fieldnames must be supplied when "
                "writing an empty CSV."
            )

        fieldnames = list(
            rows[0].keys()
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                make_json_safe(row)
            )


# Manifest helpers

def load_manifest(
    path: Path,
    required_columns: Iterable[str] = (
        "audio_path",
        "transcript",
    ),
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest does not exist:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(
            input_file
        )

        if reader.fieldnames is None:
            raise ValueError(
                f"Manifest has no header:\n{path}"
            )

        missing = (
            set(required_columns)
            - set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                f"{sorted(missing)}"
            )

        rows = [
            dict(row)
            for row in reader
        ]

    if not rows:
        raise RuntimeError(
            f"Manifest is empty:\n{path}"
        )

    return rows


def manifest_audio_keys(
    rows: Sequence[dict[str, Any]],
    audio_column: str = "audio_path",
) -> set[str]:
    return {
        str(row[audio_column]).strip()
        for row in rows
    }


def validate_no_audio_overlap(
    first_rows: Sequence[dict[str, Any]],
    second_rows: Sequence[dict[str, Any]],
    *,
    first_name: str,
    second_name: str,
    audio_column: str = "audio_path",
) -> None:
    first_audio = manifest_audio_keys(
        first_rows,
        audio_column=audio_column,
    )

    second_audio = manifest_audio_keys(
        second_rows,
        audio_column=audio_column,
    )

    overlap = (
        first_audio
        & second_audio
    )

    if overlap:
        preview = sorted(
            overlap
        )[:20]

        raise RuntimeError(
            f"{first_name}/{second_name} audio leakage "
            f"detected: {len(overlap)} overlapping "
            f"recording(s).\n"
            f"First overlaps: {preview}"
        )


def validate_no_duplicate_audio(
    rows: Sequence[dict[str, Any]],
    *,
    dataset_name: str,
    audio_column: str = "audio_path",
) -> None:
    values = [
        str(row[audio_column]).strip()
        for row in rows
    ]

    unique = set(values)

    if len(values) != len(unique):
        raise RuntimeError(
            f"{dataset_name} contains "
            f"{len(values) - len(unique)} duplicate "
            f"audio row(s)."
        )


def validate_manifest_files_exist(
    rows: Sequence[dict[str, Any]],
    *,
    audio_column: str = "audio_path",
    maximum_missing_to_print: int = 20,
) -> None:
    missing: list[str] = []

    for row in rows:
        audio_path = Path(
            str(row[audio_column])
        )

        if not audio_path.exists():
            missing.append(
                str(audio_path)
            )

    if missing:
        preview = missing[
            :maximum_missing_to_print
        ]

        raise FileNotFoundError(
            f"{len(missing)} audio file(s) from the "
            f"manifest do not exist.\n"
            f"First missing files:\n"
            + "\n".join(preview)
        )


def extract_fold_number(
    path: Path,
) -> int:
    match = re.fullmatch(
        r"fold_(\d+)",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"Invalid fold directory: {path}"
        )

    return int(
        match.group(1)
    )


def discover_fold_directories(
    processed_root: Path,
    run_folds: Sequence[int] | None = None,
    fold_pattern: str = "fold_*",
) -> list[Path]:
    fold_directories = [
        path
        for path in processed_root.glob(
            fold_pattern
        )
        if (
            path.is_dir()
            and re.fullmatch(
                r"fold_\d+",
                path.name,
            )
        )
    ]

    fold_directories.sort(
        key=extract_fold_number
    )

    if run_folds is not None:
        selected = set(
            run_folds
        )

        fold_directories = [
            path
            for path in fold_directories
            if extract_fold_number(path)
            in selected
        ]

    if not fold_directories:
        raise RuntimeError(
            f"No folds found under:\n"
            f"{processed_root}"
        )

    return fold_directories


def select_subset(
    rows: Sequence[dict[str, Any]],
    maximum_rows: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    selected = [
        dict(row)
        for row in rows
    ]

    random.Random(
        seed
    ).shuffle(
        selected
    )

    if maximum_rows is not None:
        selected = selected[
            :maximum_rows
        ]

    return selected


def validate_manifest_disjointness(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    *,
    check_audio: bool = True,
    check_speakers: bool = False,
    check_prompts: bool = False,
    audio_column: str = "audio_path",
    speaker_column: str = "speaker",
    task_column: str = "task",
    prompt_column: str = "prompt_key",
) -> None:
    if check_audio:
        validate_no_audio_overlap(
            train_rows,
            validation_rows,
            first_name="train",
            second_name="validation",
            audio_column=audio_column,
        )

    if check_speakers:
        train_speakers = {
            str(
                row[speaker_column]
            ).strip()
            for row in train_rows
        }

        validation_speakers = {
            str(
                row[speaker_column]
            ).strip()
            for row in validation_rows
        }

        overlap = (
            train_speakers
            & validation_speakers
        )

        if overlap:
            raise RuntimeError(
                "Train/validation speaker leakage "
                f"detected: {sorted(overlap)}"
            )

    if check_prompts:
        train_pairs = {
            (
                str(
                    row.get(
                        task_column,
                        "",
                    )
                ).strip().casefold(),
                str(
                    row.get(
                        prompt_column,
                        "",
                    )
                ).strip(),
            )
            for row in train_rows
        }

        validation_pairs = {
            (
                str(
                    row.get(
                        task_column,
                        "",
                    )
                ).strip().casefold(),
                str(
                    row.get(
                        prompt_column,
                        "",
                    )
                ).strip(),
            )
            for row in validation_rows
        }

        overlap = (
            train_pairs
            & validation_pairs
        )

        if overlap:
            raise RuntimeError(
                "Train/validation task+prompt "
                "leakage detected: "
                f"{sorted(overlap)}"
            )


# Normalisation

class UnionFind:
    def __init__(
        self,
    ) -> None:
        self.parent: dict[
            str,
            str,
        ] = {}

    def add(
        self,
        item: str,
    ) -> None:
        if item not in self.parent:
            self.parent[item] = item

    def find(
        self,
        item: str,
    ) -> str:
        self.add(
            item
        )

        if (
            self.parent[item]
            != item
        ):
            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(
        self,
        first: str,
        second: str,
    ) -> None:
        first_root = self.find(
            first
        )

        second_root = self.find(
            second
        )

        if (
            first_root
            != second_root
        ):
            self.parent[
                second_root
            ] = first_root


def normalise_mapping_token(
    token: str,
) -> str:
    return unicodedata.normalize(
        "NFKC",
        str(token),
    ).strip().casefold()


def tokenise_transcript(
    text: str,
) -> list[str]:
    text = (
        unicodedata.normalize(
            "NFKC",
            str(text),
        )
        .casefold()
        .replace(
            "_",
            " ",
        )
    )

    characters = [
        (
            " "
            if unicodedata.category(
                character
            ).startswith("P")
            else character
        )
        for character in text
    ]

    cleaned = " ".join(
        "".join(
            characters
        ).split()
    )

    return (
        cleaned.split()
        if cleaned
        else []
    )


def parse_equivalence_mapping(
    section_text: str,
) -> dict[str, str]:
    pattern = re.compile(
        r"^\s*\[([^\]]+)\]\s*=>\s*"
        r"\[\{([^}]+)\}\]\s*$",
        flags=re.MULTILINE,
    )

    union_find = UnionFind()

    preferred_forms: list[
        str
    ] = []

    matches = list(
        pattern.finditer(
            section_text
        )
    )

    if not matches:
        raise ValueError(
            "No mappings found in a "
            "normalisation section."
        )

    for match in matches:
        source = normalise_mapping_token(
            match.group(1)
        )

        alternatives = [
            normalise_mapping_token(
                value
            )
            for value
            in match.group(2).split("/")
            if normalise_mapping_token(
                value
            )
        ]

        if not alternatives:
            continue

        preferred = alternatives[0]

        preferred_forms.append(
            preferred
        )

        union_find.add(
            source
        )

        for alternative in alternatives:
            union_find.union(
                preferred,
                alternative,
            )

            union_find.union(
                preferred,
                source,
            )

    preferred_by_root: dict[
        str,
        str,
    ] = {}

    for preferred in preferred_forms:
        preferred_by_root.setdefault(
            union_find.find(
                preferred
            ),
            preferred,
        )

    return {
        token: preferred_by_root.get(
            union_find.find(
                token
            ),
            union_find.find(
                token
            ),
        )
        for token
        in union_find.parent
    }


def load_task_mappings(
    normalisation_file: Path | None,
    *,
    section_markers: Mapping[
        str,
        str,
    ] | None = None,
) -> dict[
    str,
    dict[str, str],
]:
    if normalisation_file is None:
        return {}

    if not normalisation_file.exists():
        raise FileNotFoundError(
            "Normalisation file not found:\n"
            f"{normalisation_file}"
        )

    text = normalisation_file.read_text(
        encoding="utf-8-sig"
    )

    if section_markers is None:
        section_markers = {
            "passage": (
                "SECTION 1 - "
                "PASSAGE RECORDINGS ONLY"
            ),
            "sentence": (
                "SECTION 2 - "
                "SENTENCE-TASK RECORDINGS ONLY"
            ),
        }

    marker_positions: list[
        tuple[int, str, str]
    ] = []

    for task_name, marker in (
        section_markers.items()
    ):
        position = text.find(
            marker
        )

        if position == -1:
            raise ValueError(
                "Normalisation file is missing "
                f"the section marker for "
                f"{task_name!r}:\n{marker}"
            )

        marker_positions.append(
            (
                position,
                task_name,
                marker,
            )
        )

    marker_positions.sort()

    mappings: dict[
        str,
        dict[str, str],
    ] = {}

    for index, (
        position,
        task_name,
        _,
    ) in enumerate(
        marker_positions
    ):
        if (
            index + 1
            < len(
                marker_positions
            )
        ):
            end = marker_positions[
                index + 1
            ][0]
        else:
            end = len(
                text
            )

        section = text[
            position:end
        ]

        mappings[
            task_name
        ] = parse_equivalence_mapping(
            section
        )

    return mappings


def canonicalise_tokens(
    tokens: Iterable[str],
    mapping: Mapping[str, str] | None,
) -> list[str]:
    mapping = (
        mapping
        if mapping is not None
        else {}
    )

    return [
        mapping.get(
            normalise_mapping_token(
                token
            ),
            normalise_mapping_token(
                token
            ),
        )
        for token in tokens
    ]


# Edit distance and WER

def calculate_edit_counts(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> tuple[
    int,
    int,
    int,
]:
    reference_length = len(
        reference
    )

    hypothesis_length = len(
        hypothesis
    )

    costs = [
        [0]
        * (
            hypothesis_length
            + 1
        )
        for _ in range(
            reference_length
            + 1
        )
    ]

    operations = [
        [""]
        * (
            hypothesis_length
            + 1
        )
        for _ in range(
            reference_length
            + 1
        )
    ]

    for index in range(
        1,
        reference_length + 1,
    ):
        costs[index][0] = index
        operations[index][0] = "D"

    for index in range(
        1,
        hypothesis_length + 1,
    ):
        costs[0][index] = index
        operations[0][index] = "I"

    for reference_index in range(
        1,
        reference_length + 1,
    ):
        for hypothesis_index in range(
            1,
            hypothesis_length + 1,
        ):
            equal = (
                reference[
                    reference_index - 1
                ]
                == hypothesis[
                    hypothesis_index - 1
                ]
            )

            substitution = (
                costs[
                    reference_index - 1
                ][
                    hypothesis_index - 1
                ]
                + (
                    0
                    if equal
                    else 1
                )
            )

            deletion = (
                costs[
                    reference_index - 1
                ][
                    hypothesis_index
                ]
                + 1
            )

            insertion = (
                costs[
                    reference_index
                ][
                    hypothesis_index - 1
                ]
                + 1
            )

            best = min(
                substitution,
                deletion,
                insertion,
            )

            costs[
                reference_index
            ][
                hypothesis_index
            ] = best

            if substitution == best:
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = (
                    "C"
                    if equal
                    else "S"
                )

            elif deletion == best:
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "D"

            else:
                operations[
                    reference_index
                ][
                    hypothesis_index
                ] = "I"

    substitutions = 0
    deletions = 0
    insertions = 0

    reference_index = (
        reference_length
    )

    hypothesis_index = (
        hypothesis_length
    )

    while (
        reference_index > 0
        or hypothesis_index > 0
    ):
        operation = operations[
            reference_index
        ][
            hypothesis_index
        ]

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

    return (
        substitutions,
        deletions,
        insertions,
    )


def calculate_single_utterance_metrics(
    reference: str,
    hypothesis: str,
    *,
    task_name: str = "",
    task_mappings: Mapping[
        str,
        Mapping[str, str],
    ] | None = None,
) -> dict[str, Any]:
    task_mappings = (
        task_mappings
        if task_mappings is not None
        else {}
    )

    raw_reference_tokens = (
        tokenise_transcript(
            reference
        )
    )

    raw_hypothesis_tokens = (
        tokenise_transcript(
            hypothesis
        )
    )

    (
        raw_substitutions,
        raw_deletions,
        raw_insertions,
    ) = calculate_edit_counts(
        raw_reference_tokens,
        raw_hypothesis_tokens,
    )

    mapping = task_mappings.get(
        str(
            task_name
        ).strip().casefold(),
        {},
    )

    normalised_reference_tokens = (
        canonicalise_tokens(
            raw_reference_tokens,
            mapping,
        )
    )

    normalised_hypothesis_tokens = (
        canonicalise_tokens(
            raw_hypothesis_tokens,
            mapping,
        )
    )

    (
        normalised_substitutions,
        normalised_deletions,
        normalised_insertions,
    ) = calculate_edit_counts(
        normalised_reference_tokens,
        normalised_hypothesis_tokens,
    )

    raw_reference_words = len(
        raw_reference_tokens
    )

    normalised_reference_words = len(
        normalised_reference_tokens
    )

    raw_errors = (
        raw_substitutions
        + raw_deletions
        + raw_insertions
    )

    normalised_errors = (
        normalised_substitutions
        + normalised_deletions
        + normalised_insertions
    )

    return {
        "reference_original": reference,
        "hypothesis_original": hypothesis,
        "reference_post_normalisation": (
            " ".join(
                normalised_reference_tokens
            )
        ),
        "hypothesis_post_normalisation": (
            " ".join(
                normalised_hypothesis_tokens
            )
        ),
        "raw_substitutions": (
            raw_substitutions
        ),
        "raw_deletions": (
            raw_deletions
        ),
        "raw_insertions": (
            raw_insertions
        ),
        "raw_reference_words": (
            raw_reference_words
        ),
        "raw_wer_percent": (
            raw_errors
            / raw_reference_words
            * 100.0
            if raw_reference_words
            else 0.0
        ),
        "normalised_substitutions": (
            normalised_substitutions
        ),
        "normalised_deletions": (
            normalised_deletions
        ),
        "normalised_insertions": (
            normalised_insertions
        ),
        "normalised_reference_words": (
            normalised_reference_words
        ),
        "normalised_wer_percent": (
            normalised_errors
            / normalised_reference_words
            * 100.0
            if normalised_reference_words
            else 0.0
        ),
        "errors_removed_by_normalisation": (
            raw_errors
            - normalised_errors
        ),
    }


def calculate_corpus_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    task_names: Sequence[str] | None = None,
    task_mappings: Mapping[
        str,
        Mapping[str, str],
    ] | None = None,
) -> dict[str, float | int]:
    if len(
        references
    ) != len(
        hypotheses
    ):
        raise ValueError(
            "Reference/hypothesis length mismatch."
        )

    if task_names is None:
        task_names = [
            ""
        ] * len(
            references
        )

    if len(
        task_names
    ) != len(
        references
    ):
        raise ValueError(
            "task_names length does not match "
            "the references."
        )

    task_mappings = (
        task_mappings
        if task_mappings is not None
        else {}
    )

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

    for (
        reference,
        hypothesis,
        task_name,
    ) in zip(
        references,
        hypotheses,
        task_names,
        strict=True,
    ):
        metrics = (
            calculate_single_utterance_metrics(
                reference,
                hypothesis,
                task_name=task_name,
                task_mappings=task_mappings,
            )
        )

        for key in totals:
            totals[key] += int(
                metrics[key]
            )

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

    raw_reference_words = totals[
        "raw_reference_words"
    ]

    normalised_reference_words = totals[
        "normalised_reference_words"
    ]

    return {
        "wer": (
            raw_errors
            / raw_reference_words
            * 100.0
            if raw_reference_words
            else 0.0
        ),
        "substitutions": totals[
            "raw_substitutions"
        ],
        "deletions": totals[
            "raw_deletions"
        ],
        "insertions": totals[
            "raw_insertions"
        ],
        "reference_words": (
            raw_reference_words
        ),
        "normalised_wer": (
            normalised_errors
            / normalised_reference_words
            * 100.0
            if normalised_reference_words
            else 0.0
        ),
        "normalised_substitutions": totals[
            "normalised_substitutions"
        ],
        "normalised_deletions": totals[
            "normalised_deletions"
        ],
        "normalised_insertions": totals[
            "normalised_insertions"
        ],
        "normalised_reference_words": (
            normalised_reference_words
        ),
        "errors_removed_by_normalisation": (
            raw_errors
            - normalised_errors
        ),
    }


# Dataset

class WhisperManifestDataset(
    Dataset
):
    def __init__(
        self,
        rows: Sequence[
            dict[str, Any]
        ],
        processor: WhisperProcessor,
        *,
        audio_column: str = "audio_path",
        transcript_column: str = "transcript",
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        print_audio_check: bool = False,
    ) -> None:
        self.rows = list(
            rows
        )

        self.processor = processor
        self.audio_column = (
            audio_column
        )

        self.transcript_column = (
            transcript_column
        )

        self.target_sample_rate = (
            target_sample_rate
        )

        self.print_audio_check = (
            print_audio_check
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self.rows
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.rows[
            index
        ]

        audio = load_wav_for_whisper(
            Path(
                str(
                    row[
                        self.audio_column
                    ]
                )
            ),
            target_sample_rate=(
                self.target_sample_rate
            ),
            print_audio_check=(
                self.print_audio_check
            ),
        )

        processed = (
            self.processor.feature_extractor(
                audio,
                sampling_rate=(
                    self.target_sample_rate
                ),
                return_attention_mask=True,
                return_tensors="np",
            )
        )

        labels = (
            self.processor.tokenizer(
                str(
                    row[
                        self.transcript_column
                    ]
                ),
                add_special_tokens=True,
            ).input_ids
        )

        return {
            "input_features": (
                processed.input_features[
                    0
                ]
            ),
            "attention_mask": (
                processed.attention_mask[
                    0
                ]
            ),
            "labels": labels,
        }


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self,
        features: list[
            dict[str, Any]
        ],
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        audio_features = [
            {
                "input_features": (
                    feature[
                        "input_features"
                    ]
                ),
                "attention_mask": (
                    feature[
                        "attention_mask"
                    ]
                ),
            }
            for feature in features
        ]

        batch = (
            self.processor
            .feature_extractor
            .pad(
                audio_features,
                return_tensors="pt",
            )
        )

        label_features = [
            {
                "input_ids": (
                    feature[
                        "labels"
                    ]
                )
            }
            for feature in features
        ]

        labels_batch = (
            self.processor
            .tokenizer
            .pad(
                label_features,
                return_tensors="pt",
            )
        )

        labels = labels_batch[
            "input_ids"
        ].masked_fill(
            labels_batch[
                "attention_mask"
            ].ne(1),
            -100,
        )

        if (
            labels.shape[1] > 0
            and torch.all(
                labels[:, 0]
                == self.decoder_start_token_id
            ).item()
        ):
            labels = labels[
                :,
                1:,
            ]

        batch[
            "labels"
        ] = labels

        return batch


# Validation metrics

def build_compute_metrics(
    processor: WhisperProcessor,
    validation_rows: Sequence[
        dict[str, Any]
    ],
    task_mappings: Mapping[
        str,
        Mapping[str, str],
    ] | None = None,
    *,
    task_column: str = "task",
    reference_text_transform: Callable[[str], str] | None = None,
    hypothesis_text_transform: Callable[[str], str] | None = None,
) -> Callable[
    [Any],
    dict[str, float | int],
]:
    tasks = [
        str(
            row.get(
                task_column,
                "",
            )
        ).strip().casefold()
        for row in validation_rows
    ]

    expected_examples = len(
        tasks
    )

    if expected_examples == 0:
        raise ValueError(
            "validation_rows must "
            "not be empty."
        )

    task_mappings = (
        task_mappings
        if task_mappings is not None
        else {}
    )

    def compute_metrics(
        output: Any,
    ) -> dict[
        str,
        float | int,
    ]:
        predictions = (
            output.predictions
        )

        labels = (
            output.label_ids
        )

        if isinstance(
            predictions,
            tuple,
        ):
            predictions = (
                predictions[0]
            )

        labels = np.where(
            labels == -100,
            processor.tokenizer.pad_token_id,
            labels,
        )

        hypotheses = (
            processor.tokenizer
            .batch_decode(
                predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        references = (
            processor.tokenizer
            .batch_decode(
                labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

        # Optional language/script-specific canonicalisation.
        #
        # Llanito callers do not pass these functions and therefore
        # retain exactly the historical behaviour.
        #
        # For Odia:
        #   reference  -> conservative Odia cleanup
        #   hypothesis -> Bengali-to-Odia script conversion followed
        #                 by the same conservative cleanup.
        #
        # Generic NFKC/punctuation/whitespace tokenisation still
        # happens later inside calculate_corpus_metrics().

        if reference_text_transform is not None:
            references = [
                reference_text_transform(
                    text
                )
                for text in references
            ]

        if hypothesis_text_transform is not None:
            hypotheses = [
                hypothesis_text_transform(
                    text
                )
                for text in hypotheses
            ]

        if (
            len(hypotheses)
            != len(references)
        ):
            raise RuntimeError(
                "Validation prediction/reference "
                "count mismatch: "
                f"{len(hypotheses)} hypotheses vs "
                f"{len(references)} references."
            )

        if (
            len(hypotheses)
            < expected_examples
        ):
            raise RuntimeError(
                "Distributed validation returned "
                "fewer examples than the manifest "
                "contains: "
                f"{len(hypotheses)} predictions "
                f"for {expected_examples} "
                "manifest rows."
            )

        if (
            len(hypotheses)
            > expected_examples
        ):
            extra = (
                len(hypotheses)
                - expected_examples
            )

            print(
                "Validation gather contains "
                f"{extra} padded distributed "
                "sample(s); trimming back to "
                "manifest length."
            )

            hypotheses = hypotheses[
                :expected_examples
            ]

            references = references[
                :expected_examples
            ]

        return calculate_corpus_metrics(
            references,
            hypotheses,
            tasks,
            task_mappings,
        )

    return compute_metrics


class ValidationMetricsCallback(
    TrainerCallback
):
    def __init__(
        self,
        experiment_name: str,
        metrics_path: Path,
        *,
        fold_number: int | None = None,
        dataset_name: str = "validation",
    ) -> None:
        self.experiment_name = (
            experiment_name
        )

        self.fold_number = (
            fold_number
        )

        self.metrics_path = (
            metrics_path
        )

        self.dataset_name = (
            dataset_name
        )

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
        ):
            return control

        row = {
            "experiment": (
                self.experiment_name
            ),
            "dataset": (
                self.dataset_name
            ),
            "fold": (
                self.fold_number
            ),
            "epoch": state.epoch,
            "global_step": (
                state.global_step
            ),
            "validation_loss": (
                metrics.get(
                    "eval_loss"
                )
            ),
            "raw_wer_percent": (
                metrics.get(
                    "eval_wer"
                )
            ),
            "normalised_wer_percent": (
                metrics.get(
                    "eval_normalised_wer"
                )
            ),
            "substitutions": (
                metrics.get(
                    "eval_substitutions"
                )
            ),
            "deletions": (
                metrics.get(
                    "eval_deletions"
                )
            ),
            "insertions": (
                metrics.get(
                    "eval_insertions"
                )
            ),
            "reference_words": (
                metrics.get(
                    "eval_reference_words"
                )
            ),
            "normalised_substitutions": (
                metrics.get(
                    "eval_normalised_substitutions"
                )
            ),
            "normalised_deletions": (
                metrics.get(
                    "eval_normalised_deletions"
                )
            ),
            "normalised_insertions": (
                metrics.get(
                    "eval_normalised_insertions"
                )
            ),
            "normalised_reference_words": (
                metrics.get(
                    "eval_normalised_reference_words"
                )
            ),
        }

        append_csv(
            self.metrics_path,
            row,
        )

        return control


# Whisper generation

def configure_whisper_generation(
    model: torch.nn.Module,
    *,
    language: str | None,
    task: str | None,
    use_cache: bool,
    forced_decoder_ids: Any = None,
    return_timestamps: bool = False,
) -> None:
    if hasattr(
        model,
        "config",
    ):
        model.config.use_cache = (
            use_cache
        )

    generation_config = getattr(
        model,
        "generation_config",
        None,
    )

    if generation_config is None:
        return

    if language is not None:
        generation_config.language = (
            language
        )

    if task is not None:
        generation_config.task = (
            task
        )

    if hasattr(
        generation_config,
        "forced_decoder_ids",
    ):
        generation_config.forced_decoder_ids = (
            forced_decoder_ids
        )

    if hasattr(
        generation_config,
        "return_timestamps",
    ):
        generation_config.return_timestamps = (
            return_timestamps
        )


# Training arguments

def build_training_arguments(
    checkpoint_directory: Path,
    seed: int,
    *,
    epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    use_fp16: bool,
    use_bf16: bool,
    use_gradient_checkpointing: bool,
    logging_steps: int,
    save_total_limit: int,
    dataloader_workers: int,
    warmup_ratio: float = 0.0,
    validation_batch_size: int | None = None,
    has_validation: bool = True,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool | None = None,
    metric_for_best_model: str | None = "normalised_wer",
    greater_is_better: bool | None = False,
    predict_with_generate: bool = True,
    generation_max_length: int = 225,
    generation_num_beams: int = 1,
    lr_scheduler_type: str = "linear",
    max_grad_norm: float = 1.0,
    eval_accumulation_steps: int | None = 1,
    optimizer: str = "adamw_torch",
    report_to: str | list[str] = "none",
    remove_unused_columns: bool = False,
    dataloader_pin_memory: bool = True,
    ddp_find_unused_parameters: bool | None = False,
    save_only_model: bool = False,
    additional_arguments: Mapping[
        str,
        Any,
    ] | None = None,
) -> Seq2SeqTrainingArguments:
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        warmup_steps > 0
        and warmup_ratio > 0.0
    ):
        raise ValueError(
            "Specify warmup_steps or warmup_ratio, "
            "not both."
        )

    if not (
        0.0
        <= warmup_ratio
        <= 1.0
    ):
        raise ValueError(
            "warmup_ratio must be between "
            "0.0 and 1.0."
        )

    if validation_batch_size is None:
        validation_batch_size = (
            train_batch_size
        )

    if not has_validation:
        evaluation_strategy = "no"
        load_best_model_at_end = False
        metric_for_best_model = None
        greater_is_better = None

    elif load_best_model_at_end is None:
        load_best_model_at_end = True

    if (
        load_best_model_at_end
        and evaluation_strategy == "no"
    ):
        raise ValueError(
            "load_best_model_at_end=True "
            "requires validation/evaluation."
        )

    if (
        load_best_model_at_end
        and save_strategy
        != evaluation_strategy
    ):
        raise ValueError(
            "When load_best_model_at_end=True, "
            "save_strategy and "
            "evaluation_strategy should match."
        )

    values: dict[
        str,
        Any,
    ] = {
        "output_dir": str(
            checkpoint_directory
        ),
        "num_train_epochs": (
            epochs
        ),
        "per_device_train_batch_size": (
            train_batch_size
        ),
        "per_device_eval_batch_size": (
            validation_batch_size
        ),
        "gradient_accumulation_steps": (
            gradient_accumulation_steps
        ),
        "learning_rate": (
            learning_rate
        ),
        "weight_decay": (
            weight_decay
        ),
        "warmup_steps": (
            warmup_steps
        ),
        "warmup_ratio": (
            warmup_ratio
        ),
        "lr_scheduler_type": (
            lr_scheduler_type
        ),
        "max_grad_norm": (
            max_grad_norm
        ),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": (
            use_gradient_checkpointing
        ),
        "logging_strategy": (
            "steps"
        ),
        "logging_steps": (
            logging_steps
        ),
        "logging_first_step": (
            True
        ),
        "save_strategy": (
            save_strategy
        ),
        "save_total_limit": (
            save_total_limit
        ),
        "load_best_model_at_end": (
            load_best_model_at_end
        ),
        "predict_with_generate": (
            predict_with_generate
        ),
        "generation_max_length": (
            generation_max_length
        ),
        "generation_num_beams": (
            generation_num_beams
        ),
        "remove_unused_columns": (
            remove_unused_columns
        ),
        "dataloader_num_workers": (
            dataloader_workers
        ),
        "dataloader_pin_memory": (
            dataloader_pin_memory
        ),
        "optim": optimizer,
        "report_to": report_to,
        "seed": seed,
        "data_seed": seed,
        "save_only_model": (
            save_only_model
        ),
    }

    if (
        eval_accumulation_steps
        is not None
    ):
        values[
            "eval_accumulation_steps"
        ] = (
            eval_accumulation_steps
        )

    if (
        ddp_find_unused_parameters
        is not None
    ):
        values[
            "ddp_find_unused_parameters"
        ] = (
            ddp_find_unused_parameters
        )

    if (
        metric_for_best_model
        is not None
    ):
        values[
            "metric_for_best_model"
        ] = (
            metric_for_best_model
        )

    if (
        greater_is_better
        is not None
    ):
        values[
            "greater_is_better"
        ] = (
            greater_is_better
        )

    parameters = inspect.signature(
        Seq2SeqTrainingArguments.__init__
    ).parameters

    if (
        "gradient_checkpointing_kwargs"
        in parameters
    ):
        values[
            "gradient_checkpointing_kwargs"
        ] = {
            "use_reentrant": False
        }

    if (
        "eval_strategy"
        in parameters
    ):
        values[
            "eval_strategy"
        ] = (
            evaluation_strategy
        )

    else:
        values[
            "evaluation_strategy"
        ] = (
            evaluation_strategy
        )

    if additional_arguments:
        values.update(
            additional_arguments
        )

    unsupported = [
        key
        for key in values
        if key not in parameters
    ]

    for key in unsupported:
        values.pop(
            key
        )

    return Seq2SeqTrainingArguments(
        **values
    )


# =============================================================================
# AUTOREGRESSIVE REPETITION LOOP GUARD
# =============================================================================
#
# This is intentionally a DECODING rule, not a WER correction.
#
# When enabled, generation terminates once ANY contiguous token span has
# appeared LOOP_GUARD_REPEAT_COUNT times consecutively at the end of the
# hypothesis.
#
# Example for repeat_count=4:
#
#     [A] [A] [A] [A]                     -> stop
#     [A B] [A B] [A B] [A B]             -> stop
#     [A B C] [A B C] [A B C] [A B C]     -> stop
#
# Nothing is deleted or forgiven.  The fourth copy remains in the hypothesis
# and ordinary WER is computed on the terminated sequence.
#
# Default is OFF so historical experiments do not silently change.
# Enable explicitly with:
#
#     WHISPER_LOOP_GUARD=1
#
# HPO runs below will set it explicitly.
# =============================================================================

LOOP_GUARD_ENABLED = (
    os.environ.get(
        "WHISPER_LOOP_GUARD",
        "0",
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

LOOP_GUARD_REPEAT_COUNT = int(
    os.environ.get(
        "WHISPER_LOOP_GUARD_REPEAT_COUNT",
        "4",
    )
)

LOOP_GUARD_MAX_SPAN_TOKENS = int(
    os.environ.get(
        "WHISPER_LOOP_GUARD_MAX_SPAN_TOKENS",
        "64",
    )
)

if LOOP_GUARD_REPEAT_COUNT < 2:
    raise ValueError(
        "WHISPER_LOOP_GUARD_REPEAT_COUNT must be >= 2."
    )

if LOOP_GUARD_MAX_SPAN_TOKENS < 1:
    raise ValueError(
        "WHISPER_LOOP_GUARD_MAX_SPAN_TOKENS must be >= 1."
    )


class ConsecutiveRepeatStoppingCriteria(
    StoppingCriteria
):
    """
    Stop a generated sequence when its suffix consists of the same
    contiguous token block repeated `repeat_count` consecutive times.

    The check is performed on token IDs.  It is therefore independent of
    punctuation normalisation, WER normalisation, language, and transcript
    text post-processing.
    """

    def __init__(
        self,
        *,
        repeat_count: int = LOOP_GUARD_REPEAT_COUNT,
        max_span_tokens: int = LOOP_GUARD_MAX_SPAN_TOKENS,
    ) -> None:
        super().__init__()

        self.repeat_count = int(
            repeat_count
        )

        self.max_span_tokens = int(
            max_span_tokens
        )

        self._reported = False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor | None,
        **kwargs,
    ) -> torch.BoolTensor:
        del scores, kwargs

        if input_ids.ndim != 2:
            raise RuntimeError(
                "Loop guard expected input_ids shaped [batch, sequence], "
                f"got {tuple(input_ids.shape)}."
            )

        batch_size = int(
            input_ids.shape[0]
        )

        should_stop = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )

        for batch_index in range(
            batch_size
        ):
            tokens = input_ids[
                batch_index
            ]

            sequence_length = int(
                tokens.numel()
            )

            maximum_span = min(
                self.max_span_tokens,
                sequence_length
                // self.repeat_count,
            )

            triggered_span = None

            for span in range(
                1,
                maximum_span + 1,
            ):
                suffix_length = (
                    self.repeat_count
                    * span
                )

                suffix = tokens[
                    -suffix_length:
                ]

                blocks = suffix.reshape(
                    self.repeat_count,
                    span,
                )

                # All four consecutive blocks must be token-identical.
                if bool(
                    torch.all(
                        blocks
                        == blocks[
                            0:1,
                            :,
                        ]
                    ).item()
                ):
                    should_stop[
                        batch_index
                    ] = True

                    triggered_span = span

                    break

            if (
                triggered_span is not None
                and not self._reported
            ):
                print(
                    "[WHISPER LOOP GUARD] "
                    f"stopping after "
                    f"{self.repeat_count} consecutive copies "
                    f"of a {triggered_span}-token span "
                    f"at generated sequence length "
                    f"{sequence_length}.",
                    flush=True,
                )

                self._reported = True

        return should_stop


class LoopGuardSeq2SeqTrainer(
    Seq2SeqTrainer
):
    """
    Seq2SeqTrainer with an opt-in generation-only repetition-loop guard.

    Training forward/backward is untouched.
    Teacher-forced loss is untouched.
    Calibration is untouched.
    Only autoregressive generation used for prediction/evaluation changes.
    """

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
        **gen_kwargs,
    ):
        if (
            LOOP_GUARD_ENABLED
            and self.args.predict_with_generate
            and not prediction_loss_only
        ):
            existing = gen_kwargs.get(
                "stopping_criteria"
            )

            if existing is None:
                criteria = (
                    StoppingCriteriaList()
                )

            else:
                criteria = (
                    StoppingCriteriaList(
                        list(
                            existing
                        )
                    )
                )

            if not any(
                isinstance(
                    criterion,
                    ConsecutiveRepeatStoppingCriteria,
                )
                for criterion
                in criteria
            ):
                criteria.append(
                    ConsecutiveRepeatStoppingCriteria()
                )

            gen_kwargs[
                "stopping_criteria"
            ] = criteria

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )


# Trainer

def make_trainer(
    *,
    model: torch.nn.Module,
    training_args: Seq2SeqTrainingArguments,
    processor: WhisperProcessor,
    train_dataset: Dataset,
    collator: WhisperDataCollator,
    validation_dataset: Dataset | None = None,
    compute_metrics: Callable[
        [Any],
        dict[str, float | int],
    ] | None = None,
    callbacks: Sequence[
        TrainerCallback
    ] = (),
) -> Seq2SeqTrainer:
    if (
        validation_dataset is None
        and compute_metrics is not None
    ):
        raise ValueError(
            "compute_metrics was supplied "
            "without a validation dataset."
        )

    kwargs: dict[
        str,
        Any,
    ] = {
        "model": model,
        "args": training_args,
        "train_dataset": (
            train_dataset
        ),
        "data_collator": (
            collator
        ),
        "callbacks": list(
            callbacks
        ),
    }

    if (
        validation_dataset
        is not None
    ):
        kwargs[
            "eval_dataset"
        ] = (
            validation_dataset
        )

    if (
        compute_metrics
        is not None
    ):
        kwargs[
            "compute_metrics"
        ] = (
            compute_metrics
        )

    parameters = inspect.signature(
        Seq2SeqTrainer.__init__
    ).parameters

    if (
        "processing_class"
        in parameters
    ):
        kwargs[
            "processing_class"
        ] = processor

    else:
        kwargs[
            "tokenizer"
        ] = (
            processor
            .feature_extractor
        )

    trainer_class = (
        LoopGuardSeq2SeqTrainer
        if LOOP_GUARD_ENABLED
        else Seq2SeqTrainer
    )

    if LOOP_GUARD_ENABLED:
        print(
            "[WHISPER LOOP GUARD] ENABLED: "
            f"stop after {LOOP_GUARD_REPEAT_COUNT} consecutive copies "
            f"of any token span up to "
            f"{LOOP_GUARD_MAX_SPAN_TOKENS} tokens.",
            flush=True,
        )

    return trainer_class(
        **kwargs
    )


# Parameters

def parameter_summary(
    model: torch.nn.Module,
) -> dict[
    str,
    int | float,
]:
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    frozen = (
        total
        - trainable
    )

    return {
        "total_parameters": (
            total
        ),
        "trainable_parameters": (
            trainable
        ),
        "frozen_parameters": (
            frozen
        ),
        "trainable_percent": (
            trainable
            / total
            * 100.0
            if total
            else 0.0
        ),
    }


def parameter_summary_by_component(
    model: torch.nn.Module,
) -> dict[
    str,
    dict[str, int | float],
]:
    components: dict[
        str,
        dict[str, int],
    ] = {}

    for (
        name,
        parameter,
    ) in model.named_parameters():
        if "." in name:
            component = name.split(
                ".",
                1,
            )[0]

        else:
            component = name

        if (
            component
            not in components
        ):
            components[
                component
            ] = {
                "total_parameters": 0,
                "trainable_parameters": 0,
            }

        count = (
            parameter.numel()
        )

        components[
            component
        ][
            "total_parameters"
        ] += count

        if parameter.requires_grad:
            components[
                component
            ][
                "trainable_parameters"
            ] += count

    output: dict[
        str,
        dict[str, int | float],
    ] = {}

    for (
        component,
        counts,
    ) in components.items():
        total = counts[
            "total_parameters"
        ]

        trainable = counts[
            "trainable_parameters"
        ]

        output[
            component
        ] = {
            "total_parameters": (
                total
            ),
            "trainable_parameters": (
                trainable
            ),
            "frozen_parameters": (
                total
                - trainable
            ),
            "trainable_percent": (
                trainable
                / total
                * 100.0
                if total
                else 0.0
            ),
        }

    return output


# Metadata helpers

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

    if (
        transfer_type
        not in valid_transfer_types
    ):
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


def training_arguments_summary(
    training_args: Seq2SeqTrainingArguments,
) -> dict[str, Any]:
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    per_device_train_batch_size = int(
        training_args.per_device_train_batch_size
    )

    gradient_accumulation_steps = int(
        training_args.gradient_accumulation_steps
    )

    effective_batch_size = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * world_size
    )

    return {
        "epochs": float(
            training_args.num_train_epochs
        ),
        "learning_rate": float(
            training_args.learning_rate
        ),
        "warmup_steps": int(
            getattr(
                training_args,
                "warmup_steps",
                0,
            )
        ),
        "warmup_ratio": float(
            getattr(
                training_args,
                "warmup_ratio",
                0.0,
            )
        ),
        "batch_size_per_device": (
            per_device_train_batch_size
        ),
        "gradient_accumulation_steps": (
            gradient_accumulation_steps
        ),
        "world_size": (
            world_size
        ),
        "effective_batch_size": (
            effective_batch_size
        ),
        "optimizer": str(
            getattr(
                training_args,
                "optim",
                "",
            )
        ),
        "scheduler": str(
            getattr(
                training_args,
                "lr_scheduler_type",
                "",
            )
        ),
        "weight_decay": float(
            training_args.weight_decay
        ),
        "max_grad_norm": float(
            training_args.max_grad_norm
        ),
        "fp16": bool(
            training_args.fp16
        ),
        "bf16": bool(
            training_args.bf16
        ),
        "gradient_checkpointing": bool(
            training_args.gradient_checkpointing
        ),
        "seed": int(
            training_args.seed
        ),
        "data_seed": int(
            getattr(
                training_args,
                "data_seed",
                training_args.seed,
            )
        ),
        "dataloader_workers": int(
            training_args.dataloader_num_workers
        ),
    }


def build_training_metadata(
    *,
    model_name: str,
    model_type: str,
    model: torch.nn.Module,
    training_args: Seq2SeqTrainingArguments,
    train_manifest: Path,
    training_examples: int,
    classification: Mapping[
        str,
        Any,
    ],
    base_model: str | None = None,
    validation_manifest: Path | None = None,
    validation_examples: int | None = None,
    language: str | None = None,
    task: str | None = None,
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
    parameters = parameter_summary(
        model
    )

    training = {
        "train_manifest": str(
            train_manifest
        ),
        "training_examples": (
            training_examples
        ),
        "validation_manifest": (
            str(
                validation_manifest
            )
            if validation_manifest
            is not None
            else None
        ),
        "validation_examples": (
            validation_examples
        ),
        "language": language,
        "whisper_task": task,
        **training_arguments_summary(
            training_args
        ),
    }

    metadata: dict[
        str,
        Any,
    ] = {
        "model_name": (
            model_name
        ),
        "model_type": (
            model_type
        ),
        "base_model": (
            base_model
        ),
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
        "parameters": {
            **parameters,
            "count_basis": (
                "requires_grad_at_training_start"
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
    path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    metadata = build_training_metadata(
        **kwargs
    )

    if is_main_process():
        save_json(
            path,
            metadata,
        )

    return metadata


def trainer_result_summary(
    trainer: Seq2SeqTrainer,
    train_result: Any,
) -> dict[str, Any]:
    metrics = getattr(
        train_result,
        "metrics",
        {},
    )

    state = trainer.state

    return {
        "global_step": (
            state.global_step
        ),
        "epoch": (
            state.epoch
        ),
        "best_metric": (
            state.best_metric
        ),
        "best_model_checkpoint": (
            state.best_model_checkpoint
        ),
        "train_metrics": (
            make_json_safe(
                metrics
            )
        ),
    }


def update_training_metadata_after_training(
    path: Path,
    *,
    existing_metadata: Mapping[
        str,
        Any,
    ],
    trainer: Seq2SeqTrainer,
    train_result: Any,
    final_model_path: Path | None = None,
) -> dict[str, Any]:
    metadata = dict(
        existing_metadata
    )

    metadata[
        "status"
    ] = "complete"

    metadata[
        "training_result"
    ] = trainer_result_summary(
        trainer,
        train_result,
    )

    if (
        final_model_path
        is not None
    ):
        metadata[
            "model_directory"
        ] = str(
            final_model_path
        )

    if is_main_process():
        save_json(
            path,
            metadata,
        )

    return metadata