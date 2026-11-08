from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import WhisperProcessor

from scripts.data_utils import (
    TARGET_SAMPLE_RATE,
    load_wav_for_whisper,
)

from scripts.train import (
    calculate_corpus_metrics,
    calculate_single_utterance_metrics,
    configure_whisper_generation,
    load_task_mappings,
    save_json,
    write_csv,
)


# Helpers

def calculate_percent(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100.0
    )


def get_normalisation_mapping_name(
    row: Mapping[str, Any],
    mapping_column: str | None,
) -> str:
    if mapping_column is None:
        return ""

    return str(
        row.get(
            mapping_column,
            "",
        )
    ).strip().casefold()


# Transcription

def transcribe_one(
    *,
    row: Mapping[str, Any],
    processor: WhisperProcessor,
    model: torch.nn.Module,
    device: torch.device,
    audio_column: str,
    generation_max_length: int,
    generation_num_beams: int,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> str:
    audio_path = Path(
        str(
            row[
                audio_column
            ]
        )
    )

    audio = load_wav_for_whisper(
        audio_path,
        target_sample_rate=TARGET_SAMPLE_RATE,
        print_audio_check=False,
    )

    inputs = (
        processor.feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_attention_mask=True,
            return_tensors="pt",
        )
    )

    input_features = (
        inputs.input_features.to(
            device
        )
    )

    attention_mask = getattr(
        inputs,
        "attention_mask",
        None,
    )

    if attention_mask is not None:
        attention_mask = (
            attention_mask.to(
                device
            )
        )

    # Match the model's floating-point dtype.

    try:
        model_dtype = next(
            parameter.dtype
            for parameter
            in model.parameters()
            if parameter.is_floating_point()
        )

    except StopIteration:
        model_dtype = (
            input_features.dtype
        )

    if (
        input_features.is_floating_point()
        and model_dtype
        in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )
    ):
        input_features = (
            input_features.to(
                dtype=model_dtype
            )
        )

    generate_arguments: dict[str, Any] = {
        "input_features": (
            input_features
        ),
        "max_length": (
            generation_max_length
        ),
        "num_beams": (
            generation_num_beams
        ),
        "do_sample": False,
    }

    if attention_mask is not None:
        generate_arguments[
            "attention_mask"
        ] = attention_mask

    if generation_kwargs:
        generate_arguments.update(
            generation_kwargs
        )

    with torch.inference_mode():
        predicted_ids = model.generate(
            **generate_arguments
        )

    hypothesis = (
        processor.tokenizer
        .batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )

    return hypothesis


# Evaluation

def evaluate_loaded_model(
    *,
    model_name: str,
    processor: WhisperProcessor,
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    output_root: Path,
    normalisation_file: Path | None,
    language: str | None,
    task: str | None,
    generation_max_length: int,
    generation_num_beams: int,
    device: torch.device | None = None,
    audio_column: str = "audio_path",
    transcript_column: str = "transcript",
    normalisation_mapping_column: str | None = None,
    normalisation_section_markers: Mapping[
        str,
        str,
    ] | None = None,
    generation_kwargs: Mapping[
        str,
        Any,
    ] | None = None,
    dataset_name: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not rows:
        raise ValueError(
            "Cannot evaluate an empty dataset."
        )

    device = (
        device
        or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    # Load optional corpus-specific normalisation.

    if normalisation_file is None:
        normalisation_mappings: dict[
            str,
            dict[str, str],
        ] = {}

    else:
        normalisation_mappings = (
            load_task_mappings(
                normalisation_file,
                section_markers=(
                    normalisation_section_markers
                ),
            )
        )

    # Configure Whisper decoding.

    if (
        language is not None
        and task is not None
    ):
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

    model.to(
        device
    )

    model.eval()

    # Evaluation storage.

    prediction_rows: list[
        dict[str, Any]
    ] = []

    references: list[str] = []
    hypotheses: list[str] = []
    mapping_names: list[str] = []

    print()
    print("=" * 80)
    print(
        f"EVALUATING: {model_name}"
    )
    print("=" * 80)

    print(
        f"Recordings: {len(rows)}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Language: {language}"
    )

    print(
        f"Whisper task: {task}"
    )

    print(
        "Normalisation: "
        + (
            str(
                normalisation_file
            )
            if normalisation_file
            is not None
            else "none"
        )
    )

    # Transcribe every recording.

    for index, row in enumerate(
        rows,
        start=1,
    ):
        if audio_column not in row:
            raise KeyError(
                f"Row {index} has no "
                f"{audio_column!r} column."
            )

        if transcript_column not in row:
            raise KeyError(
                f"Row {index} has no "
                f"{transcript_column!r} column."
            )

        reference = str(
            row[
                transcript_column
            ]
        )

        mapping_name = (
            get_normalisation_mapping_name(
                row,
                normalisation_mapping_column,
            )
        )

        hypothesis = transcribe_one(
            row=row,
            processor=processor,
            model=model,
            device=device,
            audio_column=audio_column,
            generation_max_length=(
                generation_max_length
            ),
            generation_num_beams=(
                generation_num_beams
            ),
            generation_kwargs=(
                generation_kwargs
            ),
        )

        # Calculate per-recording normalised strings.
        #
        # We deliberately do NOT save all the
        # per-recording S/D/I/WER statistics.

        row_metrics = (
            calculate_single_utterance_metrics(
                reference,
                hypothesis,
                task_name=mapping_name,
                task_mappings=(
                    normalisation_mappings
                ),
            )
        )

        output_row: dict[
            str,
            Any,
        ] = {
            "model_name": (
                model_name
            ),
        }

        # Preserve the manifest metadata.
        #
        # The transcript itself is replaced by
        # the explicit gold-standard field below.

        for (
            field,
            value,
        ) in row.items():
            if (
                field
                == transcript_column
            ):
                continue

            output_row[
                field
            ] = value

        output_row.update(
            {
                "gold_standard_original": (
                    reference
                ),
                "predicted_hypothesis_original": (
                    hypothesis
                ),
                "gold_standard_post_normalisation": (
                    row_metrics[
                        "reference_post_normalisation"
                    ]
                ),
                "predicted_hypothesis_post_normalisation": (
                    row_metrics[
                        "hypothesis_post_normalisation"
                    ]
                ),
            }
        )

        prediction_rows.append(
            output_row
        )

        references.append(
            reference
        )

        hypotheses.append(
            hypothesis
        )

        mapping_names.append(
            mapping_name
        )

        recording_name = (
            row.get(
                "recording"
            )
            or row.get(
                "utterance_id"
            )
            or Path(
                str(
                    row[
                        audio_column
                    ]
                )
            ).name
        )

        print(
            f"[{index:>4}/{len(rows)}] "
            f"{recording_name}"
        )

    # Calculate final corpus metrics.

    corpus_metrics = (
        calculate_corpus_metrics(
            references,
            hypotheses,
            mapping_names,
            normalisation_mappings,
        )
    )

    normalised_n = int(
        corpus_metrics[
            "normalised_reference_words"
        ]
    )

    substitutions = int(
        corpus_metrics[
            "normalised_substitutions"
        ]
    )

    deletions = int(
        corpus_metrics[
            "normalised_deletions"
        ]
    )

    insertions = int(
        corpus_metrics[
            "normalised_insertions"
        ]
    )

    wer_percent = float(
        corpus_metrics[
            "normalised_wer"
        ]
    )

    # Keep the permanent evaluation result lean.

    evaluation: dict[
        str,
        Any,
    ] = {
        "model_name": (
            model_name
        ),
        "dataset": (
            dataset_name
        ),
        "recordings": (
            len(rows)
        ),
        "normalised_n": (
            normalised_n
        ),
        "substitutions": (
            substitutions
        ),
        "deletions": (
            deletions
        ),
        "insertions": (
            insertions
        ),
        "sub_percent": (
            calculate_percent(
                substitutions,
                normalised_n,
            )
        ),
        "del_percent": (
            calculate_percent(
                deletions,
                normalised_n,
            )
        ),
        "ins_percent": (
            calculate_percent(
                insertions,
                normalised_n,
            )
        ),
        "wer_percent": (
            wer_percent
        ),
    }

    # Save authoritative artifacts.

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_root
        / "predictions.csv"
    )

    evaluation_path = (
        output_root
        / "evaluation.json"
    )

    write_csv(
        predictions_path,
        prediction_rows,
    )

    save_json(
        evaluation_path,
        evaluation,
    )

    # Console summary.

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Recordings: "
        f"{len(rows)}"
    )

    print(
        f"Normalised N: "
        f"{normalised_n}"
    )

    print(
        f"S/D/I: "
        f"{substitutions}/"
        f"{deletions}/"
        f"{insertions}"
    )

    print(
        f"S%: "
        f"{evaluation['sub_percent']:.4f}%"
    )

    print(
        f"D%: "
        f"{evaluation['del_percent']:.4f}%"
    )

    print(
        f"I%: "
        f"{evaluation['ins_percent']:.4f}%"
    )

    print(
        f"WER: "
        f"{wer_percent:.4f}%"
    )

    print()

    print(
        f"Predictions: "
        f"{predictions_path}"
    )

    print(
        f"Evaluation: "
        f"{evaluation_path}"
    )

    return (
        prediction_rows,
        evaluation,
    )


# Cleanup

def cleanup_model(
    *objects: Any,
) -> None:
    for value in objects:
        del value

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()