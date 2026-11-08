from __future__ import annotations

import sys
from pathlib import Path

import torch


# Project

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts.evaluate import (
    evaluate_loaded_model,
)

from scripts.model_loader import (
    load_model,
)

from scripts.model_metadata import (
    build_classification,
    build_static_model_metadata,
    ensure_static_model_metadata,
)

from scripts.train import (
    load_manifest,
)


# Model

MODEL_NAME = "vanilla_whisper_medium"

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / MODEL_NAME
)

METADATA_PATH = (
    MODEL_ROOT
    / "metadata.json"
)


# Classification

CLASSIFICATION = build_classification(
    family="vanilla",
    transfer_type="plain",
    domain_category=None,
)


# Data

TEST_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llanito_cv"
    / "test_manifest.csv"
)

DATASET_NAME = "llanito_fixed_test"

EXPECTED_TEST_RECORDINGS = 57


# Normalisation

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)

NORMALISATION_MAPPING_COLUMN = "task"

NORMALISATION_SECTION_MARKERS = {
    "passage": (
        "SECTION 1 - "
        "PASSAGE RECORDINGS ONLY"
    ),
    "sentence": (
        "SECTION 2 - "
        "SENTENCE-TASK RECORDINGS ONLY"
    ),
}


# Whisper

LANGUAGE = "es"
TASK = "transcribe"

GENERATION_MAX_LENGTH = 225
GENERATION_NUM_BEAMS = 1


# Output

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / MODEL_NAME
)


def create_metadata_if_needed() -> None:
    metadata = (
        build_static_model_metadata(
            model_name=MODEL_NAME,
            model_type="whisper_full",
            base_model=(
                "openai/whisper-medium"
            ),
            classification=(
                CLASSIFICATION
            ),
            description=(
                "Unmodified pretrained OpenAI "
                "Whisper-medium baseline."
            ),
        )
    )

    ensure_static_model_metadata(
        path=METADATA_PATH,
        metadata=metadata,
    )


def main() -> None:
    print()
    print("=" * 80)
    print(
        "VANILLA WHISPER-MEDIUM BASELINE"
    )
    print("=" * 80)

    print()
    print(
        f"Model root:  {MODEL_ROOT}"
    )

    print(
        f"Test data:   {TEST_MANIFEST}"
    )

    print(
        f"Output root: {OUTPUT_ROOT}"
    )

    # Check model directory.

    model_directory = (
        MODEL_ROOT
        / "model"
    )

    if not model_directory.exists():
        raise FileNotFoundError(
            "Model directory does not exist:\n"
            f"{model_directory}"
        )

    # Create permanent static metadata
    # if this baseline has not yet been
    # registered in V2.

    create_metadata_if_needed()

    # Check data.

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            "Test manifest does not exist:\n"
            f"{TEST_MANIFEST}"
        )

    if not NORMALISATION_FILE.exists():
        raise FileNotFoundError(
            "Normalisation file does not exist:\n"
            f"{NORMALISATION_FILE}"
        )

    # Load fixed test manifest.

    rows = load_manifest(
        TEST_MANIFEST
    )

    if (
        len(rows)
        != EXPECTED_TEST_RECORDINGS
    ):
        raise RuntimeError(
            "Unexpected Llanito fixed-test "
            "size.\n"
            f"Expected: "
            f"{EXPECTED_TEST_RECORDINGS}\n"
            f"Found:    {len(rows)}"
        )

    print()
    print(
        f"Fixed-test recordings: "
        f"{len(rows)}"
    )

    # Evaluation device.

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )

        torch_dtype = (
            torch.float16
        )

    else:
        device = torch.device(
            "cpu"
        )

        torch_dtype = (
            torch.float32
        )

    print(
        f"Device:      {device}"
    )

    print(
        f"Model dtype: {torch_dtype}"
    )

    # Load model.

    print()
    print("Loading model...")

    (
        model,
        processor,
        metadata,
    ) = load_model(
        model_root=MODEL_ROOT,
        torch_dtype=torch_dtype,
    )

    metadata_model_name = (
        metadata.get(
            "model_name"
        )
    )

    if (
        metadata_model_name
        is not None
        and metadata_model_name
        != MODEL_NAME
    ):
        raise RuntimeError(
            "Model-name mismatch.\n"
            f"Expected: {MODEL_NAME}\n"
            f"Metadata: "
            f"{metadata_model_name}"
        )

    print(
        f"Loaded:     "
        f"{metadata.get('model_name', MODEL_NAME)}"
    )

    print(
        f"Model type: "
        f"{metadata['model_type']}"
    )

    print(
        f"Base model: "
        f"{metadata.get('base_model', '')}"
    )

    # Evaluate.

    evaluate_loaded_model(
        model_name=MODEL_NAME,
        processor=processor,
        model=model,
        rows=rows,
        output_root=OUTPUT_ROOT,
        normalisation_file=(
            NORMALISATION_FILE
        ),
        language=LANGUAGE,
        task=TASK,
        generation_max_length=(
            GENERATION_MAX_LENGTH
        ),
        generation_num_beams=(
            GENERATION_NUM_BEAMS
        ),
        device=device,
        audio_column="audio_path",
        transcript_column="transcript",
        normalisation_mapping_column=(
            NORMALISATION_MAPPING_COLUMN
        ),
        normalisation_section_markers=(
            NORMALISATION_SECTION_MARKERS
        ),
        generation_kwargs=None,
        dataset_name=DATASET_NAME,
    )

    print()
    print("=" * 80)
    print(
        "VANILLA BASELINE COMPLETE"
    )
    print("=" * 80)

    print()
    print("Permanent artifacts:")

    print(
        f"  {METADATA_PATH}"
    )

    print(
        f"  "
        f"{OUTPUT_ROOT / 'predictions.csv'}"
    )

    print(
        f"  "
        f"{OUTPUT_ROOT / 'evaluation.json'}"
    )


if __name__ == "__main__":
    main()