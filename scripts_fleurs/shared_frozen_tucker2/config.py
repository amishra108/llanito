from __future__ import annotations

import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANK = 128
CALIBRATION_SIZE = 256
CALIBRATION_SEED = 2026
DAMPING = 1.0e-6
SCALE = 1.0

EXPECTED_TARGETS = 288
FCCA128_BUDGET = EXPECTED_TARGETS * RANK * RANK

REGIMES = (
    "per_layer",
    "layer_splits",
    "enc_dec",
)

LANGUAGE_RUNNERS = {
    "ast_es": "scripts_fleurs.train_ast_es_fft",
    "es_419": "scripts_fleurs.train_es_419",
}


def load_runner(language: str):
    if language not in LANGUAGE_RUNNERS:
        raise ValueError(language)

    return importlib.import_module(
        LANGUAGE_RUNNERS[language]
    )


def result_root(language: str) -> Path:
    return (
        PROJECT_ROOT / "outputs" / "new_results"
        / language
    )


def task_stats_path(language: str) -> Path:
    return (
        result_root(language)
        / "_task_stats"
        / "task_stats_256_all_attention.pt"
    )


def task_stats_metadata_path(language: str) -> Path:
    return (
        result_root(language)
        / "_task_stats"
        / "task_stats_256_all_attention.json"
    )


def tucker_calibration_root(language: str) -> Path:
    return (
        result_root(language)
        / "shared_frozen_tucker2"
        / "calibration"
    )


def tucker_basis_path(language: str) -> Path:
    return (
        tucker_calibration_root(language)
        / "shared_frozen_tucker2_r128.pt"
    )


def parse_location(name: str):
    if ".encoder.layers." in name:
        tail = name.split(
            ".encoder.layers.",
            1,
        )[1]

        return (
            "encoder",
            int(
                tail.split(
                    ".",
                    1,
                )[0]
            ),
        )

    if ".decoder.layers." in name:
        tail = name.split(
            ".decoder.layers.",
            1,
        )[1]

        return (
            "decoder",
            int(
                tail.split(
                    ".",
                    1,
                )[0]
            ),
        )

    raise RuntimeError(
        f"Cannot identify Whisper layer: {name}"
    )


def structural_family(
    name: str,
):
    leaf = name.rsplit(
        ".",
        1,
    )[-1]

    if leaf not in {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "o_proj",
    }:
        raise RuntimeError(
            f"Unknown projection family: {name}"
        )

    if (
        ".encoder.layers." in name
        and
        ".self_attn." in name
    ):
        return (
            f"encoder_self_{leaf}"
        )

    if (
        ".decoder.layers." in name
        and
        ".self_attn." in name
    ):
        return (
            f"decoder_self_{leaf}"
        )

    if (
        ".decoder.layers." in name
        and
        ".encoder_attn." in name
    ):
        return (
            f"decoder_cross_{leaf}"
        )

    raise RuntimeError(
        f"Cannot identify structural family: {name}"
    )


def group_key(
    name: str,
    regime: str,
):
    side, layer = parse_location(
        name
    )

    family = structural_family(
        name
    )

    if regime == "per_layer":
        return (
            f"{family}"
            f"_layer_{layer:02d}"
        )

    if regime == "layer_splits":
        if side == "encoder":
            if layer <= 3:
                region = "00_03"

            elif layer <= 7:
                region = "04_07"

            else:
                region = "08_23"

            return (
                f"{family}"
                f"_{region}"
            )

        return (
            f"{family}_all"
        )

    if regime == "enc_dec":
        return (
            f"{family}_all"
        )

    raise ValueError(
        regime
    )


def make_groups(
    target_names,
    regime: str,
):
    groups = {}

    for name in sorted(
        target_names
    ):
        key = group_key(
            name,
            regime,
        )

        groups.setdefault(
            key,
            [],
        ).append(
            name
        )

    return groups
