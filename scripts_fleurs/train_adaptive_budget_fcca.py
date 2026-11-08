from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


# =====================================================================
# Experiment constants
# =====================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

BASE_MODEL = "openai/whisper-medium"

SEED = 2026

UNIFORM_REFERENCE_RANK = 128
TARGET_COUNT = 288

BUDGET = (
    TARGET_COUNT
    * UNIFORM_REFERENCE_RANK
    * UNIFORM_REFERENCE_RANK
)

CALIBRATION_SIZE = 256
CALIBRATION_SEED = 2026
DAMPING = 1.0e-6
SCALE = 1.0

MAX_EPOCHS = 12
PATIENCE = 5

MICROBATCH = 4
EVAL_BATCH_SIZE = 4

WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

RECIPES = {
    "ast_es": {
        "learning_rate": 6.0e-4,
        "effective_batch_size": 8,
        "warmup_ratio": 0.05,
        "runner_candidates": [
            "scripts_fleurs/train_ast_es_fft.py",
            "scripts_fleurs/train_ast_es.py",
        ],
    },
    "es_419": {
        "learning_rate": 1.0e-4,
        "effective_batch_size": 8,
        "warmup_ratio": 0.05,
        "runner_candidates": [
            "scripts_fleurs/train_es_419.py",
        ],
    },
}


# =====================================================================
# Utilities
# =====================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fields: list[str] = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    with tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        tmp,
        path,
    )


def load_runner(language: str):
    recipe = RECIPES[
        language
    ]

    path = (
        PROJECT_ROOT / "scripts_fleurs"
        / "train_whisper_fleurs.py"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    print(
        f"[runner] {language} -> {path}",
        flush=True,
    )

    spec = importlib.util.spec_from_file_location(
        f"adaptive_budget_runner_{language}",
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import runner {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    configure = getattr(
        module,
        "configure_language",
        None,
    )

    if not callable(configure):
        raise RuntimeError(
            "Generic runner has no configure_language()"
        )

    whisper_language = (
        recipe[
            "whisper_language"
        ]
    )

    parameters = inspect.signature(
        configure
    ).parameters

    if len(parameters) >= 2:
        configure(
            language,
            whisper_language,
        )
    else:
        configure(
            language
        )

        module.WHISPER_LANGUAGE = (
            whisper_language
        )

    required = [
        "load_split",
        "build_processor",
        "collate",
        "collect_attention_targets",
        "evaluate_validation_loss",
        "evaluate_wer",
    ]

    missing = [
        key
        for key in required
        if not callable(
            getattr(
                module,
                key,
                None,
            )
        )
    ]

    if missing:
        raise RuntimeError(
            f"Runner {path} missing required APIs: {missing}"
        )

    print(
        "[runner AUDIT PASS] "
        + ", ".join(required),
        flush=True,
    )

    return module, path


def task_stats_path(
    language: str,
) -> Path:
    filename = (
        "task_stats_256_all_attention.pt"
        if language == "es_419"
        else "task_stats_256_all_attention.pt"
    )

    return (
        PROJECT_ROOT / "outputs" / "new_results"
        / language
        / "_task_stats"
        / filename
    )

def basis_path(
    language: str,
) -> Path:
    return (
        PROJECT_ROOT / "outputs" / "new_results"
        / language
        / "adaptive_budget_fcca"
        / "calibration"
        / "adaptive_budget_basis.pt"
    )


def output_dir(
    language: str,
) -> Path:
    return (
        PROJECT_ROOT / "outputs" / "new_results"
        / language
        / "adaptive_budget_fcca"
        / "training"
        / "trial_0001"
    )


# =====================================================================
# Task-stat compatibility
# =====================================================================

def find_stats_mapping(
    payload: Any,
) -> dict:
    """
    Return the exact per-module statistics mapping.

    The authoritative Week-8 cache stores statistics under

        payload["modules"][module_name]

    but a few older caches used stats/task_stats.
    """
    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Task-stat payload is not a dict."
        )

    for key in (
        "modules",
        "stats",
        "task_stats",
    ):
        value = payload.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):
            if len(value) == TARGET_COUNT:
                print(
                    f"[task-stats] mapping={key!r} "
                    f"modules={len(value)}",
                    flush=True,
                )
                return value

    raise RuntimeError(
        "Could not locate the 288-module task-stat mapping. "
        f"top-level keys={list(payload)[:40]}"
    )


def _dict_tensor(
    record: dict,
    aliases: tuple[str, ...],
):
    for key in aliases:
        value = record.get(
            key
        )

        if torch.is_tensor(
            value
        ):
            return key, value

    return None, None


def _semantic_tensor(
    record: dict,
    *,
    positive_tokens: tuple[str, ...],
    forbidden_tokens: tuple[str, ...],
    ndim: int,
    expected_numel: int | None = None,
):
    candidates = []

    for key, value in record.items():
        if not torch.is_tensor(
            value
        ):
            continue

        if value.ndim != ndim:
            continue

        if (
            expected_numel is not None
            and value.numel()
            != expected_numel
        ):
            continue

        folded = str(
            key
        ).casefold()

        if any(
            token in folded
            for token in forbidden_tokens
        ):
            continue

        score = sum(
            token in folded
            for token in positive_tokens
        )

        if score > 0:
            candidates.append(
                (
                    score,
                    str(key),
                    value,
                )
            )

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda item:
            (
                -item[0],
                item[1],
            )
    )

    best_score = candidates[0][0]

    tied = [
        item
        for item in candidates
        if item[0] == best_score
    ]

    if len(tied) != 1:
        return None, None

    _, key, value = tied[0]

    return key, value


def _normalise_accumulator(
    tensor: torch.Tensor,
    key: str,
    positions: int,
) -> torch.Tensor:
    """
    Week-8 *_sum fields are raw accumulated sums and therefore
    must be divided by the captured position count.

    Already-normalised fields (G/a/d/means) are left unchanged.
    """
    folded = key.casefold()

    is_sum = (
        "sum" in folded
        and
        "mean" not in folded
    )

    result = tensor.float()

    if is_sum:
        result = (
            result
            / float(
                positions
            )
        )

    return result


def extract_stat_components(
    record: Any,
    *,
    name: str,
    device: torch.device,
):
    if not isinstance(
        record,
        dict,
    ):
        raise RuntimeError(
            f"{name}: task-stat record is not a dict; "
            f"type={type(record)!r}"
        )

    # --------------------------------------------------------
    # Dimensions
    #
    # Some canonical Week-8 task-stat caches store only the
    # sufficient statistics:
    #
    #   cross_sum                  [out, in]
    #   activation_second_sum      [in]
    #   gradient_second_sum        [out]
    #   positions
    #
    # Explicit in_features/out_features are therefore optional.
    # When absent, infer them exactly from the cross-moment shape.
    # This is schema handling only; no statistic is modified.
    # --------------------------------------------------------

    explicit_dims = (
        "out_features" in record
        and "in_features" in record
    )

    if explicit_dims:
        try:
            out_features = int(
                record["out_features"]
            )

            in_features = int(
                record["in_features"]
            )

        except Exception as exc:
            raise RuntimeError(
                f"{name}: invalid "
                "in_features/out_features. "
                f"keys={sorted(record)}"
            ) from exc

    else:
        cross_for_shape = None

        for key in (
            "cross_sum",
            "cross",
            "G_sum",
            "g_sum",
            "G",
            "g",
        ):
            value = record.get(key)

            if (
                torch.is_tensor(value)
                and value.ndim == 2
            ):
                cross_for_shape = value
                break

        if cross_for_shape is None:
            raise RuntimeError(
                f"{name}: cannot infer "
                "in_features/out_features; "
                "no 2-D cross moment found. "
                f"keys={sorted(record)}"
            )

        out_features = int(
            cross_for_shape.shape[0]
        )

        in_features = int(
            cross_for_shape.shape[1]
        )

    # --------------------------------------------------------
    # Number of captured positions
    # --------------------------------------------------------

    position_key = None
    positions = None

    for key in (
        "positions",
        "position_count",
        "num_positions",
        "n_positions",
        "count",
    ):
        if key in record:
            try:
                candidate = int(
                    record[key]
                )
            except Exception:
                continue

            if candidate > 0:
                position_key = key
                positions = candidate
                break

    if positions is None:
        raise RuntimeError(
            f"{name}: cannot identify positive "
            "position count. "
            f"keys={sorted(record)}"
        )

    # --------------------------------------------------------
    # Cross moment G
    # --------------------------------------------------------

    cross_key, cross = _dict_tensor(
        record,
        (
            "cross_sum",
            "cross",
            "G_sum",
            "g_sum",
            "G",
            "g",
        ),
    )

    if cross is None:
        two_d = [
            (
                str(key),
                value,
            )
            for key, value
            in record.items()
            if (
                torch.is_tensor(value)
                and
                value.ndim == 2
                and
                tuple(value.shape)
                == (
                    out_features,
                    in_features,
                )
            )
        ]

        if len(two_d) == 1:
            cross_key, cross = (
                two_d[0]
            )

    if cross is None:
        raise RuntimeError(
            f"{name}: cannot identify cross moment. "
            f"keys={sorted(record)}"
        )

    if tuple(
        cross.shape
    ) != (
        out_features,
        in_features,
    ):
        raise RuntimeError(
            f"{name}: cross shape "
            f"{tuple(cross.shape)} != "
            f"{(out_features, in_features)}"
        )

    # --------------------------------------------------------
    # Activation diagonal A
    # --------------------------------------------------------

    a_key, a = _dict_tensor(
        record,
        (
            "activation_second_sum",
            "activation_squared_sum",
            "activation_sq_sum",
            "input_second_sum",
            "input_squared_sum",
            "input_sq_sum",
            "x_second_sum",
            "x_squared_sum",
            "x_sq_sum",
            "a_sum",
            "activation_second",
            "activation_squared",
            "activation_sq",
            "input_second",
            "input_squared",
            "input_sq",
            "a",
        ),
    )

    if a is None:
        a_key, a = _semantic_tensor(
            record,
            positive_tokens=(
                "activation",
                "input",
                "x_sq",
                "x2",
                "a_sum",
                "second_a",
            ),
            forbidden_tokens=(
                "grad",
                "gradient",
                "delta",
                "cross",
            ),
            ndim=1,
            expected_numel=
                in_features,
        )

    # --------------------------------------------------------
    # Gradient diagonal D
    # --------------------------------------------------------

    d_key, d = _dict_tensor(
        record,
        (
            "gradient_second_sum",
            "gradient_squared_sum",
            "gradient_sq_sum",
            "grad_second_sum",
            "grad_squared_sum",
            "grad_sq_sum",
            "delta_second_sum",
            "delta_squared_sum",
            "delta_sq_sum",
            "output_gradient_second_sum",
            "output_gradient_sq_sum",
            "d_sum",
            "gradient_second",
            "gradient_squared",
            "gradient_sq",
            "grad_second",
            "grad_squared",
            "grad_sq",
            "delta_second",
            "delta_squared",
            "delta_sq",
            "d",
        ),
    )

    if d is None:
        d_key, d = _semantic_tensor(
            record,
            positive_tokens=(
                "gradient",
                "grad",
                "delta",
                "d_sum",
                "second_d",
            ),
            forbidden_tokens=(
                "activation",
                "input",
                "cross",
            ),
            ndim=1,
            expected_numel=
                out_features,
        )

    if a is None or d is None:
        one_d = [
            (
                str(key),
                tuple(value.shape),
            )
            for key, value
            in record.items()
            if (
                torch.is_tensor(value)
                and
                value.ndim == 1
            )
        ]

        raise RuntimeError(
            f"{name}: could not unambiguously identify "
            "activation/gradient second moments. "
            f"1-D tensors={one_d}; "
            f"all keys={sorted(record)}"
        )

    if a.numel() != in_features:
        raise RuntimeError(
            f"{name}: activation moment length "
            f"{a.numel()} != {in_features}"
        )

    if d.numel() != out_features:
        raise RuntimeError(
            f"{name}: gradient moment length "
            f"{d.numel()} != {out_features}"
        )

    # --------------------------------------------------------
    # Convert sums -> means exactly as ordinary FCCA.
    # --------------------------------------------------------

    G = _normalise_accumulator(
        cross,
        cross_key,
        positions,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    a = _normalise_accumulator(
        a,
        a_key,
        positions,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    d = _normalise_accumulator(
        d,
        d_key,
        positions,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    if (
        not torch.isfinite(
            G
        ).all()
        or
        not torch.isfinite(
            a
        ).all()
        or
        not torch.isfinite(
            d
        ).all()
    ):
        raise RuntimeError(
            f"{name}: non-finite task statistics"
        )

    if (
        torch.any(
            a < 0
        )
        or
        torch.any(
            d < 0
        )
    ):
        raise RuntimeError(
            f"{name}: negative diagonal second moment"
        )

    schema = {
        "cross_key":
            cross_key,

        "activation_key":
            a_key,

        "gradient_key":
            d_key,

        "position_key":
            position_key,

        "in_features":
            in_features,

        "out_features":
            out_features,
    }

    return (
        G,
        a,
        d,
        positions,
        schema,
    )


def whitened_from_record(
    record: Any,
    device: torch.device,
    *,
    name: str,
):
    (
        G,
        a,
        d,
        positions,
        schema,
    ) = extract_stat_components(
        record,
        name=name,
        device=device,
    )

    inv_a = torch.rsqrt(
        a + DAMPING
    )

    inv_d = torch.rsqrt(
        d + DAMPING
    )

    whitened = (
        inv_d[:, None]
        * G
        * inv_a[None, :]
    )

    if not torch.isfinite(
        whitened
    ).all():
        raise RuntimeError(
            f"{name}: non-finite whitened gradient"
        )

    return (
        whitened,
        inv_a,
        inv_d,
        positions,
        schema,
    )


def audit_task_stats(
    payload: Any,
    stats: dict,
    device: torch.device,
) -> dict:
    if len(stats) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_COUNT} task-stat modules, "
            f"found {len(stats)}"
        )

    names = sorted(
        stats
    )

    schemas = {}

    for index, name in enumerate(
        names,
        start=1,
    ):
        (
            G,
            a,
            d,
            positions,
            schema,
        ) = extract_stat_components(
            stats[name],
            name=name,
            device=device,
        )

        schemas.setdefault(
            (
                schema[
                    "cross_key"
                ],
                schema[
                    "activation_key"
                ],
                schema[
                    "gradient_key"
                ],
                schema[
                    "position_key"
                ],
            ),
            0,
        )

        schemas[
            (
                schema[
                    "cross_key"
                ],
                schema[
                    "activation_key"
                ],
                schema[
                    "gradient_key"
                ],
                schema[
                    "position_key"
                ],
            )
        ] += 1

        del (
            G,
            a,
            d,
        )

    if len(schemas) != 1:
        raise RuntimeError(
            "Task-stat modules do not share one consistent "
            f"schema: {schemas}"
        )

    schema_tuple = next(
        iter(
            schemas
        )
    )

    schema = {
        "cross_key":
            schema_tuple[0],

        "activation_key":
            schema_tuple[1],

        "gradient_key":
            schema_tuple[2],

        "position_key":
            schema_tuple[3],

        "module_count":
            len(stats),
    }

    print(
        "[task-stats AUDIT PASS] "
        f"modules={len(stats)} "
        f"cross={schema['cross_key']} "
        f"A={schema['activation_key']} "
        f"D={schema['gradient_key']} "
        f"positions={schema['position_key']}",
        flush=True,
    )

    # Validate any metadata that the cache provides.
    for key, expected in (
        (
            "calibration_size",
            CALIBRATION_SIZE,
        ),
        (
            "calibration_seed",
            CALIBRATION_SEED,
        ),
    ):
        if key in payload:
            actual = int(
                payload[key]
            )

            if actual != expected:
                raise RuntimeError(
                    f"Task-stat {key} mismatch: "
                    f"{actual} != {expected}"
                )

    return schema

# =====================================================================
# Exact global rank allocation
# =====================================================================

def solve_exact_allocation(
    names: list[str],
    spectra: dict[str, torch.Tensor],
    rank_step: int,
    solver_time_limit: float,
):
    """
    Multiple-choice integer program:

      choose exactly one rank for each matrix
      sum_l r_l^2 = BUDGET
      maximize sum_l V_l(r_l)

    V_l(0)=0 and
    V_l(r)=sum_{k<=r} sigma_{l,k}^2.

    When rank_step=1 this is the full integer-rank problem.
    """

    if rank_step <= 0:
        raise ValueError(
            "rank_step must be positive"
        )

    candidates: dict[str, list[int]] = {}
    utilities: dict[str, dict[int, float]] = {}

    for name in names:
        s = spectra[name].double()

        max_rank = int(
            s.numel()
        )

        ranks = list(
            range(
                0,
                max_rank + 1,
                rank_step,
            )
        )

        if (
            UNIFORM_REFERENCE_RANK
            <= max_rank
            and
            UNIFORM_REFERENCE_RANK
            not in ranks
        ):
            ranks.append(
                UNIFORM_REFERENCE_RANK
            )

        ranks = sorted(
            set(ranks)
        )

        cumulative = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=torch.float64,
                ),
                torch.cumsum(
                    s.square(),
                    dim=0,
                ),
            ]
        )

        candidates[name] = ranks

        utilities[name] = {
            rank:
                float(
                    cumulative[rank].item()
                )
            for rank in ranks
        }

    # Uniform rank-128 must be a feasible point.
    for name in names:
        if (
            UNIFORM_REFERENCE_RANK
            not in candidates[name]
        ):
            raise RuntimeError(
                f"{name}: rank 128 unavailable"
            )

    if (
        len(names)
        * UNIFORM_REFERENCE_RANK ** 2
        != BUDGET
    ):
        raise RuntimeError(
            "Reference budget arithmetic failure"
        )

    variable_meta: list[
        tuple[str, int]
    ] = []

    for name in names:
        for rank in candidates[name]:
            variable_meta.append(
                (
                    name,
                    rank,
                )
            )

    n_variables = len(
        variable_meta
    )

    raw_utilities = np.asarray(
        [
            utilities[name][rank]
            for name, rank
            in variable_meta
        ],
        dtype=np.float64,
    )

    utility_scale = max(
        float(
            np.max(
                np.abs(
                    raw_utilities
                )
            )
        ),
        1.0e-30,
    )

    # scipy.milp minimizes, hence the minus sign.
    objective = (
        -raw_utilities
        / utility_scale
    )

    row_indices = []
    col_indices = []
    values = []

    name_to_row = {
        name: index
        for index, name
        in enumerate(names)
    }

    budget_row = len(
        names
    )

    for column, (
        name,
        rank,
    ) in enumerate(
        variable_meta
    ):
        row_indices.append(
            name_to_row[name]
        )

        col_indices.append(
            column
        )

        values.append(
            1.0
        )

        row_indices.append(
            budget_row
        )

        col_indices.append(
            column
        )

        values.append(
            float(
                rank * rank
            )
        )

    A = coo_matrix(
        (
            np.asarray(
                values,
                dtype=np.float64,
            ),
            (
                np.asarray(
                    row_indices,
                    dtype=np.int64,
                ),
                np.asarray(
                    col_indices,
                    dtype=np.int64,
                ),
            ),
        ),
        shape=(
            len(names) + 1,
            n_variables,
        ),
    ).tocsr()

    lower = np.concatenate(
        [
            np.ones(
                len(names),
                dtype=np.float64,
            ),
            np.asarray(
                [float(BUDGET)],
                dtype=np.float64,
            ),
        ]
    )

    upper = lower.copy()

    constraints = LinearConstraint(
        A,
        lower,
        upper,
    )

    print(
        "[allocation] "
        f"matrices={len(names)} "
        f"binary_variables={n_variables:,} "
        f"budget={BUDGET:,} "
        f"rank_step={rank_step}",
        flush=True,
    )

    result = milp(
        c=objective,
        integrality=np.ones(
            n_variables,
            dtype=np.int8,
        ),
        bounds=Bounds(
            np.zeros(
                n_variables,
                dtype=np.float64,
            ),
            np.ones(
                n_variables,
                dtype=np.float64,
            ),
        ),
        constraints=constraints,
        options={
            "presolve":
                True,

            "time_limit":
                float(
                    solver_time_limit
                ),

            "mip_rel_gap":
                0.0,
        },
    )

    if (
        not result.success
        or
        result.x is None
    ):
        raise RuntimeError(
            "Exact allocation solver did not certify an optimum: "
            f"status={result.status}, "
            f"message={result.message}"
        )

    selected: dict[str, int] = {}

    x = np.asarray(
        result.x
    )

    for name in names:
        selected_columns = [
            column
            for column, (
                variable_name,
                rank,
            ) in enumerate(
                variable_meta
            )
            if (
                variable_name == name
                and
                x[column] > 0.5
            )
        ]

        if len(
            selected_columns
        ) != 1:
            raise RuntimeError(
                f"{name}: expected one selected rank, "
                f"got {len(selected_columns)}"
            )

        column = selected_columns[0]

        _, rank = variable_meta[
            column
        ]

        selected[name] = int(
            rank
        )

    used_budget = sum(
        rank * rank
        for rank in selected.values()
    )

    if used_budget != BUDGET:
        raise RuntimeError(
            "Budget mismatch after MIP solve: "
            f"{used_budget:,} != {BUDGET:,}"
        )

    adaptive_utility = sum(
        utilities[name][
            selected[name]
        ]
        for name in names
    )

    uniform_utility = sum(
        utilities[name][
            UNIFORM_REFERENCE_RANK
        ]
        for name in names
    )

    mip_gap = getattr(
        result,
        "mip_gap",
        None,
    )

    if (
        adaptive_utility
        + max(
            1.0e-10,
            1.0e-10
            * abs(
                uniform_utility
            ),
        )
        < uniform_utility
    ):
        raise RuntimeError(
            "Allocator theorem audit failed: "
            "certified adaptive optimum is below the "
            "feasible uniform-rank-128 objective."
        )

    print(
        "[allocation OPTIMAL] "
        f"budget={used_budget:,} "
        f"objective={adaptive_utility:.12e} "
        f"uniform128={uniform_utility:.12e} "
        f"ratio="
        f"{adaptive_utility / uniform_utility:.8f} "
        f"mip_gap={mip_gap}",
        flush=True,
    )

    return (
        selected,
        {
            "solver":
                "scipy.optimize.milp / HiGHS",

            "solver_success":
                bool(
                    result.success
                ),

            "solver_status":
                int(
                    result.status
                ),

            "solver_message":
                str(
                    result.message
                ),

            "solver_mip_gap":
                (
                    None
                    if mip_gap is None
                    else float(
                        mip_gap
                    )
                ),

            "rank_step":
                rank_step,

            "adaptive_utility":
                adaptive_utility,

            "uniform_rank128_utility":
                uniform_utility,

            "utility_ratio":
                adaptive_utility
                / uniform_utility,

            "budget":
                BUDGET,

            "used_budget":
                used_budget,
        },
        utilities,
    )


# =====================================================================
# Build adaptive FCCA bases
# =====================================================================

@torch.no_grad()
def build_basis(
    language: str,
    device: torch.device,
    rank_step: int,
    solver_time_limit: float,
    force: bool,
) -> None:
    path = basis_path(
        language
    )

    if path.exists() and not force:
        existing = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(
            existing,
            dict,
        ):
            raise RuntimeError(
                f"Existing basis is not a dict: {path}"
            )

        if existing.get(
            "method"
        ) != "adaptive_budget_fcca":
            raise RuntimeError(
                "Existing basis method mismatch"
            )

        if int(
            existing.get(
                "reference_budget",
                -1,
            )
        ) != BUDGET:
            raise RuntimeError(
                "Existing basis budget mismatch"
            )

        existing_bases = existing.get(
            "bases",
            {}
        )

        if len(
            existing_bases
        ) != TARGET_COUNT:
            raise RuntimeError(
                "Existing basis target-count mismatch"
            )

        used = sum(
            int(
                entry[
                    "rank"
                ]
            ) ** 2
            for entry
            in existing_bases.values()
        )

        if used != BUDGET:
            raise RuntimeError(
                f"Existing basis uses {used:,} "
                f"parameters, expected {BUDGET:,}"
            )

        solver = existing.get(
            "solver",
            {}
        )

        if not bool(
            solver.get(
                "solver_success",
                False,
            )
        ):
            raise RuntimeError(
                "Existing basis has no certified "
                "optimal allocation."
            )

        print(
            "[basis AUDIT PASS] "
            f"reusing {path}; "
            f"budget={used:,}",
            flush=True,
        )

        return

    stats_path = task_stats_path(
        language
    )

    if not stats_path.exists():
        raise FileNotFoundError(
            stats_path
        )

    payload = torch.load(
        stats_path,
        map_location="cpu",
        weights_only=False,
    )

    stats = find_stats_mapping(
        payload
    )

    names = sorted(
        stats
    )

    task_stat_schema = audit_task_stats(
        payload,
        stats,
        device,
    )

    if len(names) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_COUNT} matrices, "
            f"found {len(names)}"
        )

    spectra: dict[
        str,
        torch.Tensor
    ] = {}

    diagnostics: dict[
        str,
        dict
    ] = {}

    print(
        "[spectra] computing complete "
        "Fisher-whitened singular spectra",
        flush=True,
    )

    started = time.time()

    # Pass 1: spectra only.
    for index, name in enumerate(
        names,
        start=1,
    ):
        (
            whitened,
            inv_a,
            inv_d,
            positions,
            stat_schema,
        ) = whitened_from_record(
            stats[name],
            device,
            name=name,
        )

        svals = torch.linalg.svdvals(
            whitened
        )

        spectra[name] = (
            svals.detach()
            .double()
            .cpu()
        )

        diagnostics[name] = {
            "positions":
                positions,

            "out_features":
                int(
                    whitened.shape[0]
                ),

            "in_features":
                int(
                    whitened.shape[1]
                ),

            "maximum_rank":
                int(
                    svals.numel()
                ),

            "sigma_1":
                float(
                    svals[0].item()
                ),

            "sigma_128":
                float(
                    svals[127].item()
                ),
        }

        del (
            whitened,
            inv_a,
            inv_d,
            svals,
        )

        if (
            index % 16 == 0
            or
            index == len(names)
        ):
            print(
                f"[spectra] "
                f"{index}/{len(names)}",
                flush=True,
            )

    (
        allocation,
        solver_info,
        utility_tables,
    ) = solve_exact_allocation(
        names,
        spectra,
        rank_step=rank_step,
        solver_time_limit=solver_time_limit,
    )

    ranks = list(
        allocation.values()
    )

    rank_rows = []

    for name in names:
        rank = allocation[name]

        rank_rows.append(
            {
                "module":
                    name,

                "rank":
                    rank,

                "core_parameters":
                    rank * rank,

                "local_utility":
                    utility_tables[
                        name
                    ][rank],

                "uniform_r128_utility":
                    utility_tables[
                        name
                    ][
                        UNIFORM_REFERENCE_RANK
                    ],
            }
        )

    rank_csv = (
        path.parent
        / "rank_allocation.csv"
    )

    write_csv(
        rank_csv,
        rank_rows,
    )

    print(
        "[allocation summary] "
        f"min={min(ranks)} "
        f"max={max(ranks)} "
        f"mean={np.mean(ranks):.4f} "
        f"median={np.median(ranks):.4f} "
        f"zero_rank={sum(r == 0 for r in ranks)}",
        flush=True,
    )

    # Pass 2: now construct ONLY the selected FCCA basis for each matrix.
    bases = {}

    print(
        "[basis] constructing selected "
        "ordinary FCCA subspaces",
        flush=True,
    )

    for index, name in enumerate(
        names,
        start=1,
    ):
        rank = allocation[name]

        if rank == 0:
            bases[name] = {
                "rank":
                    0,

                "P":
                    torch.empty(
                        diagnostics[name][
                            "out_features"
                        ],
                        0,
                        dtype=torch.float32,
                    ),

                "Q":
                    torch.empty(
                        diagnostics[name][
                            "in_features"
                        ],
                        0,
                        dtype=torch.float32,
                    ),
            }

            diagnostics[name][
                "selected_rank"
            ] = 0

            diagnostics[name][
                "selected_utility"
            ] = 0.0

            if (
                index % 16 == 0
                or
                index == len(names)
            ):
                print(
                    f"[basis] "
                    f"{index}/{len(names)}",
                    flush=True,
                )

            continue

        (
            whitened,
            inv_a,
            inv_d,
            positions,
            stat_schema,
        ) = whitened_from_record(
            stats[name],
            device,
            name=name,
        )

        U, S, Vh = torch.linalg.svd(
            whitened,
            full_matrices=False,
        )

        U_r = U[:, :rank]
        V_r = Vh[:rank, :].T

        P0 = (
            inv_d[:, None]
            * U_r
        )

        Q0 = (
            inv_a[:, None]
            * V_r
        )

        # QR merely changes coordinates inside the same selected spans.
        P, _ = torch.linalg.qr(
            P0.double(),
            mode="reduced",
        )

        Q, _ = torch.linalg.qr(
            Q0.double(),
            mode="reduced",
        )

        P = (
            P[:, :rank]
            .float()
        )

        Q = (
            Q[:, :rank]
            .float()
        )

        eye = torch.eye(
            rank,
            device=device,
            dtype=torch.float32,
        )

        p_error = float(
            torch.linalg.matrix_norm(
                P.T @ P - eye,
                ord="fro",
            ).item()
        )

        q_error = float(
            torch.linalg.matrix_norm(
                Q.T @ Q - eye,
                ord="fro",
            ).item()
        )

        if (
            p_error > 5.0e-4
            or
            q_error > 5.0e-4
        ):
            raise RuntimeError(
                f"{name}: QR audit failed "
                f"P={p_error:.6e} "
                f"Q={q_error:.6e}"
            )

        bases[name] = {
            "rank":
                rank,

            "P":
                P.detach().cpu(),

            "Q":
                Q.detach().cpu(),
        }

        diagnostics[name][
            "selected_rank"
        ] = rank

        diagnostics[name][
            "selected_utility"
        ] = utility_tables[
            name
        ][rank]

        diagnostics[name][
            "p_orthogonality_error"
        ] = p_error

        diagnostics[name][
            "q_orthogonality_error"
        ] = q_error

        del (
            whitened,
            inv_a,
            inv_d,
            U,
            S,
            Vh,
            U_r,
            V_r,
            P0,
            Q0,
            P,
            Q,
        )

        torch.cuda.empty_cache()

        if (
            index % 16 == 0
            or
            index == len(names)
        ):
            print(
                f"[basis] "
                f"{index}/{len(names)}",
                flush=True,
            )

    used_budget = sum(
        int(entry["rank"]) ** 2
        for entry in bases.values()
    )

    if used_budget != BUDGET:
        raise RuntimeError(
            f"Final basis budget mismatch: "
            f"{used_budget:,} != {BUDGET:,}"
        )

    basis_payload = {
        "method":
            "adaptive_budget_fcca",

        "objective":
            "sum_l sum_{k<=r_l} sigma_{l,k}^2",

        "optimality_scope":
            (
                "globally optimal rank allocation under the "
                "separable FCCA local Fisher objective over the "
                "specified allowed integer ranks"
            ),

        "language":
            language,

        "source_task_stats":
            str(
                stats_path
            ),

        "source_task_stat_schema":
            task_stat_schema,

        "calibration_size":
            CALIBRATION_SIZE,

        "calibration_seed":
            CALIBRATION_SEED,

        "damping":
            DAMPING,

        "scale":
            SCALE,

        "reference_uniform_rank":
            UNIFORM_REFERENCE_RANK,

        "reference_budget":
            BUDGET,

        "trainable_core_budget":
            used_budget,

        "target_count":
            len(names),

        "rank_allocation":
            allocation,

        "rank_summary": {
            "minimum":
                min(ranks),

            "maximum":
                max(ranks),

            "mean":
                float(
                    np.mean(ranks)
                ),

            "median":
                float(
                    np.median(ranks)
                ),

            "zero_rank_matrices":
                int(
                    sum(
                        r == 0
                        for r in ranks
                    )
                ),
        },

        "solver":
            solver_info,

        "diagnostics":
            diagnostics,

        "bases":
            bases,

        "build_seconds":
            time.time()
            - started,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        basis_payload,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )

    save_json(
        path.with_suffix(
            ".json"
        ),
        {
            key: value
            for key, value
            in basis_payload.items()
            if key != "bases"
        },
    )

    print(
        f"[basis COMPLETE] {path}",
        flush=True,
    )


# =====================================================================
# Variable-rank FCCA architecture
# =====================================================================

class AdaptiveFCCALinear(nn.Module):
    """
    Ordinary FCCA at one matrix-specific fixed rank.

        Delta W = P R Q^T

    P and Q are frozen.
    Only the r x r core R is trainable.
    """

    def __init__(
        self,
        base: nn.Linear,
        P: torch.Tensor,
        Q: torch.Tensor,
        rank: int,
    ):
        super().__init__()

        self.base = base
        self.rank = int(
            rank
        )

        for parameter in (
            self.base.parameters()
        ):
            parameter.requires_grad = False

        if self.rank <= 0:
            raise ValueError(
                "AdaptiveFCCALinear requires rank > 0"
            )

        expected_p = (
            base.out_features,
            self.rank,
        )

        expected_q = (
            base.in_features,
            self.rank,
        )

        if tuple(
            P.shape
        ) != expected_p:
            raise RuntimeError(
                f"P shape {tuple(P.shape)} "
                f"!= {expected_p}"
            )

        if tuple(
            Q.shape
        ) != expected_q:
            raise RuntimeError(
                f"Q shape {tuple(Q.shape)} "
                f"!= {expected_q}"
            )

        device = (
            base.weight.device
        )

        self.register_buffer(
            "adaptive_fcca_P",
            P.to(
                device=device,
                dtype=torch.float32,
            ).contiguous(),
        )

        self.register_buffer(
            "adaptive_fcca_Q",
            Q.to(
                device=device,
                dtype=torch.float32,
            ).contiguous(),
        )

        self.adaptive_fcca_core = nn.Parameter(
            torch.zeros(
                self.rank,
                self.rank,
                device=device,
                dtype=torch.float32,
            )
        )

        # Construction audit.
        if (
            self.adaptive_fcca_P.device
            != device
            or
            self.adaptive_fcca_Q.device
            != device
            or
            self.adaptive_fcca_core.device
            != device
        ):
            raise RuntimeError(
                "Adaptive FCCA device-placement failure"
            )

    def forward(
        self,
        x: torch.Tensor,
    ):
        base_out = (
            self.base(
                x
            )
        )

        projected = (
            x.float()
            @ self.adaptive_fcca_Q
        )

        mixed = (
            projected
            @ self.adaptive_fcca_core.T
        )

        update = (
            mixed
            @ self.adaptive_fcca_P.T
        )

        return (
            base_out
            + (
                SCALE
                * update
            ).to(
                base_out.dtype
            )
        )

def get_parent(
    model: nn.Module,
    name: str,
):
    parts = name.split(".")
    parent = model

    for part in parts[:-1]:
        parent = getattr(
            parent,
            part,
        )

    return (
        parent,
        parts[-1],
    )


def attach_adaptive_fcca(
    model: nn.Module,
    runner,
    basis_payload: dict,
):
    targets = (
        runner
        .collect_attention_targets(
            model
        )
    )

    if basis_payload.get(
        "method"
    ) != "adaptive_budget_fcca":
        raise RuntimeError(
            "Not an Adaptive-Budget FCCA basis payload"
        )

    if int(
        basis_payload.get(
            "reference_budget",
            -1,
        )
    ) != BUDGET:
        raise RuntimeError(
            "Adaptive basis reference-budget mismatch"
        )

    solver = basis_payload.get(
        "solver",
        {}
    )

    if not bool(
        solver.get(
            "solver_success",
            False,
        )
    ):
        raise RuntimeError(
            "Adaptive basis allocation was not "
            "solver-certified optimal."
        )

    bases = basis_payload[
        "bases"
    ]

    if set(targets) != set(bases):
        raise RuntimeError(
            "Adaptive basis/target mismatch"
        )

    if len(targets) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected 288 targets, "
            f"found {len(targets)}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    installed = []
    skipped_rank_zero = []

    for name in sorted(
        targets
    ):
        entry = bases[name]

        rank = int(
            entry["rank"]
        )

        P = entry[
            "P"
        ]

        Q = entry[
            "Q"
        ]

        if (
            P.ndim != 2
            or
            Q.ndim != 2
            or
            P.shape[1] != rank
            or
            Q.shape[1] != rank
        ):
            raise RuntimeError(
                f"{name}: stored basis/rank mismatch "
                f"rank={rank} "
                f"P={tuple(P.shape)} "
                f"Q={tuple(Q.shape)}"
            )

        if rank == 0:
            skipped_rank_zero.append(
                name
            )
            continue

        p_eye = torch.eye(
            rank,
            dtype=torch.float32,
        )

        q_eye = torch.eye(
            rank,
            dtype=torch.float32,
        )

        p_err = float(
            torch.linalg.matrix_norm(
                P.float().T
                @ P.float()
                - p_eye,
                ord="fro",
            ).item()
        )

        q_err = float(
            torch.linalg.matrix_norm(
                Q.float().T
                @ Q.float()
                - q_eye,
                ord="fro",
            ).item()
        )

        if (
            p_err > 1.0e-3
            or
            q_err > 1.0e-3
        ):
            raise RuntimeError(
                f"{name}: stored basis "
                "orthogonality failure "
                f"P={p_err:.6e} "
                f"Q={q_err:.6e}"
            )

        parent, leaf = get_parent(
            model,
            name,
        )

        base = getattr(
            parent,
            leaf,
        )

        setattr(
            parent,
            leaf,
            AdaptiveFCCALinear(
                base=base,
                P=P,
                Q=Q,
                rank=rank,
            ),
        )

        installed.append(
            name
        )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    allocated = sum(
        int(entry["rank"]) ** 2
        for entry in bases.values()
    )

    if (
        allocated != BUDGET
        or
        trainable != BUDGET
    ):
        raise RuntimeError(
            "Adaptive FCCA budget audit failed: "
            f"allocated={allocated:,}, "
            f"trainable={trainable:,}, "
            f"expected={BUDGET:,}"
        )

    print(
        "[architecture] "
        f"adapted={len(installed)} "
        f"rank0={len(skipped_rank_zero)} "
        f"trainable={trainable:,}",
        flush=True,
    )

    return (
        installed,
        skipped_rank_zero,
    )



def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()

    if isinstance(value, dict):
        return {
            key: _cpu_tree(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _cpu_tree(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _cpu_tree(item)
            for item in value
        )

    return value


def save_resumable_epoch_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    path: Path,
    epoch: int,
    global_step: int,
    best_wer: float,
    best_epoch: int,
    bad_epochs: int,
    validation_wer: float,
    validation_loss: float,
    train_loss: float,
    language: str,
    basis_file: Path,
):
    payload = {
        "kind":
            "adaptive_budget_fcca_resumable_epoch",

        "method":
            "adaptive_budget_fcca",

        "language":
            language,

        "epoch":
            int(epoch),

        "next_epoch":
            int(epoch + 1),

        "global_step":
            int(global_step),

        "best_validation_normalised_wer":
            float(best_wer),

        "best_epoch":
            int(best_epoch),

        "bad_epochs":
            int(bad_epochs),

        "validation_normalised_wer":
            float(validation_wer),

        "validation_loss":
            float(validation_loss),

        "train_loss":
            float(train_loss),

        "basis_path":
            str(basis_file),

        "state_dict": {
            key:
                value.detach().cpu()
            for key, value
            in model.state_dict().items()
        },

        "optimizer_state_dict":
            _cpu_tree(
                optimizer.state_dict()
            ),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "rng_state": {
            "python":
                random.getstate(),

            "numpy":
                np.random.get_state(),

            "torch":
                torch.get_rng_state(),

            "cuda":
                (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
        },
    }

    tmp = path.with_suffix(
        path.suffix
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        payload,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


# =====================================================================
# Training
# =====================================================================

def move_batch(
    batch: dict,
    device: torch.device,
):
    result = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(
                device,
                non_blocking=True,
            )

            if value.is_floating_point():
                value = value.to(
                    dtype=torch.bfloat16
                )

        result[key] = value

    return result


def make_train_loader(
    runner,
    ds,
    processor,
    epoch: int,
):
    generator = (
        torch.Generator()
        .manual_seed(
            SEED + epoch
        )
    )

    return DataLoader(
        ds,
        batch_size=MICROBATCH,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda rows:
            runner.collate(
                processor,
                rows,
            ),
    )


def train(
    language: str,
    device: torch.device,
    force: bool,
    no_early_stop: bool,
):
    recipe = RECIPES[
        language
    ]

    basis_file = basis_path(
        language
    )

    if not basis_file.exists():
        raise FileNotFoundError(
            f"Build basis first: {basis_file}"
        )

    out = output_dir(
        language
    )

    if out.exists():
        contents = list(
            out.iterdir()
        )

        if contents and not force:
            raise RuntimeError(
                f"Refusing to overwrite non-empty {out}. "
                "Use --force only intentionally."
            )

        if force:
            shutil.rmtree(
                out
            )

        elif not contents:
            print(
                f"[output] reusing empty directory {out}",
                flush=True,
            )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    runner, runner_path = (
        load_runner(
            language
        )
    )

    train_ds = runner.load_split(
        "train"
    )

    val_ds = runner.load_split(
        "validation"
    )

    processor = (
        runner.build_processor()
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
        )
        .to(device)
    )

    model.config.use_cache = False

    basis_payload = torch.load(
        basis_file,
        map_location="cpu",
        weights_only=False,
    )

    (
        installed,
        rank_zero,
    ) = attach_adaptive_fcca(
        model,
        runner,
        basis_payload,
    )

    parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    trainable = sum(
        parameter.numel()
        for parameter in parameters
    )

    if trainable != BUDGET:
        raise RuntimeError(
            f"Trainable budget mismatch "
            f"{trainable:,} != {BUDGET:,}"
        )

    learning_rate = float(
        recipe[
            "learning_rate"
        ]
    )

    ebs = int(
        recipe[
            "effective_batch_size"
        ]
    )

    warmup_ratio = float(
        recipe[
            "warmup_ratio"
        ]
    )

    if ebs % MICROBATCH != 0:
        raise RuntimeError(
            "EBS not divisible by microbatch"
        )

    grad_acc = (
        ebs
        // MICROBATCH
    )

    first_loader = make_train_loader(
        runner,
        train_ds,
        processor,
        epoch=1,
    )

    updates_per_epoch = math.ceil(
        len(first_loader)
        / grad_acc
    )

    total_steps = (
        updates_per_epoch
        * MAX_EPOCHS
    )

    warmup_steps = int(
        round(
            total_steps
            * warmup_ratio
        )
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                warmup_steps,
            num_training_steps=
                total_steps,
        )
    )

    hyperparameters = {
        "language":
            language,

        "method":
            "adaptive_budget_fcca",

        "base_model":
            BASE_MODEL,

        "runner_file":
            str(
                runner_path
            ),

        "learning_rate":
            learning_rate,

        "effective_batch_size":
            ebs,

        "per_device_train_batch_size":
            MICROBATCH,

        "gradient_accumulation_steps":
            grad_acc,

        "warmup_ratio":
            warmup_ratio,

        "warmup_steps":
            warmup_steps,

        "planned_total_optimizer_steps":
            total_steps,

        "maximum_epochs":
            MAX_EPOCHS,

        "early_stopping_patience":
            (
                None
                if no_early_stop
                else PATIENCE
            ),

        "optimizer":
            "AdamW",

        "optimizer_betas":
            [0.9, 0.999],

        "optimizer_eps":
            1.0e-8,

        "weight_decay":
            WEIGHT_DECAY,

        "max_grad_norm":
            MAX_GRAD_NORM,

        "precision":
            "bf16",

        "adapter_core_precision":
            "fp32",

        "seed":
            SEED,

        "basis_path":
            str(
                basis_file
            ),

        "trainable_parameters":
            trainable,

        "reference_uniform_fcca_rank":
            UNIFORM_REFERENCE_RANK,

        "reference_uniform_fcca_budget":
            BUDGET,

        "rank_allocation":
            basis_payload[
                "rank_allocation"
            ],

        "rank_summary":
            basis_payload[
                "rank_summary"
            ],

        "allocation_solver":
            basis_payload[
                "solver"
            ],

        "adapted_target_count":
            len(installed),

        "rank_zero_target_count":
            len(rank_zero),
    }

    save_json(
        out
        / "hyperparameters.json",
        hyperparameters,
    )

    history = []

    best_wer = math.inf
    best_epoch = 0
    bad_epochs = 0
    global_step = 0

    best_checkpoint = (
        out
        / "best_checkpoint.pt"
    )

    started = time.time()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        loader = make_train_loader(
            runner,
            train_ds,
            processor,
            epoch,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        train_weighted_loss = 0.0
        train_tokens = 0
        accumulated = 0

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            batch = move_batch(
                batch,
                device,
            )

            supervised = int(
                batch["labels"]
                .ne(-100)
                .sum()
                .item()
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    **batch
                )

                loss = outputs.loss

            if (
                loss is None
                or
                not torch.isfinite(
                    loss
                )
            ):
                raise RuntimeError(
                    f"Non-finite loss "
                    f"epoch={epoch} "
                    f"batch={batch_index}"
                )

            train_weighted_loss += (
                float(
                    loss.item()
                )
                * supervised
            )

            train_tokens += supervised

            if accumulated == 0:
                accumulation_target = min(
                    grad_acc,
                    len(loader)
                    - batch_index
                    + 1,
                )

            (
                loss
                / accumulation_target
            ).backward()

            accumulated += 1

            if (
                accumulated
                == accumulation_target
            ):
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    MAX_GRAD_NORM,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1
                accumulated = 0

            if (
                batch_index % 100 == 0
                or
                batch_index == len(loader)
            ):
                print(
                    f"[train] epoch={epoch} "
                    f"batch={batch_index}/{len(loader)} "
                    f"global_step={global_step} "
                    f"loss="
                    f"{train_weighted_loss / max(train_tokens, 1):.6f}",
                    flush=True,
                )

        epoch_train_loss = (
            train_weighted_loss
            / max(
                train_tokens,
                1,
            )
        )

        loss_metrics = (
            runner
            .evaluate_validation_loss(
                model,
                processor,
                val_ds,
                device,
            )
        )

        current_predictions = (
            out
            / f"_validation_epoch_{epoch:02d}.csv"
        )

        wer_metrics = (
            runner.evaluate_wer(
                model,
                processor,
                val_ds,
                device,
                current_predictions,
            )
        )

        current_wer = float(
            wer_metrics[
                "normalised_wer"
            ]
        )

        improved = (
            current_wer
            < best_wer
            - 1.0e-12
        )

        if improved:
            best_wer = current_wer
            best_epoch = epoch
            bad_epochs = 0

            best_predictions = (
                out
                / "validation_predictions.csv"
            )

            os.replace(
                current_predictions,
                best_predictions,
            )

        else:
            bad_epochs += 1

            current_predictions.unlink(
                missing_ok=True
            )

        row = {
            "epoch":
                epoch,

            "global_step":
                global_step,

            "learning_rate_after_epoch":
                optimizer
                .param_groups[0][
                    "lr"
                ],

            "train_loss":
                epoch_train_loss,

            "validation_loss":
                loss_metrics[
                    "validation_loss"
                ],

            "validation_normalised_wer":
                current_wer,

            "best_validation_normalised_wer_so_far":
                best_wer,

            "best_epoch_so_far":
                best_epoch,

            "bad_epochs":
                bad_epochs,

            "is_new_best":
                int(
                    improved
                ),

            "runtime_seconds_total":
                time.time()
                - started,
        }

        history.append(
            row
        )

        write_csv(
            out
            / "training_history.csv",
            history,
        )

        epoch_checkpoint = (
            out
            / f"epoch_{epoch:02d}.pt"
        )

        save_resumable_epoch_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=epoch_checkpoint,
            epoch=epoch,
            global_step=global_step,
            best_wer=best_wer,
            best_epoch=best_epoch,
            bad_epochs=bad_epochs,
            validation_wer=current_wer,
            validation_loss=float(
                loss_metrics[
                    "validation_loss"
                ]
            ),
            train_loss=epoch_train_loss,
            language=language,
            basis_file=basis_file,
        )

        print(
            f"[epoch checkpoint] "
            f"{epoch_checkpoint}",
            flush=True,
        )

        if improved:
            shutil.copy2(
                epoch_checkpoint,
                best_checkpoint,
            )

            print(
                f"[best checkpoint] "
                f"epoch={epoch} -> "
                f"{best_checkpoint}",
                flush=True,
            )

        print(
            f"[epoch] {epoch} "
            f"train_loss={epoch_train_loss:.6f} "
            f"val_loss="
            f"{loss_metrics['validation_loss']:.6f} "
            f"WER={current_wer:.4f}% "
            f"best={best_wer:.4f}% "
            f"bad={bad_epochs}",
            flush=True,
        )

        if (
            not no_early_stop
            and
            bad_epochs >= PATIENCE
        ):
            print(
                f"[early stopping] "
                f"epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_WER={best_wer:.4f}%",
                flush=True,
            )
            break

    result = {
        "method":
            "adaptive_budget_fcca",

        "language":
            language,

        "best_validation_normalised_wer":
            best_wer,

        "best_epoch":
            best_epoch,

        "epochs_ran":
            len(history),

        "global_steps":
            global_step,

        "trainable_parameters":
            BUDGET,

        "checkpoint":
            str(
                best_checkpoint
            ),

        "basis":
            str(
                basis_file
            ),

        "runtime_seconds":
            time.time()
            - started,
    }

    save_json(
        out
        / "trial_result.json",
        result,
    )

    (
        out
        / "COMPLETE"
    ).write_text(
        "complete\n",
        encoding="utf-8",
    )

    print(
        f"[COMPLETE] {out}",
        flush=True,
    )


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
    )

    parser.add_argument(
        "--whisper-language",
        required=True,
        help="Whisper language token, e.g. es, fa, zh, en, kk.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--microbatch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "build",
            "train",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--rank-step",
        type=int,
        default=1,
        help=(
            "Allowed rank spacing. "
            "1 = exact full integer-rank allocation."
        ),
    )

    parser.add_argument(
        "--solver-time-limit",
        type=float,
        default=7200.0,
    )

    parser.add_argument(
        "--no-early-stop",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    # Existing historical entries remain useful defaults, but arbitrary
    # languages are accepted. Training hyperparameters are explicit.
    existing = RECIPES.get(
        args.language,
        {}
    )

    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else existing.get(
            "learning_rate"
        )
    )

    effective_batch_size = (
        args.effective_batch_size
        if args.effective_batch_size is not None
        else existing.get(
            "effective_batch_size"
        )
    )

    warmup_ratio = (
        args.warmup_ratio
        if args.warmup_ratio is not None
        else existing.get(
            "warmup_ratio"
        )
    )

    if args.mode in {
        "train",
        "all",
    }:
        if learning_rate is None:
            raise RuntimeError(
                "--learning-rate is required for this language"
            )

        if effective_batch_size is None:
            raise RuntimeError(
                "--effective-batch-size is required for this language"
            )

        if warmup_ratio is None:
            raise RuntimeError(
                "--warmup-ratio is required for this language"
            )

    # Build-only still needs a recipe object because the common runner
    # configuration uses the language token.
    RECIPES[
        args.language
    ] = {
        "learning_rate":
            (
                0.0
                if learning_rate is None
                else float(
                    learning_rate
                )
            ),

        "effective_batch_size":
            (
                8
                if effective_batch_size is None
                else int(
                    effective_batch_size
                )
            ),

        "warmup_ratio":
            (
                0.05
                if warmup_ratio is None
                else float(
                    warmup_ratio
                )
            ),

        "whisper_language":
            args.whisper_language,
    }

    global MICROBATCH

    MICROBATCH = int(
        args.microbatch
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    device = torch.device(
        args.device
    )

    torch.cuda.set_device(
        device
    )

    set_seed()

    if args.mode in {
        "build",
        "all",
    }:
        build_basis(
            language=args.language,
            device=device,
            rank_step=args.rank_step,
            solver_time_limit=
                args.solver_time_limit,
            force=args.force,
        )

    if args.mode in {
        "train",
        "all",
    }:
        train(
            language=args.language,
            device=device,
            force=args.force,
            no_early_stop=
                args.no_early_stop,
        )


if __name__ == "__main__":
    main()
