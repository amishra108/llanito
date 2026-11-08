from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import os
import sys
import time

import optuna
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import (
    TrainerCallback,
    WhisperForConditionalGeneration,
)
from transformers.trainer_utils import (
    IntervalStrategy,
)


# =============================================================================
# PROJECT
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from scripts_fcca.hpo_shared_resumable import (
    ArchitecturePlugin,
    DataSpec,
    ExperimentSpec,
    OptimisationSearchSpec,
    PlacementSpec,
    TrainingProtocol,
    run_hpo_experiment,
)


# =============================================================================
# STAGE
# =============================================================================

def _extract_stage() -> int:
    stage = os.environ.get(
        "FCCA_STAGE"
    )

    if "--stage" in sys.argv:
        index = sys.argv.index(
            "--stage"
        )

        if index + 1 >= len(
            sys.argv
        ):
            raise ValueError(
                "--stage requires 1, 2, 3, 4, 5, or 6"
            )

        stage = sys.argv[
            index + 1
        ]

        del sys.argv[
            index:index + 2
        ]

    if stage is None:
        stage = "1"

    if stage not in {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    }:
        raise ValueError(
            "FCCA stage must be "
            "1, 2, 3, 4, 5, or 6"
        )

    os.environ[
        "FCCA_STAGE"
    ] = stage

    return int(
        stage
    )


STAGE = _extract_stage()

IS_WORKER = (
    "--worker"
    in sys.argv
)


# Fold selection

def _selected_folds() -> tuple[int, ...]:
    raw = os.environ.get(
        "FCCA_FOLDS",
        "1,2,3,4,5",
    )

    folds = tuple(
        int(x.strip())
        for x in raw.split(",")
        if x.strip()
    )

    if not folds:
        raise ValueError("No folds selected")

    if any(
        fold not in {1, 2, 3, 4, 5}
        for fold in folds
    ):
        raise ValueError(
            f"Invalid FCCA_FOLDS: {folds}"
        )

    return folds


SELECTED_FOLDS = (
    1,
    2,
    3,
    4,
    5,
)


def _selected_lr() -> float | None:
    if STAGE != 6:
        return None

    raw = os.environ.get(
        "FCCA_LR"
    )

    if raw is None:
        raise RuntimeError(
            "Stage 6 requires "
            "FCCA_LR=1.8e-4, 1.5e-4, or 0.8e-5"
        )

    value = float(raw)

    allowed = (
        1.8e-4,
        1.5e-4,
        0.8e-5,
    )

    if not any(
        abs(value - candidate) < 1e-15
        for candidate in allowed
    ):
        raise ValueError(
            f"Invalid FCCA_LR={value}. "
            "Use 1.8e-4, 1.5e-4, or 0.8e-5."
        )

    return value


SELECTED_LR = _selected_lr()

if STAGE == 6:
    if abs(
        float(SELECTED_LR)
        - 1.8e-4
    ) < 1e-15:
        LR_TAG = "lr_1p8e-4"
    elif abs(
        float(SELECTED_LR)
        - 1.5e-4
    ) < 1e-15:
        LR_TAG = "lr_1p5e-4"
    else:
        LR_TAG = "lr_0p8e-5"
else:
    LR_TAG = (
        f"stage_{STAGE}"
    )


# =============================================================================
# FCCA CONSTANTS
# =============================================================================

FCCA_DEFAULT_RANK = 16

FCCA_SCALE = 1.0

FCCA_DAMPING = 1.0e-6

FCCA_CALIBRATION_SEED = 2026

FCCA_CALIBRATION_SIZES = (
    128,
    256,
)

FCCA_STAGE1_WARMUPS = (
    0.0,
    0.03,
    0.10,
)

FCCA_TARGET_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)

CACHE_FORMAT_VERSION = 5


TARGET_MODE_ALL = (
    "all_attention"
)

TARGET_MODE_SELF = (
    "self_attention_only"
)


EXPECTED_TARGETS = {
    TARGET_MODE_ALL:
        288,
    TARGET_MODE_SELF:
        192,
}


# =============================================================================
# STAGE 1/2 CONSTANTS
# =============================================================================

LORA_BEST_LR = 1.0e-4

LORA_BEST_EBS = 32

LORA_ALT_LR = 2.5e-5

LORA_ALT_EBS = 16


# =============================================================================
# STAGE 3 CONSTANTS
# =============================================================================

STAGE3_CALIBRATION_SIZE = 256

STAGE3_WARMUP = 0.03

STAGE3_LR = 1.0e-4


# =============================================================================
# STAGE 4 CONSTANTS
#
# Controlled rank-only experiment.
#
# Hold:
#   target placement = all attention q/k/v/o
#   LR               = 1e-4
#   EBS              = 8
#   per-device batch = 1
#   GA               = 4
#   world size       = 2
#   calibration      = 256
#   warmup           = .03
#
# Change only:
#   rank = 32 vs 64
# =============================================================================

STAGE4_RANKS = (
    32,
    64,
)

STAGE4_TARGET_MODE = (
    TARGET_MODE_ALL
)

STAGE4_CALIBRATION_SIZE = 256

STAGE4_WARMUP = 0.03

STAGE4_LR = 1.0e-4

STAGE4_EBS = 8

STAGE4_GA = 4


# =============================================================================
# STAGE 5 CONSTANTS
#
# Targeted continuation after Stage 4.
#
# Exact five trials:
#
#   0: rank 64, LR 5e-5,   EBS 8,  GA 4
#   1: rank 48, LR 1e-4,   EBS 8,  GA 4
#   2: rank 48, LR 5e-5,   EBS 8,  GA 4
#   3: rank 64, LR 2.5e-5, EBS 8,  GA 4
#   4: rank 64, LR 1e-4,   EBS 32, GA 8
#
# Everything else fixed:
#   all-attention q/k/v/o
#   calibration = 256
#   warmup = .03
#   scale = 1
#   damping = 1e-6
# =============================================================================

STAGE5_RANKS = (
    48,
    64,
)

STAGE5_TARGET_MODE = (
    TARGET_MODE_ALL
)

STAGE5_CALIBRATION_SIZE = 256

STAGE5_WARMUP = 0.03

STAGE5_LR_LOW = 2.5e-5

STAGE5_LR_HIGH = 1.0e-4

STAGE5_EBS_VALUES = (
    8,
    32,
)


# =============================================================================
# STAGE 6 CONSTANTS
#
# Exact five-trial continuation:
#
# 0: r64, LR=6.25e-5, EBS=8,  GA=4
# 1: r64, LR=5.00e-5, EBS=16, GA=8
# 2: r32, LR=1.00e-4, EBS=4,  GA=2
# 3: r32, LR=7.50e-5, EBS=4,  GA=2
# 4: r32, LR=1.50e-4, EBS=8,  GA=4
#
# Fixed:
# all-attention q/k/v/o
# calibration=256
# warmup=.03
# scale=1
# damping=1e-6
# =============================================================================

STAGE6_RANKS = (
    128,
)

STAGE6_TARGET_MODE = (
    TARGET_MODE_ALL
)

STAGE6_CALIBRATION_SIZE = 256

STAGE6_WARMUP = 0.05

STAGE6_LR_LOW = (
    float(SELECTED_LR)
    if SELECTED_LR is not None
    else 0.0
)

STAGE6_LR_HIGH = (
    float(SELECTED_LR)
    if SELECTED_LR is not None
    else 0.0
)

STAGE6_EBS_VALUES = (
    8,
)


# =============================================================================
# PATHS
# =============================================================================

BASIS_CACHE_ROOT = (
    PROJECT_ROOT
    / "week9"
    / "basis_cache"
    / "fcca_fresh_shared"
    / LR_TAG
    / "bases"
)

STAGE1_OUTPUT_ROOT = (
    SCRIPT_PATH.parent
    / "results_stage1"
)

STAGE2_OUTPUT_ROOT = (
    SCRIPT_PATH.parent
    / "results_stage2"
)

STAGE3_OUTPUT_ROOT = (
    SCRIPT_PATH.parent
    / "results_stage3"
)

STAGE4_OUTPUT_ROOT = (
    SCRIPT_PATH.parent
    / "results_stage4"
)

STAGE5_OUTPUT_ROOT = (
    SCRIPT_PATH.parent
    / "results_stage5"
)

STAGE6_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "scripts_fcca"
    / "results"
    / "fcca_fresh_runs"
    / LR_TAG
)


# =============================================================================
# HELPERS
# =============================================================================

def _rank() -> int:
    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        return int(
            dist.get_rank()
        )

    return int(
        os.environ.get(
            "RANK",
            "0",
        )
    )


def _local_rank() -> int:
    return int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )


def _wait_for_file(
    path: Path,
    timeout_seconds: int = 7200,
) -> None:
    deadline = (
        time.time()
        + timeout_seconds
    )

    while not path.exists():
        if (
            time.time()
            >= deadline
        ):
            raise TimeoutError(
                "Timed out waiting for "
                "FCCA basis cache: "
                f"{path}"
            )

        time.sleep(
            1.0
        )


def _rows_fingerprint(
    rows: list[
        dict[str, Any]
    ],
) -> str:
    digest = (
        hashlib.sha256()
    )

    for row in rows:
        encoded = json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )

        digest.update(
            encoded
        )

        digest.update(
            b"\n"
        )

    return (
        digest
        .hexdigest()[:24]
    )


def _basis_cache_path(
    *,
    base_model: str,
    fold_fingerprint: str,
    calibration_size: int,
    rank: int,
    damping: float,
    target_mode: str,
) -> Path:
    model_tag = (
        hashlib.sha1(
            base_model.encode(
                "utf-8"
            )
        )
        .hexdigest()[:10]
    )

    if (
        target_mode
        == TARGET_MODE_ALL
    ):
        target_tag = "all"

    elif (
        target_mode
        == TARGET_MODE_SELF
    ):
        target_tag = "self"

    else:
        raise ValueError(
            "Unknown FCCA target mode: "
            f"{target_mode}"
        )

    return (
        BASIS_CACHE_ROOT
        / (
            f"fcca_v"
            f"{CACHE_FORMAT_VERSION}_"
            f"{model_tag}_"
            f"{fold_fingerprint}_"
            f"n{calibration_size}_"
            f"r{rank}_"
            f"eps{damping:.0e}_"
            f"seed"
            f"{FCCA_CALIBRATION_SEED}_"
            f"{target_tag}.pt"
        )
    )


def _get_parent(
    model: nn.Module,
    name: str,
) -> tuple[
    nn.Module,
    str,
]:
    parts = name.split(
        "."
    )

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


# =============================================================================
# TARGET PLACEMENT
# =============================================================================

def _is_target(
    name: str,
    module: nn.Module,
    target_mode: str,
) -> bool:
    if not isinstance(
        module,
        nn.Linear,
    ):
        return False

    if (
        name.rsplit(
            ".",
            1,
        )[-1]
        not in FCCA_TARGET_LEAVES
    ):
        return False

    # Encoder self-attention.

    if (
        ".encoder.layers."
        in name
        and ".self_attn."
        in name
    ):
        return True

    # Decoder self-attention.

    if (
        ".decoder.layers."
        in name
        and ".self_attn."
        in name
    ):
        return True

    # Decoder cross-attention.

    if (
        target_mode
        == TARGET_MODE_ALL
        and ".decoder.layers."
        in name
        and ".encoder_attn."
        in name
    ):
        return True

    return False


def _collect_targets(
    model: nn.Module,
    *,
    target_mode: str,
) -> dict[
    str,
    nn.Linear,
]:
    if (
        target_mode
        not in EXPECTED_TARGETS
    ):
        raise ValueError(
            "Unknown FCCA target mode: "
            f"{target_mode}"
        )

    targets = {
        name:
            module
        for (
            name,
            module,
        )
        in model.named_modules()
        if _is_target(
            name,
            module,
            target_mode,
        )
    }

    encoder_self = sum(
        (
            ".encoder.layers."
            in name
            and ".self_attn."
            in name
        )
        for name in targets
    )

    decoder_self = sum(
        (
            ".decoder.layers."
            in name
            and ".self_attn."
            in name
        )
        for name in targets
    )

    decoder_cross = sum(
        (
            ".decoder.layers."
            in name
            and ".encoder_attn."
            in name
        )
        for name in targets
    )

    expected_cross = (
        96
        if (
            target_mode
            == TARGET_MODE_ALL
        )
        else 0
    )

    expected_total = (
        EXPECTED_TARGETS[
            target_mode
        ]
    )

    if (
        encoder_self,
        decoder_self,
        decoder_cross,
    ) != (
        96,
        96,
        expected_cross,
    ):
        raise RuntimeError(
            "Unexpected Whisper-medium "
            "FCCA targets: "
            f"mode={target_mode}, "
            f"encoder_self="
            f"{encoder_self}, "
            f"decoder_self="
            f"{decoder_self}, "
            f"decoder_cross="
            f"{decoder_cross}, "
            f"total={len(targets)}"
        )

    if (
        len(targets)
        != expected_total
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_total} "
            "FCCA targets for "
            f"mode={target_mode}, "
            f"found {len(targets)}"
        )

    return targets


def _uses_encoder_time(
    name: str,
) -> bool:
    if (
        ".encoder.layers."
        in name
        and ".self_attn."
        in name
    ):
        return True

    if (
        ".decoder.layers."
        in name
        and ".encoder_attn."
        in name
    ):
        leaf = (
            name.rsplit(
                ".",
                1,
            )[-1]
        )

        return (
            leaf
            in {
                "k_proj",
                "v_proj",
            }
        )

    return False


def _resize_prefix_mask(
    mask: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    if (
        mask.ndim
        != 2
    ):
        raise RuntimeError(
            "Expected 2-D "
            "attention mask, "
            f"got "
            f"{tuple(mask.shape)}"
        )

    mask = mask.to(
        dtype=torch.bool
    )

    source_length = int(
        mask.shape[1]
    )

    if (
        source_length
        == target_length
    ):
        return mask

    valid = (
        mask.long()
        .sum(
            dim=1
        )
    )

    if (
        source_length
        == 2 * target_length
    ):
        target_valid = (
            valid + 1
        ) // 2

    else:
        target_valid = (
            torch.ceil(
                valid.float()
                * (
                    float(
                        target_length
                    )
                    / float(
                        source_length
                    )
                )
            )
            .long()
        )

    target_valid = (
        target_valid
        .clamp(
            min=0,
            max=target_length,
        )
    )

    positions = (
        torch.arange(
            target_length,
            device=mask.device,
        )[None, :]
    )

    return (
        positions
        < target_valid[:, None]
    )


# =============================================================================
# FCCA LINEAR
# =============================================================================

class FCCALinear(
    nn.Module
):
    def __init__(
        self,
        base: nn.Linear,
        P: torch.Tensor,
        Q: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        super().__init__()

        if (
            P.ndim != 2
            or Q.ndim != 2
        ):
            raise ValueError(
                "P/Q must be matrices"
            )

        if (
            P.shape[0]
            != base.out_features
        ):
            raise ValueError(
                "P output dimension mismatch"
            )

        if (
            Q.shape[0]
            != base.in_features
        ):
            raise ValueError(
                "Q input dimension mismatch"
            )

        if (
            P.shape[1]
            != Q.shape[1]
        ):
            raise ValueError(
                "P/Q rank mismatch"
            )

        self.base = base

        self.rank = int(
            P.shape[1]
        )

        self.scale = float(
            scale
        )

        for parameter in (
            base.parameters()
        ):
            parameter.requires_grad = (
                False
            )
            parameter.grad = None

        self.fcca_P = nn.Parameter(
            P.detach()
            .to(
                device=(
                    base.weight.device
                ),
                dtype=(
                    base.weight.dtype
                ),
            )
            .contiguous(),
            requires_grad=False,
        )

        self.fcca_Q = nn.Parameter(
            Q.detach()
            .to(
                device=(
                    base.weight.device
                ),
                dtype=(
                    base.weight.dtype
                ),
            )
            .contiguous(),
            requires_grad=False,
        )

        # R remains FP32, matching previous
        # FCCA experiments.

        self.fcca_core = (
            nn.Parameter(
                torch.zeros(
                    self.rank,
                    self.rank,
                    device=(
                        base.weight.device
                    ),
                    dtype=(
                        torch.float32
                    ),
                )
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        base_output = (
            self.base(
                x
            )
        )

        projected = (
            torch.matmul(
                x,
                self.fcca_Q,
            )
        )

        mixed = (
            torch.matmul(
                projected,
                self.fcca_core
                .transpose(
                    0,
                    1,
                ),
            )
        )

        update = (
            torch.matmul(
                mixed,
                self.fcca_P
                .transpose(
                    0,
                    1,
                ),
            )
        )

        return (
            base_output
            + self.scale
            * update
        )

    @torch.no_grad()
    def merged_weight(
        self,
    ) -> torch.Tensor:
        delta = (
            self.fcca_P.float()
            @ self.fcca_core.float()
            @ self.fcca_Q.float()
            .transpose(
                0,
                1,
            )
        )

        return (
            (
                self.base
                .weight
                .detach()
                .float()
            )
            + self.scale
            * delta
        ).to(
            self.base
            .weight
            .dtype
        )


# =============================================================================
# CALIBRATION CONTEXT
# =============================================================================

class _CalibrationContext:
    encoder_attention_mask: (
        torch.Tensor
        | None
    ) = None

    decoder_mask: (
        torch.Tensor
        | None
    ) = None


CALIBRATION_CONTEXT = (
    _CalibrationContext()
)


class FCCAStatistics:
    def __init__(
        self,
        name: str,
        module: nn.Linear,
    ) -> None:
        self.name = name
        self.module = module

        device = (
            module.weight.device
        )

        self.cross_sum = (
            torch.zeros(
                module.out_features,
                module.in_features,
                device=device,
                dtype=torch.float32,
            )
        )

        self.input_sq_sum = (
            torch.zeros(
                module.in_features,
                device=device,
                dtype=torch.float32,
            )
        )

        self.error_sq_sum = (
            torch.zeros(
                module.out_features,
                device=device,
                dtype=torch.float32,
            )
        )

        self.position_count = 0

        self._handle = None

    def _mask(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = int(
            x.shape[-2]
        )

        if _uses_encoder_time(
            self.name
        ):
            source = (
                CALIBRATION_CONTEXT
                .encoder_attention_mask
            )

            if (
                source is None
            ):
                raise RuntimeError(
                    "Missing encoder mask "
                    f"for {self.name}"
                )

            return (
                _resize_prefix_mask(
                    source,
                    sequence_length,
                )
            )

        source = (
            CALIBRATION_CONTEXT
            .decoder_mask
        )

        if source is None:
            raise RuntimeError(
                "Missing decoder mask "
                f"for {self.name}"
            )

        if (
            int(
                source.shape[1]
            )
            != sequence_length
        ):
            raise RuntimeError(
                "Decoder mask length "
                "mismatch for "
                f"{self.name}: "
                f"mask="
                f"{source.shape[1]}, "
                f"sequence="
                f"{sequence_length}"
            )

        return source.to(
            dtype=torch.bool
        )

    def install(
        self,
    ) -> None:
        def forward_hook(
            module,
            inputs,
            output,
        ):
            del module

            if not torch.is_tensor(
                output
            ):
                raise RuntimeError(
                    "Non-tensor FCCA "
                    "target output: "
                    f"{self.name}"
                )

            x = (
                inputs[0]
                .detach()
            )

            if (
                x.ndim != 3
                or output.ndim != 3
            ):
                raise RuntimeError(
                    "Expected "
                    "[batch,time,hidden] "
                    f"at {self.name}; "
                    f"x="
                    f"{tuple(x.shape)}, "
                    f"y="
                    f"{tuple(output.shape)}"
                )

            mask = (
                self._mask(
                    x
                )
            )

            if (
                tuple(
                    mask.shape
                )
                != tuple(
                    x.shape[:2]
                )
            ):
                raise RuntimeError(
                    "Mask shape mismatch "
                    f"at {self.name}: "
                    f"mask="
                    f"{tuple(mask.shape)}, "
                    f"x="
                    f"{tuple(x.shape)}"
                )

            if (
                not output
                .requires_grad
            ):
                output.requires_grad_(
                    True
                )

            x_selected = (
                x[mask]
                .float()
            )

            if (
                x_selected
                .numel()
                == 0
            ):
                raise RuntimeError(
                    "No valid calibration "
                    f"positions at "
                    f"{self.name}"
                )

            self.input_sq_sum.add_(
                x_selected
                .pow(2)
                .sum(
                    dim=0
                )
            )

            def error_hook(
                grad: torch.Tensor,
            ) -> torch.Tensor:
                grad_selected = (
                    grad.detach()[
                        mask
                    ]
                    .float()
                )

                if (
                    grad_selected.shape[0]
                    != x_selected.shape[0]
                ):
                    raise RuntimeError(
                        "x/delta count "
                        "mismatch at "
                        f"{self.name}"
                    )

                self.cross_sum.add_(
                    grad_selected
                    .transpose(
                        0,
                        1,
                    )
                    @ x_selected
                )

                self.error_sq_sum.add_(
                    grad_selected
                    .pow(2)
                    .sum(
                        dim=0
                    )
                )

                self.position_count += (
                    int(
                        x_selected
                        .shape[0]
                    )
                )

                return grad

            output.register_hook(
                error_hook
            )

        self._handle = (
            self.module
            .register_forward_hook(
                forward_hook
            )
        )

    def remove(
        self,
    ) -> None:
        if (
            self._handle
            is not None
        ):
            self._handle.remove()
            self._handle = None


# =============================================================================
# BASIS CONSTRUCTION
# =============================================================================

def _build_bases(
    *,
    model: nn.Module,
    train_dataset,
    collator,
    calibration_size: int,
    rank: int,
    damping: float,
    target_mode: str,
    fold_fingerprint: str | None = None,
) -> tuple[
    dict[
        str,
        dict[
            str,
            torch.Tensor,
        ],
    ],
    dict[str, Any],
]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FCCA calibration "
            "requires CUDA"
        )

    device = torch.device(
        "cuda",
        _local_rank(),
    )

    torch.cuda.set_device(
        device
    )

    model.to(
        device
    )

    model.eval()

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            False
        )
        parameter.grad = None

    targets = (
        _collect_targets(
            model,
            target_mode=(
                target_mode
            ),
        )
    )

    statistics = {
        name:
            FCCAStatistics(
                name,
                module,
            )
        for (
            name,
            module,
        )
        in targets.items()
    }

    for stat in (
        statistics.values()
    ):
        stat.install()

    if (
        len(train_dataset)
        < calibration_size
    ):
        raise RuntimeError(
            f"Fold has only "
            f"{len(train_dataset)} "
            "training examples but "
            f"calibration_size="
            f"{calibration_size}"
        )

    generator = (
        torch.Generator()
        .manual_seed(
            FCCA_CALIBRATION_SEED
        )
    )

    selected_indices = (
        torch.randperm(
            len(
                train_dataset
            ),
            generator=generator,
        )[
            :calibration_size
        ]
        .tolist()
    )

    loader = DataLoader(
        Subset(
            train_dataset,
            selected_indices,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "FCCA CALIBRATION "
        f"rank={_rank()} "
        f"gpu={_local_rank()} "
        f"n="
        f"{calibration_size} "
        f"targets="
        f"{len(targets)} "
        f"target_mode="
        f"{target_mode} "
        f"r={rank}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    try:
        for (
            batch_index,
            batch,
        ) in enumerate(
            loader,
            start=1,
        ):
            clean_batch = {
                key: (
                    value.to(
                        device,
                        non_blocking=True,
                    )
                    if torch.is_tensor(
                        value
                    )
                    else value
                )
                for (
                    key,
                    value,
                )
                in batch.items()
            }

            attention_mask = (
                clean_batch.get(
                    "attention_mask"
                )
            )

            labels = (
                clean_batch.get(
                    "labels"
                )
            )

            if (
                attention_mask
                is None
            ):
                raise RuntimeError(
                    "Whisper calibration "
                    "batch has no "
                    "attention_mask"
                )

            if labels is None:
                raise RuntimeError(
                    "Whisper calibration "
                    "batch has no labels"
                )

            (
                CALIBRATION_CONTEXT
                .encoder_attention_mask
            ) = (
                attention_mask.bool()
            )

            (
                CALIBRATION_CONTEXT
                .decoder_mask
            ) = (
                labels.ne(
                    -100
                )
            )

            outputs = model(
                **clean_batch
            )

            loss = outputs.loss

            if (
                loss is None
                or not torch.isfinite(
                    loss
                )
            ):
                raise RuntimeError(
                    "Non-finite FCCA "
                    "calibration loss at "
                    f"example "
                    f"{batch_index}: "
                    f"{loss}"
                )

            valid_labels = int(
                labels.ne(
                    -100
                )
                .sum()
                .item()
            )

            if (
                valid_labels
                < 1
            ):
                raise RuntimeError(
                    "Calibration example "
                    "has no supervised "
                    "labels"
                )

            (
                loss
                * float(
                    valid_labels
                )
            ).backward()

            if (
                batch_index % 16
                == 0
                or batch_index
                == calibration_size
            ):
                print(
                    "[FCCA] calibration "
                    f"{batch_index}/"
                    f"{calibration_size}",
                    flush=True,
                )

    finally:
        for stat in (
            statistics.values()
        ):
            stat.remove()

        (
            CALIBRATION_CONTEXT
            .encoder_attention_mask
        ) = None

        (
            CALIBRATION_CONTEXT
            .decoder_mask
        ) = None

    if STAGE == 6 and fold_fingerprint is not None:
        raw_modules = {}

        for raw_name in sorted(statistics):
            raw_stat = statistics[raw_name]

            raw_modules[raw_name] = {
                "cross_sum":
                    raw_stat.cross_sum.detach().cpu().clone(),
                "activation_second_sum":
                    raw_stat.input_sq_sum.detach().cpu().clone(),
                "gradient_second_sum":
                    raw_stat.error_sq_sum.detach().cpu().clone(),
                "positions":
                    int(raw_stat.position_count),
            }

        raw_payload = {
            "schema":
                "llanito_whisper_task_stats_v1",
            "source":
                "fresh_standard_fcca",
            "base_model":
                "openai/whisper-medium",
            "language":
                "es",
            "task":
                "transcribe",
            "fold_fingerprint":
                fold_fingerprint,
            "calibration_size":
                calibration_size,
            "calibration_seed":
                FCCA_CALIBRATION_SEED,
            "damping":
                damping,
            "selected_indices":
                selected_indices,
            "target_count":
                len(raw_modules),
            "modules":
                raw_modules,
        }

        raw_path = (
            PROJECT_ROOT
            / "week9"
            / "basis_cache"
            / "fcca_fresh_shared"
            / LR_TAG
            / "task_stats"
            / f"{fold_fingerprint}.pt"
        )

        raw_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_tmp = raw_path.with_suffix(
            raw_path.suffix
            + f".tmp.{os.getpid()}"
        )

        torch.save(
            raw_payload,
            raw_tmp,
        )

        os.replace(
            raw_tmp,
            raw_path,
        )

        print(
            "[FCCA] saved exact raw task stats: "
            f"{raw_path}",
            flush=True,
        )

    bases: dict[
        str,
        dict[
            str,
            torch.Tensor,
        ],
    ] = {}

    diagnostics: dict[
        str,
        Any,
    ] = {
        "calibration_examples":
            calibration_size,
        "rank":
            rank,
        "damping":
            damping,
        "target_mode":
            target_mode,
        "target_count":
            len(targets),
        "targets":
            {},
    }

    print(
        "[FCCA] constructing "
        "Fisher-whitened bases",
        flush=True,
    )

    for (
        index,
        (
            name,
            module,
        ),
    ) in enumerate(
        targets.items(),
        start=1,
    ):
        stat = statistics[
            name
        ]

        if (
            stat.position_count
            < 1
        ):
            raise RuntimeError(
                "No calibration "
                "positions for "
                f"{name}"
            )

        for (
            label,
            tensor,
        ) in (
            (
                "G",
                stat.cross_sum,
            ),
            (
                "a",
                stat.input_sq_sum,
            ),
            (
                "d",
                stat.error_sq_sum,
            ),
        ):
            if not torch.isfinite(
                tensor
            ).all():
                raise RuntimeError(
                    f"Non-finite "
                    f"{label} "
                    f"for {name}"
                )

        G = stat.cross_sum

        a = (
            stat.input_sq_sum
        )

        d = (
            stat.error_sq_sum
        )

        inv_sqrt_a = (
            torch.rsqrt(
                a
                + float(
                    damping
                )
            )
        )

        inv_sqrt_d = (
            torch.rsqrt(
                d
                + float(
                    damping
                )
            )
        )

        whitened = (
            inv_sqrt_d[
                :,
                None,
            ]
            * G
            * inv_sqrt_a[
                None,
                :,
            ]
        )

        if not torch.isfinite(
            whitened
        ).all():
            raise RuntimeError(
                "Non-finite whitened "
                f"G for {name}"
            )

        (
            U,
            singular_values,
            Vh,
        ) = torch.linalg.svd(
            whitened,
            full_matrices=False,
        )

        available_rank = min(
            int(
                U.shape[1]
            ),
            int(
                Vh.shape[0]
            ),
        )

        if (
            rank
            > available_rank
        ):
            raise RuntimeError(
                f"Rank {rank} "
                "impossible for "
                f"{name}; "
                f"available="
                f"{available_rank}"
            )

        U_r = U[
            :,
            :rank,
        ]

        V_r = (
            Vh[
                :rank,
                :,
            ]
            .transpose(
                0,
                1,
            )
        )

        P0 = (
            inv_sqrt_d[
                :,
                None,
            ]
            * U_r
        )

        Q0 = (
            inv_sqrt_a[
                :,
                None,
            ]
            * V_r
        )

        P, _ = (
            torch.linalg.qr(
                P0,
                mode="reduced",
            )
        )

        Q, _ = (
            torch.linalg.qr(
                Q0,
                mode="reduced",
            )
        )

        identity = (
            torch.eye(
                rank,
                device=device,
                dtype=P.dtype,
            )
        )

        p_error = (
            torch.linalg
            .matrix_norm(
                (
                    P.T
                    @ P
                    - identity
                ),
                ord="fro",
            )
        )

        q_error = (
            torch.linalg
            .matrix_norm(
                (
                    Q.T
                    @ Q
                    - identity
                ),
                ord="fro",
            )
        )

        if (
            float(
                p_error.item()
            )
            > 1e-3
            or float(
                q_error.item()
            )
            > 1e-3
        ):
            raise RuntimeError(
                "FCCA QR "
                "orthogonality failure "
                f"at {name}: "
                f"P="
                f"{p_error.item()}, "
                f"Q="
                f"{q_error.item()}"
            )

        bases[
            name
        ] = {
            "P":
                P.detach().cpu(),
            "Q":
                Q.detach().cpu(),
        }

        diagnostics[
            "targets"
        ][
            name
        ] = {
            "in_features":
                int(
                    module.in_features
                ),
            "out_features":
                int(
                    module.out_features
                ),
            "positions":
                int(
                    stat.position_count
                ),
            "top_singular_value":
                float(
                    singular_values[
                        0
                    ].item()
                ),
            "rank_r_singular_value":
                float(
                    singular_values[
                        rank - 1
                    ].item()
                ),
            "P_orthogonality_error":
                float(
                    p_error.item()
                ),
            "Q_orthogonality_error":
                float(
                    q_error.item()
                ),
        }

        if (
            index % 24 == 0
            or index
            == len(targets)
        ):
            print(
                "[FCCA] bases "
                f"{index}/"
                f"{len(targets)}",
                flush=True,
            )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            False
        )
        parameter.grad = None

    torch.cuda.empty_cache()

    return (
        bases,
        diagnostics,
    )


# =============================================================================
# ATTACH BASES
# =============================================================================

def _attach_bases(
    *,
    model: nn.Module,
    bases: dict[
        str,
        dict[
            str,
            torch.Tensor,
        ],
    ],
    scale: float,
    target_mode: str,
    rank: int,
) -> None:
    targets = (
        _collect_targets(
            model,
            target_mode=(
                target_mode
            ),
        )
    )

    if (
        set(targets)
        != set(bases)
    ):
        missing = sorted(
            set(targets)
            - set(bases)
        )

        extra = sorted(
            set(bases)
            - set(targets)
        )

        raise RuntimeError(
            "FCCA basis/target "
            "mismatch "
            f"missing="
            f"{missing[:5]} "
            f"extra="
            f"{extra[:5]}"
        )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            False
        )
        parameter.grad = None

    for name in sorted(
        targets
    ):
        parent, leaf = (
            _get_parent(
                model,
                name,
            )
        )

        base = getattr(
            parent,
            leaf,
        )

        P = bases[
            name
        ][
            "P"
        ]

        Q = bases[
            name
        ][
            "Q"
        ]

        if (
            int(
                P.shape[1]
            )
            != rank
            or int(
                Q.shape[1]
            )
            != rank
        ):
            raise RuntimeError(
                "Cached FCCA basis "
                "rank mismatch at "
                f"{name}: "
                f"expected={rank}, "
                f"P={P.shape}, "
                f"Q={Q.shape}"
            )

        setattr(
            parent,
            leaf,
            FCCALinear(
                base,
                P,
                Q,
                scale=scale,
            ),
        )

    trainable = [
        (
            name,
            parameter,
        )
        for (
            name,
            parameter,
        )
        in model.named_parameters()
        if parameter.requires_grad
    ]

    expected_target_count = (
        EXPECTED_TARGETS[
            target_mode
        ]
    )

    expected_params = (
        expected_target_count
        * rank
        * rank
    )

    actual_params = sum(
        parameter.numel()
        for (
            _,
            parameter,
        )
        in trainable
    )

    if (
        actual_params
        != expected_params
    ):
        raise RuntimeError(
            "FCCA trainable count "
            "mismatch: "
            f"expected "
            f"{expected_params}, "
            f"got "
            f"{actual_params}"
        )

    if any(
        (
            "fcca_core"
            not in name
        )
        for (
            name,
            _,
        )
        in trainable
    ):
        raise RuntimeError(
            "A non-core FCCA "
            "parameter is trainable"
        )

    if (
        len(trainable)
        != expected_target_count
    ):
        raise RuntimeError(
            f"Expected "
            f"{expected_target_count} "
            "trainable FCCA cores, "
            f"got "
            f"{len(trainable)}"
        )

    max_abs = max(
        float(
            parameter
            .detach()
            .abs()
            .max()
            .item()
        )
        for (
            _,
            parameter,
        )
        in trainable
    )

    if (
        max_abs
        != 0.0
    ):
        raise RuntimeError(
            "FCCA cores are not "
            "exactly zero initialized"
        )

    print(
        "[FCCA] attached "
        f"{expected_target_count} "
        f"rank-{rank} bases; "
        f"trainable parameters="
        f"{actual_params}",
        flush=True,
    )


# =============================================================================
# EPOCH CORE SNAPSHOTS
# =============================================================================

class FCCACoreSnapshotCallback(
    TrainerCallback
):
    def __init__(
        self,
        model: nn.Module,
        directory: Path,
        basis_cache: Path,
        expected_target_count: int,
        rank: int,
    ) -> None:
        self.model = model

        self.directory = (
            directory
        )

        self.basis_cache = (
            basis_cache
        )

        self.expected_target_count = int(
            expected_target_count
        )

        self.rank = int(
            rank
        )

    def on_evaluate(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        del args, kwargs

        if not getattr(
            state,
            "is_world_process_zero",
            True,
        ):
            return control

        if (
            state.epoch
            is None
        ):
            return control

        epoch = int(
            round(
                float(
                    state.epoch
                )
            )
        )

        if (
            epoch < 1
        ):
            return control

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        cores = {
            name: (
                parameter
                .detach()
                .float()
                .cpu()
                .clone()
            )
            for (
                name,
                parameter,
            )
            in self.model
            .named_parameters()
            if name.endswith(
                "fcca_core"
            )
        }

        if (
            len(cores)
            != self
            .expected_target_count
        ):
            raise RuntimeError(
                "Snapshot found "
                f"{len(cores)} "
                "FCCA cores, "
                f"expected "
                f"{self.expected_target_count}"
            )

        for (
            name,
            core,
        ) in cores.items():
            if (
                tuple(
                    core.shape
                )
                != (
                    self.rank,
                    self.rank,
                )
            ):
                raise RuntimeError(
                    "Snapshot core rank "
                    "mismatch for "
                    f"{name}: "
                    f"expected "
                    f"{self.rank}x"
                    f"{self.rank}, "
                    f"got "
                    f"{tuple(core.shape)}"
                )

        payload = {
            "format_version":
                1,
            "epoch":
                epoch,
            "global_step":
                int(
                    state.global_step
                ),
            "basis_cache":
                str(
                    self.basis_cache
                ),
            "rank":
                self.rank,
            "scale":
                FCCA_SCALE,
            "expected_target_count":
                self
                .expected_target_count,
            "cores":
                cores,
        }

        final = (
            self.directory
            / "latest.pt"
        )

        temporary = (
            final.with_suffix(
                final.suffix
                + f".tmp."
                f"{os.getpid()}"
            )
        )

        torch.save(
            payload,
            temporary,
        )

        os.replace(
            temporary,
            final,
        )

        return control


# =============================================================================
# STAGE-1 WINNER
# =============================================================================

def _load_stage1_winner(
) -> dict[str, Any]:
    path = (
        STAGE1_OUTPUT_ROOT
        / "best_recipe.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Stage 2 requires "
            "completed Stage 1 "
            "best_recipe.json: "
            + str(path)
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    params = payload.get(
        "best_params",
        {},
    )

    required = {
        "fcca_calibration_size",
        "fcca_warmup_ratio",
    }

    if not required.issubset(
        params
    ):
        raise RuntimeError(
            "Stage-1 best recipe "
            "lacks FCCA parameters: "
            f"{params}"
        )

    return payload


STAGE1_WINNER = (
    _load_stage1_winner()
    if STAGE == 2
    else None
)


# =============================================================================
# PLUGIN
# =============================================================================

class FCCAPlugin(
    ArchitecturePlugin
):
    method_name = (
        "fcca_frozen_core"
    )

    def __init__(
        self,
    ) -> None:
        self._fold_fingerprint: (
            str
            | None
        ) = None

    # -------------------------------------------------------------------------
    # HYPERPARAMETER SAMPLING
    # -------------------------------------------------------------------------

    def sample_hparams(
        self,
        trial: optuna.Trial,
        experiment: ExperimentSpec,
    ) -> dict[
        str,
        Any,
    ]:
        del experiment

        if (
            STAGE == 1
        ):
            rank = (
                FCCA_DEFAULT_RANK
            )

            calibration_size = (
                trial.suggest_categorical(
                    "fcca_calibration_size",
                    list(
                        FCCA_CALIBRATION_SIZES
                    ),
                )
            )

            warmup_ratio = (
                trial.suggest_categorical(
                    "fcca_warmup_ratio",
                    list(
                        FCCA_STAGE1_WARMUPS
                    ),
                )
            )

            target_mode = (
                TARGET_MODE_ALL
            )

            ga_override = None

        elif (
            STAGE == 2
        ):
            assert (
                STAGE1_WINNER
                is not None
            )

            params = (
                STAGE1_WINNER[
                    "best_params"
                ]
            )

            rank = (
                FCCA_DEFAULT_RANK
            )

            calibration_size = int(
                params[
                    "fcca_calibration_size"
                ]
            )

            warmup_ratio = float(
                params[
                    "fcca_warmup_ratio"
                ]
            )

            target_mode = (
                TARGET_MODE_ALL
            )

            ga_override = None

        elif (
            STAGE == 3
        ):
            rank = (
                FCCA_DEFAULT_RANK
            )

            calibration_size = (
                STAGE3_CALIBRATION_SIZE
            )

            warmup_ratio = (
                STAGE3_WARMUP
            )

            target_mode = (
                trial.suggest_categorical(
                    "fcca_target_mode",
                    [
                        TARGET_MODE_ALL,
                        TARGET_MODE_SELF,
                    ],
                )
            )

            ga_override = int(
                trial.suggest_categorical(
                    "fcca_gradient_accumulation_steps",
                    [
                        4,
                        8,
                    ],
                )
            )

        elif (
            STAGE == 4
        ):
            # -------------------------------------------------------------
            # STAGE 4
            # -------------------------------------------------------------
            #
            # Rank-only experiment:
            # 32 vs 64.
            # -------------------------------------------------------------

            rank = int(
                trial.suggest_categorical(
                    "fcca_rank",
                    list(
                        STAGE4_RANKS
                    ),
                )
            )

            calibration_size = (
                STAGE4_CALIBRATION_SIZE
            )

            warmup_ratio = (
                STAGE4_WARMUP
            )

            target_mode = (
                STAGE4_TARGET_MODE
            )

            ga_override = (
                STAGE4_GA
            )

        elif (
            STAGE == 5
        ):
            rank = int(
                trial.suggest_categorical(
                    "fcca_rank",
                    list(
                        STAGE5_RANKS
                    ),
                )
            )

            calibration_size = (
                STAGE5_CALIBRATION_SIZE
            )

            warmup_ratio = (
                STAGE5_WARMUP
            )

            target_mode = (
                STAGE5_TARGET_MODE
            )

            ga_override = int(
                trial.suggest_categorical(
                    "fcca_gradient_accumulation_steps",
                    [
                        4,
                        8,
                    ],
                )
            )

        else:
            rank = int(
                trial.suggest_categorical(
                    "fcca_rank",
                    list(
                        STAGE6_RANKS
                    ),
                )
            )

            calibration_size = (
                STAGE6_CALIBRATION_SIZE
            )

            warmup_ratio = (
                STAGE6_WARMUP
            )

            target_mode = (
                STAGE6_TARGET_MODE
            )

            ga_override = int(
                trial.suggest_categorical(
                    "fcca_gradient_accumulation_steps",
                    [
                        4,
                    ],
                )
            )

        result = {
            "fcca_rank":
                int(rank),
            "fcca_calibration_size":
                int(
                    calibration_size
                ),
            "fcca_warmup_ratio":
                float(
                    warmup_ratio
                ),
            "fcca_scale":
                FCCA_SCALE,
            "fcca_damping":
                FCCA_DAMPING,
            "fcca_target_mode":
                target_mode,
        }

        if (
            ga_override
            is not None
        ):
            result[
                "gradient_accumulation_steps"
            ] = int(
                ga_override
            )

        return result

    # -------------------------------------------------------------------------
    # EXACT STARTUP RECIPES
    # -------------------------------------------------------------------------

    def startup_hparams(
        self,
        experiment: ExperimentSpec,
    ) -> list[
        dict[str, Any]
    ]:
        del experiment

        # ---------------------------------------------------------------------
        # STAGE 1
        # ---------------------------------------------------------------------

        if STAGE == 1:
            return [
                {
                    "fcca_calibration_size":
                        calibration_size,
                    "fcca_warmup_ratio":
                        warmup,
                }
                for warmup
                in FCCA_STAGE1_WARMUPS
                for calibration_size
                in FCCA_CALIBRATION_SIZES
            ]

        # ---------------------------------------------------------------------
        # STAGE 2
        # ---------------------------------------------------------------------

        if STAGE == 2:
            return []

        # ---------------------------------------------------------------------
        # STAGE 3
        # ---------------------------------------------------------------------

        if STAGE == 3:
            return [
                {
                    "fcca_target_mode":
                        TARGET_MODE_SELF,
                    "fcca_gradient_accumulation_steps":
                        8,
                    "effective_global_batch_size":
                        32,
                },
                {
                    "fcca_target_mode":
                        TARGET_MODE_ALL,
                    "fcca_gradient_accumulation_steps":
                        4,
                    "effective_global_batch_size":
                        8,
                },
                {
                    "fcca_target_mode":
                        TARGET_MODE_SELF,
                    "fcca_gradient_accumulation_steps":
                        4,
                    "effective_global_batch_size":
                        8,
                },
            ]

        # ---------------------------------------------------------------------
        # STAGE 4
        #
        # Trial 0: rank 32
        # Trial 1: rank 64
        # ---------------------------------------------------------------------

        if STAGE == 4:
            return [
                {
                    "fcca_rank":
                        32,
                    "fcca_target_mode":
                        STAGE4_TARGET_MODE,
                    "fcca_gradient_accumulation_steps":
                        STAGE4_GA,
                    "effective_global_batch_size":
                        STAGE4_EBS,
                },
                {
                    "fcca_rank":
                        64,
                    "fcca_target_mode":
                        STAGE4_TARGET_MODE,
                    "fcca_gradient_accumulation_steps":
                        STAGE4_GA,
                    "effective_global_batch_size":
                        STAGE4_EBS,
                },
            ]

        # ---------------------------------------------------------------------
        # STAGE 5
        #
        # Exact original five trials.
        # ---------------------------------------------------------------------

        if STAGE == 5:
            return [
                {
                    "fcca_rank":
                        64,
                    "fcca_target_mode":
                        STAGE5_TARGET_MODE,
                    "fcca_gradient_accumulation_steps":
                        4,
                    "learning_rate":
                        5.0e-5,
                    "effective_global_batch_size":
                        8,
                },
                {
                    "fcca_rank":
                        48,
                    "fcca_target_mode":
                        STAGE5_TARGET_MODE,
                    "fcca_gradient_accumulation_steps":
                        4,
                    "learning_rate":
                        1.0e-4,
                    "effective_global_batch_size":
                        8,
                },
                {
                    "fcca_rank":
                        48,
                    "fcca_target_mode":
                        STAGE5_TARGET_MODE,
                    "fcca_gradient_accumulation_steps":
                        4,
                    "learning_rate":
                        5.0e-5,
                    "effective_global_batch_size":
                        8,
                },
                {
                    "fcca_rank":
                        64,
                    "fcca_target_mode":
                        STAGE5_TARGET_MODE,
                    "fcca_gradient_accumulation_steps":
                        4,
                    "learning_rate":
                        2.5e-5,
                    "effective_global_batch_size":
                        8,
                },
                {
                    "fcca_rank":
                        64,
                    "fcca_target_mode":
                        STAGE5_TARGET_MODE,
                    "fcca_gradient_accumulation_steps":
                        8,
                    "learning_rate":
                        1.0e-4,
                    "effective_global_batch_size":
                        32,
                },
            ]

        # ---------------------------------------------------------------------
        # STAGE 6
        #
        # Exact five targeted experiments:
        #
        # T0: r64 / LR6.25e-5 / EBS8  / GA4
        # T1: r64 / LR5e-5    / EBS16 / GA8
        # T2: r32 / LR1e-4    / EBS4  / GA2
        # T3: r32 / LR7.5e-5  / EBS4  / GA2
        # T4: r32 / LR1.5e-4  / EBS8  / GA4
        # ---------------------------------------------------------------------

        if STAGE == 6:
            return [
                {
                    "fcca_rank":
                        128,
                    "fcca_target_mode":
                        STAGE6_TARGET_MODE,
                    "fcca_gradient_accumulation_steps":
                        4,
                    "learning_rate":
                        float(SELECTED_LR),
                    "effective_global_batch_size":
                        8,
                },
            ]

        raise RuntimeError(
            f"Unhandled FCCA stage: {STAGE}"
        )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    def build_model(
        self,
        *,
        experiment: ExperimentSpec,
        architecture_hparams: dict[
            str,
            Any,
        ],
    ) -> nn.Module:
        del architecture_hparams

        model = (
            WhisperForConditionalGeneration
            .from_pretrained(
                experiment.base_model,
                local_files_only=True,
            )
        )

        for parameter in (
            model.parameters()
        ):
            parameter.requires_grad = (
                False
            )

        return model

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    def build_datasets(
        self,
        *,
        default_dataset_cls,
        train_rows,
        validation_rows,
        processor,
        experiment: ExperimentSpec,
        architecture_hparams: dict[
            str,
            Any,
        ],
    ):
        del (
            experiment,
            architecture_hparams,
        )

        self._fold_fingerprint = (
            _rows_fingerprint(
                [
                    dict(
                        row
                    )
                    for row
                    in train_rows
                ]
            )
        )

        return (
            default_dataset_cls(
                train_rows,
                processor,
            ),
            default_dataset_cls(
                validation_rows,
                processor,
            ),
        )

    # -------------------------------------------------------------------------
    # TRAINER
    # -------------------------------------------------------------------------

    def create_trainer(
        self,
        *,
        default_make_trainer,
        model,
        training_args,
        processor,
        train_dataset,
        validation_dataset,
        collator,
        compute_metrics,
        experiment: ExperimentSpec,
        architecture_hparams: dict[
            str,
            Any,
        ],
    ):
        if (
            self._fold_fingerprint
            is None
        ):
            raise RuntimeError(
                "FCCA fold fingerprint "
                "was not initialized"
            )

        rank = int(
            architecture_hparams[
                "fcca_rank"
            ]
        )

        calibration_size = int(
            architecture_hparams[
                "fcca_calibration_size"
            ]
        )

        warmup_ratio = float(
            architecture_hparams[
                "fcca_warmup_ratio"
            ]
        )

        scale = float(
            architecture_hparams[
                "fcca_scale"
            ]
        )

        damping = float(
            architecture_hparams[
                "fcca_damping"
            ]
        )

        target_mode = str(
            architecture_hparams[
                "fcca_target_mode"
            ]
        )

        # Exact rank validation.

        if STAGE in {
            1,
            2,
            3,
        }:
            if (
                rank
                != FCCA_DEFAULT_RANK
            ):
                raise RuntimeError(
                    "FCCA Stage "
                    f"{STAGE} must use "
                    "rank 16; got "
                    f"{rank}"
                )

        elif (
            STAGE == 4
        ):
            if (
                rank
                not in STAGE4_RANKS
            ):
                raise RuntimeError(
                    "FCCA Stage 4 rank "
                    "must be 32 or 64; "
                    f"got {rank}"
                )

            if (
                target_mode
                != STAGE4_TARGET_MODE
            ):
                raise RuntimeError(
                    "FCCA Stage 4 must "
                    "use all-attention "
                    "placement."
                )

            if (
                calibration_size
                != STAGE4_CALIBRATION_SIZE
            ):
                raise RuntimeError(
                    "FCCA Stage 4 must "
                    "use calibration=256."
                )

            if (
                abs(
                    warmup_ratio
                    - STAGE4_WARMUP
                )
                > 1e-12
            ):
                raise RuntimeError(
                    "FCCA Stage 4 must "
                    "use warmup=.03."
                )

        elif (
            STAGE == 5
        ):
            if (
                rank
                not in STAGE5_RANKS
            ):
                raise RuntimeError(
                    "FCCA Stage 5 rank "
                    "must be 48 or 64; "
                    f"got {rank}"
                )

            if (
                target_mode
                != STAGE5_TARGET_MODE
            ):
                raise RuntimeError(
                    "FCCA Stage 5 must "
                    "use all-attention "
                    "placement."
                )

            if (
                calibration_size
                != STAGE5_CALIBRATION_SIZE
            ):
                raise RuntimeError(
                    "FCCA Stage 5 must "
                    "use calibration=256."
                )

            if (
                abs(
                    warmup_ratio
                    - STAGE5_WARMUP
                )
                > 1e-12
            ):
                raise RuntimeError(
                    "FCCA Stage 5 must "
                    "use warmup=.03."
                )

        else:
            if rank != 128:
                raise RuntimeError(
                    "Week 9 FCCA must use "
                    f"rank 128; got {rank}"
                )

            if (
                target_mode
                != STAGE6_TARGET_MODE
            ):
                raise RuntimeError(
                    "FCCA Stage 6 must "
                    "use all-attention "
                    "placement."
                )

            if (
                calibration_size
                != STAGE6_CALIBRATION_SIZE
            ):
                raise RuntimeError(
                    "FCCA Stage 6 must "
                    "use calibration=256."
                )

            if (
                abs(
                    warmup_ratio
                    - STAGE6_WARMUP
                )
                > 1e-12
            ):
                raise RuntimeError(
                    "FCCA Stage 6 must "
                    "use warmup=.05."
                )

        if (
            target_mode
            not in EXPECTED_TARGETS
        ):
            raise RuntimeError(
                "Invalid FCCA "
                "target mode: "
                f"{target_mode}"
            )

        # Trial-resolved warmup.
        #
        # This is the value actually used
        # by Hugging Face Trainer.

        training_args.warmup_steps = (
            0
        )

        training_args.warmup_ratio = (
            warmup_ratio
        )

        training_args.save_strategy = (
            IntervalStrategy.EPOCH
        )

        (
            training_args
            .load_best_model_at_end
        ) = False

        (
            training_args
            .save_total_limit
        ) = 1

        basis_cache = (
            _basis_cache_path(
                base_model=(
                    experiment.base_model
                ),
                fold_fingerprint=(
                    self._fold_fingerprint
                ),
                calibration_size=(
                    calibration_size
                ),
                rank=rank,
                damping=damping,
                target_mode=(
                    target_mode
                ),
            )
        )

        # -------------------------------------------------------------
        # BUILD/CACHE FCCA BASES ON RANK ZERO
        # -------------------------------------------------------------

        if (
            _rank() == 0
        ):
            BASIS_CACHE_ROOT.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not (
                basis_cache.exists()
            ):
                (
                    bases,
                    diagnostics,
                ) = _build_bases(
                    model=model,
                    train_dataset=(
                        train_dataset
                    ),
                    collator=collator,
                    calibration_size=(
                        calibration_size
                    ),
                    rank=rank,
                    damping=damping,
                    target_mode=(
                        target_mode
                    ),
                    fold_fingerprint=(
                        self._fold_fingerprint
                    ),
                )

                payload = {
                    "format_version":
                        CACHE_FORMAT_VERSION,
                    "method":
                        "FCCA",
                    "paper":
                        "arXiv:2609.00762",
                    "base_model":
                        experiment.base_model,
                    "fold_fingerprint":
                        self._fold_fingerprint,
                    "calibration_seed":
                        FCCA_CALIBRATION_SEED,
                    "calibration_size":
                        calibration_size,
                    "rank":
                        rank,
                    "scale":
                        scale,
                    "damping":
                        damping,
                    "target_mode":
                        target_mode,
                    "targets":
                        sorted(
                            bases
                        ),
                    "bases":
                        bases,
                    "diagnostics":
                        diagnostics,
                }

                temporary = (
                    basis_cache
                    .with_suffix(
                        basis_cache.suffix
                        + f".tmp."
                        f"{os.getpid()}"
                    )
                )

                torch.save(
                    payload,
                    temporary,
                )

                os.replace(
                    temporary,
                    basis_cache,
                )

                print(
                    "[FCCA] cached bases: "
                    f"{basis_cache}",
                    flush=True,
                )

            else:
                print(
                    "[FCCA] reusing bases: "
                    f"{basis_cache}",
                    flush=True,
                )

        else:
            _wait_for_file(
                basis_cache
            )

        payload = torch.load(
            basis_cache,
            map_location="cpu",
            weights_only=True,
        )

        checks = {
            "format_version":
                CACHE_FORMAT_VERSION,
            "base_model":
                experiment.base_model,
            "fold_fingerprint":
                self._fold_fingerprint,
            "calibration_seed":
                FCCA_CALIBRATION_SEED,
            "calibration_size":
                calibration_size,
            "rank":
                rank,
            "target_mode":
                target_mode,
        }

        for (
            key,
            expected,
        ) in checks.items():
            if (
                payload.get(
                    key
                )
                != expected
            ):
                raise RuntimeError(
                    "FCCA cache mismatch "
                    f"for {key}: "
                    f"expected="
                    f"{expected!r}, "
                    f"got="
                    f"{payload.get(key)!r}"
                )

        _attach_bases(
            model=model,
            bases=(
                payload[
                    "bases"
                ]
            ),
            scale=scale,
            target_mode=(
                target_mode
            ),
            rank=rank,
        )

        trainer = (
            default_make_trainer(
                model=model,
                training_args=(
                    training_args
                ),
                processor=processor,
                train_dataset=(
                    train_dataset
                ),
                validation_dataset=(
                    validation_dataset
                ),
                collator=collator,
                compute_metrics=(
                    compute_metrics
                ),
            )
        )

        snapshot_dir = (
            Path(
                training_args
                .output_dir
            )
            .parent
            .parent
            / "fcca_core_snapshots"
        )

        trainer.add_callback(
            FCCACoreSnapshotCallback(
                model=model,
                directory=(
                    snapshot_dir
                ),
                basis_cache=(
                    basis_cache
                ),
                expected_target_count=(
                    EXPECTED_TARGETS[
                        target_mode
                    ]
                ),
                rank=rank,
            )
        )

        return trainer

    # -------------------------------------------------------------------------
    # METADATA
    # -------------------------------------------------------------------------

    def metadata(
        self,
        *,
        experiment: ExperimentSpec,
        architecture_hparams: dict[
            str,
            Any,
        ],
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "method_name":
                self.method_name,
            "paper":
                "arXiv:2609.00762",
            "stage":
                STAGE,
            "placement": {
                "name":
                    experiment
                    .placement
                    .name,
                "region":
                    experiment
                    .placement
                    .region,
                "extra":
                    experiment
                    .placement
                    .extra,
            },
            "architecture_hparams":
                architecture_hparams,
            "basis_rule": (
                "G,a,d sums -> "
                "D^-1/2 G A^-1/2 -> "
                "rank-r SVD -> "
                "inverse scaling -> "
                "thin QR"
            ),
            "calibration_data":
                (
                    "current CV fold "
                    "training set only"
                ),
            "validation_used_for_calibration":
                False,
        }


# =============================================================================
# EXPERIMENT CONSTRUCTION
# =============================================================================

def _build_experiment(
) -> ExperimentSpec:
    if (
        STAGE == 1
    ):
        output_root = (
            STAGE1_OUTPUT_ROOT
        )

        learning_rate_low = (
            LORA_BEST_LR
        )

        learning_rate_high = (
            LORA_BEST_LR
        )

        batches = (
            LORA_BEST_EBS,
        )

        startup = (
            (
                (
                    "learning_rate",
                    LORA_BEST_LR,
                ),
                (
                    "effective_global_batch_size",
                    LORA_BEST_EBS,
                ),
            ),
        )

        protocol_warmup = 0.0

        protocol_ga = 8

        study_name = (
            "week5_fcca_l_stage1"
        )

        description = (
            "FCCA Stage 1: rank-16 "
            "all-attention q/k/v/o "
            "Whisper adaptation. "
            "Fix LR=1e-4, EBS=32; "
            "search calibration "
            "{128,256} x warmup "
            "{0,.03,.10}."
        )

    elif (
        STAGE == 2
    ):
        output_root = (
            STAGE2_OUTPUT_ROOT
        )

        learning_rate_low = (
            LORA_ALT_LR
        )

        learning_rate_high = (
            LORA_BEST_LR
        )

        batches = (
            LORA_ALT_EBS,
            LORA_BEST_EBS,
        )

        startup = (
            (
                (
                    "learning_rate",
                    LORA_BEST_LR,
                ),
                (
                    "effective_global_batch_size",
                    LORA_ALT_EBS,
                ),
            ),
            (
                (
                    "learning_rate",
                    LORA_ALT_LR,
                ),
                (
                    "effective_global_batch_size",
                    LORA_BEST_EBS,
                ),
            ),
            (
                (
                    "learning_rate",
                    LORA_ALT_LR,
                ),
                (
                    "effective_global_batch_size",
                    LORA_ALT_EBS,
                ),
            ),
        )

        # Stage 2's actual warmup is loaded from
        # the Stage-1 winner inside the plugin.
        #
        # The winner is known to be .03 here,
        # so make the static protocol metadata
        # truthful as well.

        protocol_warmup = 0.03

        protocol_ga = 8

        study_name = (
            "week5_fcca_l_stage2"
        )

        description = (
            "FCCA Stage 2: hold "
            "Stage-1 winning "
            "calibration/warmup; "
            "run three non-duplicate "
            "LR/EBS optimizer points."
        )

    elif (
        STAGE == 3
    ):
        output_root = (
            STAGE3_OUTPUT_ROOT
        )

        learning_rate_low = (
            STAGE3_LR
        )

        learning_rate_high = (
            STAGE3_LR
        )

        batches = (
            8,
            32,
        )

        startup = (
            (
                (
                    "learning_rate",
                    STAGE3_LR,
                ),
            ),
        )

        protocol_warmup = (
            STAGE3_WARMUP
        )

        protocol_ga = 8

        study_name = (
            "week5_fcca_l_stage3"
        )

        description = (
            "FCCA Stage 3 controlled "
            "2x2 completion: hold "
            "LR=1e-4, calibration=256, "
            "warmup=.03, rank=16. "
            "Run only "
            "self/EBS32/GA8, "
            "all/EBS8/GA4, "
            "self/EBS8/GA4. "
            "Existing all/EBS32/GA8 "
            "Stage-1 winner is "
            "the fourth cell."
        )

    elif (
        STAGE == 4
    ):
        # ================================================================
        # STAGE 4
        # ================================================================

        output_root = (
            STAGE4_OUTPUT_ROOT
        )

        learning_rate_low = (
            STAGE4_LR
        )

        learning_rate_high = (
            STAGE4_LR
        )

        batches = (
            STAGE4_EBS,
        )

        startup = (
            (
                (
                    "learning_rate",
                    STAGE4_LR,
                ),
                (
                    "effective_global_batch_size",
                    STAGE4_EBS,
                ),
            ),
        )

        protocol_warmup = (
            STAGE4_WARMUP
        )

        protocol_ga = (
            STAGE4_GA
        )

        study_name = (
            "week5_fcca_l_stage4"
        )

        description = (
            "FCCA Stage 4 rank-only "
            "ablation. Hold "
            "all-attention q/k/v/o "
            "placement, LR=1e-4, "
            "EBS=8, per-device batch=1, "
            "GA=4, calibration=256, "
            "warmup=.03, scale=1, "
            "damping=1e-6. "
            "Run exactly rank 32 "
            "and rank 64."
        )

    elif (
        STAGE == 5
    ):
        # ================================================================
        # STAGE 5
        # ================================================================

        output_root = (
            STAGE5_OUTPUT_ROOT
        )

        learning_rate_low = (
            STAGE5_LR_LOW
        )

        learning_rate_high = (
            STAGE5_LR_HIGH
        )

        batches = (
            STAGE5_EBS_VALUES
        )

        # Only ONE common startup recipe.
        #
        # The five exact architecture startup
        # recipes override LR/EBS individually.
        # This prevents a 5 x N cross-product.

        startup = (
            (
                (
                    "learning_rate",
                    1.0e-4,
                ),
                (
                    "effective_global_batch_size",
                    8,
                ),
            ),
        )

        protocol_warmup = (
            STAGE5_WARMUP
        )

        # Stage-5 recipes override GA where needed.
        # Base value 4 matches the EBS8 cases.

        protocol_ga = 4

        study_name = (
            "week5_fcca_l_stage5"
        )

        description = (
            "FCCA Stage 5 targeted rank/LR/EBS "
            "search after the rank-64 Fold-5 rescue. "
            "Hold all-attention q/k/v/o placement, "
            "calibration=256, warmup=.03, "
            "scale=1, damping=1e-6. "
            "Run exactly five recipes: "
            "r64/LR5e-5/EBS8, "
            "r48/LR1e-4/EBS8, "
            "r48/LR5e-5/EBS8, "
            "r64/LR2.5e-5/EBS8, "
            "r64/LR1e-4/EBS32."
        )

    else:
        # ================================================================
        # STAGE 6
        # ================================================================

        output_root = (
            STAGE6_OUTPUT_ROOT
        )

        learning_rate_low = (
            STAGE6_LR_LOW
        )

        learning_rate_high = (
            STAGE6_LR_HIGH
        )

        batches = (
            STAGE6_EBS_VALUES
        )

        startup = (
            (
                (
                    "learning_rate",
                    float(SELECTED_LR),
                ),
                (
                    "effective_global_batch_size",
                    8,
                ),
            ),
        )

        protocol_warmup = (
            STAGE6_WARMUP
        )

        protocol_ga = 4

        study_name = (
            f"week9_fcca_{LR_TAG}"
        )

        description = (
            "Week 9 normal FCCA LR endpoint run. "
            "Whisper-medium; five-fold CV; "
            "rank=128; calibration=256; "
            "all-attention q/k/v/o; "
            "EBS=8; GA=4; world_size=2; "
            "warmup=.05; scale=1; damping=1e-6; "
            "max_epochs=10. "
            "Only learning rate differs."
        )

    # ---------------------------------------------------------------------
    # PLACEMENT METADATA
    # ---------------------------------------------------------------------

    placement_extra: dict[
        str,
        Any,
    ] = {
        "target_leaves":
            list(
                FCCA_TARGET_LEAVES
            ),
        "scale":
            FCCA_SCALE,
        "damping":
            FCCA_DAMPING,
    }

    if (
        STAGE == 4
    ):
        placement_extra.update(
            {
                "ranks":
                    list(
                        STAGE4_RANKS
                    ),
                "target_mode":
                    STAGE4_TARGET_MODE,
                "encoder_self_attention":
                    True,
                "decoder_self_attention":
                    True,
                "decoder_cross_attention":
                    True,
            }
        )

    elif (
        STAGE == 5
    ):
        placement_extra.update(
            {
                "ranks":
                    list(
                        STAGE5_RANKS
                    ),
                "target_mode":
                    STAGE5_TARGET_MODE,
                "encoder_self_attention":
                    True,
                "decoder_self_attention":
                    True,
                "decoder_cross_attention":
                    True,
                "calibration_size":
                    STAGE5_CALIBRATION_SIZE,
                "warmup_ratio":
                    STAGE5_WARMUP,
            }
        )

    elif (
        STAGE == 6
    ):
        placement_extra.update(
            {
                "ranks":
                    list(
                        STAGE6_RANKS
                    ),
                "target_mode":
                    STAGE6_TARGET_MODE,
                "encoder_self_attention":
                    True,
                "decoder_self_attention":
                    True,
                "decoder_cross_attention":
                    True,
                "calibration_size":
                    STAGE6_CALIBRATION_SIZE,
                "warmup_ratio":
                    STAGE6_WARMUP,
            }
        )

    else:
        placement_extra[
            "rank"
        ] = (
            FCCA_DEFAULT_RANK
        )

        if (
            STAGE in {
                1,
                2,
            }
        ):
            placement_extra.update(
                {
                    "target_mode":
                        TARGET_MODE_ALL,
                    "encoder_self_attention":
                        True,
                    "decoder_self_attention":
                        True,
                    "decoder_cross_attention":
                        True,
                }
            )

        elif (
            STAGE == 3
        ):
            placement_extra.update(
                {
                    "stage3_target_modes":
                        [
                            TARGET_MODE_ALL,
                            TARGET_MODE_SELF,
                        ],
                }
            )

    return ExperimentSpec(
        name=(
            f"fcca_l_stage"
            f"{STAGE}"
        ),
        model_type=(
            "whisper_fcca"
        ),
        domain_category="l",
        project_root=str(
            PROJECT_ROOT
        ),
        output_root=str(
            output_root
        ),
        base_model=(
            "openai/whisper-medium"
        ),
        language="es",
        whisper_task=(
            "transcribe"
        ),
        normalisation_file=str(
            PROJECT_ROOT
            / "scripts"
            / "resources"
            / (
                "LLanito_"
                "normalisation_v2.txt"
            )
        ),
        data=DataSpec(
            mode="cv",
            fold_train_template=(
                "data/processed/"
                "llanito_cv/"
                "fold_{fold}/"
                "train.csv"
            ),
            fold_validation_template=(
                "data/processed/"
                "llanito_cv/"
                "fold_{fold}/"
                "validation.csv"
            ),
            folds=SELECTED_FOLDS,
        ),
        search=(
            OptimisationSearchSpec(
                learning_rate_low=(
                    learning_rate_low
                ),
                learning_rate_high=(
                    learning_rate_high
                ),
                effective_global_batch_sizes=(
                    batches
                ),
                startup_recipes=(
                    startup
                ),
            )
        ),
        training=TrainingProtocol(
            gradient_accumulation_steps=(
                protocol_ga
            ),
            world_size=2,
            per_device_eval_batch_size=1,
            optimizer="adamw_torch",
            lr_scheduler_type="linear",
            weight_decay=0.01,
            max_grad_norm=1.0,

            # For fixed-warmup stages this now
            # matches the actual run and therefore
            # the search_protocol JSON.
            #
            # Stage 1 still has 0 here because
            # warmup is itself a searched plugin
            # hyperparameter.

            warmup_ratio=(
                protocol_warmup
            ),
            warmup_steps=0,

            fp16=True,
            bf16=False,
            gradient_checkpointing=False,
            dataloader_workers=4,
            logging_steps=10,
            generation_num_beams=1,
            generation_max_length=225,
            max_epochs=10,
            retain_all_epoch_checkpoints=False,
            hpo_seed=2026,
            final_model_seed=2026,
            optuna_sampler_seed=2026,
        ),
        placement=PlacementSpec(
            name=(
                "fcca_attention_qkvo"
            ),
            region=(
                "encoder_decoder"
            ),
            extra=(
                placement_extra
            ),
        ),
        wandb_project=(
            "speech-hpo"
        ),
        study_name=(
            study_name
        ),
        description=(
            description
        ),
    )


EXPERIMENT = (
    _build_experiment()
)

PLUGIN = FCCAPlugin()


# =============================================================================
# STAGE-2 SUMMARY
# =============================================================================

def _write_stage2_optimizer_summary(
) -> None:
    if (
        STAGE != 2
        or IS_WORKER
    ):
        return

    assert (
        STAGE1_WINNER
        is not None
    )

    rows: list[
        dict[str, Any]
    ] = [
        {
            "source":
                "stage1_reused",
            "learning_rate":
                LORA_BEST_LR,
            "effective_global_batch_size":
                LORA_BEST_EBS,
            "objective_normalised_wer":
                float(
                    STAGE1_WINNER[
                        "best_objective_normalised_wer"
                    ]
                ),
            "trial_number":
                int(
                    STAGE1_WINNER[
                        "best_trial"
                    ]
                ),
        }
    ]

    storage = (
        "sqlite:///"
        + str(
            STAGE2_OUTPUT_ROOT
            / "optuna_study.sqlite3"
        )
    )

    study = (
        optuna.load_study(
            study_name=(
                EXPERIMENT
                .study_name
            ),
            storage=storage,
        )
    )

    for trial in (
        study.trials
    ):
        if (
            trial.state
            != (
                optuna.trial
                .TrialState
                .COMPLETE
            )
            or trial.value
            is None
        ):
            continue

        rows.append(
            {
                "source":
                    "stage2_new",
                "learning_rate":
                    float(
                        trial.params[
                            "learning_rate"
                        ]
                    ),
                "effective_global_batch_size":
                    int(
                        trial.params[
                            "effective_global_batch_size"
                        ]
                    ),
                "objective_normalised_wer":
                    float(
                        trial.value
                    ),
                "trial_number":
                    int(
                        trial.number
                    ),
            }
        )

    unique = {
        (
            float(
                row[
                    "learning_rate"
                ]
            ),
            int(
                row[
                    "effective_global_batch_size"
                ]
            ),
        )
        for row in rows
    }

    expected = {
        (
            LORA_BEST_LR,
            LORA_BEST_EBS,
        ),
        (
            LORA_BEST_LR,
            LORA_ALT_EBS,
        ),
        (
            LORA_ALT_LR,
            LORA_BEST_EBS,
        ),
        (
            LORA_ALT_LR,
            LORA_ALT_EBS,
        ),
    }

    if (
        unique != expected
        or len(rows) != 4
    ):
        raise RuntimeError(
            "Stage-2 optimizer "
            "grid incomplete: "
            f"expected="
            f"{sorted(expected)}, "
            f"found="
            f"{sorted(unique)}, "
            f"rows={len(rows)}"
        )

    rows.sort(
        key=lambda row: (
            float(
                row[
                    "learning_rate"
                ]
            ),
            int(
                row[
                    "effective_global_batch_size"
                ]
            ),
        )
    )

    best = min(
        rows,
        key=lambda row: float(
            row[
                "objective_normalised_wer"
            ]
        ),
    )

    STAGE2_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        STAGE2_OUTPUT_ROOT
        / "optimizer_grid_summary.json"
    )

    csv_path = (
        STAGE2_OUTPUT_ROOT
        / "optimizer_grid_summary.csv"
    )

    json_path.write_text(
        json.dumps(
            {
                "duplicate_anchor_retrained":
                    False,
                "stage1_winning_calibration_size":
                    int(
                        STAGE1_WINNER[
                            "best_params"
                        ][
                            "fcca_calibration_size"
                        ]
                    ),
                "stage1_winning_warmup":
                    float(
                        STAGE1_WINNER[
                            "best_params"
                        ][
                            "fcca_warmup_ratio"
                        ]
                    ),
                "rows":
                    rows,
                "best_over_full_2x2":
                    best,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        "[FCCA] complete optimizer "
        "grid summary: "
        f"{json_path}",
        flush=True,
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    run_hpo_experiment(
        experiment=(
            EXPERIMENT
        ),
        plugin=PLUGIN,
        launcher_path=(
            SCRIPT_PATH
        ),
    )

    _write_stage2_optimizer_summary()
