from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    WhisperForConditionalGeneration,
)

from .config import (
    CALIBRATION_SEED,
    CALIBRATION_SIZE,
    DAMPING,
    EXPECTED_TARGETS,
    FCCA128_BUDGET,
    RANK,
    REGIMES,
    SCALE,
    load_runner,
    make_groups,
    result_root,
    task_stats_metadata_path,
    task_stats_path,
    tucker_basis_path,
    tucker_calibration_root,
)


class CalibrationContext:
    encoder_attention_mask: torch.Tensor | None = None
    decoder_mask: torch.Tensor | None = None


CTX = CalibrationContext()


def atomic_torch_save(
    value,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        value,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def save_json(
    path: Path,
    value,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + f".tmp.{os.getpid()}"
    )

    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def sha256_file(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                16 * 1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def resize_prefix_mask(
    mask: torch.Tensor,
    target_length: int,
):
    if mask.ndim != 2:
        raise RuntimeError(
            f"Expected 2D mask, got {mask.shape}"
        )

    mask = mask.bool()

    source_length = int(
        mask.shape[1]
    )

    if source_length == target_length:
        return mask

    valid = (
        mask.long()
        .sum(
            dim=1
        )
    )

    if source_length == 2 * target_length:
        target_valid = (
            valid + 1
        ) // 2

    else:
        target_valid = torch.ceil(
            valid.float()
            * float(
                target_length
            )
            / float(
                source_length
            )
        ).long()

    target_valid = target_valid.clamp(
        0,
        target_length,
    )

    positions = torch.arange(
        target_length,
        device=mask.device,
    )[None, :]

    return (
        positions
        < target_valid[:, None]
    )


def whisper_uses_encoder_time(
    name: str,
):
    if (
        ".encoder.layers." in name
        and
        ".self_attn." in name
    ):
        return True

    if (
        ".decoder.layers." in name
        and
        ".encoder_attn." in name
    ):
        return (
            name.rsplit(
                ".",
                1,
            )[-1]
            in {
                "k_proj",
                "v_proj",
            }
        )

    return False


class TaskStat:
    def __init__(
        self,
        name: str,
        module: nn.Linear,
    ):
        self.name = name
        self.module = module

        device = (
            module.weight.device
        )

        self.cross = torch.zeros(
            module.out_features,
            module.in_features,
            dtype=torch.float32,
            device=device,
        )

        self.activation_second = (
            torch.zeros(
                module.in_features,
                dtype=torch.float32,
                device=device,
            )
        )

        self.gradient_second = (
            torch.zeros(
                module.out_features,
                dtype=torch.float32,
                device=device,
            )
        )

        self.positions = 0
        self.handle = None

    def mask(
        self,
        x,
    ):
        source = (
            CTX.encoder_attention_mask
            if whisper_uses_encoder_time(
                self.name
            )
            else CTX.decoder_mask
        )

        if source is None:
            raise RuntimeError(
                f"Missing calibration mask for "
                f"{self.name}"
            )

        return resize_prefix_mask(
            source,
            int(
                x.shape[-2]
            ),
        )

    def install(
        self,
    ):
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
                    f"Non-tensor output: {self.name}"
                )

            x = inputs[0].detach()

            if (
                x.ndim != 3
                or
                output.ndim != 3
            ):
                raise RuntimeError(
                    f"Expected [B,T,H] at "
                    f"{self.name}; "
                    f"x={tuple(x.shape)}, "
                    f"y={tuple(output.shape)}"
                )

            if not output.requires_grad:
                output.requires_grad_(
                    True
                )

            mask = self.mask(
                x
            )

            if tuple(
                mask.shape
            ) != tuple(
                x.shape[:2]
            ):
                raise RuntimeError(
                    f"Mask mismatch at {self.name}"
                )

            x_selected = (
                x[
                    mask
                ]
                .float()
            )

            self.activation_second.add_(
                x_selected
                .square()
                .sum(
                    dim=0
                )
            )

            def grad_hook(
                grad,
            ):
                g = (
                    grad.detach()[
                        mask
                    ]
                    .float()
                )

                if (
                    g.shape[0]
                    != x_selected.shape[0]
                ):
                    raise RuntimeError(
                        f"x/delta count mismatch "
                        f"at {self.name}"
                    )

                self.cross.add_(
                    g.T
                    @ x_selected
                )

                self.gradient_second.add_(
                    g.square()
                    .sum(
                        dim=0
                    )
                )

                self.positions += int(
                    g.shape[0]
                )

                return grad

            output.register_hook(
                grad_hook
            )

        self.handle = (
            self.module
            .register_forward_hook(
                forward_hook
            )
        )

    def remove(
        self,
    ):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def safe_name(
    index: int,
    name: str,
):
    cleaned = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        name,
    )

    return (
        f"{index:03d}_{cleaned}.pt"
    )


def collect_task_stats(
    language: str,
    force: bool,
):
    runner = load_runner(
        language
    )

    path = task_stats_path(
        language
    )

    if (
        path.exists()
        and
        not force
    ):
        print(
            f"[reuse task stats] {path}",
            flush=True,
        )

        return path

    device = torch.device(
        "cuda:0"
    )

    runner.set_seed(
        CALIBRATION_SEED
    )

    train_ds = runner.load_split(
        "train"
    )

    if len(
        train_ds
    ) < CALIBRATION_SIZE:
        raise RuntimeError(
            f"{language}: only "
            f"{len(train_ds)} train examples."
        )

    generator = (
        torch.Generator()
        .manual_seed(
            CALIBRATION_SEED
        )
    )

    indices = torch.randperm(
        len(
            train_ds
        ),
        generator=generator,
    )[
        :CALIBRATION_SIZE
    ].tolist()

    calibration_ds = (
        train_ds.select(
            indices
        )
    )

    processor = (
        runner.build_processor()
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            runner.BASE_MODEL,
            dtype=torch.bfloat16,
        )
        .to(
            device
        )
    )

    model.config.use_cache = False
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None

    targets = (
        runner.collect_attention_targets(
            model
        )
    )

    if len(
        targets
    ) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGETS} "
            f"attention matrices, "
            f"found {len(targets)}."
        )

    stats = {
        name: TaskStat(
            name,
            module,
        )
        for name, module
        in targets.items()
    }

    for stat in stats.values():
        stat.install()

    loader = DataLoader(
        calibration_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda rows:
            runner.collate(
                processor,
                rows,
            ),
    )

    print(
        f"[task stats] {language}: "
        f"256 examples, "
        f"{len(targets)} matrices",
        flush=True,
    )

    try:
        for index, batch in enumerate(
            loader,
            start=1,
        ):
            batch = runner.move_batch(
                batch,
                device,
            )

            CTX.encoder_attention_mask = (
                batch.get(
                    "attention_mask"
                )
            )

            labels = batch.get(
                "labels"
            )

            if labels is None:
                raise RuntimeError(
                    "Calibration labels missing."
                )

            CTX.decoder_mask = (
                labels.ne(
                    -100
                )
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
                    f"Non-finite calibration "
                    f"loss at {index}: {loss}"
                )

            valid = int(
                labels.ne(
                    -100
                )
                .sum()
                .item()
            )

            (
                loss
                * float(
                    max(
                        valid,
                        1,
                    )
                )
            ).backward()

            model.zero_grad(
                set_to_none=True
            )

            if (
                index % 16 == 0
                or
                index
                == CALIBRATION_SIZE
            ):
                print(
                    f"[task stats] "
                    f"{index}/"
                    f"{CALIBRATION_SIZE}",
                    flush=True,
                )

    finally:
        for stat in stats.values():
            stat.remove()

        CTX.encoder_attention_mask = None
        CTX.decoder_mask = None

    module_payload = {}

    for name in sorted(
        stats
    ):
        stat = stats[
            name
        ]

        if stat.positions <= 0:
            raise RuntimeError(
                f"No positions captured: {name}"
            )

        tensors = {
            "cross_sum":
                stat.cross.detach().cpu(),
            "activation_second_sum":
                stat.activation_second
                .detach()
                .cpu(),
            "gradient_second_sum":
                stat.gradient_second
                .detach()
                .cpu(),
        }

        for key, tensor in tensors.items():
            if tensor.dtype != torch.float32:
                raise RuntimeError(
                    f"{name}/{key} not FP32."
                )

            if not torch.isfinite(
                tensor
            ).all():
                raise RuntimeError(
                    f"{name}/{key} non-finite."
                )

        module_payload[
            name
        ] = {
            **tensors,
            "positions":
                int(
                    stat.positions
                ),
            "in_features":
                int(
                    stat.module.in_features
                ),
            "out_features":
                int(
                    stat.module.out_features
                ),
        }

    calibration_rows = []

    for calibration_index, source_index in enumerate(
        indices
    ):
        row = train_ds[
            source_index
        ]

        calibration_rows.append(
            {
                "calibration_index":
                    int(
                        calibration_index
                    ),
                "source_index":
                    int(
                        source_index
                    ),
                "id":
                    str(
                        row.get(
                            "id",
                            source_index,
                        )
                    ),
                "path":
                    str(
                        row.get(
                            "path",
                            "",
                        )
                    ),
                "num_samples":
                    int(
                        row.get(
                            "num_samples",
                            0,
                        )
                    ),
                "transcription":
                    str(
                        row[
                            "transcription"
                        ]
                    ),
            }
        )

    fingerprint = hashlib.sha256(
        json.dumps(
            calibration_rows,
            ensure_ascii=False,
            sort_keys=True,
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    payload = {
        "schema":
            "shared_frozen_tucker2_task_stats_v2",
        "language":
            language,
        "base_model":
            runner.BASE_MODEL,
        "whisper_language":
            runner.WHISPER_LANGUAGE,
        "whisper_task":
            runner.WHISPER_TASK,
        "calibration_size":
            CALIBRATION_SIZE,
        "calibration_seed":
            CALIBRATION_SEED,
        "calibration_indices":
            indices,
        "calibration_rows":
            calibration_rows,
        "calibration_fingerprint_sha256":
            fingerprint,
        "target_count":
            len(targets),
        "target_names":
            sorted(
                targets
            ),
        "capture_dtype":
            "float32",
        "model_forward_precision":
            "bfloat16 autocast",
        "statistics_definition": {
            "cross_sum":
                "sum_t delta_t x_t^T",
            "activation_second_sum":
                "sum_t x_t^2",
            "gradient_second_sum":
                "sum_t delta_t^2",
            "positions":
                "number of valid time/token positions",
        },
        "modules":
            module_payload,
    }

    atomic_torch_save(
        payload,
        path,
    )

    metadata = {
        key: value
        for key, value
        in payload.items()
        if key
        not in {
            "modules",
        }
    }

    metadata[
        "task_stats_sha256"
    ] = sha256_file(
        path
    )

    save_json(
        task_stats_metadata_path(
            language
        ),
        metadata,
    )

    print(
        f"[task stats saved] {path}",
        flush=True,
    )

    del model
    del processor
    del stats
    del module_payload

    torch.cuda.empty_cache()

    return path


def build_fcca_geometry(
    language: str,
    force: bool,
):
    runner = load_runner(
        language
    )

    stats_path = task_stats_path(
        language
    )

    stats_payload = torch.load(
        stats_path,
        map_location="cpu",
        weights_only=False,
    )

    target_names = (
        stats_payload[
            "target_names"
        ]
    )

    output_path = (
        runner.FCCA_BASIS_PATH
    )

    full_svd_root = (
        output_path.parent
        / "full_svd"
    )

    if force:
        if full_svd_root.exists():
            import shutil
            shutil.rmtree(
                full_svd_root
            )

    full_svd_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    bases = {
        str(
            RANK
        ): {}
    }

    diagnostics = {}
    full_svd_files = {}

    device = torch.device(
        "cuda:0"
    )

    for index, name in enumerate(
        target_names,
        start=1,
    ):
        stat = (
            stats_payload[
                "modules"
            ][
                name
            ]
        )

        count = float(
            stat[
                "positions"
            ]
        )

        G = (
            stat[
                "cross_sum"
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
            / count
        )

        a = (
            stat[
                "activation_second_sum"
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
            / count
        )

        d = (
            stat[
                "gradient_second_sum"
            ]
            .to(
                device=device,
                dtype=torch.float32,
            )
            / count
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
                f"Non-finite FCCA statistics: {name}"
            )

        inv_a = torch.rsqrt(
            a
            + DAMPING
        )

        inv_d = torch.rsqrt(
            d
            + DAMPING
        )

        whitened = (
            inv_d[:, None]
            * G
            * inv_a[None, :]
        )

        U, singular, Vh = (
            torch.linalg.svd(
                whitened,
                full_matrices=False,
            )
        )

        if min(
            U.shape[1],
            Vh.shape[0],
        ) < RANK:
            raise RuntimeError(
                f"Rank 128 impossible: {name}"
            )

        P0 = (
            inv_d[:, None]
            * U[
                :,
                :RANK,
            ]
        )

        Q0 = (
            inv_a[:, None]
            * Vh[
                :RANK,
                :,
            ].T
        )

        P, _ = torch.linalg.qr(
            P0,
            mode="reduced",
        )

        Q, _ = torch.linalg.qr(
            Q0,
            mode="reduced",
        )

        eye = torch.eye(
            RANK,
            device=device,
            dtype=torch.float32,
        )

        p_error = float(
            torch.linalg.matrix_norm(
                P.T @ P
                - eye,
                ord="fro",
            ).item()
        )

        q_error = float(
            torch.linalg.matrix_norm(
                Q.T @ Q
                - eye,
                ord="fro",
            ).item()
        )

        if (
            p_error > 1.0e-3
            or
            q_error > 1.0e-3
        ):
            raise RuntimeError(
                f"FCCA orthogonality failure "
                f"{name}: "
                f"P={p_error}, Q={q_error}"
            )

        bases[
            str(
                RANK
            )
        ][
            name
        ] = {
            "P":
                P.detach().cpu(),
            "Q":
                Q.detach().cpu(),
        }

        svd_file = (
            full_svd_root
            / safe_name(
                index,
                name,
            )
        )

        svd_payload = {
            "name":
                name,
            "positions":
                int(
                    stat[
                        "positions"
                    ]
                ),
            "mean_cross":
                G.detach().cpu(),
            "mean_activation_second":
                a.detach().cpu(),
            "mean_gradient_second":
                d.detach().cpu(),
            "inv_sqrt_activation_metric":
                inv_a.detach().cpu(),
            "inv_sqrt_gradient_metric":
                inv_d.detach().cpu(),
            "U":
                U.detach().cpu(),
            "singular_values":
                singular.detach().cpu(),
            "Vh":
                Vh.detach().cpu(),
            "P_rank128":
                P.detach().cpu(),
            "Q_rank128":
                Q.detach().cpu(),
        }

        atomic_torch_save(
            svd_payload,
            svd_file,
        )

        full_svd_files[
            name
        ] = str(
            svd_file
        )

        energy = (
            singular.square()
        )

        diagnostics[
            name
        ] = {
            "positions":
                int(
                    stat[
                        "positions"
                    ]
                ),
            "top_singular_value":
                float(
                    singular[
                        0
                    ].item()
                ),
            "rank128_singular_value":
                float(
                    singular[
                        RANK - 1
                    ].item()
                ),
            "rank128_energy_fraction":
                float(
                    (
                        energy[
                            :RANK
                        ].sum()
                        /
                        energy.sum().clamp_min(
                            1.0e-30
                        )
                    ).item()
                ),
            "P_orthogonality_error":
                p_error,
            "Q_orthogonality_error":
                q_error,
        }

        if (
            index % 16 == 0
            or
            index
            == len(
                target_names
            )
        ):
            print(
                f"[FCCA geometry] "
                f"{index}/"
                f"{len(target_names)}",
                flush=True,
            )

        del (
            G,
            a,
            d,
            inv_a,
            inv_d,
            whitened,
            U,
            singular,
            Vh,
            P0,
            Q0,
            P,
            Q,
            eye,
            energy,
        )

        torch.cuda.empty_cache()

    payload = {
        "method":
            "FCCA",
        "paper":
            "arXiv:2609.00762",
        "language":
            language,
        "model":
            "whisper",
        "base_model":
            runner.BASE_MODEL,
        "rank":
            RANK,
        "calibration_size":
            CALIBRATION_SIZE,
        "calibration_seed":
            CALIBRATION_SEED,
        "damping":
            DAMPING,
        "scale":
            SCALE,
        "target_count":
            len(
                target_names
            ),
        "target_names":
            target_names,
        "task_stats_path":
            str(
                stats_path
            ),
        "task_stats_sha256":
            sha256_file(
                stats_path
            ),
        "full_svd_files":
            full_svd_files,
        "bases":
            bases,
        "diagnostics":
            diagnostics,
    }

    atomic_torch_save(
        payload,
        output_path,
    )

    save_json(
        output_path.with_suffix(
            ".json"
        ),
        {
            key: value
            for key, value
            in payload.items()
            if key
            not in {
                "bases",
            }
        },
    )

    print(
        f"[FCCA geometry saved] "
        f"{output_path}",
        flush=True,
    )


def build_tucker_geometry(
    language: str,
    force: bool,
):
    runner = load_runner(
        language
    )

    stats_path = task_stats_path(
        language
    )

    task_payload = torch.load(
        stats_path,
        map_location="cpu",
        weights_only=False,
    )

    names = task_payload[
        "target_names"
    ]

    if (
        len(
            names
        )
        != EXPECTED_TARGETS
    ):
        raise RuntimeError(
            "Tucker target-count mismatch."
        )

    trainable_budget = (
        len(
            names
        )
        * RANK
        * RANK
    )

    if (
        trainable_budget
        != FCCA128_BUDGET
    ):
        raise RuntimeError(
            f"Budget mismatch: "
            f"{trainable_budget:,} "
            f"!= {FCCA128_BUDGET:,}"
        )

    root = tucker_calibration_root(
        language
    )

    geometry_root = (
        root
        / "full_group_geometry"
    )

    if force and geometry_root.exists():
        import shutil
        shutil.rmtree(
            geometry_root
        )

    geometry_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    regimes_payload = {}

    device = torch.device(
        "cuda:0"
    )

    for regime in REGIMES:
        groups = make_groups(
            names,
            regime,
        )

        membership = {
            name: group
            for group, members
            in groups.items()
            for name
            in members
        }

        stored_groups = {}

        regime_root = (
            geometry_root
            / regime
        )

        regime_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[Tucker geometry] "
            f"{language} {regime}: "
            f"{len(groups)} groups",
            flush=True,
        )

        for group_index, (
            group_name,
            members,
        ) in enumerate(
            sorted(
                groups.items()
            ),
            start=1,
        ):
            first = (
                task_payload[
                    "modules"
                ][
                    members[0]
                ]
            )

            d_out = int(
                first[
                    "out_features"
                ]
            )

            d_in = int(
                first[
                    "in_features"
                ]
            )

            if (
                d_out < RANK
                or
                d_in < RANK
            ):
                raise RuntimeError(
                    f"Rank 128 impossible "
                    f"for {group_name}"
                )

            left_cov = torch.zeros(
                d_out,
                d_out,
                device=device,
                dtype=torch.float32,
            )

            right_cov = torch.zeros(
                d_in,
                d_in,
                device=device,
                dtype=torch.float32,
            )

            task_norms = {}

            for name in members:
                stat = (
                    task_payload[
                        "modules"
                    ][
                        name
                    ]
                )

                G = (
                    stat[
                        "cross_sum"
                    ]
                    .to(
                        device=device,
                        dtype=torch.float32,
                    )
                    / float(
                        stat[
                            "positions"
                        ]
                    )
                )

                if tuple(
                    G.shape
                ) != (
                    d_out,
                    d_in,
                ):
                    raise RuntimeError(
                        f"Incompatible Tucker "
                        f"shape in {group_name}: "
                        f"{name}={G.shape}"
                    )

                left_cov.add_(
                    G
                    @ G.T
                )

                right_cov.add_(
                    G.T
                    @ G
                )

                task_norms[
                    name
                ] = float(
                    torch.linalg.matrix_norm(
                        G,
                        ord="fro",
                    ).item()
                )

                del G

            left_cov = (
                0.5
                * (
                    left_cov
                    + left_cov.T
                )
            )

            right_cov = (
                0.5
                * (
                    right_cov
                    + right_cov.T
                )
            )

            left_values, left_vectors = (
                torch.linalg.eigh(
                    left_cov
                )
            )

            right_values, right_vectors = (
                torch.linalg.eigh(
                    right_cov
                )
            )

            left_order = torch.argsort(
                left_values,
                descending=True,
            )

            right_order = torch.argsort(
                right_values,
                descending=True,
            )

            left_values = (
                left_values[
                    left_order
                ]
            )

            left_vectors = (
                left_vectors[
                    :,
                    left_order,
                ]
            )

            right_values = (
                right_values[
                    right_order
                ]
            )

            right_vectors = (
                right_vectors[
                    :,
                    right_order,
                ]
            )

            A = (
                left_vectors[
                    :,
                    :RANK,
                ]
                .contiguous()
            )

            B = (
                right_vectors[
                    :,
                    :RANK,
                ]
                .contiguous()
            )

            eye = torch.eye(
                RANK,
                device=device,
                dtype=torch.float32,
            )

            a_error = float(
                torch.linalg.matrix_norm(
                    A.T @ A
                    - eye,
                    ord="fro",
                ).item()
            )

            b_error = float(
                torch.linalg.matrix_norm(
                    B.T @ B
                    - eye,
                    ord="fro",
                ).item()
            )

            if (
                a_error > 1.0e-3
                or
                b_error > 1.0e-3
            ):
                raise RuntimeError(
                    f"Tucker orthogonality "
                    f"failure: {group_name}"
                )

            projected_cores = {}
            reconstruction = {}

            for name in members:
                stat = (
                    task_payload[
                        "modules"
                    ][
                        name
                    ]
                )

                G = (
                    stat[
                        "cross_sum"
                    ]
                    .to(
                        device=device,
                        dtype=torch.float32,
                    )
                    / float(
                        stat[
                            "positions"
                        ]
                    )
                )

                core = (
                    A.T
                    @ G
                    @ B
                )

                reconstruction_G = (
                    A
                    @ core
                    @ B.T
                )

                denominator = (
                    torch.linalg.matrix_norm(
                        G,
                        ord="fro",
                    )
                    .square()
                    .clamp_min(
                        1.0e-30
                    )
                )

                residual_fraction = float(
                    (
                        torch.linalg.matrix_norm(
                            G
                            - reconstruction_G,
                            ord="fro",
                        )
                        .square()
                        /
                        denominator
                    ).item()
                )

                projected_cores[
                    name
                ] = core.detach().cpu()

                reconstruction[
                    name
                ] = {
                    "relative_squared_residual":
                        residual_fraction,
                    "captured_fraction":
                        1.0
                        - residual_fraction,
                }

                del (
                    G,
                    core,
                    reconstruction_G,
                    denominator,
                )

            left_singular_values = torch.sqrt(
                torch.clamp(
                    left_values,
                    min=0.0,
                )
            )

            right_singular_values = torch.sqrt(
                torch.clamp(
                    right_values,
                    min=0.0,
                )
            )

            group_file = (
                regime_root
                / f"{group_index:03d}_{group_name}.pt"
            )

            full_group = {
                "group_name":
                    group_name,
                "members":
                    members,
                "left_covariance":
                    left_cov.detach().cpu(),
                "right_covariance":
                    right_cov.detach().cpu(),
                "left_eigenvalues":
                    left_values.detach().cpu(),
                "left_singular_values":
                    left_singular_values.detach().cpu(),
                "left_eigenvectors":
                    left_vectors.detach().cpu(),
                "right_eigenvalues":
                    right_values.detach().cpu(),
                "right_singular_values":
                    right_singular_values.detach().cpu(),
                "right_eigenvectors":
                    right_vectors.detach().cpu(),
                "A_rank128":
                    A.detach().cpu(),
                "B_rank128":
                    B.detach().cpu(),
                "projected_task_cores":
                    projected_cores,
                "task_direction_frobenius_norms":
                    task_norms,
                "reconstruction":
                    reconstruction,
                "A_orthogonality_error":
                    a_error,
                "B_orthogonality_error":
                    b_error,
            }

            atomic_torch_save(
                full_group,
                group_file,
            )

            stored_groups[
                group_name
            ] = {
                "members":
                    members,
                "A":
                    A.detach().cpu(),
                "B":
                    B.detach().cpu(),
                "full_geometry_file":
                    str(
                        group_file
                    ),
                "A_orthogonality_error":
                    a_error,
                "B_orthogonality_error":
                    b_error,
            }

            print(
                f"[Tucker geometry] "
                f"{regime} "
                f"{group_index}/"
                f"{len(groups)} "
                f"{group_name} "
                f"M={len(members)}",
                flush=True,
            )

            del (
                left_cov,
                right_cov,
                left_values,
                left_vectors,
                right_values,
                right_vectors,
                left_order,
                right_order,
                A,
                B,
                eye,
                projected_cores,
            )

            torch.cuda.empty_cache()

        regimes_payload[
            regime
        ] = {
            "membership":
                membership,
            "group_count":
                len(
                    groups
                ),
            "groups":
                stored_groups,
        }

    payload = {
        "method":
            "shared_frozen_tucker2",
        "construction":
            (
                "shared frozen task-gradient Tucker-2; "
                "ordinary Euclidean HOSVD of raw mean "
                "task-gradient matrices G_l=E[delta_l x_l^T]; "
                "no Fisher whitening; no damping; "
                "layer/matrix mode uncompressed"
            ),
        "tucker_uses_fisher_whitening":
            False,
        "tucker_uses_fisher_damping":
            False,
        "basis_source":
            "raw_mean_task_gradients",
        "shared_factors_trainable":
            False,
        "matrix_specific_cores_trainable":
            True,
        "language":
            language,
        "base_model":
            runner.BASE_MODEL,
        "rank":
            RANK,
        "calibration_size":
            CALIBRATION_SIZE,
        "calibration_seed":
            CALIBRATION_SEED,
        "task_stats_path":
            str(
                stats_path
            ),
        "task_stats_sha256":
            sha256_file(
                stats_path
            ),
        "target_count":
            len(
                names
            ),
        "target_names":
            names,
        "trainable_parameters_per_matrix":
            RANK
            * RANK,
        "reference_trainable_budget":
            FCCA128_BUDGET,
        "actual_trainable_budget":
            trainable_budget,
        "budget_difference":
            trainable_budget
            - FCCA128_BUDGET,
        "shared_A_B_trainable":
            False,
        "matrix_specific_core_trainable":
            True,
        "regimes":
            regimes_payload,
    }

    path = tucker_basis_path(
        language
    )

    atomic_torch_save(
        payload,
        path,
    )

    save_json(
        path.with_suffix(
            ".json"
        ),
        {
            "method":
                payload[
                    "method"
                ],
            "construction":
                payload[
                    "construction"
                ],
            "language":
                language,
            "base_model":
                runner.BASE_MODEL,
            "rank":
                RANK,
            "calibration_size":
                CALIBRATION_SIZE,
            "calibration_seed":
                CALIBRATION_SEED,
            "target_count":
                len(
                    names
                ),
            "reference_trainable_budget":
                FCCA128_BUDGET,
            "actual_trainable_budget":
                trainable_budget,
            "budget_difference":
                0,
            "regimes": {
                regime: {
                    "group_count":
                        regimes_payload[
                            regime
                        ][
                            "group_count"
                        ],
                    "groups":
                        sorted(
                            regimes_payload[
                                regime
                            ][
                                "groups"
                            ]
                        ),
                }
                for regime in REGIMES
            },
        },
    )

    print(
        f"[Tucker geometry saved] {path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        choices=[
            "ast_es",
            "es_419",
        ],
        required=True,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    collect_task_stats(
        args.language,
        force=args.force,
    )

    build_fcca_geometry(
        args.language,
        force=args.force,
    )

    build_tucker_geometry(
        args.language,
        force=args.force,
    )


if __name__ == "__main__":
    main()
