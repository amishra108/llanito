from __future__ import annotations
from scripts_fleurs.fleurs_paths import dataset_root, load_fleurs_split, test_manifest_path

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from datasets import Audio, load_from_disk
from transformers import StoppingCriteria, StoppingCriteriaList


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path("/data/milsrg1/corpora/fleurs-selected-languages")
RESULT_ROOT = ROOT / "outputs" / "new_results_qwen"

BASE_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"

SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 30.0

EXPECTED_TARGETS = 208
REFERENCE_RANK = 128
REFERENCE_CORE_BUDGET = EXPECTED_TARGETS * REFERENCE_RANK**2


LANGUAGES = {
    "ast_es": "Spanish",
    "cmn_hans_cn": "Chinese",
    "es_419": "Spanish",
    "ky_kg": "Turkish",
    "ckb_iq": "Persian",
}


METHOD_DIR = {
    "fft": "fft",
    "lora": "lora_r128",
    "fcca": "fcca_r128",
    "adaptive_fcca": "adaptive_budget_fcca",
    "asym_hard": "asym_hard_r128_pairwise",
    "asym_shared_private":
        "asym_shared_private_r128_s64_p64_pairwise",
}


CANONICAL_BASIS_FILENAME = {
    "fcca": "basis_r128.pt",
    "adaptive_fcca": "adaptive_budget_basis.pt",
    "asym_hard": "basis_r128.pt",
    "asym_shared_private": "basis_r128.pt",
}


# =====================================================================
# General utilities
# =====================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(16 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def atomic_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
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
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
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

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    tmp = path.with_suffix(
        path.suffix
        + f".tmp.{os.getpid()}"
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


def setup_distributed(
    device_arg: str,
):
    distributed = (
        "LOCAL_RANK" in os.environ
        and
        "WORLD_SIZE" in os.environ
    )

    if distributed:
        local_rank = int(
            os.environ["LOCAL_RANK"]
        )

        rank = int(
            os.environ["RANK"]
        )

        world_size = int(
            os.environ["WORLD_SIZE"]
        )

        torch.cuda.set_device(
            local_rank
        )

        dist.init_process_group(
            backend="nccl"
        )

        device = torch.device(
            f"cuda:{local_rank}"
        )

    else:
        local_rank = 0
        rank = 0
        world_size = 1

        device = torch.device(
            device_arg
        )

        torch.cuda.set_device(
            device
        )

    return (
        distributed,
        local_rank,
        rank,
        world_size,
        device,
    )


def barrier(
    distributed: bool,
):
    if distributed:
        dist.barrier()


# =====================================================================
# Load the exact Qwen training runner
# =====================================================================

def find_runner() -> Path:
    path = ROOT / "scripts_fleurs" / "train_qwen3_fcca_variants.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def import_runner():
    path = find_runner()

    spec = (
        importlib.util
        .spec_from_file_location(
            "qwen_week8_runner",
            path,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    # The training runner imports OpenCC only for Mandarin CER.
    # Asturian/Sorani evaluation must not depend on OpenCC being installed.
    try:
        import opencc  # noqa: F401
    except ModuleNotFoundError:
        import sys
        import types

        stub = types.ModuleType("opencc")

        class OpenCC:
            def __init__(self, *args, **kwargs):
                pass

            def convert(self, text):
                return str(text)

        stub.OpenCC = OpenCC
        sys.modules["opencc"] = stub

    spec.loader.exec_module(
        module
    )

    required = [
        "build_model",
        "build_processor",
        "collect_attention_targets",
        "get_parent",
        "FCCALinear",
        "decode_audio_value",
        "normalise_text",
        "corpus_wer",
    ]

    missing = [
        name
        for name in required
        if not hasattr(
            module,
            name,
        )
    ]

    if missing:
        raise RuntimeError(
            "Qwen runner missing required API: "
            f"{missing}"
        )

    return module, path


# =====================================================================
# Same repetition guard as Qwen training/validation
# =====================================================================

class TrackingRepeatedTokenCycleGuard(
    StoppingCriteria
):
    """
    Same policy as the Qwen runner:

      cycle length:      1..8
      repeats required:  6
      inspect:           generated tokens only
      action:            stop only
      repeated cycles:   remain visible
    """

    def __init__(
        self,
        prompt_length: int,
        max_cycle_length: int = 8,
        repeats_required: int = 6,
    ):
        super().__init__()

        self.prompt_length = int(
            prompt_length
        )

        self.max_cycle_length = int(
            max_cycle_length
        )

        self.repeats_required = int(
            repeats_required
        )

        self.triggered = False

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):
        batch_size = int(
            input_ids.shape[0]
        )

        stop = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )

        generated = (
            input_ids[
                :,
                self.prompt_length:
            ]
        )

        generated_length = int(
            generated.shape[1]
        )

        for cycle_length in range(
            1,
            self.max_cycle_length + 1,
        ):
            needed = (
                cycle_length
                * self.repeats_required
            )

            if (
                generated_length
                < needed
            ):
                continue

            tail = (
                generated[
                    :,
                    -needed:
                ]
            )

            pattern = (
                tail[
                    :,
                    -cycle_length:
                ]
            )

            expected = pattern.repeat(
                1,
                self.repeats_required,
            )

            stop |= (
                tail
                == expected
            ).all(
                dim=1
            )

        if bool(
            stop.any().item()
        ):
            self.triggered = True

        return stop


# =====================================================================
# Standard Qwen FFT / LoRA reconstruction
# =====================================================================

def import_standard_runner():
    path = (
        ROOT / "scripts_fleurs"
        / "train_qwen3_fleurs.py"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "qwen_standard_runner",
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    required = [
        "BASE_MODEL",
        "AutoProcessor",
        "Qwen3ASRForConditionalGeneration",
        "configure",
    ]

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "Standard Qwen trainer missing: "
            f"{missing}"
        )

    return module, path


def exact_standard_reconstruction(
    checkpoint: Path,
    language: str,
    method: str,
    device: torch.device,
):
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        payload,
        Mapping,
    ):
        raise RuntimeError(
            "Checkpoint payload is not a mapping."
        )

    config = dict(
        payload.get(
            "config",
            {},
        )
    )

    checkpoint_language = payload.get(
        "language",
        config.get("language"),
    )

    if checkpoint_language not in {
        None,
        language,
    }:
        raise RuntimeError(
            "Checkpoint language mismatch: "
            f"{checkpoint_language} != {language}"
        )

    checkpoint_method = payload.get(
        "method",
        config.get("method"),
    )

    if checkpoint_method not in {
        None,
        method,
    }:
        raise RuntimeError(
            "Checkpoint method mismatch: "
            f"{checkpoint_method} != {method}"
        )

    standard, standard_path = (
        import_standard_runner()
    )

    checkpoint_base = payload.get(
        "base_model",
        config.get(
            "base_model",
            standard.BASE_MODEL,
        ),
    )

    if checkpoint_base != standard.BASE_MODEL:
        raise RuntimeError(
            "Base-model mismatch: "
            f"{checkpoint_base} != "
            f"{standard.BASE_MODEL}"
        )

    # Restore exact LoRA architecture BEFORE strict state load.
    if method == "lora":
        lora_rank = int(
            config.get(
                "lora_rank",
                128,
            )
        )

        lora_alpha = float(
            config.get(
                "lora_alpha",
                2.0 * lora_rank,
            )
        )

        lora_dropout = float(
            config.get(
                "lora_dropout",
                0.05,
            )
        )

        standard.LORA_RANK = lora_rank
        standard.LORA_ALPHA = lora_alpha
        standard.LORA_DROPOUT = lora_dropout

    else:
        lora_rank = None
        lora_alpha = None
        lora_dropout = None

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    processor = (
        standard.AutoProcessor
        .from_pretrained(
            standard.BASE_MODEL
        )
    )

    model = (
        standard
        .Qwen3ASRForConditionalGeneration
        .from_pretrained(
            standard.BASE_MODEL,
            dtype=dtype,
        )
        .to(device)
    )

    model.config.use_cache = False

    if method == "lora":
        targets = standard.configure(
            model,
            "lora",
        )

        if len(targets) != EXPECTED_TARGETS:
            raise RuntimeError(
                "Expected "
                f"{EXPECTED_TARGETS} LoRA targets; "
                f"got {len(targets)}"
            )

    elif method == "fft":
        targets = []

    else:
        raise RuntimeError(
            f"Unsupported standard method: {method}"
        )

    full_state = payload.get(
        "state_dict"
    )

    if not isinstance(
        full_state,
        Mapping,
    ):
        full_state = payload.get(
            "full_model_state_dict"
        )

    if not isinstance(
        full_state,
        Mapping,
    ):
        raise RuntimeError(
            "best_checkpoint.pt does not contain "
            "a complete state_dict."
        )

    load_result = model.load_state_dict(
        full_state,
        strict=True,
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict state load returned key mismatches."
        )

    # Tensor-by-tensor exact verification.
    reconstructed = model.state_dict()

    if set(reconstructed) != set(full_state):
        raise RuntimeError(
            "State-key set mismatch after load."
        )

    checked_tensors = 0
    checked_elements = 0

    for name, expected in full_state.items():
        actual = (
            reconstructed[name]
            .detach()
            .cpu()
        )

        expected = (
            expected
            .detach()
            .cpu()
        )

        if actual.dtype != expected.dtype:
            raise RuntimeError(
                f"dtype mismatch: {name}"
            )

        if actual.shape != expected.shape:
            raise RuntimeError(
                f"shape mismatch: {name}"
            )

        if not torch.equal(
            actual,
            expected,
        ):
            raise RuntimeError(
                "BIT-EXACT reconstruction failed: "
                f"{name}"
            )

        checked_tensors += 1
        checked_elements += int(
            actual.numel()
        )

    audit = {
        "checkpoint":
            str(checkpoint.resolve()),

        "checkpoint_sha256":
            sha256_file(checkpoint),

        "checkpoint_epoch":
            payload.get("epoch"),

        "language":
            language,

        "method":
            method,

        "base_model":
            standard.BASE_MODEL,

        "standard_runner":
            str(standard_path.resolve()),

        "strict_state_dict_load":
            True,

        "tensor_by_tensor_bit_exact":
            True,

        "checked_state_tensors":
            checked_tensors,

        "checked_state_elements":
            checked_elements,

        "target_count":
            len(targets),

        "lora_rank":
            lora_rank,

        "lora_alpha":
            lora_alpha,

        "lora_dropout":
            lora_dropout,

        "reconstruction_passed":
            True,
    }

    del full_state
    del payload

    model.eval()
    model.config.use_cache = True

    return (
        model,
        processor,
        audit,
        standard_path,
    )


# =====================================================================
# Checkpoint / basis reconstruction
# =====================================================================

def training_run_dir(
    language: str,
    method: str,
    trial_name: str,
) -> Path:
    return (
        RESULT_ROOT
        / language
        / METHOD_DIR[method]
        / "training"
        / trial_name
    )


def default_checkpoint(language, method, trial_name):
    if method in {"fft", "lora"}:
        return (RESULT_ROOT / language / METHOD_DIR[method]
                / trial_name / "best_checkpoint.pt")
    return (training_run_dir(language, method, trial_name)
            / "checkpoints" / "best_checkpoint.pt")


def canonical_basis(
    language: str,
    method: str,
) -> Path:
    return (
        RESULT_ROOT
        / language
        / METHOD_DIR[method]
        / "calibration"
        / CANONICAL_BASIS_FILENAME[
            method
        ]
    )


def basis_rank(
    entry: Mapping[str, Any],
) -> int:
    if "rank" in entry:
        return int(
            entry["rank"]
        )

    P = torch.as_tensor(
        entry["P"]
    )

    Q = torch.as_tensor(
        entry["Q"]
    )

    if (
        P.ndim != 2
        or
        Q.ndim != 2
        or
        P.shape[1]
        != Q.shape[1]
    ):
        raise RuntimeError(
            "Cannot infer basis rank."
        )

    return int(
        P.shape[1]
    )


def load_exact_basis_payload(
    checkpoint_payload,
    run_dir: Path,
    language: str,
    method: str,
):
    """
    Priority:
      1. basis payload embedded in checkpoint
      2. immutable trial-local frozen basis snapshot
      3. canonical calibration basis

    The third option is only safe because full model state is REQUIRED
    below; stored P/Q are subsequently overwritten from checkpoint.
    """

    embedded = (
        checkpoint_payload.get(
            "basis_payload"
        )
    )

    if isinstance(
        embedded,
        Mapping,
    ):
        return (
            dict(embedded),
            "checkpoint:basis_payload",
            None,
        )

    snapshot = (
        run_dir
        / "frozen_basis_snapshot.pt"
    )

    if snapshot.exists():
        payload = torch.load(
            snapshot,
            map_location="cpu",
            weights_only=False,
        )

        expected_sha = (
            checkpoint_payload.get(
                "basis_sha256"
            )
        )

        actual_sha = (
            sha256_file(
                snapshot
            )
        )

        if (
            expected_sha is not None
            and
            str(expected_sha)
            != actual_sha
        ):
            raise RuntimeError(
                "Frozen basis snapshot SHA mismatch:\n"
                f"checkpoint={expected_sha}\n"
                f"current={actual_sha}"
            )

        return (
            payload,
            str(snapshot.resolve()),
            actual_sha,
        )

    path = canonical_basis(
        language,
        method,
    )

    if not path.exists():
        raise FileNotFoundError(
            "No embedded basis, frozen snapshot, "
            f"or canonical basis: {path}"
        )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return (
        payload,
        str(path.resolve()),
        sha256_file(path),
    )


def attach_geometry(
    runner,
    model,
    basis_payload,
    language: str,
    method: str,
):
    bases = (
        basis_payload.get(
            "bases"
        )
    )

    if not isinstance(
        bases,
        Mapping,
    ):
        raise RuntimeError(
            "Basis payload has no bases dict."
        )

    targets = (
        runner
        .collect_attention_targets(
            model
        )
    )

    if len(targets) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGETS} "
            f"targets; got {len(targets)}"
        )

    if set(targets) != set(bases):
        missing = sorted(
            set(targets)
            - set(bases)
        )

        extra = sorted(
            set(bases)
            - set(targets)
        )

        raise RuntimeError(
            "Model/basis target mismatch. "
            f"missing={missing[:10]} "
            f"extra={extra[:10]}"
        )

    payload_language = (
        basis_payload.get(
            "language"
        )
    )

    if (
        payload_language is not None
        and
        payload_language != language
    ):
        raise RuntimeError(
            "Basis language mismatch: "
            f"{payload_language} != {language}"
        )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False
        parameter.grad = None

    rank_map = {}

    for name in sorted(
        targets
    ):
        item = bases[name]

        if (
            "P" not in item
            or
            "Q" not in item
        ):
            raise RuntimeError(
                f"Basis missing P/Q: {name}"
            )

        P = torch.as_tensor(
            item["P"]
        )

        Q = torch.as_tensor(
            item["Q"]
        )

        rank = basis_rank(
            item
        )

        if (
            P.shape[1] != rank
            or
            Q.shape[1] != rank
        ):
            raise RuntimeError(
                f"Basis rank/shape mismatch "
                f"at {name}"
            )

        rank_map[name] = rank

        if rank == 0:
            continue

        parent, leaf = (
            runner.get_parent(
                model,
                name,
            )
        )

        base = getattr(
            parent,
            leaf,
        )

        setattr(
            parent,
            leaf,
            runner.FCCALinear(
                base,
                P=P,
                Q=Q,
                scale=1.0,
            ),
        )

    used_budget = sum(
        rank * rank
        for rank in rank_map.values()
    )

    if method in {
        "fcca",
        "asym_hard",
        "asym_shared_private",
    }:
        bad = {
            name: rank
            for name, rank
            in rank_map.items()
            if rank != 128
        }

        if bad:
            raise RuntimeError(
                "Expected rank 128 everywhere. "
                f"Bad={list(bad.items())[:10]}"
            )

        if (
            used_budget
            != REFERENCE_CORE_BUDGET
        ):
            raise RuntimeError(
                "Fixed-rank core budget mismatch: "
                f"{used_budget:,} != "
                f"{REFERENCE_CORE_BUDGET:,}"
            )

    if method == "adaptive_fcca":
        allocation = (
            basis_payload.get(
                "rank_allocation"
            )
        )

        if allocation is not None:
            if (
                set(allocation)
                != set(rank_map)
            ):
                raise RuntimeError(
                    "Adaptive rank-allocation "
                    "target mismatch."
                )

            for name, rank in (
                rank_map.items()
            ):
                if (
                    int(
                        allocation[name]
                    )
                    != rank
                ):
                    raise RuntimeError(
                        "Adaptive rank mismatch "
                        f"at {name}: "
                        f"{allocation[name]} "
                        f"!= {rank}"
                    )

        expected_budget = (
            basis_payload.get(
                "trainable_core_budget",
                basis_payload.get(
                    "actual_core_budget",
                    basis_payload.get(
                        "reference_core_budget",
                        basis_payload.get(
                            "reference_budget"
                        ),
                    ),
                ),
            )
        )

        if (
            expected_budget is not None
            and
            int(expected_budget)
            != used_budget
        ):
            raise RuntimeError(
                "Adaptive budget mismatch: "
                f"{expected_budget} "
                f"!= {used_budget}"
            )

    return (
        sorted(targets),
        rank_map,
        used_budget,
    )


def exact_reconstruction(
    runner,
    checkpoint: Path,
    language: str,
    method: str,
    device: torch.device,
    trial_name: str,
):
    checkpoint_payload = (
        torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    )

    if not isinstance(
        checkpoint_payload,
        Mapping,
    ):
        raise RuntimeError(
            "Checkpoint is not a mapping."
        )

    config = dict(
        checkpoint_payload.get(
            "config",
            {},
        )
    )

    if (
        config.get("language")
        not in {None, language}
    ):
        raise RuntimeError(
            "Checkpoint language mismatch: "
            f"{config.get('language')} "
            f"!= {language}"
        )

    if (
        config.get("method")
        not in {None, method}
    ):
        raise RuntimeError(
            "Checkpoint method mismatch: "
            f"{config.get('method')} "
            f"!= {method}"
        )

    checkpoint_base = (
        checkpoint_payload.get(
            "base_model",
            config.get(
                "base_model"
            ),
        )
    )

    if (
        checkpoint_base is not None
        and
        checkpoint_base
        != BASE_MODEL
    ):
        raise RuntimeError(
            "Base-model mismatch: "
            f"{checkpoint_base} "
            f"!= {BASE_MODEL}"
        )

    run_dir = training_run_dir(
        language,
        method,
        trial_name,
    )

    basis_payload, basis_source, basis_sha = (
        load_exact_basis_payload(
            checkpoint_payload,
            run_dir,
            language,
            method,
        )
    )

    processor = (
        runner.build_processor()
    )

    model = (
        runner.build_model(
            device
        )
    )

    (
        targets,
        rank_map,
        core_budget,
    ) = attach_geometry(
        runner,
        model,
        basis_payload,
        language,
        method,
    )

    stored_rank_map = (
        checkpoint_payload.get(
            "rank_map"
        )
    )

    if stored_rank_map is not None:
        canonical_stored = {
            str(name):
                int(rank)
            for name, rank
            in stored_rank_map.items()
        }

        if (
            canonical_stored
            != rank_map
        ):
            raise RuntimeError(
                "Checkpoint rank_map does "
                "not match reconstructed basis."
            )

    # --------------------------------------------------------------
    # STRICT RULE:
    # We only evaluate fully self-contained checkpoints.
    #
    # This lets us prove that pretrained weights, frozen P/Q,
    # trainable FCCA cores and every persistent buffer are exactly
    # the state selected as "best".
    # --------------------------------------------------------------
    full_state = (
        checkpoint_payload.get(
            "full_model_state_dict"
        )
    )

    if full_state is None:
        full_state = (
            checkpoint_payload.get(
                "state_dict"
            )
        )

    if not isinstance(
        full_state,
        Mapping,
    ):
        raise RuntimeError(
            "best_checkpoint.pt does not contain a COMPLETE "
            "model state. Refusing held-out evaluation because "
            "100% exact reconstruction cannot be proven."
        )

    load_result = (
        model.load_state_dict(
            full_state,
            strict=True,
        )
    )

    if (
        load_result.missing_keys
        or
        load_result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict state reconstruction unexpectedly "
            "reported key mismatches."
        )

    # --------------------------------------------------------------
    # Tensor-by-tensor BIT-EXACT verification after loading.
    # This is deliberately stronger than just strict=True.
    # --------------------------------------------------------------
    reconstructed = (
        model.state_dict()
    )

    if (
        set(reconstructed)
        != set(full_state)
    ):
        raise RuntimeError(
            "State-key set changed after strict load."
        )

    checked_tensors = 0
    checked_elements = 0

    for name, expected in (
        full_state.items()
    ):
        actual = (
            reconstructed[name]
            .detach()
            .cpu()
        )

        expected_cpu = (
            expected
            .detach()
            .cpu()
        )

        if (
            actual.dtype
            != expected_cpu.dtype
        ):
            raise RuntimeError(
                f"State dtype mismatch: {name}: "
                f"{actual.dtype} != "
                f"{expected_cpu.dtype}"
            )

        if (
            tuple(actual.shape)
            != tuple(
                expected_cpu.shape
            )
        ):
            raise RuntimeError(
                f"State shape mismatch: {name}"
            )

        if not torch.equal(
            actual,
            expected_cpu,
        ):
            raise RuntimeError(
                "BIT-EXACT reconstruction failed "
                f"for tensor: {name}"
            )

        checked_tensors += 1
        checked_elements += int(
            actual.numel()
        )

    task_stats_path = (
        basis_payload.get(
            "source_task_stats",
            basis_payload.get(
                "task_stats_path"
            ),
        )
    )

    task_stats_sha = None

    if task_stats_path:
        stats = Path(
            str(task_stats_path)
        )

        if not stats.is_absolute():
            stats = (
                ROOT
                / stats
            )

        if stats.exists():
            task_stats_sha = (
                sha256_file(
                    stats
                )
            )

            expected_stats_sha = (
                checkpoint_payload.get(
                    "task_stats_sha256"
                )
            )

            if (
                expected_stats_sha
                is not None
                and
                task_stats_sha
                != str(
                    expected_stats_sha
                )
            ):
                raise RuntimeError(
                    "Task-stat SHA changed since training."
                )

    audit = {
        "checkpoint":
            str(
                checkpoint.resolve()
            ),

        "checkpoint_sha256":
            sha256_file(
                checkpoint
            ),

        "checkpoint_epoch":
            checkpoint_payload.get(
                "epoch"
            ),

        "language":
            language,

        "method":
            method,

        "base_model":
            BASE_MODEL,

        "strict_state_dict_load":
            True,

        "tensor_by_tensor_bit_exact":
            True,

        "checked_state_tensors":
            checked_tensors,

        "checked_state_elements":
            checked_elements,

        "basis_source":
            basis_source,

        "basis_sha256":
            basis_sha,

        "source_task_stats":
            task_stats_path,

        "current_task_stats_sha256":
            task_stats_sha,

        "target_count":
            len(targets),

        "rank_min":
            min(
                rank_map.values()
            ),

        "rank_max":
            max(
                rank_map.values()
            ),

        "core_parameter_budget":
            core_budget,

        "rank_allocation_matches_checkpoint":
            (
                stored_rank_map is None
                or
                {
                    str(k): int(v)
                    for k, v
                    in stored_rank_map.items()
                }
                == rank_map
            ),

        "reconstruction_passed":
            True,
    }

    # Full CPU state is no longer needed.
    del full_state
    del checkpoint_payload

    model.eval()

    if hasattr(
        model.config,
        "use_cache",
    ):
        model.config.use_cache = True

    return (
        model,
        processor,
        audit,
    )


# =====================================================================
# Held-out test manifest
# =====================================================================

def load_manifest(
    language: str,
):
    path = (
        test_manifest_path(language)
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    if not rows:
        raise RuntimeError(
            f"Empty manifest: {path}"
        )

    return path, rows


def parse_float(
    value,
):
    if (
        value is None
        or
        str(value).strip() == ""
    ):
        return None

    return float(
        value
    )


def resolve_manifest_rows(
    runner,
    language: str,
):
    manifest_path, manifest_rows = (
        load_manifest(
            language
        )
    )

    dataset_path = dataset_root(language)

    if not dataset_path.exists():
        raise FileNotFoundError(
            dataset_path
        )

    ds = load_fleurs_split(language, "test")

    if "audio" in ds.column_names:
        ds = ds.cast_column(
            "audio",
            Audio(decode=False),
        )

    # --------------------------------------------------------------
    # IMPORTANT:
    # FLEURS IDs are NOT assumed unique.
    #
    # Map every ID/path to a LIST of dataset rows.  During manifest
    # matching we disambiguate by exact path/transcription whenever
    # possible and consume duplicate matches without reusing rows.
    # --------------------------------------------------------------
    id_map = {}

    if "id" in ds.column_names:
        for index, value in enumerate(
            ds["id"]
        ):
            key = str(value).strip()

            if key:
                id_map.setdefault(
                    key,
                    [],
                ).append(
                    index
                )

    path_map = {}

    if "path" in ds.column_names:
        for index, value in enumerate(
            ds["path"]
        ):
            key = str(value).strip()

            if key:
                path_map.setdefault(
                    key,
                    [],
                ).append(
                    index
                )

    used_dataset_indices = set()

    included = []
    excluded = []

    for manifest_position, m in enumerate(
        manifest_rows
    ):
        manifest_id = str(
            m.get(
                "id",
                "",
            )
        ).strip()

        manifest_path_value = str(
            m.get(
                "path",
                "",
            )
        ).strip()

        manifest_reference = (
            m.get(
                "transcription"
            )
        )

        candidates = []

        # Path is strongest when available.
        if (
            manifest_path_value
            and
            manifest_path_value in path_map
        ):
            candidates.extend(
                path_map[
                    manifest_path_value
                ]
            )

        if (
            manifest_id
            and
            manifest_id in id_map
        ):
            candidates.extend(
                id_map[
                    manifest_id
                ]
            )

        # Deduplicate candidate indices while preserving order.
        candidates = list(
            dict.fromkeys(
                candidates
            )
        )

        # Never reuse a dataset row.
        candidates = [
            index
            for index in candidates
            if index not in used_dataset_indices
        ]

        # Exact transcription is an additional disambiguator.
        if (
            manifest_reference is not None
            and
            candidates
        ):
            reference_matches = [
                index
                for index in candidates
                if str(
                    ds[index].get(
                        "transcription"
                    )
                )
                == str(
                    manifest_reference
                )
            ]

            if reference_matches:
                candidates = (
                    reference_matches
                )

        dataset_index = None

        if candidates:
            # Duplicate IDs / even duplicate metadata are valid.
            # Consume matching rows deterministically in dataset order.
            dataset_index = min(
                candidates
            )

        # Explicit row index is a fallback only.
        if dataset_index is None:
            candidate = (
                m.get(
                    "row_index"
                )
            )

            if candidate in {
                None,
                "",
            }:
                candidate = (
                    m.get(
                        "index"
                    )
                )

            if candidate not in {
                None,
                "",
            }:
                value = int(
                    candidate
                )

                if (
                    0 <= value < len(ds)
                    and
                    value not in used_dataset_indices
                ):
                    dataset_index = value

        # Final safe positional fallback when manifest and Arrow split
        # have corresponding row order.
        if (
            dataset_index is None
            and
            manifest_position < len(ds)
            and
            manifest_position
            not in used_dataset_indices
        ):
            candidate_row = ds[
                manifest_position
            ]

            id_ok = (
                not manifest_id
                or
                str(
                    candidate_row.get(
                        "id",
                        "",
                    )
                ).strip()
                == manifest_id
            )

            ref_ok = (
                manifest_reference is None
                or
                str(
                    candidate_row.get(
                        "transcription"
                    )
                )
                == str(
                    manifest_reference
                )
            )

            if id_ok and ref_ok:
                dataset_index = (
                    manifest_position
                )

        if dataset_index is None:
            raise RuntimeError(
                "Could not uniquely map manifest row "
                f"{manifest_position} to an unused "
                "Arrow test row.\n"
                f"id={manifest_id!r}\n"
                f"path={manifest_path_value!r}\n"
                f"transcription={manifest_reference!r}"
            )

        used_dataset_indices.add(
            dataset_index
        )

        row = ds[
            dataset_index
        ]

        # Validate the transcription if the manifest supplied one.
        dataset_reference = (
            row.get(
                "transcription"
            )
        )

        if (
            manifest_reference is not None
            and
            str(manifest_reference)
            != str(dataset_reference)
        ):
            raise RuntimeError(
                "Manifest/dataset transcription mismatch "
                f"at manifest row {manifest_position}: "
                f"dataset_index={dataset_index}"
            )

        reference = str(
            manifest_reference
            if manifest_reference is not None
            else dataset_reference
        )

        duration = None

        num_samples = (
            row.get(
                "num_samples"
            )
        )

        if (
            num_samples is not None
            and
            str(num_samples).strip()
            != ""
        ):
            duration = (
                float(
                    num_samples
                )
                / SAMPLE_RATE
            )

        if duration is None:
            duration = parse_float(
                m.get(
                    "duration_seconds",
                    m.get(
                        "duration"
                    ),
                )
            )

        if duration is None:
            audio = (
                runner
                .decode_audio_value(
                    row["audio"]
                )
            )

            duration = (
                len(audio)
                / SAMPLE_RATE
            )

        item = {
            "manifest_position":
                manifest_position,

            "dataset_index":
                dataset_index,

            "id":
                row.get(
                    "id",
                    m.get(
                        "id",
                        dataset_index,
                    ),
                ),

            "duration_seconds":
                float(duration),

            "reference":
                reference,
        }

        if (
            float(duration)
            > MAX_DURATION_SECONDS
        ):
            excluded.append(
                item
            )

        else:
            included.append(
                item
            )

    if not included:
        raise RuntimeError(
            "No <=30s test examples remain."
        )

    print(
        "[manifest] "
        f"rows={len(manifest_rows)} "
        f"unique_dataset_rows_used={len(used_dataset_indices)} "
        f"duplicate_ids_allowed=True "
        f"eligible={len(included)} "
        f"excluded_over_30s={len(excluded)}",
        flush=True,
    )

    return (
        manifest_path,
        ds,
        manifest_rows,
        included,
        excluded,
    )


# =====================================================================
# Exact Qwen decoding
# =====================================================================

@torch.inference_mode()
def transcribe(
    runner,
    model,
    processor,
    audio,
    language: str,
    device: torch.device,
):
    inputs = (
        processor
        .apply_transcription_request(
            audio=audio,
            language=LANGUAGES[
                language
            ],
        )
    )

    moved = {}

    for key, value in dict(
        inputs
    ).items():
        if torch.is_tensor(
            value
        ):
            value = value.to(
                device
            )

            if value.is_floating_point():
                value = value.to(
                    dtype=torch.bfloat16
                )

        moved[key] = value

    prompt_length = int(
        moved[
            "input_ids"
        ].shape[1]
    )

    guard = (
        TrackingRepeatedTokenCycleGuard(
            prompt_length=
                prompt_length,

            max_cycle_length=8,
            repeats_required=6,
        )
    )

    output_ids = (
        model.generate(
            **moved,
            max_new_tokens=512,
            do_sample=False,
            stopping_criteria=
                StoppingCriteriaList(
                    [guard]
                ),
        )
    )

    generated = (
        output_ids[
            :,
            prompt_length:
        ]
    )

    decoded = (
        processor.decode(
            generated,
            return_format=
                "transcription_only",
        )
    )

    return (
        str(
            decoded[0]
        ).strip(),
        bool(
            guard.triggered
        ),
        int(
            generated.shape[1]
        ),
    )


# =====================================================================
# Main evaluator
# =====================================================================

def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    (
        distributed,
        local_rank,
        rank,
        world_size,
        device,
    ) = setup_distributed(
        args.device
    )

    runner, runner_path = (
        import_runner()
    )

    language = args.language
    method = args.method

    if method in {
        "fft",
        "lora",
    }:
        run_dir = (
            RESULT_ROOT
            / language
            / METHOD_DIR[method]
            / args.trial_name
        )
    else:
        run_dir = training_run_dir(
            language,
            method,
            args.trial_name,
        )

    checkpoint = (
        args.checkpoint
        if args.checkpoint
        is not None
        else default_checkpoint(
            language,
            method,
            args.trial_name,
        )
    )

    checkpoint = (
        checkpoint
        .expanduser()
        .resolve()
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            "Best checkpoint missing: "
            f"{checkpoint}"
        )

    architecture = (
        METHOD_DIR[
            method
        ]
    )

    # Standard LoRA checkpoints can have arbitrary rank.
    # Name held-out outputs using the rank stored in the checkpoint.
    if method == "lora":
        _payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        _cfg = dict(
            _payload.get(
                "config",
                {},
            )
        )
        architecture = (
            f"lora_r{int(_cfg.get('lora_rank', 128))}"
        )
        del _payload
        del _cfg

    output_root = (
        RESULT_ROOT
        / language
        / "test"
        / architecture
    )

    if method in {
        "fft",
        "lora",
    }:
        output_root = (
            output_root
            / args.trial_name
        )

    predictions_path = (
        output_root
        / "predictions.csv"
    )

    metrics_path = (
        output_root
        / "wer.json"
    )

    audit_path = (
        output_root
        / "reconstruction_audit.json"
    )

    if rank == 0:
        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            not args.force
            and
            (
                predictions_path.exists()
                or
                metrics_path.exists()
                or
                audit_path.exists()
            )
        ):
            raise RuntimeError(
                "Held-out output already exists at "
                f"{output_root}. "
                "Use --force only if intentional."
            )

    barrier(
        distributed
    )

    print(
        f"[rank {rank}] "
        f"device={device} "
        f"checkpoint={checkpoint}",
        flush=True,
    )

    if method in {
        "fft",
        "lora",
    }:
        (
            model,
            processor,
            reconstruction_audit,
            standard_runner_path,
        ) = exact_standard_reconstruction(
            checkpoint,
            language,
            method,
            device,
        )

        runner_path = (
            standard_runner_path
        )

    else:
        (
            model,
            processor,
            reconstruction_audit,
        ) = exact_reconstruction(
            runner,
            checkpoint,
            language,
            method,
            device,
            args.trial_name,
        )

    print(
        f"[rank {rank}] "
        "BIT-EXACT MODEL RECONSTRUCTION: PASS",
        flush=True,
    )

    (
        manifest_path,
        test_ds,
        all_manifest_rows,
        included,
        excluded,
    ) = resolve_manifest_rows(
        runner,
        language,
    )

    shard = (
        included[
            rank::world_size
        ]
    )

    print(
        f"[rank {rank}] "
        f"manifest_total={len(all_manifest_rows)} "
        f"excluded_over_30s={len(excluded)} "
        f"eligible={len(included)} "
        f"local={len(shard)}",
        flush=True,
    )

    rows = []

    for local_i, item in enumerate(
        shard,
        start=1,
    ):
        ds_row = test_ds[
            item[
                "dataset_index"
            ]
        ]

        audio = (
            runner
            .decode_audio_value(
                ds_row[
                    "audio"
                ]
            )
        )

        actual_duration = (
            len(audio)
            / SAMPLE_RATE
        )

        # Strong final duration check using actual decoded waveform.
        if (
            actual_duration
            > MAX_DURATION_SECONDS
        ):
            raise RuntimeError(
                "Manifest/num_samples said <=30s but "
                "decoded audio is >30s: "
                f"id={item['id']} "
                f"actual={actual_duration:.9f}"
            )

        (
            hypothesis,
            guard_triggered,
            generated_tokens,
        ) = transcribe(
            runner,
            model,
            processor,
            audio,
            language,
            device,
        )

        reference = (
            item[
                "reference"
            ]
        )

        rows.append(
            {
                "manifest_position":
                    item[
                        "manifest_position"
                    ],

                "dataset_index":
                    item[
                        "dataset_index"
                    ],

                "id":
                    item[
                        "id"
                    ],

                "duration_seconds":
                    actual_duration,

                "reference_original":
                    reference,

                "hypothesis_original":
                    hypothesis,

                "reference_normalised":
                    runner.normalise_text(
                        reference
                    ),

                "hypothesis_normalised":
                    runner.normalise_text(
                        hypothesis
                    ),

                "loop_guard_triggered":
                    int(
                        guard_triggered
                    ),

                "generated_tokens":
                    generated_tokens,

                "rank":
                    rank,
            }
        )

        if (
            local_i % 25 == 0
            or
            local_i == len(shard)
        ):
            print(
                f"[rank {rank}] "
                f"decoded "
                f"{local_i}/{len(shard)}",
                flush=True,
            )

    shard_path = (
        output_root
        / (
            f".predictions"
            f".rank{rank}.csv"
        )
    )

    write_csv(
        shard_path,
        rows,
    )

    shard_audit_path = (
        output_root
        / (
            f".audit"
            f".rank{rank}.json"
        )
    )

    atomic_json(
        shard_audit_path,
        reconstruction_audit,
    )

    barrier(
        distributed
    )

    if rank == 0:
        combined = []
        audits = []

        for shard_rank in range(
            world_size
        ):
            path = (
                output_root
                / (
                    f".predictions"
                    f".rank{shard_rank}.csv"
                )
            )

            if path.stat().st_size:
                with path.open(
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    combined.extend(
                        csv.DictReader(
                            handle
                        )
                    )

            audit = json.loads(
                (
                    output_root
                    / (
                        f".audit"
                        f".rank{shard_rank}.json"
                    )
                ).read_text(
                    encoding="utf-8"
                )
            )

            if not audit.get(
                "reconstruction_passed"
            ):
                raise RuntimeError(
                    "A GPU rank failed reconstruction audit."
                )

            if not audit.get(
                "tensor_by_tensor_bit_exact"
            ):
                raise RuntimeError(
                    "A GPU rank was not bit-exact."
                )

            audits.append(
                audit
            )

        combined.sort(
            key=lambda row:
                int(
                    row[
                        "manifest_position"
                    ]
                )
        )

        if len(combined) != len(
            included
        ):
            raise RuntimeError(
                "Merged prediction count mismatch: "
                f"{len(combined)} != "
                f"{len(included)}"
            )

        refs = [
            row[
                "reference_original"
            ]
            for row in combined
        ]

        hyps = [
            row[
                "hypothesis_original"
            ]
            for row in combined
        ]

        wer = (
            runner.corpus_wer(
                refs,
                hyps,
                language,
            )
        )

        loop_guard_count = sum(
            int(
                row[
                    "loop_guard_triggered"
                ]
            )
            for row in combined
        )

        result = {
            "language":
                language,

            "architecture":
                architecture,

            "method":
                method,

            "split":
                "heldout_test_manifest",

            "manifest":
                str(
                    manifest_path.resolve()
                ),

            "checkpoint":
                str(
                    checkpoint
                ),

            "checkpoint_sha256":
                sha256_file(
                    checkpoint
                ),

            "runner":
                str(
                    runner_path.resolve()
                ),

            "world_size":
                world_size,

            "manifest_examples":
                len(
                    all_manifest_rows
                ),

            "excluded_over_30_seconds":
                len(
                    excluded
                ),

            "evaluated_examples":
                len(
                    combined
                ),

            "duration_policy":
                "exclude decoded recordings with duration > 30.0 seconds",

            "max_duration_seconds":
                MAX_DURATION_SECONDS,

            "generation":
                {
                    "do_sample":
                        False,

                    "max_new_tokens":
                        512,

                    "loop_guard":
                        {
                            "generated_tokens_only":
                                True,

                            "max_cycle_length":
                                8,

                            "repeats_required":
                                6,

                            "action":
                                "stop only; six repeated cycles remain visible",

                            "triggered_examples":
                                loop_guard_count,
                        },
                },

            "reconstruction":
                {
                    "all_gpu_ranks_passed":
                        True,

                    "strict_state_dict":
                        True,

                    "tensor_by_tensor_bit_exact":
                        True,

                    "per_rank":
                        audits,
                },

            **wer,
        }

        write_csv(
            predictions_path,
            combined,
        )

        atomic_json(
            metrics_path,
            result,
        )

        atomic_json(
            audit_path,
            {
                "all_gpu_ranks_passed":
                    True,

                "world_size":
                    world_size,

                "per_rank":
                    audits,
            },
        )

        print()
        print("=" * 100)
        print(
            "HELD-OUT TEST COMPLETE"
        )
        print(
            f"language={language}"
        )
        print(
            f"architecture={architecture}"
        )
        print(
            f"evaluated={len(combined)}"
        )
        print(
            f"excluded_over_30s={len(excluded)}"
        )
        print(
            f"loop_guard_triggers={loop_guard_count}"
        )
        print(
            f"{wer['metric']}="
            f"{wer['normalised_wer']:.8f}%"
        )
        print(
            "BIT-EXACT RECONSTRUCTION=PASS"
        )
        print(
            f"predictions={predictions_path}"
        )
        print(
            f"metrics={metrics_path}"
        )
        print(
            f"audit={audit_path}"
        )
        print("=" * 100)

        for shard_rank in range(
            world_size
        ):
            (
                output_root
                / (
                    f".predictions"
                    f".rank{shard_rank}.csv"
                )
            ).unlink(
                missing_ok=True
            )

            (
                output_root
                / (
                    f".audit"
                    f".rank{shard_rank}.json"
                )
            ).unlink(
                missing_ok=True
            )

    barrier(
        distributed
    )

    if distributed:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        choices=sorted(
            LANGUAGES
        ),
        required=True,
    )

    parser.add_argument(
        "--method",
        choices=sorted(
            METHOD_DIR
        ),
        required=True,
    )

    parser.add_argument(
        "--trial-name",
        default="trial_0000",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional explicit best_checkpoint.pt. "
            "Default is the selected method/trial best checkpoint."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(
        parse_args()
    )
