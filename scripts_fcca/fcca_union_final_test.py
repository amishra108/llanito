from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from statistics import mean, pstdev
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, Subset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate_loaded_model

from scripts.train import (
    WhisperDataCollator,
    WhisperManifestDataset,
    build_compute_metrics,
    build_training_arguments,
    configure_whisper_generation,
    load_manifest,
    make_trainer,
    set_seed,
    validate_cuda_environment,
    validate_manifest_files_exist,
    validate_no_duplicate_audio,
)

PAPER = "https://arxiv.org/pdf/2609.29800"
BASE_MODEL = "openai/whisper-medium"
LANGUAGE = "es"
TASK = "transcribe"

METHODS = ("standard", "ac-h", "ac-sp", "ar")

RANK = 128
SHARED_RANK = 64
PRIVATE_RANK = 64
TARGET_COUNT = 288
BUDGET = TARGET_COUNT * RANK * RANK

EXPECTED_TRAINING_EXAMPLES = 904
EXPECTED_TEST_RECORDINGS = 57
CALIBRATION_SIZE = 256
CALIBRATION_SEED = 2026
DAMPING = 1.0e-6
SCALE = 1.0

LEARNING_RATE = 1.8e-4
MAX_EPOCHS = 5
EFFECTIVE_BATCH_SIZE = 8
DEFAULT_MICROBATCH = 1
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 2026
DATALOADER_WORKERS = 4
LOGGING_STEPS = 10
GENERATION_NUM_BEAMS = 1
GENERATION_MAX_LENGTH = 225
SAVE_TOTAL_LIMIT = MAX_EPOCHS

AC_MAX_ITERS = 8
AC_TOL = 1.0e-5

TARGET_LEAVES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "o_proj",
}

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "llanito_cv"
TRAIN_MANIFEST = PROCESSED_ROOT / "train_validation_union_manifest.csv"
TEST_MANIFEST = PROCESSED_ROOT / "fixed_test.csv"

CACHE_ROOT = OUTPUT_ROOT / "basis_cache"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "scripts_fcca" / "fcca_union_final_test"
NORMALISATION_FILE = PROJECT_ROOT / "scripts" / "resources" / "LLanito_normalisation_v2.txt"

NORMALISATION_MAPPING_COLUMN = "task"
NORMALISATION_SECTION_MARKERS = {
    "passage": "SECTION 1 - PASSAGE RECORDINGS ONLY",
    "sentence": "SECTION 2 - SENTENCE-TASK RECORDINGS ONLY",
}
DATASET_NAME = "llanito_fixed_test"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def save_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def process_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_rank_zero() -> bool:
    return process_rank() == 0


def wait_for_file(path: Path, timeout_seconds: int = 7200) -> None:
    deadline = time.time() + timeout_seconds
    while not path.exists():
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(1.0)


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()[:24]


def method_tag(method: str) -> str:
    return {
        "standard": "fcca",
        "ac-h": "ac_fcca_h",
        "ac-sp": "ac_fcca_sp",
        "ar": "ar_fcca",
    }[method]


def method_output(method: str) -> Path:
    return OUTPUT_ROOT / method_tag(method)


def checkpoint_root(method: str) -> Path:
    return method_output(method) / "checkpoints"


def evaluation_root(method: str) -> Path:
    return method_output(method) / "test"


def task_stats_path() -> Path:
    return CACHE_ROOT / "task_stats_256.pt"


def basis_path(method: str) -> Path:
    return CACHE_ROOT / "bases" / f"{method_tag(method)}_basis.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    found = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        found.append((step, path))
    if not found:
        return None
    return max(found, key=lambda item: item[0])[1]


def get_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def collect_attention_targets(model: nn.Module) -> dict[str, nn.Linear]:
    targets = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in TARGET_LEAVES:
            continue
        if (
            ".encoder.layers." in name
            and ".self_attn." in name
        ):
            targets[name] = module
        elif (
            ".decoder.layers." in name
            and ".self_attn." in name
        ):
            targets[name] = module
        elif (
            ".decoder.layers." in name
            and ".encoder_attn." in name
        ):
            targets[name] = module

    encoder_self = sum(
        ".encoder.layers." in name and ".self_attn." in name
        for name in targets
    )
    decoder_self = sum(
        ".decoder.layers." in name and ".self_attn." in name
        for name in targets
    )
    decoder_cross = sum(
        ".decoder.layers." in name and ".encoder_attn." in name
        for name in targets
    )

    if (encoder_self, decoder_self, decoder_cross, len(targets)) != (96, 96, 96, 288):
        raise RuntimeError(
            "Unexpected Whisper-medium attention targets: "
            f"encoder_self={encoder_self}, decoder_self={decoder_self}, "
            f"decoder_cross={decoder_cross}, total={len(targets)}"
        )
    return targets


TARGET_RE = re.compile(
    r"^model\."
    r"(encoder|decoder)"
    r"\.layers\.(\d+)\."
    r"(self_attn|encoder_attn)\."
    r"(q_proj|k_proj|v_proj|out_proj|o_proj)$"
)


def parse_target(name: str) -> dict[str, Any]:
    match = TARGET_RE.match(name)
    if match is None:
        raise RuntimeError(f"Unrecognised target: {name}")

    stack = match.group(1)
    layer = int(match.group(2))
    attn = match.group(3)
    leaf = match.group(4)

    if stack == "encoder":
        if attn != "self_attn":
            raise RuntimeError(name)
        family = "encoder_self"
    elif attn == "self_attn":
        family = "decoder_self"
    else:
        family = "decoder_cross"

    projection = "o" if leaf in {"out_proj", "o_proj"} else leaf[0]

    return {
        "stack": stack,
        "layer": layer,
        "family": family,
        "projection": projection,
    }


def make_pair_groups(names) -> dict[tuple[str, str, int], list[str]]:
    groups: dict[tuple[str, str, int], list[str]] = {}

    for name in names:
        info = parse_target(name)
        layer = info["layer"]

        if not 0 <= layer < 24:
            raise RuntimeError(f"Expected layers 0..23: {name}")

        pair_start = layer if layer % 2 == 0 else layer - 1
        key = (info["family"], info["projection"], pair_start)
        groups.setdefault(key, []).append(name)

    if len(groups) != 144:
        raise RuntimeError(
            f"Expected 144 adjacent-pair groups, found {len(groups)}"
        )

    for key, members in groups.items():
        members.sort(key=lambda x: parse_target(x)["layer"])
        if len(members) != 2:
            raise RuntimeError(f"{key}: expected 2 matrices, got {len(members)}")
        layers = [parse_target(x)["layer"] for x in members]
        if layers != [key[2], key[2] + 1]:
            raise RuntimeError(f"Bad pair {key}: {layers}")

    return groups


def uses_encoder_time(name: str) -> bool:
    if ".encoder.layers." in name and ".self_attn." in name:
        return True
    if ".decoder.layers." in name and ".encoder_attn." in name:
        leaf = name.rsplit(".", 1)[-1]
        return leaf in {"k_proj", "v_proj"}
    return False


def resize_prefix_mask(mask: torch.Tensor, target_length: int) -> torch.Tensor:
    if mask.ndim != 2:
        raise RuntimeError(f"Expected 2-D mask, got {tuple(mask.shape)}")

    mask = mask.bool()
    source_length = int(mask.shape[1])

    if source_length == target_length:
        return mask

    valid = mask.long().sum(dim=1)

    if source_length == 2 * target_length:
        target_valid = (valid + 1) // 2
    else:
        target_valid = torch.ceil(
            valid.float() * (float(target_length) / float(source_length))
        ).long()

    target_valid = target_valid.clamp(min=0, max=target_length)
    positions = torch.arange(target_length, device=mask.device)[None, :]
    return positions < target_valid[:, None]


class CalibrationContext:
    encoder_attention_mask: torch.Tensor | None = None
    decoder_mask: torch.Tensor | None = None


CTX = CalibrationContext()


class TaskStat:
    def __init__(self, name: str, module: nn.Linear) -> None:
        self.name = name
        self.module = module
        device = module.weight.device

        self.cross_sum = torch.zeros(
            module.out_features,
            module.in_features,
            device=device,
            dtype=torch.float32,
        )
        self.activation_second_sum = torch.zeros(
            module.in_features,
            device=device,
            dtype=torch.float32,
        )
        self.gradient_second_sum = torch.zeros(
            module.out_features,
            device=device,
            dtype=torch.float32,
        )
        self.positions = 0
        self._handle = None

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = int(x.shape[-2])

        if uses_encoder_time(self.name):
            source = CTX.encoder_attention_mask
            if source is None:
                raise RuntimeError(f"Missing encoder mask for {self.name}")
            return resize_prefix_mask(source, sequence_length)

        source = CTX.decoder_mask
        if source is None:
            raise RuntimeError(f"Missing decoder mask for {self.name}")
        if int(source.shape[1]) != sequence_length:
            raise RuntimeError(
                f"Decoder mask length mismatch for {self.name}: "
                f"mask={source.shape[1]}, sequence={sequence_length}"
            )
        return source.bool()

    def install(self) -> None:
        def forward_hook(module, inputs, output):
            del module

            if not torch.is_tensor(output):
                raise RuntimeError(f"Non-tensor target output: {self.name}")

            x = inputs[0].detach()

            if x.ndim != 3 or output.ndim != 3:
                raise RuntimeError(
                    f"Expected [batch,time,hidden] at {self.name}; "
                    f"x={tuple(x.shape)}, output={tuple(output.shape)}"
                )

            mask = self._mask(x)

            if tuple(mask.shape) != tuple(x.shape[:2]):
                raise RuntimeError(
                    f"Mask shape mismatch at {self.name}: "
                    f"mask={tuple(mask.shape)}, x={tuple(x.shape)}"
                )

            if not output.requires_grad:
                output.requires_grad_(True)

            x_selected = x[mask].float()

            if x_selected.numel() == 0:
                raise RuntimeError(f"No valid positions at {self.name}")

            self.activation_second_sum.add_(
                x_selected.pow(2).sum(dim=0)
            )

            def error_hook(grad: torch.Tensor) -> torch.Tensor:
                grad_selected = grad.detach()[mask].float()

                if grad_selected.shape[0] != x_selected.shape[0]:
                    raise RuntimeError(f"x/delta count mismatch at {self.name}")

                self.cross_sum.add_(grad_selected.T @ x_selected)
                self.gradient_second_sum.add_(
                    grad_selected.pow(2).sum(dim=0)
                )
                self.positions += int(x_selected.shape[0])
                return grad

            output.register_hook(error_hook)

        self._handle = self.module.register_forward_hook(forward_hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def build_task_stats(
    *,
    train_rows: list[dict[str, Any]],
    train_dataset,
    collator,
    fingerprint: str,
    path: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    ).to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None

    targets = collect_attention_targets(model)
    if len(targets) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_COUNT} attention matrices, found {len(targets)}"
        )

    stats = {
        name: TaskStat(name, module)
        for name, module in targets.items()
    }
    for stat in stats.values():
        stat.install()

    if len(train_dataset) != EXPECTED_TRAINING_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_EXAMPLES} union examples, "
            f"found {len(train_dataset)}"
        )

    # Calibrate on one deterministic 256-example subset of the 904-example union.
    generator = torch.Generator().manual_seed(CALIBRATION_SEED)
    selected_indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    )[:CALIBRATION_SIZE].tolist()

    if len(selected_indices) != CALIBRATION_SIZE:
        raise RuntimeError("Calibration subset size mismatch")

    loader = DataLoader(
        Subset(train_dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    print(
        f"[task-stats] calibration={CALIBRATION_SIZE} "
        f"seed={CALIBRATION_SEED} targets={len(targets)}",
        flush=True,
    )

    try:
        for index, batch in enumerate(loader, start=1):
            clean_batch = {
                key: (
                    value.to(device, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in batch.items()
            }

            attention_mask = clean_batch.get("attention_mask")
            labels = clean_batch.get("labels")
            if attention_mask is None or labels is None:
                raise RuntimeError(
                    "Calibration batch lacks attention_mask or labels"
                )

            CTX.encoder_attention_mask = attention_mask.bool()
            CTX.decoder_mask = labels.ne(-100)

            outputs = model(**clean_batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite calibration loss at example {index}"
                )

            valid_tokens = int(labels.ne(-100).sum().item())
            if valid_tokens < 1:
                raise RuntimeError("Calibration example has no labels")

            (loss * float(valid_tokens)).backward()

            CTX.encoder_attention_mask = None
            CTX.decoder_mask = None

            if index % 25 == 0 or index == CALIBRATION_SIZE:
                print(
                    f"[task-stats] {index}/{CALIBRATION_SIZE}",
                    flush=True,
                )

    finally:
        CTX.encoder_attention_mask = None
        CTX.decoder_mask = None
        for stat in stats.values():
            stat.remove()

    modules = {}
    for name in sorted(stats):
        stat = stats[name]
        if stat.positions <= 0:
            raise RuntimeError(f"No positions for {name}")
        modules[name] = {
            "cross_sum": stat.cross_sum.detach().cpu().clone(),
            "activation_second_sum":
                stat.activation_second_sum.detach().cpu().clone(),
            "gradient_second_sum":
                stat.gradient_second_sum.detach().cpu().clone(),
            "positions": int(stat.positions),
        }

    payload = {
        "schema": "llanito_whisper_task_stats_v1",
        "source": "union_256_single_calibration",
        "paper": PAPER,
        "base_model": BASE_MODEL,
        "language": LANGUAGE,
        "task": TASK,
        "dataset": str(TRAIN_MANIFEST),
        "fold_fingerprint": fingerprint,
        "calibration_size": CALIBRATION_SIZE,
        "calibration_seed": CALIBRATION_SEED,
        "damping": DAMPING,
        "selected_indices": selected_indices,
        "target_count": len(modules),
        "modules": modules,
    }

    save_torch(path, payload)
    print(f"[task-stats] saved once: {path}", flush=True)
    print(f"[task-stats] sha256={file_sha256(path)}", flush=True)

    del model
    del stats
    torch.cuda.empty_cache()
    return payload


def get_task_stats(
    *,
    train_rows: list[dict[str, Any]],
    train_dataset,
    collator,
    rebuild: bool,
) -> tuple[dict[str, Any], Path, str]:
    fingerprint = rows_fingerprint(train_rows)
    path = task_stats_path()

    if is_rank_zero():
        if rebuild and path.exists():
            path.unlink()
        if not path.exists():
            build_task_stats(
                train_rows=train_rows,
                train_dataset=train_dataset,
                collator=collator,
                fingerprint=fingerprint,
                path=path,
            )
    else:
        wait_for_file(path)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    checks = {
        "source": "union_256_single_calibration",
        "fold_fingerprint": fingerprint,
        "calibration_size": CALIBRATION_SIZE,
        "calibration_seed": CALIBRATION_SEED,
        "target_count": TARGET_COUNT,
        "damping": DAMPING,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Task-stat cache mismatch for {key}: "
                f"expected={expected!r}, got={payload.get(key)!r}"
            )

    digest = file_sha256(path)
    print(
        f"[task-stats] shared calibration loaded: {path} sha256={digest}",
        flush=True,
    )
    return payload, path, digest


def mean_stats(
    stat: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    count = int(stat["positions"])
    if count <= 0:
        raise RuntimeError(f"Invalid task-stat count={count}")

    G = (
        stat["cross_sum"].to(
            device=device,
            dtype=torch.float64,
        )
        / float(count)
    )

    a = (
        stat["activation_second_sum"].to(
            device=device,
            dtype=torch.float64,
        )
        / float(count)
    )

    d = (
        stat["gradient_second_sum"].to(
            device=device,
            dtype=torch.float64,
        )
        / float(count)
    )

    if G.ndim != 2 or a.ndim != 1 or d.ndim != 1:
        raise RuntimeError("Invalid task-stat dimensions")

    if G.shape[0] != d.numel() or G.shape[1] != a.numel():
        raise RuntimeError("Task-stat dimensions disagree")

    if not torch.isfinite(G).all():
        raise RuntimeError("Non-finite G")
    if not torch.isfinite(a).all():
        raise RuntimeError("Non-finite A diagonal")
    if not torch.isfinite(d).all():
        raise RuntimeError("Non-finite D diagonal")
    if torch.any(a < 0) or torch.any(d < 0):
        raise RuntimeError("Negative second moment")

    return G, a, d, count


def whiten(
    stat: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    G, a, d, count = mean_stats(stat, device)

    inv_a = torch.rsqrt(a + DAMPING)
    inv_d = torch.rsqrt(d + DAMPING)
    Gt = inv_d[:, None] * G * inv_a[None, :]

    if not torch.isfinite(Gt).all():
        raise RuntimeError("Non-finite Fisher-whitened cross covariance")

    return {
        "G": G,
        "a": a,
        "d": d,
        "inv_a": inv_a,
        "inv_d": inv_d,
        "Gt": Gt,
        "count": count,
    }


def symmetrise(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.T)


def top_eig(A: torch.Tensor, rank: int) -> torch.Tensor:
    A = symmetrise(A)
    _, evecs = torch.linalg.eigh(A)

    if rank > evecs.shape[1]:
        raise RuntimeError(
            f"rank={rank} impossible for {tuple(A.shape)}"
        )

    V = evecs[:, -rank:]
    V = torch.flip(V, dims=(1,))
    return V.contiguous()


def orth(
    X: torch.Tensor,
    rank: int,
    name: str,
) -> torch.Tensor:
    if X.ndim != 2:
        raise RuntimeError(
            f"{name}: expected matrix, got {tuple(X.shape)}"
        )
    if rank > X.shape[0]:
        raise RuntimeError(
            f"{name}: rank={rank} exceeds ambient dimension {X.shape[0]}"
        )

    U, _, _ = torch.linalg.svd(
        X,
        full_matrices=True,
    )
    Q = U[:, :rank].contiguous()

    eye = torch.eye(
        rank,
        device=Q.device,
        dtype=Q.dtype,
    )
    error = float(
        torch.linalg.matrix_norm(
            Q.T @ Q - eye,
            ord="fro",
        ).item()
    )

    if error > 1.0e-8:
        raise RuntimeError(
            f"{name}: orthogonality failure {error:.12e}"
        )

    return Q


def column_span_basis(
    X: torch.Tensor,
    rtol: float = 1.0e-10,
) -> torch.Tensor:
    if X.ndim != 2:
        raise RuntimeError(
            f"Expected matrix, got {tuple(X.shape)}"
        )

    if X.shape[1] == 0:
        return X[:, :0]

    U, S, _ = torch.linalg.svd(
        X,
        full_matrices=False,
    )

    if S.numel() == 0:
        return X[:, :0]

    threshold = float(S[0].item()) * float(rtol)
    numerical_rank = int((S > threshold).sum().item())

    return U[:, :numerical_rank].contiguous()


def projector_complement(V: torch.Tensor) -> torch.Tensor:
    n = V.shape[0]
    I = torch.eye(
        n,
        device=V.device,
        dtype=V.dtype,
    )
    return I - V @ V.T


def independent_fcca_score(
    Gt: torch.Tensor,
    rank: int,
) -> float:
    s = torch.linalg.svdvals(Gt)
    return float(
        torch.sum(s[:rank] ** 2).item()
    )


def side_covariance(
    Gt: torch.Tensor,
    side: str,
) -> torch.Tensor:
    if side == "right":
        return Gt.T @ Gt
    if side == "left":
        return Gt @ Gt.T
    raise ValueError(side)


def side_score(
    C: torch.Tensor,
    V: torch.Tensor,
) -> float:
    return float(
        torch.trace(V.T @ C @ V).item()
    )


def solve_hard_group(
    members: list[str],
    whitened: dict[str, dict[str, Any]],
    shared_side: str,
):
    C = {
        name: side_covariance(
            whitened[name]["Gt"],
            shared_side,
        )
        for name in members
    }

    pooled = sum(C.values())
    shared = top_eig(pooled, RANK)

    bases = {}
    shared_objective = 0.0
    independent_objective = 0.0

    for name in members:
        Gt = whitened[name]["Gt"]

        if shared_side == "right":
            V = shared
            U = orth(
                Gt @ V,
                RANK,
                f"{name}: U=orth(GV)",
            )
        else:
            U = shared
            V = orth(
                Gt.T @ U,
                RANK,
                f"{name}: V=orth(G^T U)",
            )

        shared_objective += float(
            torch.sum(
                (U.T @ Gt @ V) ** 2
            ).item()
        )

        independent_objective += (
            independent_fcca_score(
                Gt,
                RANK,
            )
        )

        bases[name] = {
            "U": U,
            "V": V,
        }

    tolerance = (
        1.0e-7
        * max(
            independent_objective,
            1.0,
        )
    )

    if (
        shared_objective
        > independent_objective
        + tolerance
    ):
        raise RuntimeError(
            "AC-FCCA-H theorem audit failed: "
            f"{shared_objective:.12e} > "
            f"{independent_objective:.12e}"
        )

    return bases, {
        "shared_objective":
            shared_objective,
        "independent_fcca_objective":
            independent_objective,
        "ratio_to_independent":
            shared_objective
            / independent_objective,
    }


def private_update(
    C: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    P = projector_complement(shared)
    private = top_eig(
        P @ C @ P,
        PRIVATE_RANK,
    )
    return orth(
        P @ private,
        PRIVATE_RANK,
        "private update",
    )


def shared_update(
    C_by_name: dict[str, torch.Tensor],
    private_by_name: dict[str, torch.Tensor],
) -> torch.Tensor:
    raw_union = torch.cat(
        [
            private_by_name[name]
            for name in sorted(
                private_by_name
            )
        ],
        dim=1,
    )

    union = column_span_basis(
        raw_union
    )

    ambient = int(
        raw_union.shape[0]
    )

    remaining = (
        ambient
        - int(
            union.shape[1]
        )
    )

    if remaining < SHARED_RANK:
        raise RuntimeError(
            "Private union leaves insufficient room "
            f"for shared rank: ambient={ambient}, "
            f"private_union_rank={union.shape[1]}, "
            f"remaining={remaining}"
        )

    P = projector_complement(
        union
    )

    pooled = sum(
        C_by_name.values()
    )

    shared = top_eig(
        P @ pooled @ P,
        SHARED_RANK,
    )

    shared = orth(
        P @ shared,
        SHARED_RANK,
        "shared conditional update",
    )

    for name, private in (
        private_by_name.items()
    ):
        overlap = float(
            torch.linalg.matrix_norm(
                shared.T @ private,
                ord="fro",
            ).item()
        )

        if overlap > 1.0e-7:
            raise RuntimeError(
                "Shared/private feasibility "
                f"failure for {name}: "
                f"{overlap:.12e}"
            )

    return shared


def shared_private_objective(
    C_by_name,
    shared,
    private_by_name,
) -> float:
    total = 0.0

    for name, C in C_by_name.items():
        total += side_score(
            C,
            shared,
        )
        total += side_score(
            C,
            private_by_name[name],
        )

    return total


def make_shared_private_initializations(
    pooled: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    pooled_top128 = top_eig(
        pooled,
        RANK,
    )

    starts = [
        pooled_top128[
            :,
            :SHARED_RANK,
        ].contiguous()
    ]

    for seed in (
        20261,
        20262,
        20263,
    ):
        generator = (
            torch.Generator(
                device="cpu"
            )
            .manual_seed(seed)
        )

        random_matrix = torch.randn(
            RANK,
            SHARED_RANK,
            generator=generator,
            dtype=pooled.dtype,
        ).to(device)

        rotation, _ = (
            torch.linalg.qr(
                random_matrix,
                mode="reduced",
            )
        )

        candidate = (
            pooled_top128
            @ rotation
        )

        candidate = orth(
            candidate,
            SHARED_RANK,
            f"initial shared seed={seed}",
        )

        starts.append(
            candidate
        )

    return starts


def run_shared_private_start(
    C_by_name: dict[str, torch.Tensor],
    shared_initial: torch.Tensor,
    max_iters: int,
    tol: float,
):
    shared = shared_initial

    private = {
        name: private_update(
            C_by_name[name],
            shared,
        )
        for name in sorted(
            C_by_name
        )
    }

    history = [
        shared_private_objective(
            C_by_name,
            shared,
            private,
        )
    ]

    converged = False

    for iteration in range(
        1,
        max_iters + 1,
    ):
        shared_new = shared_update(
            C_by_name,
            private,
        )

        private_new = {
            name: private_update(
                C_by_name[name],
                shared_new,
            )
            for name in sorted(
                C_by_name
            )
        }

        value = (
            shared_private_objective(
                C_by_name,
                shared_new,
                private_new,
            )
        )

        previous = history[-1]

        monotonic_tolerance = (
            1.0e-8
            * max(
                abs(previous),
                1.0,
            )
        )

        if (
            value
            + monotonic_tolerance
            < previous
        ):
            raise RuntimeError(
                "Monotonic block-coordinate "
                "ascent audit failed: "
                f"iteration={iteration}, "
                f"new={value:.12e}, "
                f"old={previous:.12e}"
            )

        history.append(
            value
        )

        relative_change = (
            abs(
                value - previous
            )
            / max(
                abs(previous),
                1.0,
            )
        )

        shared = shared_new
        private = private_new

        if relative_change <= tol:
            converged = True
            break

    return {
        "shared": shared,
        "private": private,
        "objective":
            history[-1],
        "objective_history":
            history,
        "iterations":
            len(history) - 1,
        "converged":
            converged,
    }


def solve_shared_private_group(
    members: list[str],
    whitened: dict[str, dict[str, Any]],
    shared_side: str,
    max_iters: int,
    tol: float,
):
    C = {
        name: side_covariance(
            whitened[name]["Gt"],
            shared_side,
        )
        for name in members
    }

    pooled = sum(C.values())

    starts = (
        make_shared_private_initializations(
            pooled,
            pooled.device,
        )
    )

    solutions = []

    for start_index, shared_initial in enumerate(
        starts
    ):
        solution = (
            run_shared_private_start(
                C_by_name=C,
                shared_initial=(
                    shared_initial
                ),
                max_iters=max_iters,
                tol=tol,
            )
        )

        solution["start_index"] = (
            start_index
        )

        solutions.append(
            solution
        )

    best = max(
        solutions,
        key=lambda x: x[
            "objective"
        ],
    )

    shared = best["shared"]
    private = best["private"]

    bases = {}
    constrained_objective = 0.0
    independent_objective = 0.0

    for name in members:
        Gt = whitened[name]["Gt"]

        if shared_side == "right":
            V = torch.cat(
                [
                    shared,
                    private[name],
                ],
                dim=1,
            )

            V = orth(
                V,
                RANK,
                f"{name}: combined right "
                "shared/private basis",
            )

            U = orth(
                Gt @ V,
                RANK,
                f"{name}: U=orth(GV)",
            )

        elif shared_side == "left":
            U = torch.cat(
                [
                    shared,
                    private[name],
                ],
                dim=1,
            )

            U = orth(
                U,
                RANK,
                f"{name}: combined left "
                "shared/private basis",
            )

            V = orth(
                Gt.T @ U,
                RANK,
                f"{name}: V=orth(G^T U)",
            )

        else:
            raise ValueError(
                shared_side
            )

        matrix_value = float(
            torch.sum(
                (
                    U.T
                    @ Gt
                    @ V
                ) ** 2
            ).item()
        )

        constrained_objective += (
            matrix_value
        )

        independent_objective += (
            independent_fcca_score(
                Gt,
                RANK,
            )
        )

        bases[name] = {
            "U": U,
            "V": V,
        }

    theorem_tolerance = (
        1.0e-7
        * max(
            independent_objective,
            1.0,
        )
    )

    if (
        constrained_objective
        > independent_objective
        + theorem_tolerance
    ):
        raise RuntimeError(
            "AC-FCCA-SP theorem audit failed: "
            f"{constrained_objective:.12e} > "
            f"{independent_objective:.12e}"
        )

    start_summaries = [
        {
            "start_index":
                int(
                    solution[
                        "start_index"
                    ]
                ),
            "final_objective":
                float(
                    solution[
                        "objective"
                    ]
                ),
            "iterations":
                int(
                    solution[
                        "iterations"
                    ]
                ),
            "converged":
                bool(
                    solution[
                        "converged"
                    ]
                ),
            "objective_history":
                [
                    float(x)
                    for x
                    in solution[
                        "objective_history"
                    ]
                ],
        }
        for solution in solutions
    ]

    return bases, {
        "shared_private_objective":
            constrained_objective,
        "independent_fcca_objective":
            independent_objective,
        "ratio_to_independent":
            constrained_objective
            / independent_objective,
        "number_of_starts":
            len(solutions),
        "selected_start":
            int(
                best[
                    "start_index"
                ]
            ),
        "iterations":
            int(
                best[
                    "iterations"
                ]
            ),
        "converged":
            bool(
                best[
                    "converged"
                ]
            ),
        "objective_history":
            [
                float(x)
                for x
                in best[
                    "objective_history"
                ]
            ],
        "all_starts":
            start_summaries,
    }


def native_basis(
    U: torch.Tensor,
    V: torch.Tensor,
    stat: dict[str, Any],
    name: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    P0 = (
        stat["inv_d"][:, None]
        * U
    )

    Q0 = (
        stat["inv_a"][:, None]
        * V
    )

    P, _ = torch.linalg.qr(
        P0,
        mode="reduced",
    )

    Q, _ = torch.linalg.qr(
        Q0,
        mode="reduced",
    )

    P = P[:, :U.shape[1]]
    Q = Q[:, :V.shape[1]]

    rank = int(
        P.shape[1]
    )

    eye = torch.eye(
        rank,
        device=P.device,
        dtype=P.dtype,
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

    if p_error > 1.0e-8:
        raise RuntimeError(
            f"{name}: native P QR error "
            f"{p_error}"
        )

    if q_error > 1.0e-8:
        raise RuntimeError(
            f"{name}: native Q QR error "
            f"{q_error}"
        )

    return (
        P.float().cpu(),
        Q.float().cpu(),
    )


def build_ac_basis(
    *,
    variant: str,
    stats_payload: dict[str, Any],
    output: Path,
    max_iters: int,
    tol: float,
) -> dict[str, Any]:
    device = torch.device(
        "cuda:0"
    )

    modules = stats_payload[
        "modules"
    ]

    if len(modules) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_COUNT} task-stat "
            f"modules, got {len(modules)}"
        )

    groups = make_pair_groups(
        modules.keys()
    )

    payload = {
        "method":
            "asymmetric_one_sided_fcca",
        "paper":
            PAPER,
        "variant":
            variant,
        "base_model":
            BASE_MODEL,
        "rank":
            RANK,
        "shared_rank":
            (
                RANK
                if variant == "hard"
                else SHARED_RANK
            ),
        "private_rank":
            (
                0
                if variant == "hard"
                else PRIVATE_RANK
            ),
        "calibration_size":
            CALIBRATION_SIZE,
        "calibration_seed":
            CALIBRATION_SEED,
        "damping":
            DAMPING,
        "pairing":
            "fixed adjacent non-overlapping "
            "pairs (0,1),(2,3),...,(22,23)",
        "sharing_rule":
            "Q/K/V share Fisher-coordinate "
            "right/input side; O shares "
            "Fisher-coordinate left/output side",
        "target_count":
            TARGET_COUNT,
        "trainable_core_budget":
            BUDGET,
        "rank_allocation":
            {
                name: RANK
                for name in modules
            },
        "bases": {},
        "groups": {},
    }

    total_constrained = 0.0
    total_independent = 0.0

    ordered_groups = sorted(
        groups.items(),
        key=lambda kv: kv[0],
    )

    for group_index, (
        key,
        members,
    ) in enumerate(
        ordered_groups,
        start=1,
    ):
        (
            family,
            projection,
            pair_start,
        ) = key

        shared_side = (
            "left"
            if projection == "o"
            else "right"
        )

        whitened = {
            name: whiten(
                modules[name],
                device,
            )
            for name in members
        }

        if variant == "hard":
            (
                fisher_bases,
                diagnostics,
            ) = solve_hard_group(
                members,
                whitened,
                shared_side,
            )

            constrained = (
                diagnostics[
                    "shared_objective"
                ]
            )

        elif variant == "shared_private":
            (
                fisher_bases,
                diagnostics,
            ) = (
                solve_shared_private_group(
                    members,
                    whitened,
                    shared_side,
                    max_iters=max_iters,
                    tol=tol,
                )
            )

            constrained = (
                diagnostics[
                    "shared_private_objective"
                ]
            )

        else:
            raise ValueError(
                variant
            )

        independent = diagnostics[
            "independent_fcca_objective"
        ]

        total_constrained += (
            constrained
        )

        total_independent += (
            independent
        )

        group_name = (
            f"{family}_{projection}_"
            f"{pair_start:02d}_"
            f"{pair_start + 1:02d}"
        )

        payload[
            "groups"
        ][group_name] = {
            "family":
                family,
            "projection":
                projection,
            "layers":
                [
                    pair_start,
                    pair_start + 1,
                ],
            "shared_side":
                shared_side,
            "members":
                members,
            **diagnostics,
        }

        for name in members:
            P, Q = native_basis(
                fisher_bases[name]["U"],
                fisher_bases[name]["V"],
                whitened[name],
                name,
            )

            payload[
                "bases"
            ][name] = {
                "rank": RANK,
                "P": P,
                "Q": Q,
            }

        print(
            f"[AC {group_index:03d}/"
            f"{len(ordered_groups):03d}] "
            f"{group_name} "
            f"side={shared_side} "
            f"ratio="
            f"{constrained / independent:.8f}",
            flush=True,
        )

        del whitened
        del fisher_bases
        torch.cuda.empty_cache()

    global_ratio = (
        total_constrained
        / total_independent
    )

    if global_ratio > 1.0 + 1.0e-7:
        raise RuntimeError(
            "GLOBAL AC THEOREM AUDIT FAILED: "
            f"ratio={global_ratio}"
        )

    payload[
        "global_audit"
    ] = {
        "constrained_objective":
            total_constrained,
        "independent_fcca_objective":
            total_independent,
        "ratio_to_independent":
            global_ratio,
        "theorem_check":
            "PASS",
    }

    save_torch(
        output,
        payload,
    )

    return payload


def solve_exact_allocation(
    names: list[str],
    spectra: dict[
        str,
        torch.Tensor,
    ],
    rank_step: int,
    solver_time_limit: float,
):
    if rank_step <= 0:
        raise ValueError(
            "rank_step must be positive"
        )

    candidates: dict[
        str,
        list[int],
    ] = {}

    utilities: dict[
        str,
        dict[int, float],
    ] = {}

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
            RANK <= max_rank
            and RANK not in ranks
        ):
            ranks.append(
                RANK
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
                    cumulative[
                        rank
                    ].item()
                )
            for rank in ranks
        }

    for name in names:
        if RANK not in candidates[name]:
            raise RuntimeError(
                f"{name}: rank 128 unavailable"
            )

    if (
        len(names)
        * RANK ** 2
        != BUDGET
    ):
        raise RuntimeError(
            "Reference budget arithmetic failure"
        )

    variable_meta = [
        (
            name,
            rank,
        )
        for name in names
        for rank in candidates[name]
    ]

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
                [
                    float(
                        BUDGET
                    ),
                ],
                dtype=np.float64,
            ),
        ]
    )

    upper = lower.copy()

    print(
        "[AR allocation] "
        f"matrices={len(names)} "
        f"binary_variables="
        f"{n_variables:,} "
        f"budget={BUDGET:,} "
        f"rank_step={rank_step}",
        flush=True,
    )

    options = {
        "presolve":
            True,
        "mip_rel_gap":
            0.0,
    }

    if solver_time_limit > 0:
        options[
            "time_limit"
        ] = float(
            solver_time_limit
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
        constraints=(
            LinearConstraint(
                A,
                lower,
                upper,
            )
        ),
        options=options,
    )

    if (
        not result.success
        or result.x is None
    ):
        raise RuntimeError(
            "Exact allocation solver did not "
            "certify an optimum: "
            f"status={result.status}, "
            f"message={result.message}"
        )

    selected = {}
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
                and x[column] > 0.5
            )
        ]

        if len(
            selected_columns
        ) != 1:
            raise RuntimeError(
                f"{name}: expected one "
                "selected rank, got "
                f"{len(selected_columns)}"
            )

        column = (
            selected_columns[0]
        )

        _, rank = (
            variable_meta[column]
        )

        selected[name] = int(
            rank
        )

    used_budget = sum(
        rank * rank
        for rank in (
            selected.values()
        )
    )

    if used_budget != BUDGET:
        raise RuntimeError(
            "Budget mismatch after MIP: "
            f"{used_budget:,} != "
            f"{BUDGET:,}"
        )

    adaptive_utility = sum(
        utilities[name][
            selected[name]
        ]
        for name in names
    )

    uniform_utility = sum(
        utilities[name][RANK]
        for name in names
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
            "AR allocator theorem audit "
            "failed: adaptive optimum "
            "below feasible uniform rank-128"
        )

    mip_gap = getattr(
        result,
        "mip_gap",
        None,
    )

    print(
        "[AR allocation OPTIMAL] "
        f"budget={used_budget:,} "
        f"objective="
        f"{adaptive_utility:.12e} "
        f"uniform128="
        f"{uniform_utility:.12e} "
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


def build_ar_basis(
    *,
    stats_payload: dict[str, Any],
    output: Path,
    rank_step: int,
    solver_time_limit: float,
) -> dict[str, Any]:
    device = torch.device(
        "cuda:0"
    )

    modules = stats_payload[
        "modules"
    ]

    names = sorted(
        modules
    )

    if len(names) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_COUNT} matrices, "
            f"found {len(names)}"
        )

    spectra = {}
    diagnostics = {}

    print(
        "[AR spectra] computing complete "
        "Fisher-whitened singular spectra",
        flush=True,
    )

    for index, name in enumerate(
        names,
        start=1,
    ):
        stat = whiten(
            modules[name],
            device,
        )

        svals = (
            torch.linalg.svdvals(
                stat["Gt"]
            )
        )

        spectra[name] = (
            svals.detach()
            .double()
            .cpu()
        )

        diagnostics[
            name
        ] = {
            "positions":
                int(
                    stat["count"]
                ),
            "out_features":
                int(
                    stat["Gt"]
                    .shape[0]
                ),
            "in_features":
                int(
                    stat["Gt"]
                    .shape[1]
                ),
            "maximum_rank":
                int(
                    svals.numel()
                ),
            "sigma_1":
                float(
                    svals[0]
                    .item()
                ),
            "sigma_128":
                float(
                    svals[127]
                    .item()
                ),
        }

        if (
            index % 16 == 0
            or index == len(names)
        ):
            print(
                f"[AR spectra] "
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
        solver_time_limit=(
            solver_time_limit
        ),
    )

    bases = {}

    print(
        "[AR basis] constructing "
        "selected ordinary FCCA "
        "subspaces",
        flush=True,
    )

    for index, name in enumerate(
        names,
        start=1,
    ):
        rank = int(
            allocation[name]
        )

        stat = whiten(
            modules[name],
            device,
        )

        if rank == 0:
            bases[name] = {
                "rank": 0,
                "P": torch.empty(
                    int(
                        stat["Gt"]
                        .shape[0]
                    ),
                    0,
                    dtype=torch.float32,
                ),
                "Q": torch.empty(
                    int(
                        stat["Gt"]
                        .shape[1]
                    ),
                    0,
                    dtype=torch.float32,
                ),
            }

            diagnostics[
                name
            ][
                "selected_rank"
            ] = 0

            diagnostics[
                name
            ][
                "selected_utility"
            ] = 0.0

        else:
            U, _, Vh = (
                torch.linalg.svd(
                    stat["Gt"],
                    full_matrices=False,
                )
            )

            U_r = U[:, :rank]
            V_r = Vh[:rank, :].T

            P0 = (
                stat["inv_d"][
                    :,
                    None,
                ]
                * U_r
            )

            Q0 = (
                stat["inv_a"][
                    :,
                    None,
                ]
                * V_r
            )

            P, _ = torch.linalg.qr(
                P0,
                mode="reduced",
            )

            Q, _ = torch.linalg.qr(
                Q0,
                mode="reduced",
            )

            P = P[:, :rank]
            Q = Q[:, :rank]

            eye = torch.eye(
                rank,
                device=device,
                dtype=torch.float64,
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
                p_error > 1.0e-8
                or q_error > 1.0e-8
            ):
                raise RuntimeError(
                    f"{name}: AR QR audit "
                    f"failed P={p_error:.6e} "
                    f"Q={q_error:.6e}"
                )

            bases[name] = {
                "rank": rank,
                "P": P.float().detach().cpu(),
                "Q": Q.float().detach().cpu(),
            }

            diagnostics[
                name
            ][
                "selected_rank"
            ] = rank

            diagnostics[
                name
            ][
                "selected_utility"
            ] = (
                utility_tables[
                    name
                ][rank]
            )

            diagnostics[
                name
            ][
                "p_orthogonality_error"
            ] = p_error

            diagnostics[
                name
            ][
                "q_orthogonality_error"
            ] = q_error

        if (
            index % 16 == 0
            or index == len(names)
        ):
            print(
                f"[AR basis] "
                f"{index}/{len(names)}",
                flush=True,
            )

        torch.cuda.empty_cache()

    used_budget = sum(
        int(
            entry["rank"]
        ) ** 2
        for entry in (
            bases.values()
        )
    )

    if used_budget != BUDGET:
        raise RuntimeError(
            "Final AR basis budget "
            f"mismatch: {used_budget:,} "
            f"!= {BUDGET:,}"
        )

    ranks = list(
        allocation.values()
    )

    payload = {
        "method":
            "adaptive_rank_fcca",
        "paper":
            PAPER,
        "objective":
            "sum_l sum_{k<=r_l} "
            "sigma_{l,k}^2",
        "optimality_scope":
            "globally optimal rank "
            "allocation under Eq. (3) "
            "over allowed integer ranks",
        "base_model":
            BASE_MODEL,
        "calibration_size":
            CALIBRATION_SIZE,
        "calibration_seed":
            CALIBRATION_SEED,
        "damping":
            DAMPING,
        "scale":
            SCALE,
        "reference_uniform_rank":
            RANK,
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
                    np.median(
                        ranks
                    )
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
    }

    save_torch(
        output,
        payload,
    )

    return payload


def build_standard_basis(
    *,
    stats_payload: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    modules = stats_payload["modules"]
    names = sorted(modules)

    if len(names) != TARGET_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_COUNT} task-stat matrices, found {len(names)}"
        )

    bases = {}
    diagnostics = {}

    print(
        "[standard FCCA] constructing rank-128 bases from shared stats",
        flush=True,
    )

    for index, name in enumerate(names, start=1):
        stat = whiten(modules[name], device)
        U, singular_values, Vh = torch.linalg.svd(
            stat["Gt"],
            full_matrices=False,
        )

        if RANK > min(U.shape[1], Vh.shape[0]):
            raise RuntimeError(f"{name}: rank {RANK} unavailable")

        U_r = U[:, :RANK]
        V_r = Vh[:RANK, :].T

        P0 = stat["inv_d"][:, None] * U_r
        Q0 = stat["inv_a"][:, None] * V_r

        P, _ = torch.linalg.qr(P0, mode="reduced")
        Q, _ = torch.linalg.qr(Q0, mode="reduced")
        P = P[:, :RANK]
        Q = Q[:, :RANK]

        eye = torch.eye(RANK, device=device, dtype=torch.float64)
        p_error = float(
            torch.linalg.matrix_norm(P.T @ P - eye, ord="fro").item()
        )
        q_error = float(
            torch.linalg.matrix_norm(Q.T @ Q - eye, ord="fro").item()
        )
        if p_error > 1.0e-8 or q_error > 1.0e-8:
            raise RuntimeError(
                f"{name}: standard FCCA QR audit failed "
                f"P={p_error:.6e} Q={q_error:.6e}"
            )

        bases[name] = {
            "rank": RANK,
            "P": P.float().detach().cpu(),
            "Q": Q.float().detach().cpu(),
        }
        diagnostics[name] = {
            "positions": int(stat["count"]),
            "sigma_1": float(singular_values[0].item()),
            "sigma_128": float(singular_values[RANK - 1].item()),
            "p_orthogonality_error": p_error,
            "q_orthogonality_error": q_error,
        }

        if index % 16 == 0 or index == len(names):
            print(
                f"[standard FCCA] {index}/{len(names)}",
                flush=True,
            )
        torch.cuda.empty_cache()

    payload = {
        "method": "standard_fcca",
        "paper": PAPER,
        "base_model": BASE_MODEL,
        "calibration_size": CALIBRATION_SIZE,
        "calibration_seed": CALIBRATION_SEED,
        "damping": DAMPING,
        "scale": SCALE,
        "rank": RANK,
        "target_count": len(names),
        "trainable_core_budget": BUDGET,
        "rank_allocation": {name: RANK for name in names},
        "diagnostics": diagnostics,
        "bases": bases,
    }
    save_torch(output, payload)
    return payload


def get_basis(
    *,
    method: str,
    stats_payload: dict[str, Any],
    stats_path: Path,
    stats_sha256: str,
    rebuild: bool,
    rank_step: int,
    solver_time_limit: float,
    ac_max_iters: int,
    ac_tol: float,
) -> tuple[dict[str, Any], Path]:
    path = basis_path(method)
    ready_path = path.with_suffix(path.suffix + ".ready")

    if is_rank_zero():
        if rebuild:
            if path.exists():
                path.unlink()
            if ready_path.exists():
                ready_path.unlink()

        if not path.exists():
            if method == "standard":
                payload = build_standard_basis(
                    stats_payload=stats_payload,
                    output=path,
                )
            elif method == "ac-h":
                payload = build_ac_basis(
                    variant="hard",
                    stats_payload=stats_payload,
                    output=path,
                    max_iters=ac_max_iters,
                    tol=ac_tol,
                )
            elif method == "ac-sp":
                payload = build_ac_basis(
                    variant="shared_private",
                    stats_payload=stats_payload,
                    output=path,
                    max_iters=ac_max_iters,
                    tol=ac_tol,
                )
            elif method == "ar":
                payload = build_ar_basis(
                    stats_payload=stats_payload,
                    output=path,
                    rank_step=rank_step,
                    solver_time_limit=solver_time_limit,
                )
            else:
                raise ValueError(method)

            payload["task_stats_path"] = str(stats_path)
            payload["task_stats_sha256"] = stats_sha256
            payload["task_stats_fingerprint"] = stats_payload["fold_fingerprint"]
            payload["task_stats_source"] = stats_payload["source"]
            save_torch(path, payload)

        # Publish only after the basis file contains complete provenance.
        payload_check = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        required_provenance = {
            "task_stats_path": str(stats_path),
            "task_stats_sha256": stats_sha256,
            "task_stats_fingerprint": stats_payload["fold_fingerprint"],
            "task_stats_source": stats_payload["source"],
        }
        for key, expected in required_provenance.items():
            if payload_check.get(key) != expected:
                raise RuntimeError(
                    f"{method} basis not ready for publication: "
                    f"{key} expected={expected!r}, "
                    f"got={payload_check.get(key)!r}"
                )

        ready_path.write_text(
            stats_sha256 + "\n",
            encoding="utf-8",
        )
    else:
        wait_for_file(ready_path)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    provenance = {
        "task_stats_path": str(stats_path),
        "task_stats_sha256": stats_sha256,
        "task_stats_fingerprint": stats_payload["fold_fingerprint"],
        "task_stats_source": stats_payload["source"],
    }
    for key, expected in provenance.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"{method} basis provenance mismatch for {key}: "
                f"expected={expected!r}, got={payload.get(key)!r}. "
                "Delete/rebuild this basis before training."
            )

    if int(payload.get("trainable_core_budget", -1)) != BUDGET:
        raise RuntimeError("Existing basis budget mismatch")

    print(
        f"[basis provenance] {method}: PASS "
        f"task_stats_sha256={stats_sha256}",
        flush=True,
    )
    return payload, path


class VariableFCCALinear(
    nn.Module
):
    def __init__(
        self,
        base: nn.Linear,
        P: torch.Tensor,
        Q: torch.Tensor,
        rank: int,
    ) -> None:
        super().__init__()

        self.base = base
        self.rank = int(
            rank
        )

        for parameter in (
            self.base.parameters()
        ):
            parameter.requires_grad = (
                False
            )
            parameter.grad = None

        if self.rank <= 0:
            raise ValueError(
                "VariableFCCALinear "
                "requires rank > 0"
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
            "fcca_P",
            P.to(
                device=device,
                dtype=torch.float32,
            ).contiguous(),
        )

        self.register_buffer(
            "fcca_Q",
            Q.to(
                device=device,
                dtype=torch.float32,
            ).contiguous(),
        )

        self.fcca_core = (
            nn.Parameter(
                torch.zeros(
                    self.rank,
                    self.rank,
                    device=device,
                    dtype=torch.float32,
                )
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        base_out = self.base(x)

        projected = (
            x.float()
            @ self.fcca_Q
        )

        mixed = (
            projected
            @ self.fcca_core.T
        )

        update = (
            mixed
            @ self.fcca_P.T
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


def attach_basis(
    model: nn.Module,
    payload: dict[str, Any],
) -> tuple[
    list[str],
    list[str],
]:
    targets = (
        collect_attention_targets(
            model
        )
    )

    bases = payload[
        "bases"
    ]

    if set(
        targets
    ) != set(
        bases
    ):
        raise RuntimeError(
            "Basis/target mismatch"
        )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False
        parameter.grad = None

    installed = []
    skipped_rank_zero = []

    for name in sorted(
        targets
    ):
        entry = bases[name]
        rank = int(
            entry["rank"]
        )

        if rank == 0:
            skipped_rank_zero.append(
                name
            )
            continue

        P = entry["P"]
        Q = entry["Q"]

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
            or q_err > 1.0e-3
        ):
            raise RuntimeError(
                f"{name}: stored basis "
                "orthogonality failure "
                f"P={p_err:.6e} "
                f"Q={q_err:.6e}"
            )

        parent, leaf = (
            get_parent(
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
            VariableFCCALinear(
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
        for parameter in (
            model.parameters()
        )
        if parameter.requires_grad
    )

    allocated = sum(
        int(
            entry["rank"]
        ) ** 2
        for entry in (
            bases.values()
        )
    )

    if (
        allocated != BUDGET
        or trainable
        != BUDGET
    ):
        raise RuntimeError(
            "FCCA budget audit failed: "
            f"allocated={allocated:,}, "
            f"trainable={trainable:,}, "
            f"expected={BUDGET:,}"
        )

    print(
        "[architecture] "
        f"adapted={len(installed)} "
        f"rank0="
        f"{len(skipped_rank_zero)} "
        f"trainable={trainable:,}",
        flush=True,
    )

    return (
        installed,
        skipped_rank_zero,
    )


def load_union_and_test():
    train_rows = load_manifest(
        TRAIN_MANIFEST,
        required_columns=("audio_path", "transcript"),
    )
    test_rows = load_manifest(
        TEST_MANIFEST,
        required_columns=("audio_path", "transcript", "task"),
    )

    if len(train_rows) != EXPECTED_TRAINING_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_EXAMPLES} training examples, "
            f"found {len(train_rows)}"
        )
    if len(test_rows) != EXPECTED_TEST_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_RECORDINGS} test recordings, "
            f"found {len(test_rows)}"
        )

    validate_no_duplicate_audio(
        train_rows,
        dataset_name="FCCA train+validation union",
    )
    validate_no_duplicate_audio(
        test_rows,
        dataset_name="Llanito fixed test",
    )
    # Audio-overlap validation intentionally disabled.
    validate_manifest_files_exist(train_rows)
    validate_manifest_files_exist(test_rows)

    return train_rows, test_rows


def save_run_metadata(
    *,
    method: str,
    basis_file: Path,
    stats_path: Path,
    stats_sha256: str,
    final_checkpoint: Path,
) -> None:
    payload = {
        "method": method,
        "paper": PAPER,
        "base_model": BASE_MODEL,
        "train_manifest": str(TRAIN_MANIFEST),
        "test_manifest": str(TEST_MANIFEST),
        "training_examples": EXPECTED_TRAINING_EXAMPLES,
        "test_recordings": EXPECTED_TEST_RECORDINGS,
        "learning_rate": LEARNING_RATE,
        "epochs": MAX_EPOCHS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": SEED,
        "rank_reference": RANK,
        "calibration_examples": CALIBRATION_SIZE,
        "calibration_seed": CALIBRATION_SEED,
        "task_stats_path": str(stats_path),
        "task_stats_sha256": stats_sha256,
        "basis_path": str(basis_file),
        "final_checkpoint": str(final_checkpoint),
        "evaluation_path": str(evaluation_root(method) / "evaluation.json"),
    }
    save_json(method_output(method) / "run_metadata.json", payload)


def run_method(
    *,
    method: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    processor,
    train_dataset,
    collator,
    stats_payload: dict[str, Any],
    stats_path: Path,
    stats_sha256: str,
    microbatch: int,
    rebuild_bases: bool,
    rank_step: int,
    solver_time_limit: float,
    ac_max_iters: int,
    ac_tol: float,
) -> None:
    done_path = evaluation_root(method) / "evaluation.json"
    if done_path.exists():
        print(f"[skip] {method}: fixed-test evaluation already exists", flush=True)
        return

    ws = world_size()
    denominator = microbatch * ws
    if EFFECTIVE_BATCH_SIZE % denominator != 0:
        raise RuntimeError(
            "EBS=8 must be divisible by "
            f"microbatch*world_size={denominator}"
        )
    grad_acc = EFFECTIVE_BATCH_SIZE // denominator

    basis_payload, basis_file = get_basis(
        method=method,
        stats_payload=stats_payload,
        stats_path=stats_path,
        stats_sha256=stats_sha256,
        rebuild=rebuild_bases,
        rank_step=rank_step,
        solver_time_limit=solver_time_limit,
        ac_max_iters=ac_max_iters,
        ac_tol=ac_tol,
    )
    distributed_barrier()

    set_seed(SEED)
    validate_cuda_environment(
        use_fp16=True,
        use_bf16=False,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )
    model.config.use_cache = False
    configure_whisper_generation(
        model,
        language=LANGUAGE,
        task=TASK,
        use_cache=False,
    )

    installed, rank_zero = attach_basis(
        model,
        basis_payload,
    )

    checkpoints = checkpoint_root(method)
    checkpoints.mkdir(parents=True, exist_ok=True)

    training_args = build_training_arguments(
        checkpoint_directory=checkpoints,
        seed=SEED,
        epochs=float(MAX_EPOCHS),
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=0,
        warmup_ratio=WARMUP_RATIO,
        train_batch_size=microbatch,
        gradient_accumulation_steps=grad_acc,
        use_fp16=True,
        use_bf16=False,
        use_gradient_checkpointing=False,
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        dataloader_workers=DATALOADER_WORKERS,
        has_validation=False,
        evaluation_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        predict_with_generate=False,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,
        lr_scheduler_type="linear",
        max_grad_norm=MAX_GRAD_NORM,
        optimizer="adamw_torch",
        save_only_model=False,
    )

    trainer = make_trainer(
        model=model,
        training_args=training_args,
        processor=processor,
        train_dataset=train_dataset,
        collator=collator,
        validation_dataset=None,
        compute_metrics=None,
        callbacks=(),
    )

    checkpoint = latest_checkpoint(checkpoints)

    print()
    print("=" * 80)
    print(f"FINAL UNION TRAINING | {method}")
    print("=" * 80)
    print(f"train_examples={len(train_rows)}")
    print(f"test_examples={len(test_rows)}")
    print(f"calibration_examples={CALIBRATION_SIZE}")
    print(f"task_stats_sha256={stats_sha256}")
    print(f"learning_rate={LEARNING_RATE}")
    print(f"epochs={MAX_EPOCHS}")
    print(f"world_size={ws}")
    print(f"microbatch={microbatch}")
    print(f"GA={grad_acc}")
    print(f"EBS={EFFECTIVE_BATCH_SIZE}")
    print(f"adapted={len(installed)} rank0={len(rank_zero)}")
    print(f"resume={checkpoint}")
    print("=" * 80)

    trainer.train(
        resume_from_checkpoint=(
            str(checkpoint)
            if checkpoint is not None
            else None
        )
    )

    distributed_barrier()

    final_checkpoint = latest_checkpoint(checkpoints)
    if final_checkpoint is None:
        raise RuntimeError(f"No final checkpoint found for {method}")

    # Five epoch checkpoints must remain available.
    checkpoint_dirs = sorted(
        [p for p in checkpoints.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-", 1)[1]),
    )
    if len(checkpoint_dirs) != MAX_EPOCHS:
        raise RuntimeError(
            f"{method}: expected {MAX_EPOCHS} retained epoch checkpoints, "
            f"found {len(checkpoint_dirs)}: {[p.name for p in checkpoint_dirs]}"
        )

    if not is_rank_zero():
        distributed_barrier()
        del trainer
        del model
        torch.cuda.empty_cache()
        return

    print()
    print("=" * 80)
    print(f"FIXED TEST | {method} | FINAL EPOCH {MAX_EPOCHS}")
    print("=" * 80)
    print(f"final_checkpoint={final_checkpoint}")

    # trainer.model is the exact post-epoch-5 state that the final checkpoint saved.
    eval_model = trainer.model
    eval_model.eval()
    configure_whisper_generation(
        eval_model,
        language=LANGUAGE,
        task=TASK,
        use_cache=True,
    )

    out = evaluation_root(method)
    out.mkdir(parents=True, exist_ok=True)

    evaluate_loaded_model(
        model_name=f"week9_{method_tag(method)}_union_final",
        processor=processor,
        model=eval_model,
        rows=test_rows,
        output_root=out,
        normalisation_file=NORMALISATION_FILE,
        language=LANGUAGE,
        task=TASK,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,
        device=torch.device("cuda:0"),
        audio_column="audio_path",
        transcript_column="transcript",
        normalisation_mapping_column=NORMALISATION_MAPPING_COLUMN,
        normalisation_section_markers=NORMALISATION_SECTION_MARKERS,
        generation_kwargs=None,
        dataset_name=DATASET_NAME,
    )

    save_run_metadata(
        method=method,
        basis_file=basis_file,
        stats_path=stats_path,
        stats_sha256=stats_sha256,
        final_checkpoint=final_checkpoint,
    )

    print(f"[complete] {method}: {out / 'evaluation.json'}", flush=True)

    distributed_barrier()
    del trainer
    del model
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--microbatch",
        type=int,
        default=DEFAULT_MICROBATCH,
    )
    parser.add_argument(
        "--rank-step",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--solver-time-limit",
        type=float,
        default=7200.0,
    )
    parser.add_argument(
        "--ac-max-iters",
        type=int,
        default=AC_MAX_ITERS,
    )
    parser.add_argument(
        "--ac-tol",
        type=float,
        default=AC_TOL,
    )
    parser.add_argument(
        "--rebuild-bases",
        action="store_true",
    )
    parser.add_argument(
        "--rebuild-stats",
        action="store_true",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete only this new final-union experiment's cache/results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if world_size() != 2:
        raise RuntimeError(
            f"This experiment is fixed to 2 GPUs; WORLD_SIZE={world_size()}"
        )
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"Expected exactly 2 visible GPUs, found {torch.cuda.device_count()}"
        )

    if args.restart and is_rank_zero():
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    distributed_barrier()

    train_rows, test_rows = load_union_and_test()

    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL,
        language=LANGUAGE,
        task=TASK,
        local_files_only=True,
    )
    processor.tokenizer.set_prefix_tokens(
        language=LANGUAGE,
        task=TASK,
        predict_timestamps=False,
    )

    probe_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )
    decoder_start_token_id = probe_model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        raise RuntimeError("Whisper has no decoder_start_token_id")
    del probe_model

    train_dataset = WhisperManifestDataset(
        train_rows,
        processor,
    )
    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
    )

    # Exactly one deterministic 256-example calibration from the 904-example union.
    stats_payload, stats_path, stats_sha256 = get_task_stats(
        train_rows=train_rows,
        train_dataset=train_dataset,
        collator=collator,
        rebuild=args.rebuild_stats,
    )
    distributed_barrier()

    if is_rank_zero():
        print()
        print("=" * 80)
        print("LLANITO FINAL FCCA / AC-FCCA / AR-FCCA")
        print("=" * 80)
        print(f"methods={args.methods}")
        print(f"train_union={TRAIN_MANIFEST}")
        print(f"train_examples={len(train_rows)}")
        print(f"fixed_test={TEST_MANIFEST}")
        print(f"test_examples={len(test_rows)}")
        print(f"calibration={CALIBRATION_SIZE} sampled training examples")
        print(f"shared_task_stats={stats_path}")
        print(f"shared_task_stats_sha256={stats_sha256}")
        print(f"learning_rate={LEARNING_RATE}")
        print(f"epochs={MAX_EPOCHS}")
        print(f"checkpoints_retained_per_method={SAVE_TOTAL_LIMIT}")
        print("=" * 80)

    for method in args.methods:
        run_method(
            method=method,
            train_rows=train_rows,
            test_rows=test_rows,
            processor=processor,
            train_dataset=train_dataset,
            collator=collator,
            stats_payload=stats_payload,
            stats_path=stats_path,
            stats_sha256=stats_sha256,
            microbatch=args.microbatch,
            rebuild_bases=args.rebuild_bases,
            rank_step=args.rank_step,
            solver_time_limit=args.solver_time_limit,
            ac_max_iters=args.ac_max_iters,
            ac_tol=args.ac_tol,
        )
        distributed_barrier()

    if is_rank_zero():
        print()
        print("=" * 80)
        print("ALL REQUESTED FINAL-UNION FCCA RUNS COMPLETE")
        print(f"results={OUTPUT_ROOT}")
        print(f"shared_stats={stats_path}")
        print(f"shared_stats_sha256={stats_sha256}")
        print("=" * 80)


if __name__ == "__main__":
    main()
