from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_from_disk
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.signal import resample_poly
from scipy.sparse import coo_matrix
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
from scripts_fleurs.fleurs_paths import dataset_root, load_fleurs_split


# ---------------------------------------------------------------------
# Paths / experiment definition
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path("/data/milsrg1/corpora/fleurs-selected-languages")
RESULT_ROOT = PROJECT_ROOT / "outputs" / "new_results_qwen"

BASE_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"

LANGUAGES = [
    "ast_es",
    "cmn_hans_cn",
    "es_419",
    "ky_kg",
    "ckb_iq",
]

# Qwen3-ASR does not natively expose Asturian / Kyrgyz / Sorani labels.
# Preserve the proxy-language convention used for the Qwen experiments.
QWEN_LANGUAGE_NAME = {
    "ast_es": "Spanish",
    "cmn_hans_cn": "Chinese",
    "es_419": "Spanish",
    "ky_kg": "Turkish",
    "ckb_iq": "Persian",
}

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)

CALIBRATION_SIZE = 256
CALIBRATION_SEED = 2026

DAMPING = 1.0e-6

REFERENCE_RANK = 128

SHARED_RANK = 64
PRIVATE_RANK = 64

# Qwen targets:
#   audio encoder: 24 layers x Q/K/V/O = 96
#   language model: 28 layers x Q/K/V/O = 112
#   total = 208
EXPECTED_AUDIO_TARGETS = 96
EXPECTED_TEXT_TARGETS = 112
EXPECTED_TARGETS = 208

# Equal-core budget to uniform rank-128 FCCA:
#   208 * 128^2 = 3,407,872
REFERENCE_CORE_BUDGET = EXPECTED_TARGETS * REFERENCE_RANK**2

# Shared/private block-coordinate ascent.
SP_MAX_ITERS = 20
SP_TOL = 0.99999


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}"
    )

    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    os.replace(tmp, path)


def choose_dtype(device: torch.device):
    if device.type != "cuda":
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def clear_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Audio / dataset
# ---------------------------------------------------------------------

def normalise_audio(x, sr):
    x = np.asarray(x)

    if x.ndim == 2:
        if x.shape[0] <= 8:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=1)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)

        scale = float(
            max(abs(info.min), abs(info.max))
        )

        x = x.astype(np.float32) / scale

    else:
        x = x.astype(np.float32, copy=False)

    if int(sr) != SAMPLE_RATE:
        g = math.gcd(int(sr), SAMPLE_RATE)

        x = resample_poly(
            x,
            SAMPLE_RATE // g,
            int(sr) // g,
        ).astype(np.float32, copy=False)

    if len(x) > MAX_AUDIO_SAMPLES:
        x = x[:MAX_AUDIO_SAMPLES]

    return np.ascontiguousarray(x)


def decode_audio(value):
    if isinstance(value, dict):
        if value.get("array") is not None:
            return normalise_audio(
                value["array"],
                int(value.get("sampling_rate", SAMPLE_RATE)),
            )

        if value.get("bytes") is not None:
            import io

            x, sr = sf.read(
                io.BytesIO(value["bytes"]),
                always_2d=False,
            )

            return normalise_audio(x, sr)

        if value.get("path"):
            path = Path(value["path"])

            if path.is_file():
                x, sr = sf.read(
                    str(path),
                    always_2d=False,
                )

                return normalise_audio(x, sr)

    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()

        x = samples.data

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        return normalise_audio(
            x,
            int(samples.sample_rate),
        )

    if isinstance(value, (str, Path)):
        x, sr = sf.read(
            str(value),
            always_2d=False,
        )

        return normalise_audio(x, sr)

    raise TypeError(
        f"Unsupported audio object: {type(value)}"
    )


def transcription_of(row):
    for key in (
        "transcription",
        "sentence",
        "text",
    ):
        if key in row and row[key] is not None:
            text = str(row[key]).strip()

            if text:
                return text

    raise RuntimeError(
        f"No transcript field. Keys={list(row.keys())}"
    )


def load_train(language):
    path = dataset_root(language)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {path}"
        )

    ds = load_fleurs_split(language, "train")

    if "audio" not in ds.column_names:
        raise RuntimeError(
            f"{language}: no audio column; "
            f"columns={ds.column_names}"
        )

    try:
        ds = ds.cast_column(
            "audio",
            Audio(
                sampling_rate=SAMPLE_RATE,
                decode=False,
            ),
        )
    except Exception:
        pass

    return ds


def calibration_indices(
    dataset_length: int,
    size: int,
    seed: int,
):
    if dataset_length < size:
        raise RuntimeError(
            f"Need {size} calibration recordings "
            f"but dataset only contains {dataset_length}."
        )

    rng = np.random.default_rng(seed)

    return [
        int(x)
        for x in rng.choice(
            dataset_length,
            size=size,
            replace=False,
        )
    ]


# ---------------------------------------------------------------------
# Qwen target discovery
# ---------------------------------------------------------------------

@dataclass
class Target:
    name: str
    module: torch.nn.Linear
    stack: str
    layer: int
    projection: str


def infer_layer(name: str):
    parts = name.split(".")

    for i, value in enumerate(parts):
        if value == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass

    raise RuntimeError(
        f"Cannot infer transformer layer from {name}"
    )


def infer_stack(name: str):
    lower = name.lower()

    if "audio_tower" in lower:
        return "audio"

    if "language_model" in lower:
        return "text"

    return None


def discover_targets(model):
    projection_names = {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o",
        "out_proj": "o",
    }

    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        if ".self_attn." not in name:
            continue

        leaf = name.rsplit(".", 1)[-1]

        if leaf not in projection_names:
            continue

        stack = infer_stack(name)

        if stack is None:
            continue

        targets.append(
            Target(
                name=name,
                module=module,
                stack=stack,
                layer=infer_layer(name),
                projection=projection_names[leaf],
            )
        )

    targets.sort(
        key=lambda x: (
            x.stack,
            x.layer,
            x.projection,
            x.name,
        )
    )

    n_audio = sum(
        t.stack == "audio"
        for t in targets
    )

    n_text = sum(
        t.stack == "text"
        for t in targets
    )

    if n_audio != EXPECTED_AUDIO_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_AUDIO_TARGETS} audio targets, "
            f"found {n_audio}."
        )

    if n_text != EXPECTED_TEXT_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEXT_TARGETS} text targets, "
            f"found {n_text}."
        )

    if len(targets) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGETS} total targets, "
            f"found {len(targets)}."
        )

    return targets


# ---------------------------------------------------------------------
# Task-stat accumulation
# ---------------------------------------------------------------------

class TaskStatAccumulator:
    def __init__(self, target: Target):
        self.target = target

        out_dim = int(target.module.out_features)
        in_dim = int(target.module.in_features)

        # float32 is sufficient and halves the very large Qwen CPU footprint.
        self.cross_sum = torch.zeros(
            out_dim,
            in_dim,
            dtype=torch.float32,
            device="cpu",
        )

        self.activation_second_sum = torch.zeros(
            in_dim,
            dtype=torch.float32,
            device="cpu",
        )

        self.gradient_second_sum = torch.zeros(
            out_dim,
            dtype=torch.float32,
            device="cpu",
        )

        self.positions = 0

        self._x = None
        self._fwd = None
        self._bwd = None

    def install(self):
        def forward_hook(module, inputs, output):
            if not inputs:
                self._x = None
                return

            x = inputs[0]

            if not torch.is_tensor(x):
                self._x = None
                return

            self._x = x.detach()

        def backward_hook(module, grad_input, grad_output):
            if self._x is None:
                return

            if not grad_output:
                return

            delta = grad_output[0]

            if not torch.is_tensor(delta):
                return

            x = self._x
            self._x = None

            x = x.reshape(-1, x.shape[-1])
            delta = delta.reshape(-1, delta.shape[-1])

            n = min(x.shape[0], delta.shape[0])

            if n <= 0:
                return

            x = x[:n]
            delta = delta[:n]

            finite = (
                torch.isfinite(x).all(dim=-1)
                & torch.isfinite(delta).all(dim=-1)
            )

            active = (
                delta.abs().amax(dim=-1) > 0
            )

            keep = finite & active

            if not bool(keep.any()):
                return

            x = x[keep].float()
            delta = delta[keep].float()

            cross = delta.transpose(0, 1) @ x
            a2 = x.square().sum(dim=0)
            d2 = delta.square().sum(dim=0)

            self.cross_sum.add_(
                cross.detach().cpu()
            )

            self.activation_second_sum.add_(
                a2.detach().cpu()
            )

            self.gradient_second_sum.add_(
                d2.detach().cpu()
            )

            self.positions += int(x.shape[0])

        self._fwd = (
            self.target.module.register_forward_hook(
                forward_hook
            )
        )

        self._bwd = (
            self.target.module.register_full_backward_hook(
                backward_hook
            )
        )

    def remove(self):
        if self._fwd is not None:
            self._fwd.remove()

        if self._bwd is not None:
            self._bwd.remove()

    def payload(self):
        return {
            "cross_sum": self.cross_sum,
            "activation_second_sum": self.activation_second_sum,
            "gradient_second_sum": self.gradient_second_sum,
            "positions": int(self.positions),
            "weight_shape": [
                int(self.target.module.out_features),
                int(self.target.module.in_features),
            ],
            "stack": self.target.stack,
            "layer": int(self.target.layer),
            "projection": self.target.projection,
        }


def make_training_conversation(
    audio,
    transcript,
    qwen_language,
):
    return [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": audio,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"language {qwen_language}"
                            f"<asr_text>{transcript}"
                        ),
                    }
                ],
            },
        ]
    ]


def processor_batch(
    processor,
    audio,
    transcript,
    qwen_language,
):
    conversation = make_training_conversation(
        audio,
        transcript,
        qwen_language,
    )

    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "output_labels": True,
        },
    )

    if "labels" not in batch:
        raise RuntimeError(
            "Processor returned no labels. "
            f"Keys={list(batch.keys())}"
        )

    return batch


def move_batch(batch, device, dtype):
    result = {}

    for key, value in batch.items():
        if not torch.is_tensor(value):
            result[key] = value
            continue

        value = value.to(device)

        if torch.is_floating_point(value):
            value = value.to(dtype=dtype)

        result[key] = value

    return result


def force_input_gradient(batch):
    preferred = (
        "input_features",
        "audio_values",
        "input_values",
    )

    for key in preferred:
        if key in batch:
            value = batch[key]

            if (
                torch.is_tensor(value)
                and torch.is_floating_point(value)
            ):
                value.requires_grad_(True)
                return

    for value in batch.values():
        if (
            torch.is_tensor(value)
            and torch.is_floating_point(value)
        ):
            value.requires_grad_(True)
            return

    raise RuntimeError(
        "Could not identify floating-point audio input."
    )


def collect_one_language(
    language,
    model,
    processor,
    targets,
    device,
    dtype,
    calibration_size,
    seed,
    overwrite,
):
    output_dir = (
        RESULT_ROOT
        / language
        / "_task_stats"
    )

    output_pt = (
        output_dir
        / f"task_stats_{calibration_size}_all_attention.pt"
    )

    output_json = (
        output_dir
        / f"task_stats_{calibration_size}_all_attention.json"
    )

    if output_pt.exists() and not overwrite:
        print(
            f"\n[{language}] task stats already exist; "
            f"SKIP collection: {output_pt}",
            flush=True,
        )
        return output_pt

    print()
    print("=" * 79)
    print(f"COLLECTING LANGUAGE: {language}")
    print("=" * 79)

    dataset = load_train(language)

    indices = calibration_indices(
        len(dataset),
        calibration_size,
        seed,
    )

    accumulators = [
        TaskStatAccumulator(target)
        for target in targets
    ]

    for acc in accumulators:
        acc.install()

    losses = []
    valid_tokens = 0

    qwen_language = QWEN_LANGUAGE_NAME[language]

    try:
        for n, index in enumerate(indices, start=1):
            row = dataset[index]

            audio = decode_audio(
                row["audio"]
            )

            transcript = transcription_of(row)

            batch = processor_batch(
                processor,
                audio,
                transcript,
                qwen_language,
            )

            batch = move_batch(
                batch,
                device,
                dtype,
            )

            force_input_gradient(batch)

            labels = batch["labels"]

            n_valid = int(
                (labels != -100).sum().item()
            )

            if n_valid <= 0:
                raise RuntimeError(
                    f"{language} index={index}: zero valid labels"
                )

            model.zero_grad(set_to_none=True)

            outputs = model(**batch)

            if outputs.loss is None:
                raise RuntimeError(
                    "Qwen model returned no supervised loss."
                )

            loss = outputs.loss

            (
                loss
                * float(n_valid)
            ).backward()

            losses.append(
                float(
                    loss.detach().float().cpu()
                )
            )

            valid_tokens += n_valid

            print(
                f"[{language}] "
                f"{n:03d}/{calibration_size:03d} "
                f"idx={index:05d} "
                f"loss={losses[-1]:.6f} "
                f"tokens={n_valid}",
                flush=True,
            )

            del outputs
            del loss
            del batch

    finally:
        for acc in accumulators:
            acc.remove()

    zero = [
        acc.target.name
        for acc in accumulators
        if acc.positions <= 0
    ]

    if zero:
        raise RuntimeError(
            f"{language}: zero captured positions:\n"
            + "\n".join(zero)
        )

    modules = {
        acc.target.name: acc.payload()
        for acc in accumulators
    }

    payload = {
        "schema": "qwen3_asr_task_stats_v2_per_language",
        "base_model": BASE_MODEL,
        "language": language,
        "qwen_language_prompt": qwen_language,
        "calibration_size": int(calibration_size),
        "calibration_seed": int(seed),
        "calibration_indices": indices,
        "damping": float(DAMPING),
        "sample_rate": int(SAMPLE_RATE),
        "target_count": len(modules),
        "mean_loss": float(np.mean(losses)),
        "total_valid_tokens": int(valid_tokens),
        "modules": modules,
    }

    atomic_torch_save(
        payload,
        output_pt,
    )

    atomic_json_save(
        {
            key: value
            for key, value in payload.items()
            if key != "modules"
        },
        output_json,
    )

    print(
        f"[{language}] SAVED task stats: {output_pt}",
        flush=True,
    )

    del accumulators
    del modules
    del payload
    del dataset

    gc.collect()

    return output_pt


# ---------------------------------------------------------------------
# FCCA mathematics
# ---------------------------------------------------------------------

def whitened_matrix(
    stat,
    device,
):
    positions = int(stat["positions"])

    if positions <= 0:
        raise RuntimeError("positions <= 0")

    G = (
        stat["cross_sum"]
        .to(
            device=device,
            dtype=torch.float64,
        )
        / float(positions)
    )

    A = (
        stat["activation_second_sum"]
        .to(
            device=device,
            dtype=torch.float64,
        )
        / float(positions)
    )

    D = (
        stat["gradient_second_sum"]
        .to(
            device=device,
            dtype=torch.float64,
        )
        / float(positions)
    )

    a_inv = torch.rsqrt(
        A + DAMPING
    )

    d_inv = torch.rsqrt(
        D + DAMPING
    )

    G_tilde = (
        d_inv[:, None]
        * G
        * a_inv[None, :]
    )

    return G_tilde, a_inv, d_inv


def orth(x):
    if x.shape[1] == 0:
        return x

    q, _ = torch.linalg.qr(
        x,
        mode="reduced",
    )

    return q


def top_eigenspace(
    C,
    rank,
):
    if rank == 0:
        return C.new_zeros(
            C.shape[0],
            0,
        )

    values, vectors = torch.linalg.eigh(C)

    return vectors[:, -rank:]


def residual_covariance(
    C,
    basis,
):
    if basis.shape[1] == 0:
        return C

    # (I - BB^T) C (I - BB^T), expanded without forming I.
    BC = basis.transpose(0, 1) @ C

    result = (
        C
        - basis @ BC
        - BC.transpose(0, 1) @ basis.transpose(0, 1)
        + basis
        @ (
            basis.transpose(0, 1)
            @ C
            @ basis
        )
        @ basis.transpose(0, 1)
    )

    return 0.5 * (
        result + result.transpose(0, 1)
    )


def metric_to_native(
    fisher_left,
    fisher_right,
    d_inv,
    a_inv,
):
    if fisher_left.shape[1] == 0:
        P = fisher_left

    else:
        P = orth(
            d_inv[:, None]
            * fisher_left
        )

    if fisher_right.shape[1] == 0:
        Q = fisher_right

    else:
        Q = orth(
            a_inv[:, None]
            * fisher_right
        )

    return (
        P.detach().cpu(),
        Q.detach().cpu(),
    )


# ---------------------------------------------------------------------
# Standard FCCA -- save immediately before any expensive adaptive solve
# ---------------------------------------------------------------------

def build_standard_fcca_only(
    language,
    stats,
    task_stats_path,
    device,
    overwrite,
):
    out_dir = (
        RESULT_ROOT
        / language
        / "fcca_r128"
        / "calibration"
    )

    out_path = out_dir / "basis_r128.pt"
    meta_path = out_dir / "basis_r128.json"

    if out_path.exists() and not overwrite:
        print(
            f"[{language}] ordinary FCCA basis exists; SKIP: {out_path}",
            flush=True,
        )
        return

    names = sorted(stats["modules"])

    if len(names) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"{language}: expected {EXPECTED_TARGETS} modules, "
            f"got {len(names)}"
        )

    bases = {}
    diagnostics = {}

    for index, name in enumerate(names, start=1):
        record = stats["modules"][name]

        count = float(record["positions"])

        if count <= 0:
            raise RuntimeError(
                f"{name}: positions={count}"
            )

        G = (
            record["cross_sum"]
            .double()
            .to(device)
            / count
        )

        a = (
            record["activation_second_sum"]
            .double()
            .to(device)
            / count
        )

        d = (
            record["gradient_second_sum"]
            .double()
            .to(device)
            / count
        )

        inv_a = torch.rsqrt(a + DAMPING)
        inv_d = torch.rsqrt(d + DAMPING)

        whitened = (
            inv_d[:, None]
            * G
            * inv_a[None, :]
        )

        U, S, Vh = torch.linalg.svd(
            whitened,
            full_matrices=False,
        )

        if min(whitened.shape) < REFERENCE_RANK:
            raise RuntimeError(
                f"{name}: matrix too small for rank "
                f"{REFERENCE_RANK}: {tuple(whitened.shape)}"
            )

        U_r = U[:, :REFERENCE_RANK]
        V_r = Vh[:REFERENCE_RANK, :].T

        P0 = inv_d[:, None] * U_r
        Q0 = inv_a[:, None] * V_r

        # Exact same native-coordinate convention as the Whisper builder:
        # QR in double, then store float32.
        P, _ = torch.linalg.qr(
            P0.double(),
            mode="reduced",
        )

        Q, _ = torch.linalg.qr(
            Q0.double(),
            mode="reduced",
        )

        P = P[:, :REFERENCE_RANK].float()
        Q = Q[:, :REFERENCE_RANK].float()

        eye = torch.eye(
            REFERENCE_RANK,
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

        if p_error > 5.0e-4 or q_error > 5.0e-4:
            raise RuntimeError(
                f"{name}: QR audit failed "
                f"P={p_error:.6e} Q={q_error:.6e}"
            )

        bases[name] = {
            "rank": REFERENCE_RANK,
            "P": P.detach().cpu(),
            "Q": Q.detach().cpu(),
            "stack": record["stack"],
            "layer": int(record["layer"]),
            "projection": record["projection"],
            "weight_shape": record["weight_shape"],
        }

        diagnostics[name] = {
            "rank": REFERENCE_RANK,
            "sigma_1": float(S[0].item()),
            "sigma_r": float(S[REFERENCE_RANK - 1].item()),
            "p_orthogonality_error": p_error,
            "q_orthogonality_error": q_error,
        }

        del (
            G,
            a,
            d,
            inv_a,
            inv_d,
            whitened,
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

        if index % 16 == 0 or index == len(names):
            print(
                f"[{language}] ordinary FCCA {index}/{len(names)}",
                flush=True,
            )

    payload = {
        "schema": "qwen3_asr_fcca_basis_v1",
        "method": "FCCA",
        "base_model": BASE_MODEL,
        "language": language,
        "qwen_language_prompt": stats.get("qwen_language_prompt"),
        "calibration_size": int(stats["calibration_size"]),
        "calibration_seed": int(stats["calibration_seed"]),
        "damping": float(DAMPING),
        "rank": int(REFERENCE_RANK),
        "target_count": len(bases),
        "trainable_core_budget": int(REFERENCE_CORE_BUDGET),
        "source_task_stats": str(task_stats_path),
        "source_task_stat_schema": stats.get("schema"),
        "selected_indices": stats.get("calibration_indices"),
        "diagnostics": diagnostics,
        "bases": bases,
    }

    atomic_torch_save(payload, out_path)

    atomic_json_save(
        {
            key: value
            for key, value in payload.items()
            if key not in {"bases", "diagnostics"}
        },
        meta_path,
    )

    print(
        f"[{language}] SAVED ordinary FCCA: {out_path}",
        flush=True,
    )

    del bases
    del diagnostics
    clear_cuda()


# ---------------------------------------------------------------------
# Standard + adaptive FCCA
# ---------------------------------------------------------------------

def solve_adaptive_rank_allocation(
    names,
    spectra,
    budget,
    rank_step=1,
    solver_time_limit=7200.0,
):
    """
    Faithful Qwen port of the Whisper adaptive-budget allocator.

    Multiple-choice integer program:

      choose exactly one rank for each matrix
      sum_l r_l^2 = budget
      maximize sum_l V_l(r_l)

    V_l(0)=0 and V_l(r)=sum_{k<=r} sigma_{l,k}^2.
    rank_step=1 is the full integer-rank problem.
    """

    if rank_step <= 0:
        raise ValueError("rank_step must be positive")

    candidates = {}
    utilities = {}

    for name in names:
        svals = spectra[name].double()
        max_rank = int(svals.numel())

        ranks = list(
            range(
                0,
                max_rank + 1,
                rank_step,
            )
        )

        if (
            REFERENCE_RANK <= max_rank
            and REFERENCE_RANK not in ranks
        ):
            ranks.append(REFERENCE_RANK)

        ranks = sorted(set(ranks))

        cumulative = torch.cat(
            [
                torch.zeros(1, dtype=torch.float64),
                torch.cumsum(svals.square(), dim=0),
            ]
        )

        candidates[name] = ranks
        utilities[name] = {
            rank: float(cumulative[rank].item())
            for rank in ranks
        }

    # Uniform rank-128 must be feasible under the reference budget.
    for name in names:
        if REFERENCE_RANK not in candidates[name]:
            raise RuntimeError(
                f"{name}: rank {REFERENCE_RANK} unavailable"
            )

    if len(names) * REFERENCE_RANK ** 2 != budget:
        raise RuntimeError(
            "Reference budget arithmetic failure: "
            f"{len(names)}*{REFERENCE_RANK}^2 != {budget}"
        )

    variable_meta = []

    for name in names:
        for rank in candidates[name]:
            variable_meta.append((name, rank))

    n_variables = len(variable_meta)

    raw_utilities = np.asarray(
        [
            utilities[name][rank]
            for name, rank in variable_meta
        ],
        dtype=np.float64,
    )

    # IMPORTANT: exactly like the Whisper allocator: one GLOBAL utility
    # scaling factor, not one separate scale per module.
    utility_scale = max(
        float(np.max(np.abs(raw_utilities))),
        1.0e-30,
    )

    # scipy.optimize.milp minimizes.
    objective = -raw_utilities / utility_scale

    row_indices = []
    col_indices = []
    values = []

    name_to_row = {
        name: index
        for index, name in enumerate(names)
    }

    budget_row = len(names)

    for column, (name, rank) in enumerate(variable_meta):
        row_indices.append(name_to_row[name])
        col_indices.append(column)
        values.append(1.0)

        row_indices.append(budget_row)
        col_indices.append(column)
        values.append(float(rank * rank))

    A = coo_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(col_indices, dtype=np.int64),
            ),
        ),
        shape=(len(names) + 1, n_variables),
    ).tocsr()

    lower = np.concatenate(
        [
            np.ones(len(names), dtype=np.float64),
            np.asarray([float(budget)], dtype=np.float64),
        ]
    )
    upper = lower.copy()

    constraints = LinearConstraint(A, lower, upper)

    print(
        "[allocation] "
        f"matrices={len(names)} "
        f"binary_variables={n_variables:,} "
        f"budget={budget:,} "
        f"rank_step={rank_step}",
        flush=True,
    )

    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=np.int8),
        bounds=Bounds(
            np.zeros(n_variables, dtype=np.float64),
            np.ones(n_variables, dtype=np.float64),
        ),
        constraints=constraints,
        options={
            "presolve": True,
            "time_limit": float(solver_time_limit),
            "mip_rel_gap": 0.0,
            "disp": True,
        },
    )

    if (
        not result.success
        or result.x is None
    ):
        raise RuntimeError(
            "Exact allocation solver did not certify an optimum: "
            f"status={result.status}, message={result.message}"
        )

    selected = {}
    x = np.asarray(result.x)

    for name in names:
        selected_columns = [
            column
            for column, (variable_name, rank)
            in enumerate(variable_meta)
            if variable_name == name and x[column] > 0.5
        ]

        if len(selected_columns) != 1:
            raise RuntimeError(
                f"{name}: expected one selected rank, "
                f"got {len(selected_columns)}"
            )

        _, rank = variable_meta[selected_columns[0]]
        selected[name] = int(rank)

    used_budget = sum(
        rank * rank
        for rank in selected.values()
    )

    if used_budget != budget:
        raise RuntimeError(
            "Budget mismatch after MIP solve: "
            f"{used_budget:,} != {budget:,}"
        )

    adaptive_utility = sum(
        utilities[name][selected[name]]
        for name in names
    )

    uniform_utility = sum(
        utilities[name][REFERENCE_RANK]
        for name in names
    )

    mip_gap = getattr(result, "mip_gap", None)

    if (
        adaptive_utility
        + max(1.0e-10, 1.0e-10 * abs(uniform_utility))
        < uniform_utility
    ):
        raise RuntimeError(
            "Allocator theorem audit failed: certified adaptive "
            "optimum is below feasible uniform-rank-128 objective."
        )

    print(
        "[allocation OPTIMAL] "
        f"budget={used_budget:,} "
        f"objective={adaptive_utility:.12e} "
        f"uniform128={uniform_utility:.12e} "
        f"ratio={adaptive_utility / uniform_utility:.8f} "
        f"mip_gap={mip_gap}",
        flush=True,
    )

    solver_info = {
        "solver": "scipy.optimize.milp / HiGHS",
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_mip_gap": (
            None if mip_gap is None else float(mip_gap)
        ),
        "rank_step": int(rank_step),
        "adaptive_utility": float(adaptive_utility),
        "uniform_rank128_utility": float(uniform_utility),
        "utility_ratio": float(
            adaptive_utility / uniform_utility
        ),
        "budget": int(budget),
        "used_budget": int(used_budget),
        "binary_variables": int(n_variables),
        "solver_time_limit": float(solver_time_limit),
    }

    return selected, solver_info, utilities

def build_standard_and_adaptive(
    language,
    stats,
    task_stats_path,
    device,
    overwrite,
    rank_step=1,
    solver_time_limit=7200.0,
):
    """
    Build ONLY the adaptive-budget FCCA basis here.

    Ordinary rank-128 FCCA is saved earlier by build_standard_fcca_only().
    This mirrors the Whisper adaptive builder: pass 1 computes complete
    singular spectra, CPU HiGHS selects ranks, pass 2 constructs only the
    selected per-matrix FCCA bases.
    """

    adaptive_dir = (
        RESULT_ROOT
        / language
        / "adaptive_budget_fcca"
        / "calibration"
    )

    adaptive_path = adaptive_dir / "adaptive_budget_basis.pt"

    if adaptive_path.exists() and not overwrite:
        existing = torch.load(
            adaptive_path,
            map_location="cpu",
            weights_only=False,
        )

        if existing.get("method") != "adaptive_budget_fcca":
            raise RuntimeError(
                f"Existing adaptive basis method mismatch: {adaptive_path}"
            )

        if int(existing.get("reference_core_budget", -1)) != REFERENCE_CORE_BUDGET:
            raise RuntimeError(
                f"Existing adaptive basis budget mismatch: {adaptive_path}"
            )

        if len(existing.get("bases", {})) != EXPECTED_TARGETS:
            raise RuntimeError(
                f"Existing adaptive basis target-count mismatch: {adaptive_path}"
            )

        used = sum(
            int(entry["rank"]) ** 2
            for entry in existing["bases"].values()
        )

        if used != REFERENCE_CORE_BUDGET:
            raise RuntimeError(
                f"Existing adaptive basis uses {used:,}, "
                f"expected {REFERENCE_CORE_BUDGET:,}"
            )

        solver = existing.get("solver", {})

        if not bool(solver.get("solver_success", False)):
            raise RuntimeError(
                "Existing adaptive basis has no solver-certified allocation."
            )

        print(
            f"[{language}] adaptive basis AUDIT PASS; SKIP: {adaptive_path}",
            flush=True,
        )
        return

    names = sorted(stats["modules"].keys())

    if len(names) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"{language}: expected {EXPECTED_TARGETS} modules, "
            f"got {len(names)}"
        )

    spectra = {}
    diagnostics = {}

    print(
        f"[{language}] [spectra] computing complete "
        "Fisher-whitened singular spectra",
        flush=True,
    )

    # Pass 1: spectra only, on GPU.
    for index, name in enumerate(names, start=1):
        G, a_inv, d_inv = whitened_matrix(
            stats["modules"][name],
            device,
        )

        svals = torch.linalg.svdvals(G)

        spectra[name] = (
            svals.detach()
            .double()
            .cpu()
        )

        diagnostics[name] = {
            "positions": int(stats["modules"][name]["positions"]),
            "out_features": int(G.shape[0]),
            "in_features": int(G.shape[1]),
            "maximum_rank": int(svals.numel()),
            "sigma_1": float(svals[0].item()),
            "sigma_128": float(svals[REFERENCE_RANK - 1].item()),
        }

        del G, a_inv, d_inv, svals
        torch.cuda.empty_cache()

        if index % 16 == 0 or index == len(names):
            print(
                f"[{language}] [spectra] {index}/{len(names)}",
                flush=True,
            )

    (
        allocation,
        solver_info,
        utility_tables,
    ) = solve_adaptive_rank_allocation(
        names,
        spectra,
        REFERENCE_CORE_BUDGET,
        rank_step=rank_step,
        solver_time_limit=solver_time_limit,
    )

    ranks = list(allocation.values())

    print(
        f"[{language}] [allocation summary] "
        f"min={min(ranks)} max={max(ranks)} "
        f"mean={np.mean(ranks):.4f} median={np.median(ranks):.4f} "
        f"zero_rank={sum(r == 0 for r in ranks)}",
        flush=True,
    )

    # Pass 2: construct ONLY selected FCCA bases, on GPU.
    adaptive_bases = {}

    print(
        f"[{language}] [basis] constructing selected ordinary FCCA subspaces",
        flush=True,
    )

    for index, name in enumerate(names, start=1):
        rank = int(allocation[name])
        stat = stats["modules"][name]

        if rank == 0:
            adaptive_bases[name] = {
                "rank": 0,
                "P": torch.empty(
                    diagnostics[name]["out_features"],
                    0,
                    dtype=torch.float32,
                ),
                "Q": torch.empty(
                    diagnostics[name]["in_features"],
                    0,
                    dtype=torch.float32,
                ),
                "stack": stat["stack"],
                "layer": int(stat["layer"]),
                "projection": stat["projection"],
                "weight_shape": stat["weight_shape"],
            }

            diagnostics[name]["selected_rank"] = 0
            diagnostics[name]["selected_utility"] = 0.0

        else:
            G, a_inv, d_inv = whitened_matrix(stat, device)

            U, S, Vh = torch.linalg.svd(
                G,
                full_matrices=False,
            )

            U_r = U[:, :rank]
            V_r = Vh[:rank, :].T

            P0 = d_inv[:, None] * U_r
            Q0 = a_inv[:, None] * V_r

            # Same native-coordinate convention as Whisper adaptive FCCA:
            # QR in double, then store float32.
            P, _ = torch.linalg.qr(
                P0.double(),
                mode="reduced",
            )
            Q, _ = torch.linalg.qr(
                Q0.double(),
                mode="reduced",
            )

            P = P[:, :rank].float()
            Q = Q[:, :rank].float()

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

            if p_error > 5.0e-4 or q_error > 5.0e-4:
                raise RuntimeError(
                    f"{name}: QR audit failed "
                    f"P={p_error:.6e} Q={q_error:.6e}"
                )

            adaptive_bases[name] = {
                "rank": rank,
                "P": P.detach().cpu(),
                "Q": Q.detach().cpu(),
                "stack": stat["stack"],
                "layer": int(stat["layer"]),
                "projection": stat["projection"],
                "weight_shape": stat["weight_shape"],
            }

            diagnostics[name]["selected_rank"] = rank
            diagnostics[name]["selected_utility"] = float(
                utility_tables[name][rank]
            )
            diagnostics[name]["p_orthogonality_error"] = p_error
            diagnostics[name]["q_orthogonality_error"] = q_error

            del (
                G,
                a_inv,
                d_inv,
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

        if index % 16 == 0 or index == len(names):
            print(
                f"[{language}] [basis] {index}/{len(names)}",
                flush=True,
            )

    used_budget = sum(
        int(entry["rank"]) ** 2
        for entry in adaptive_bases.values()
    )

    if used_budget != REFERENCE_CORE_BUDGET:
        raise RuntimeError(
            f"Final basis budget mismatch: "
            f"{used_budget:,} != {REFERENCE_CORE_BUDGET:,}"
        )

    adaptive_payload = {
        "schema": "qwen3_asr_adaptive_fcca_basis_v2",
        "method": "adaptive_budget_fcca",
        "objective": "sum_l sum_{k<=r_l} sigma_{l,k}^2",
        "optimality_scope": (
            "globally optimal rank allocation under the separable FCCA "
            "local Fisher objective over the specified allowed integer ranks"
        ),
        "base_model": BASE_MODEL,
        "language": language,
        "source_task_stats": str(task_stats_path),
        "source_task_stat_schema": stats.get("schema"),
        "calibration_size": int(stats["calibration_size"]),
        "calibration_seed": int(stats["calibration_seed"]),
        "damping": float(DAMPING),
        "reference_uniform_rank": int(REFERENCE_RANK),
        "reference_core_budget": int(REFERENCE_CORE_BUDGET),
        "actual_core_budget": int(used_budget),
        "rank_allocation": allocation,
        "rank_summary": {
            "minimum": min(ranks),
            "maximum": max(ranks),
            "mean": float(np.mean(ranks)),
            "median": float(np.median(ranks)),
            "zero_rank_matrices": int(sum(r == 0 for r in ranks)),
        },
        "solver": solver_info,
        "diagnostics": diagnostics,
        "target_count": len(adaptive_bases),
        "bases": adaptive_bases,
    }

    atomic_torch_save(adaptive_payload, adaptive_path)

    atomic_json_save(
        {
            key: value
            for key, value in adaptive_payload.items()
            if key not in {"bases", "diagnostics"}
        },
        adaptive_dir / "adaptive_budget_basis.json",
    )

    print(
        f"[{language}] [adaptive basis COMPLETE] {adaptive_path}",
        flush=True,
    )

    del spectra
    del utility_tables
    del adaptive_bases
    del diagnostics
    clear_cuda()


# ---------------------------------------------------------------------
# Asymmetric hard + shared/private -- faithful port of Whisper builder
# ---------------------------------------------------------------------

def asym_symmetrise(A):
    return 0.5 * (A + A.T)


def asym_top_eig(A, rank):
    A = asym_symmetrise(A)
    evals, evecs = torch.linalg.eigh(A)

    if rank > evecs.shape[1]:
        raise RuntimeError(
            f"rank={rank} impossible for {tuple(A.shape)}"
        )

    V = evecs[:, -rank:]
    V = torch.flip(V, dims=(1,))
    return V.contiguous()


def asym_orth(X, rank, name):
    """
    Match the Whisper asymmetric FCCA orthogonalisation exactly:
    skinny FP64 QR followed by FP32 storage.
    """
    if X.ndim != 2:
        raise RuntimeError(
            f"{name}: expected matrix, got {tuple(X.shape)}"
        )

    if rank > X.shape[0]:
        raise RuntimeError(
            f"{name}: rank={rank} exceeds ambient dimension "
            f"{X.shape[0]}"
        )

    if X.shape[1] < rank:
        raise RuntimeError(
            f"{name}: only {X.shape[1]} columns available "
            f"for requested rank={rank}"
        )

    Q64, _ = torch.linalg.qr(
        X.to(dtype=torch.float64),
        mode="reduced",
    )

    Q64 = Q64[:, :rank].contiguous()

    eye64 = torch.eye(
        rank,
        device=Q64.device,
        dtype=torch.float64,
    )

    err64 = float(
        torch.linalg.matrix_norm(
            Q64.T @ Q64 - eye64,
            ord="fro",
        ).item()
    )

    if err64 > 1.0e-10:
        raise RuntimeError(
            f"{name}: FP64 QR orthogonality failure "
            f"{err64:.12e}"
        )

    Q = Q64.to(
        dtype=torch.float32
    ).contiguous()

    eye32 = torch.eye(
        rank,
        device=Q.device,
        dtype=torch.float32,
    )

    err32 = float(
        torch.linalg.matrix_norm(
            Q.T @ Q - eye32,
            ord="fro",
        ).item()
    )

    if err32 > 5.0e-5:
        raise RuntimeError(
            f"{name}: FP32 stored-basis orthogonality failure "
            f"{err32:.12e}"
        )

    return Q

def asym_column_span_basis(X, rtol=1.0e-6):
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


def asym_projector_complement(V):
    n = V.shape[0]

    I = torch.eye(
        n,
        device=V.device,
        dtype=V.dtype,
    )

    return I - V @ V.T


def asym_side_covariance(Gt, side):
    if side == "right":
        return Gt.T @ Gt

    if side == "left":
        return Gt @ Gt.T

    raise ValueError(side)


def asym_side_score(C, V):
    return float(
        torch.trace(
            V.T @ C @ V
        ).item()
    )


def asym_independent_fcca_score(Gt, rank):
    s = torch.linalg.svdvals(Gt)
    return float(torch.sum(s[:rank] ** 2).item())


def asym_pair_groups(stats):
    grouped = {}

    for name, stat in stats["modules"].items():
        key = (
            stat["stack"],
            stat["projection"],
        )

        grouped.setdefault(key, []).append(
            (
                int(stat["layer"]),
                name,
            )
        )

    result = {}

    for (stack, projection), members in sorted(grouped.items()):
        members = sorted(members)
        layers = [layer for layer, _ in members]

        if layers != list(range(min(layers), max(layers) + 1)):
            raise RuntimeError(
                f"Non-contiguous layers for {stack}/{projection}: {layers}"
            )

        if len(members) % 2:
            raise RuntimeError(
                f"Odd number of layers for {stack}/{projection}"
            )

        for i in range(0, len(members), 2):
            pair = members[i:i + 2]
            pair_start = pair[0][0]

            if [x[0] for x in pair] != [pair_start, pair_start + 1]:
                raise RuntimeError(
                    f"Bad adjacent pair for {stack}/{projection}: {pair}"
                )

            key = (
                stack,
                projection,
                pair_start,
            )

            result[key] = [name for _, name in pair]

    # audio: 24 layers -> 12 pairs x 4 projections = 48
    # text:  28 layers -> 14 pairs x 4 projections = 56
    # total = 104 groups.
    if len(result) != 104:
        raise RuntimeError(
            f"Expected 104 Qwen adjacent-pair groups, found {len(result)}"
        )

    return result


def asym_whiten(stat, device):
    count = float(stat["positions"])

    if count <= 0:
        raise RuntimeError(
            f"Invalid task-stat count={count}"
        )

    G = stat["cross_sum"].to(
        device=device,
        dtype=torch.float32,
    ) / count

    a = stat["activation_second_sum"].to(
        device=device,
        dtype=torch.float32,
    ) / count

    d = stat["gradient_second_sum"].to(
        device=device,
        dtype=torch.float32,
    ) / count

    inv_a = torch.rsqrt(a + DAMPING)
    inv_d = torch.rsqrt(d + DAMPING)

    Gt = (
        inv_d[:, None]
        * G
        * inv_a[None, :]
    )

    return {
        "G": G,
        "a": a,
        "d": d,
        "inv_a": inv_a,
        "inv_d": inv_d,
        "Gt": Gt,
        "count": int(count),
    }


def asym_solve_hard_group(
    members,
    whitened,
    shared_side,
):
    C = {
        name: asym_side_covariance(
            whitened[name]["Gt"],
            shared_side,
        )
        for name in members
    }

    pooled = sum(C.values())

    shared = asym_top_eig(
        pooled,
        REFERENCE_RANK,
    )

    bases = {}
    shared_objective = 0.0
    independent_objective = 0.0

    for name in members:
        Gt = whitened[name]["Gt"]

        if shared_side == "right":
            V = shared
            U = asym_orth(
                Gt @ V,
                REFERENCE_RANK,
                f"{name}: U=orth(GV)",
            )

        else:
            U = shared
            V = asym_orth(
                Gt.T @ U,
                REFERENCE_RANK,
                f"{name}: V=orth(G^T U)",
            )

        # Audit the ONE-SIDED constrained subspace itself.
        # Do this in FP64 and re-orthogonalise the shared side for
        # numerical auditing only.  The saved U/V bases are unchanged.
        G64 = Gt.to(dtype=torch.float64)

        if shared_side == "right":
            V64, _ = torch.linalg.qr(
                V.to(dtype=torch.float64),
                mode="reduced",
            )
            shared_objective += float(
                torch.sum(
                    (G64 @ V64[:, :REFERENCE_RANK]) ** 2
                ).item()
            )

        else:
            U64, _ = torch.linalg.qr(
                U.to(dtype=torch.float64),
                mode="reduced",
            )
            shared_objective += float(
                torch.sum(
                    (
                        U64[:, :REFERENCE_RANK].T
                        @ G64
                    ) ** 2
                ).item()
            )

        # Independent rank-r FCCA is the theoretical upper bound.
        s64 = torch.linalg.svdvals(G64)
        independent_objective += float(
            torch.sum(
                s64[:REFERENCE_RANK] ** 2
            ).item()
        )

        bases[name] = {
            "U": U,
            "V": V,
        }

    tolerance = 1.0e-5 * max(
        independent_objective,
        1.0,
    )

    if shared_objective > independent_objective + tolerance:
        raise RuntimeError(
            "THEOREM AUDIT FAILED: "
            f"hard shared objective {shared_objective:.12e} > "
            f"independent FCCA {independent_objective:.12e}"
        )

    return bases, {
        "shared_objective": shared_objective,
        "independent_fcca_objective": independent_objective,
        "ratio_to_independent": (
            shared_objective / independent_objective
        ),
    }


def asym_private_update(C, shared):
    P = asym_projector_complement(shared)
    projected = P @ C @ P

    private = asym_top_eig(
        projected,
        PRIVATE_RANK,
    )

    private = asym_orth(
        P @ private,
        PRIVATE_RANK,
        "private update",
    )

    return private


def asym_shared_update(
    C_by_name,
    private_by_name,
):
    raw_union = torch.cat(
        [
            private_by_name[name]
            for name in sorted(private_by_name)
        ],
        dim=1,
    )

    union = asym_column_span_basis(raw_union)

    ambient = int(raw_union.shape[0])
    remaining = ambient - int(union.shape[1])

    if remaining < SHARED_RANK:
        raise RuntimeError(
            "Private union leaves insufficient room for shared rank: "
            f"ambient={ambient}, "
            f"private_union_rank={union.shape[1]}, "
            f"remaining={remaining}, "
            f"shared_rank={SHARED_RANK}"
        )

    P = asym_projector_complement(union)
    pooled = sum(C_by_name.values())
    projected = P @ pooled @ P

    shared = asym_top_eig(
        projected,
        SHARED_RANK,
    )

    shared = asym_orth(
        P @ shared,
        SHARED_RANK,
        "shared conditional update",
    )

    for name, private in private_by_name.items():
        overlap = float(
            torch.linalg.matrix_norm(
                shared.T @ private,
                ord="fro",
            ).item()
        )

        if overlap > 1.0e-4:
            raise RuntimeError(
                f"Shared/private feasibility failure for {name}: "
                f"||Vg^T Vpriv||_F={overlap:.12e}"
            )

    return shared


def asym_shared_private_objective(
    C_by_name,
    shared,
    private_by_name,
):
    total = 0.0

    for name, C in C_by_name.items():
        total += asym_side_score(C, shared)
        total += asym_side_score(C, private_by_name[name])

    return total


def asym_make_shared_private_initializations(
    pooled,
    device,
):
    pooled_top128 = asym_top_eig(
        pooled,
        REFERENCE_RANK,
    )

    starts = [
        pooled_top128[:, :SHARED_RANK].contiguous()
    ]

    for seed in (20261,):
        generator = torch.Generator(
            device="cpu"
        ).manual_seed(seed)

        random_matrix = torch.randn(
            REFERENCE_RANK,
            SHARED_RANK,
            generator=generator,
            dtype=torch.float32,
        ).to(device)

        rotation, _ = torch.linalg.qr(
            random_matrix,
            mode="reduced",
        )

        candidate = pooled_top128 @ rotation

        candidate = asym_orth(
            candidate,
            SHARED_RANK,
            f"initial shared space seed={seed}",
        )

        starts.append(candidate)

    return starts


def asym_run_shared_private_start(
    C_by_name,
    shared_initial,
    max_iters,
    tol,
):
    shared = shared_initial

    private = {
        name: asym_private_update(
            C_by_name[name],
            shared,
        )
        for name in sorted(C_by_name)
    }

    history = [
        asym_shared_private_objective(
            C_by_name,
            shared,
            private,
        )
    ]

    converged = False

    for iteration in range(1, max_iters + 1):
        shared_new = asym_shared_update(
            C_by_name,
            private,
        )

        private_new = {
            name: asym_private_update(
                C_by_name[name],
                shared_new,
            )
            for name in sorted(C_by_name)
        }

        value = asym_shared_private_objective(
            C_by_name,
            shared_new,
            private_new,
        )

        previous = history[-1]

        monotonic_tolerance = (
            1.0e-5 * max(abs(previous), 1.0)
        )

        if value + monotonic_tolerance < previous:
            raise RuntimeError(
                "Monotonic block-coordinate ascent audit failed: "
                f"iteration={iteration}, "
                f"new={value:.12e}, old={previous:.12e}"
            )

        history.append(value)

        relative_change = (
            abs(value - previous)
            / max(abs(previous), 1.0)
        )

        shared = shared_new
        private = private_new

        if relative_change <= tol:
            converged = True
            break

    return {
        "shared": shared,
        "private": private,
        "objective": history[-1],
        "objective_history": history,
        "iterations": len(history) - 1,
        "converged": converged,
    }


def asym_solve_shared_private_group(
    members,
    whitened,
    shared_side,
    max_iters,
    tol,
):
    C = {
        name: asym_side_covariance(
            whitened[name]["Gt"],
            shared_side,
        )
        for name in members
    }

    pooled = sum(C.values())

    starts = asym_make_shared_private_initializations(
        pooled,
        pooled.device,
    )

    solutions = []

    for start_index, shared_initial in enumerate(starts):
        solution = asym_run_shared_private_start(
            C_by_name=C,
            shared_initial=shared_initial,
            max_iters=max_iters,
            tol=tol,
        )

        solution["start_index"] = start_index
        solutions.append(solution)

    best = max(
        solutions,
        key=lambda x: x["objective"],
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

            V = asym_orth(
                V,
                REFERENCE_RANK,
                f"{name}: combined right shared/private basis",
            )

            U = asym_orth(
                Gt @ V,
                REFERENCE_RANK,
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

            U = asym_orth(
                U,
                REFERENCE_RANK,
                f"{name}: combined left shared/private basis",
            )

            V = asym_orth(
                Gt.T @ U,
                REFERENCE_RANK,
                f"{name}: V=orth(G^T U)",
            )

        else:
            raise ValueError(shared_side)

        # Final theorem audit: measure the actual ONE-SIDED
        # rank-128 constrained span in FP64.  Do not use the
        # two-sided FP32 core, whose tiny non-orthogonality can
        # spuriously exceed the independent FCCA upper bound.
        G64 = Gt.to(dtype=torch.float64)

        if shared_side == "right":
            V64, _ = torch.linalg.qr(
                V.to(dtype=torch.float64),
                mode="reduced",
            )
            matrix_value = float(
                torch.sum(
                    (
                        G64
                        @ V64[:, :REFERENCE_RANK]
                    ) ** 2
                ).item()
            )

        else:
            U64, _ = torch.linalg.qr(
                U.to(dtype=torch.float64),
                mode="reduced",
            )
            matrix_value = float(
                torch.sum(
                    (
                        U64[:, :REFERENCE_RANK].T
                        @ G64
                    ) ** 2
                ).item()
            )

        constrained_objective += matrix_value

        s64 = torch.linalg.svdvals(G64)
        independent_objective += float(
            torch.sum(
                s64[:REFERENCE_RANK] ** 2
            ).item()
        )

        bases[name] = {
            "U": U,
            "V": V,
        }

    theorem_tolerance = (
        1.0e-5 * max(independent_objective, 1.0)
    )

    if constrained_objective > independent_objective + theorem_tolerance:
        raise RuntimeError(
            "THEOREM AUDIT FAILED: "
            f"shared/private={constrained_objective:.12e} > "
            f"independent_FCCA={independent_objective:.12e}"
        )

    start_summaries = []

    for solution in solutions:
        start_summaries.append(
            {
                "start_index": int(solution["start_index"]),
                "final_objective": float(solution["objective"]),
                "iterations": int(solution["iterations"]),
                "converged": bool(solution["converged"]),
                "objective_history": [
                    float(x)
                    for x in solution["objective_history"]
                ],
            }
        )

    return bases, {
        "shared_private_objective": constrained_objective,
        "independent_fcca_objective": independent_objective,
        "ratio_to_independent": (
            constrained_objective / independent_objective
        ),
        "number_of_starts": len(solutions),
        "selected_start": int(best["start_index"]),
        "iterations": int(best["iterations"]),
        "converged": bool(best["converged"]),
        "objective_history": [
            float(x)
            for x in best["objective_history"]
        ],
        "all_starts": start_summaries,
    }


def asym_native_basis(
    U,
    V,
    stat,
    name,
):
    P0 = stat["inv_d"][:, None] * U
    Q0 = stat["inv_a"][:, None] * V

    P, RP = torch.linalg.qr(
        P0,
        mode="reduced",
    )

    Q, RQ = torch.linalg.qr(
        Q0,
        mode="reduced",
    )

    P = P[:, :REFERENCE_RANK]
    Q = Q[:, :REFERENCE_RANK]

    eye = torch.eye(
        REFERENCE_RANK,
        device=P.device,
        dtype=P.dtype,
    )

    p_err = float(
        torch.linalg.matrix_norm(
            P.T @ P - eye,
            ord="fro",
        ).item()
    )

    q_err = float(
        torch.linalg.matrix_norm(
            Q.T @ Q - eye,
            ord="fro",
        ).item()
    )

    if p_err > 1.0e-4:
        raise RuntimeError(
            f"{name}: native P QR error {p_err}"
        )

    if q_err > 1.0e-4:
        raise RuntimeError(
            f"{name}: native Q QR error {q_err}"
        )

    return (
        P.float().cpu(),
        Q.float().cpu(),
        RP.float().cpu(),
        RQ.float().cpu(),
    )


def build_asymmetric_variant(
    language,
    variant,
    stats,
    task_stats_path,
    device,
    overwrite,
    max_iters=8,
    tol=1.0e-5,
):
    if variant not in {"hard", "shared_private"}:
        raise ValueError(variant)

    if variant == "hard":
        out_dir = (
            RESULT_ROOT
            / language
            / "asym_hard_r128_pairwise"
            / "calibration"
        )

    else:
        out_dir = (
            RESULT_ROOT
            / language
            / "asym_shared_private_r128_s64_p64_pairwise"
            / "calibration"
        )

    out_path = out_dir / "basis_r128.pt"
    meta_path = out_dir / "basis_r128.json"

    if out_path.exists() and not overwrite:
        print(
            f"[{language}] {variant} basis exists; SKIP: {out_path}",
            flush=True,
        )
        return

    modules = stats["modules"]
    groups = asym_pair_groups(stats)

    payload = {
        "schema": (
            "qwen3_asr_asym_hard_basis_v1"
            if variant == "hard"
            else "qwen3_asr_asym_shared_private_basis_v1"
        ),
        "method": "asymmetric_one_sided_fcca",
        "variant": variant,
        "base_model": BASE_MODEL,
        "language": language,
        "rank": REFERENCE_RANK,
        "shared_rank": (
            REFERENCE_RANK
            if variant == "hard"
            else SHARED_RANK
        ),
        "private_rank": (
            0
            if variant == "hard"
            else PRIVATE_RANK
        ),
        "calibration_size": int(stats["calibration_size"]),
        "calibration_seed": int(stats["calibration_seed"]),
        "damping": float(DAMPING),
        "pairing": (
            "fixed adjacent non-overlapping pairs; "
            "audio (0,1)...(22,23), text (0,1)...(26,27)"
        ),
        "sharing_rule": (
            "Q/K/V share Fisher-coordinate right/input side; "
            "O shares Fisher-coordinate left/output side"
        ),
        "task_stats_path": str(Path(task_stats_path).resolve()),
        "source_task_stat_schema": stats.get("schema"),
        "target_count": len(modules),
        "trainable_core_budget": (
            EXPECTED_TARGETS
            * REFERENCE_RANK
            * REFERENCE_RANK
        ),
        "bases": {},
        "groups": {},
        "target_names": sorted(modules),
    }

    total_constrained = 0.0
    total_independent = 0.0

    ordered_groups = sorted(
        groups.items(),
        key=lambda kv: kv[0],
    )

    for group_index, (key, members) in enumerate(
        ordered_groups,
        start=1,
    ):
        stack, projection, pair_start = key

        shared_side = (
            "left"
            if projection == "o"
            else "right"
        )

        whitened = {
            name: asym_whiten(
                modules[name],
                device,
            )
            for name in members
        }

        if variant == "hard":
            fisher_bases, diagnostics = asym_solve_hard_group(
                members,
                whitened,
                shared_side,
            )
            constrained = diagnostics["shared_objective"]

        else:
            fisher_bases, diagnostics = asym_solve_shared_private_group(
                members,
                whitened,
                shared_side,
                max_iters=max_iters,
                tol=tol,
            )
            constrained = diagnostics["shared_private_objective"]

        independent = diagnostics["independent_fcca_objective"]

        total_constrained += constrained
        total_independent += independent

        group_name = (
            f"{stack}_{projection}_"
            f"{pair_start:02d}_{pair_start + 1:02d}"
        )

        payload["groups"][group_name] = {
            "stack": stack,
            "projection": projection,
            "layers": [
                pair_start,
                pair_start + 1,
            ],
            "shared_side": shared_side,
            "members": members,
            **diagnostics,
        }

        for name in members:
            P, Q, RP, RQ = asym_native_basis(
                fisher_bases[name]["U"],
                fisher_bases[name]["V"],
                whitened[name],
                name,
            )

            payload["bases"][name] = {
                "rank": REFERENCE_RANK,
                "P": P,
                "Q": Q,
                "left_qr_R": RP,
                "right_qr_R": RQ,
                "stack": modules[name]["stack"],
                "layer": int(modules[name]["layer"]),
                "projection": modules[name]["projection"],
                "shared_side": shared_side,
                "pair": [
                    pair_start,
                    pair_start + 1,
                ],
                "weight_shape": modules[name]["weight_shape"],
            }

        print(
            f"[{language}] "
            f"[{group_index:03d}/{len(ordered_groups):03d}] "
            f"{group_name} "
            f"variant={variant} "
            f"side={shared_side} "
            f"ratio={constrained / independent:.8f}",
            flush=True,
        )

        del whitened
        del fisher_bases
        torch.cuda.empty_cache()

    if len(payload["bases"]) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"{variant}: expected {EXPECTED_TARGETS} bases, "
            f"got {len(payload['bases'])}"
        )

    global_ratio = total_constrained / total_independent

    if global_ratio > 1.0 + 1.0e-5:
        raise RuntimeError(
            "GLOBAL THEOREM AUDIT FAILED: "
            f"ratio={global_ratio}"
        )

    payload["global_audit"] = {
        "constrained_objective": total_constrained,
        "independent_fcca_objective": total_independent,
        "ratio_to_independent": global_ratio,
        "theorem_check": "PASS",
        "statement": (
            "Constrained feasible set is a subset of independent "
            "rank-128 FCCA; calibration objective must not exceed "
            "independent FCCA."
        ),
    }

    atomic_torch_save(payload, out_path)

    atomic_json_save(
        {
            key: value
            for key, value in payload.items()
            if key != "bases"
        },
        meta_path,
    )

    print("=" * 79)
    print(
        f"[{language}] PASS {variant}; "
        f"global ratio={global_ratio:.12f}",
        flush=True,
    )
    print(
        f"[{language}] SAVED {out_path}",
        flush=True,
    )
    print("=" * 79)

    clear_cuda()


def build_asymmetric_bases(
    language,
    stats,
    task_stats_path,
    device,
    overwrite,
):
    # Save HARD first, independently.  If shared/private later fails,
    # the hard basis remains reusable.
    build_asymmetric_variant(
        language=language,
        variant="hard",
        stats=stats,
        task_stats_path=task_stats_path,
        device=device,
        overwrite=overwrite,
    )

    build_asymmetric_variant(
        language=language,
        variant="shared_private",
        stats=stats,
        task_stats_path=task_stats_path,
        device=device,
        overwrite=overwrite,
        max_iters=8,
        tol=1.0e-5,
    )


# ---------------------------------------------------------------------
# All bases from one saved language-specific task-stat file
# ---------------------------------------------------------------------

def build_all_bases(
    language,
    task_stats_path,
    device,
    overwrite,
    rank_step=1,
    solver_time_limit=7200.0,
):
    print()
    print("=" * 79)
    print(f"BUILDING BASES: {language}")
    print("=" * 79)

    stats = torch.load(
        task_stats_path,
        map_location="cpu",
        weights_only=False,
    )

    if stats.get("base_model") != BASE_MODEL:
        raise RuntimeError(
            f"{language}: wrong base model in stats: "
            f"{stats.get('base_model')!r}"
        )

    if stats.get("language") != language:
        raise RuntimeError(
            f"{language}: task-stat language mismatch: "
            f"{stats.get('language')!r}"
        )

    if int(stats.get("target_count", -1)) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"{language}: expected {EXPECTED_TARGETS} targets, "
            f"got {stats.get('target_count')}"
        )

    # Save reusable bases in cheap/safe order.  Adaptive is last because
    # its exact integer allocation can be by far the slowest stage.
    build_standard_fcca_only(
        language=language,
        stats=stats,
        task_stats_path=task_stats_path,
        device=device,
        overwrite=overwrite,
    )

    build_asymmetric_bases(
        language=language,
        stats=stats,
        task_stats_path=task_stats_path,
        device=device,
        overwrite=overwrite,
    )

    build_standard_and_adaptive(
        language=language,
        stats=stats,
        task_stats_path=task_stats_path,
        device=device,
        overwrite=overwrite,
        rank_step=rank_step,
        solver_time_limit=solver_time_limit,
    )

    del stats
    clear_cuda()


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_qwen_model(
    device,
    dtype,
):
    kwargs = {
        "attn_implementation": "eager",
    }

    try:
        model = (
            Qwen3ASRForConditionalGeneration
            .from_pretrained(
                BASE_MODEL,
                dtype=dtype,
                **kwargs,
            )
        )

    except TypeError:
        model = (
            Qwen3ASRForConditionalGeneration
            .from_pretrained(
                BASE_MODEL,
                torch_dtype=dtype,
                **kwargs,
            )
        )

    model.to(device)
    model.eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGUAGES,
    )

    parser.add_argument(
        "--calibration-size",
        type=int,
        default=CALIBRATION_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CALIBRATION_SEED,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--collect-only",
        action="store_true",
    )

    parser.add_argument(
        "--bases-only",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--rank-step",
        type=int,
        default=1,
        help=(
            "Adaptive FCCA allowed-rank spacing. "
            "1 = exact full integer-rank allocation; "
            "8 = much faster restricted allocation."
        ),
    )

    parser.add_argument(
        "--solver-time-limit",
        type=float,
        default=7200.0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.collect_only and args.bases_only:
        raise ValueError(
            "Cannot use --collect-only and --bases-only together."
        )

    unknown = [
        x
        for x in args.languages
        if x not in QWEN_LANGUAGE_NAME
    ]

    if unknown:
        raise RuntimeError(
            "Unknown language IDs: "
            + ", ".join(unknown)
        )

    set_seed(args.seed)

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    dtype = choose_dtype(device)

    print(
        f"BASE_MODEL={BASE_MODEL}"
    )
    print(
        f"DEVICE={device}"
    )
    print(
        f"DTYPE={dtype}"
    )
    print(
        f"LANGUAGES={args.languages}"
    )
    print(
        f"CALIBRATION_SIZE={args.calibration_size} PER LANGUAGE"
    )
    print(
        f"CALIBRATION_SEED={args.seed}"
    )
    print(
        f"REFERENCE_RANK={REFERENCE_RANK}"
    )
    print(
        f"REFERENCE_CORE_BUDGET={REFERENCE_CORE_BUDGET:,}"
    )
    print(
        f"SHARED_PRIVATE={SHARED_RANK}+{PRIVATE_RANK}"
    )

    task_stat_paths = {}

    # -------------------------------------------------------------
    # Stage 1: collect each language independently
    # -------------------------------------------------------------

    if not args.bases_only:
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL
        )

        model = load_qwen_model(
            device,
            dtype,
        )

        targets = discover_targets(model)

        print(
            f"discovered_targets={len(targets)}"
        )
        print(
            "audio_targets="
            f"{sum(t.stack == 'audio' for t in targets)}"
        )
        print(
            "text_targets="
            f"{sum(t.stack == 'text' for t in targets)}"
        )

        for language in args.languages:
            task_stat_paths[language] = (
                collect_one_language(
                    language=language,
                    model=model,
                    processor=processor,
                    targets=targets,
                    device=device,
                    dtype=dtype,
                    calibration_size=args.calibration_size,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
            )

        del targets
        del model
        del processor

        clear_cuda()

    else:
        for language in args.languages:
            path = (
                RESULT_ROOT
                / language
                / "_task_stats"
                / (
                    f"task_stats_{args.calibration_size}"
                    "_all_attention.pt"
                )
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing task stats for {language}: {path}"
                )

            task_stat_paths[language] = path

    if args.collect_only:
        print()
        print("COLLECTION COMPLETE.")
        return

    # -------------------------------------------------------------
    # Stage 2: construct all four reusable basis sets
    # -------------------------------------------------------------

    for language in args.languages:
        build_all_bases(
            language=language,
            task_stats_path=task_stat_paths[language],
            device=device,
            overwrite=args.overwrite,
            rank_step=args.rank_step,
            solver_time_limit=args.solver_time_limit,
        )

    print()
    print("=" * 79)
    print("ALL REQUESTED LANGUAGES COMPLETE")
    print("=" * 79)

    for language in args.languages:
        root = RESULT_ROOT / language

        print()
        print(language)
        print(
            "  stats: "
            + str(
                root
                / "_task_stats"
                / (
                    f"task_stats_{args.calibration_size}"
                    "_all_attention.pt"
                )
            )
        )
        print(
            "  FCCA: "
            + str(
                root
                / "fcca_r128"
                / "calibration"
                / "basis_r128.pt"
            )
        )
        print(
            "  adaptive: "
            + str(
                root
                / "adaptive_budget_fcca"
                / "calibration"
                / "adaptive_budget_basis.pt"
            )
        )
        print(
            "  asym hard: "
            + str(
                root
                / "asym_hard_r128_pairwise"
                / "calibration"
                / "basis_r128.pt"
            )
        )
        print(
            "  asym shared/private: "
            + str(
                root
                / (
                    "asym_shared_private_"
                    "r128_s64_p64_pairwise"
                )
                / "calibration"
                / "basis_r128.pt"
            )
        )


if __name__ == "__main__":
    main()
