from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import time
import unicodedata
from opencc import OpenCC
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Audio, Dataset, load_from_disk
from scripts_fleurs.fleurs_paths import dataset_root, load_fleurs_split
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    Qwen3ASRForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    get_linear_schedule_with_warmup,
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DATA_ROOT = Path("/data/milsrg1/corpora/fleurs-selected-languages")
RESULT_ROOT = PROJECT_ROOT / "outputs" / "new_results_qwen"

BASE_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
SAMPLE_RATE = 16_000
SEED = 2026

DEFAULT_MAX_EPOCHS = 12
DEFAULT_PATIENCE = 3
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_MICROBATCH = 1
DEFAULT_EVAL_BATCH = 1

REFERENCE_RANK = 128
EXPECTED_TARGETS = 208
REFERENCE_CORE_BUDGET = EXPECTED_TARGETS * REFERENCE_RANK**2

# Evaluation-only Mandarin orthographic normalization.
# FLEURS Mandarin references are Simplified Chinese.
# Convert both reference and hypothesis to Simplified before CER
# so script choice is not counted as an ASR recognition error.
ZH_T2S = OpenCC("t2s")

LANGUAGES = {
    "ast_es": {"name": "Asturian", "qwen_language": "Spanish"},
    "cmn_hans_cn": {"name": "Mandarin Chinese", "qwen_language": "Chinese"},
    "es_419": {"name": "Spanish", "qwen_language": "Spanish"},
    "ky_kg": {"name": "Kyrgyz", "qwen_language": "Turkish"},
    "ckb_iq": {"name": "Central Kurdish / Sorani", "qwen_language": "Persian"},
}
QWEN_LANGUAGE_NAME = {k: v["qwen_language"] for k, v in LANGUAGES.items()}

METHOD_DIR = {
    "fcca": "fcca_r128",
    "adaptive_fcca": "adaptive_budget_fcca",
    "asym_hard": "asym_hard_r128_pairwise",
    "asym_shared_private": "asym_shared_private_r128_s64_p64_pairwise",
}
BASIS_FILENAME = {
    "fcca": "basis_r128.pt",
    "adaptive_fcca": "adaptive_budget_basis.pt",
    "asym_hard": "basis_r128.pt",
    "asym_shared_private": "basis_r128.pt",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)



def file_sha256(path: Path) -> str:
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(16 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def canonical_task_stats_path(
    language: str,
) -> Path:
    return (
        RESULT_ROOT
        / language
        / "_task_stats"
        / "task_stats_256_all_attention.pt"
    )


def recorded_task_stats_path(
    language: str,
    payload: Mapping[str, Any],
) -> Path:
    recorded = (
        payload.get("task_stats_path")
        or payload.get("source_task_stats")
    )

    if recorded is None:
        raise RuntimeError(
            "Basis payload lacks task_stats_path/source_task_stats."
        )

    path = Path(str(recorded))

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def complete_model_state(
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name:
            tensor.detach().cpu().clone()
        for name, tensor
        in model.state_dict().items()
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_safe(row.get(k, "")) for k in fieldnames})
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda_all"):
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def require_cuda(device_arg: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(device_arg)
    torch.cuda.set_device(device)
    return device


def _pcm_to_float(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32, copy=False)
    if audio.dtype == np.uint8:
        return ((audio.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    info = np.iinfo(audio.dtype)
    scale = float(max(abs(info.min), abs(info.max)))
    return (audio.astype(np.float32) / scale).astype(np.float32)


def decode_audio_value(value: Any, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if value is None:
        raise ValueError("Missing audio value.")
    if isinstance(value, dict):
        if value.get("array") is not None:
            audio = np.asarray(value["array"])
            sr = int(value.get("sampling_rate") or target_sr)
        elif value.get("bytes") is not None:
            audio, sr = sf.read(io.BytesIO(value["bytes"]), always_2d=False)
        elif value.get("path") and Path(value["path"]).is_file():
            audio, sr = sf.read(str(value["path"]), always_2d=False)
        else:
            raise RuntimeError(
                "Audio dict has neither array/bytes nor an existing path: "
                f"{value.get('path')!r}"
            )
    elif isinstance(value, (str, Path)):
        audio, sr = sf.read(str(value), always_2d=False)
    elif hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        audio = samples.data
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        sr = int(samples.sample_rate)
    else:
        raise TypeError(f"Unsupported audio value type: {type(value)!r}")

    audio = _pcm_to_float(np.asarray(audio))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono/convertible audio, got {audio.shape}")

    if int(sr) != int(target_sr):
        g = math.gcd(int(sr), int(target_sr))
        audio = resample_poly(audio, target_sr // g, int(sr) // g).astype(np.float32)

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    if audio.size == 0:
        raise ValueError("Decoded audio is empty.")
    return np.ascontiguousarray(audio)


def load_split(language: str, split: str) -> Dataset:
    """
    Load EVERY row in the locally saved FLEURS split.
    No 30-second filter and no subsampling are performed.
    """
    path = dataset_root(language)
    if not path.exists():
        raise FileNotFoundError(f"Missing FLEURS split: {path}")
    ds = load_fleurs_split(language, split)
    required = {"audio", "transcription"}
    if not required.issubset(ds.column_names):
        raise RuntimeError(
            f"{path} missing columns {sorted(required - set(ds.column_names))}; "
            f"actual={ds.column_names}"
        )
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    if len(ds) == 0:
        raise RuntimeError(f"Empty split: {path}")
    return ds


def dataset_fingerprint(ds: Dataset, split: str) -> str:
    digest = hashlib.sha256()
    digest.update(split.encode("utf-8"))
    digest.update(b"\n")
    digest.update(str(len(ds)).encode("utf-8"))
    digest.update(b"\n")
    for i in range(len(ds)):
        row = ds[i]
        parts = [
            str(i),
            str(row.get("id", "")),
            str(row.get("path", "")),
            str(row.get("num_samples", "")),
            str(row.get("transcription", "")),
        ]
        digest.update("\0".join(parts).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalise_decimal_digits(text: str) -> str:
    out: list[str] = []
    for ch in text:
        try:
            digit = unicodedata.digit(ch)
        except (TypeError, ValueError):
            out.append(ch)
        else:
            out.append(str(int(digit)))
    return "".join(out)


def normalise_text(text: str) -> str:
    text = _normalise_decimal_digits(str(text))
    text = unicodedata.normalize("NFKC", text).casefold()
    out: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in {"P", "S"} or category in {"Cf", "Cc"}:
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int]:
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], op[i][0] = i, "D"
    for j in range(1, m + 1):
        cost[0][j], op[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = reference[i - 1] == hypothesis[j - 1]
            choices = [
                (cost[i - 1][j - 1] + (0 if same else 1), "C" if same else "S"),
                (cost[i - 1][j] + 1, "D"),
                (cost[i][j - 1] + 1, "I"),
            ]
            cost[i][j], op[i][j] = min(choices, key=lambda x: x[0])
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        action = op[i][j]
        if action in {"C", "S"}:
            s += int(action == "S")
            i -= 1
            j -= 1
        elif action == "D":
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def corpus_wer(
    refs: Sequence[str],
    hyps: Sequence[str],
    language: str,
) -> dict[str, Any]:
    s = d = ins = n = 0

    for ref, hyp in zip(refs, hyps, strict=True):
        ref_normalised = normalise_text(ref)
        hyp_normalised = normalise_text(hyp)

        if language == "cmn_hans_cn":
            # Mandarin:
            #   1. Normalize Traditional -> Simplified Chinese.
            #   2. Remove FLEURS annotation whitespace.
            #   3. Compute character-level edit distance (CER).
            #
            # This makes evaluation invariant to Simplified/Traditional
            # script choice and to FLEURS character spacing.
            ref_normalised = ZH_T2S.convert(ref_normalised)
            hyp_normalised = ZH_T2S.convert(hyp_normalised)

            rt = [
                ch
                for ch in ref_normalised
                if not ch.isspace()
            ]
            ht = [
                ch
                for ch in hyp_normalised
                if not ch.isspace()
            ]
        else:
            # Ordinary whitespace-token WER for non-Mandarin languages.
            rt = ref_normalised.split()
            ht = hyp_normalised.split()

        ss, dd, ii = edit_counts(rt, ht)
        s += ss
        d += dd
        ins += ii
        n += len(rt)

    errors = s + d + ins

    return {
        # Keep the historical key name for checkpoint/resume compatibility.
        # For cmn_hans_cn this value is semantically CER.
        "normalised_wer": 100.0 * errors / n if n else 0.0,
        "normalised_reference_words": n,
        "normalised_substitutions": s,
        "normalised_deletions": d,
        "normalised_insertions": ins,
        "metric": "CER" if language == "cmn_hans_cn" else "WER",
        "mandarin_script_normalisation": (
            "OpenCC Traditional-to-Simplified (t2s)"
            if language == "cmn_hans_cn"
            else None
        ),
    }


class RepeatedTokenCycleGuard(StoppingCriteria):
    """
    Stop after an obvious exact short-token cycle has repeated
    six consecutive times.

    Only newly generated tokens are inspected; prompt/template
    tokens are excluded.
    """

    def __init__(
        self,
        prompt_length: int,
        max_cycle_length: int = 8,
        repeats_required: int = 6,
    ) -> None:
        super().__init__()
        self.prompt_length = int(prompt_length)
        self.max_cycle_length = int(max_cycle_length)
        self.repeats_required = int(repeats_required)

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):
        batch_size = int(input_ids.shape[0])

        stop = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )

        generated = input_ids[:, self.prompt_length:]
        generated_length = int(generated.shape[1])

        for cycle_length in range(
            1,
            self.max_cycle_length + 1,
        ):
            needed = (
                cycle_length
                * self.repeats_required
            )

            if generated_length < needed:
                continue

            tail = generated[:, -needed:]
            pattern = tail[:, -cycle_length:]

            expected = pattern.repeat(
                1,
                self.repeats_required,
            )

            stop |= (
                tail == expected
            ).all(dim=1)

        return stop


def build_processor() -> Any:
    return AutoProcessor.from_pretrained(BASE_MODEL)


def build_model(device: torch.device) -> nn.Module:
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model.to(device)


def qwen_prefix_text(processor: Any) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    rendered = processor.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(rendered, (list, tuple)):
        if len(rendered) != 1:
            raise RuntimeError(f"Unexpected prefix batch size: {len(rendered)}")
        rendered = rendered[0]
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("Qwen chat template returned an invalid prefix.")
    return rendered


def qwen_collate(
    processor: Any,
    language: str,
    features: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    if not features:
        raise ValueError("Empty Qwen batch.")

    prefix = qwen_prefix_text(processor)
    prefixes = [prefix] * len(features)
    audios = [decode_audio_value(row["audio"]) for row in features]
    language_name = QWEN_LANGUAGE_NAME[language]
    targets = [
        f"language {language_name}<asr_text>{str(row['transcription'])}"
        for row in features
    ]

    eos = processor.tokenizer.eos_token or ""
    full_texts = [
        pfx + target + eos
        for pfx, target in zip(prefixes, targets, strict=True)
    ]

    full_inputs = processor(
        text=full_texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prefix_inputs = processor(
        text=prefixes,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    labels = full_inputs["input_ids"].clone()
    prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()

    for i, prefix_len in enumerate(prefix_lens):
        prefix_len = int(prefix_len)
        if prefix_len <= 0 or prefix_len >= labels.shape[1]:
            raise RuntimeError(
                f"Invalid Qwen prefix length example={i}: "
                f"{prefix_len}/{labels.shape[1]}"
            )
        labels[i, :prefix_len] = -100

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == int(pad_id)] = -100

    if not torch.any(labels.ne(-100)):
        raise RuntimeError("No supervised Qwen tokens.")

    full_inputs["labels"] = labels
    return dict(full_inputs)


def make_loader(
    ds: Dataset,
    processor: Any,
    language: str,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda rows: qwen_collate(processor, language, rows),
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device, non_blocking=True)
            if value.is_floating_point():
                value = value.to(dtype=torch.bfloat16)
        result[key] = value
    return result


def collect_attention_targets(model: nn.Module) -> dict[str, nn.Linear]:
    modules = dict(model.named_modules())
    targets: dict[str, nn.Linear] = {}
    leaves = {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}
    for name, module in modules.items():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in leaves:
            continue
        parent_name = name.rsplit(".", 1)[0]
        parent = modules.get(parent_name)
        if parent is None:
            continue
        if "attention" not in parent.__class__.__name__.casefold():
            continue
        targets[name] = module

    if len(targets) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGETS} attention targets, found {len(targets)}."
        )
    return targets


def get_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class FCCALinear(nn.Module):
    """
    Frozen base + frozen P/Q + trainable R:
        Delta W = P R Q^T

    Works directly for rectangular Qwen matrices.
    """

    def __init__(
        self,
        base: nn.Linear,
        P: torch.Tensor,
        Q: torch.Tensor,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        P = torch.as_tensor(P)
        Q = torch.as_tensor(Q)
        if P.ndim != 2 or Q.ndim != 2:
            raise ValueError("P and Q must be matrices.")
        if P.shape[0] != base.out_features:
            raise ValueError(
                f"P rows {P.shape[0]} != out_features {base.out_features}"
            )
        if Q.shape[0] != base.in_features:
            raise ValueError(
                f"Q rows {Q.shape[0]} != in_features {base.in_features}"
            )
        if P.shape[1] != Q.shape[1]:
            raise ValueError(f"P/Q rank mismatch {P.shape} vs {Q.shape}")
        if int(P.shape[1]) <= 0:
            raise ValueError("Rank-zero matrices must not be wrapped.")

        self.rank = int(P.shape[1])
        self.scale = float(scale)

        self.register_buffer(
            "fcca_P",
            P.to(device=base.weight.device, dtype=torch.float32).contiguous(),
        )
        self.register_buffer(
            "fcca_Q",
            Q.to(device=base.weight.device, dtype=torch.float32).contiguous(),
        )
        self.fcca_core = nn.Parameter(
            torch.zeros(
                self.rank,
                self.rank,
                dtype=torch.float32,
                device=base.weight.device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        z = x.float() @ self.fcca_Q
        z = z @ self.fcca_core.T
        z = z @ self.fcca_P.T
        return base_out + (self.scale * z).to(base_out.dtype)


def basis_path(language: str, method: str) -> Path:
    return (
        RESULT_ROOT
        / language
        / METHOD_DIR[method]
        / "calibration"
        / BASIS_FILENAME[method]
    )


def _basis_rank(entry: Mapping[str, Any]) -> int:
    if "rank" in entry:
        return int(entry["rank"])
    P = torch.as_tensor(entry["P"])
    Q = torch.as_tensor(entry["Q"])
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != Q.shape[1]:
        raise RuntimeError("Cannot infer valid basis rank.")
    return int(P.shape[1])


def load_and_attach_bases(
    model: nn.Module,
    language: str,
    method: str,
    path_override: Path | None = None,
) -> tuple[list[str], dict[str, Any], Path, dict[str, int]]:
    path = (
        Path(path_override)
        if path_override is not None
        else basis_path(language, method)
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed basis: {path}\n"
            "This runner NEVER recalibrates or rebuilds bases."
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    bases = payload.get("bases")
    if not isinstance(bases, dict):
        raise RuntimeError(f"Basis payload lacks 'bases': {path}")

    targets = collect_attention_targets(model)
    if set(targets) != set(bases):
        missing = sorted(set(targets) - set(bases))[:10]
        extra = sorted(set(bases) - set(targets))[:10]
        raise RuntimeError(
            f"Basis/model target mismatch. missing={missing} extra={extra}"
        )

    if payload.get("language") not in {None, language}:
        raise RuntimeError(
            f"Basis language mismatch: wanted {language}, got {payload.get('language')}"
        )

    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    rank_map: dict[str, int] = {}
    for name in sorted(targets):
        entry = bases[name]
        if "P" not in entry or "Q" not in entry:
            raise RuntimeError(f"Basis entry missing P/Q: {name}")

        P = torch.as_tensor(entry["P"])
        Q = torch.as_tensor(entry["Q"])
        rank = _basis_rank(entry)
        rank_map[name] = rank

        if rank == 0:
            continue

        if P.shape[1] != rank or Q.shape[1] != rank:
            raise RuntimeError(
                f"Stored rank disagrees with shapes at {name}: "
                f"rank={rank}, P={tuple(P.shape)}, Q={tuple(Q.shape)}"
            )

        parent, leaf = get_parent(model, name)
        base = getattr(parent, leaf)
        setattr(parent, leaf, FCCALinear(base, P=P, Q=Q, scale=1.0))

    used_budget = sum(r * r for r in rank_map.values())

    if method in {"fcca", "asym_hard", "asym_shared_private"}:
        if any(r != REFERENCE_RANK for r in rank_map.values()):
            bad = [(n, r) for n, r in rank_map.items() if r != REFERENCE_RANK][:10]
            raise RuntimeError(f"Expected rank-128 everywhere; bad={bad}")
        if used_budget != REFERENCE_CORE_BUDGET:
            raise RuntimeError(
                f"Core budget mismatch {used_budget} != {REFERENCE_CORE_BUDGET}"
            )

    if method == "adaptive_fcca":
        expected_budget = payload.get(
            "trainable_core_budget",
            payload.get(
                "actual_core_budget",
                payload.get(
                    "reference_core_budget",
                    payload.get("reference_budget"),
                ),
            ),
        )
        if expected_budget is not None and int(expected_budget) != used_budget:
            raise RuntimeError(
                f"Adaptive core budget mismatch: bases={used_budget}, "
                f"payload={expected_budget}"
            )

    print(
        f"[basis] method={method} path={path} targets={len(rank_map)} "
        f"core_parameters={used_budget:,} "
        f"rank_min={min(rank_map.values())} rank_max={max(rank_map.values())}",
        flush=True,
    )
    return sorted(targets), payload, path, rank_map


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def load_trainable_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    named = dict(model.named_parameters())
    current = {name for name, p in named.items() if p.requires_grad}
    if current != set(state):
        missing = sorted(current - set(state))[:10]
        extra = sorted(set(state) - current)[:10]
        raise RuntimeError(
            f"Trainable-state key mismatch. missing={missing} extra={extra}"
        )
    with torch.no_grad():
        for name, tensor in state.items():
            named[name].copy_(
                tensor.to(device=named[name].device, dtype=named[name].dtype)
            )


@torch.inference_mode()
def transcribe(
    model: nn.Module,
    processor: Any,
    audio: np.ndarray,
    language: str,
    device: torch.device,
) -> str:
    inputs = processor.apply_transcription_request(
        audio=audio,
        language=QWEN_LANGUAGE_NAME[language],
    )
    moved: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype=torch.bfloat16)
        moved[key] = value

    output_ids = model.generate(
        **moved,
        max_new_tokens=512,
        do_sample=False,
        stopping_criteria=StoppingCriteriaList(
            [
                RepeatedTokenCycleGuard(
                    prompt_length=int(
                        moved["input_ids"].shape[1]
                    ),
                    max_cycle_length=8,
                    repeats_required=6,
                )
            ]
        ),
    )
    prefix_len = int(moved["input_ids"].shape[1])
    generated = output_ids[:, prefix_len:]
    decoded = processor.decode(generated, return_format="transcription_only")
    return str(decoded[0]).strip()


@torch.inference_mode()
def evaluate_wer(
    model: nn.Module,
    processor: Any,
    ds: Dataset,
    language: str,
    device: torch.device,
    output_csv: Path,
) -> dict[str, Any]:
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    rows: list[dict[str, Any]] = []
    refs: list[str] = []
    hyps: list[str] = []
    started = time.time()

    for i, row in enumerate(ds, start=1):
        audio = decode_audio_value(row["audio"])
        ref = str(row["transcription"])
        hyp = transcribe(model, processor, audio, language, device)
        refs.append(ref)
        hyps.append(hyp)
        rows.append(
            {
                "index": i - 1,
                "id": row.get("id", i - 1),
                "reference_original": ref,
                "hypothesis_original": hyp,
                "reference_normalised": (
                    ZH_T2S.convert(normalise_text(ref))
                    if language == "cmn_hans_cn"
                    else normalise_text(ref)
                ),
                "hypothesis_normalised": (
                    ZH_T2S.convert(normalise_text(hyp))
                    if language == "cmn_hans_cn"
                    else normalise_text(hyp)
                ),
            }
        )
        if i % 50 == 0 or i == len(ds):
            print(
                f"[{'CER' if language == 'cmn_hans_cn' else 'WER'}] "
                f"{language}: {i}/{len(ds)}",
                flush=True,
            )

    metrics = corpus_wer(refs, hyps, language)
    metrics.update(
        {
            "language": language,
            "examples": len(ds),
            "runtime_seconds": time.time() - started,
        }
    )
    write_csv(output_csv, rows)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return metrics


@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    processor: Any,
    ds: Dataset,
    language: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    loader = make_loader(
        ds,
        processor,
        language,
        batch_size=batch_size,
        shuffle=False,
        generator=None,
    )

    weighted = 0.0
    tokens = 0
    batches = 0

    for batch in loader:
        moved = move_batch(batch, device)
        labels = moved["labels"]
        count = int(labels.ne(-100).sum().item())
        if count <= 0:
            raise RuntimeError("Validation batch has zero supervised tokens.")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**moved)
            loss = out.loss

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid validation loss: {loss}")

        weighted += float(loss.item()) * count
        tokens += count
        batches += 1

    return {
        "validation_loss": weighted / tokens,
        "validation_supervised_tokens": tokens,
        "validation_loss_batches": batches,
    }


def checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints"


def latest_checkpoint_path(run_dir: Path) -> Path:
    return checkpoint_dir(run_dir) / "latest_checkpoint.pt"


def best_checkpoint_path(run_dir: Path) -> Path:
    return checkpoint_dir(run_dir) / "best_checkpoint.pt"


def latest_checkpoint(run_dir: Path) -> Path:
    path = latest_checkpoint_path(run_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"No latest checkpoint under {checkpoint_dir(run_dir)}"
        )

    return path


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    best_wer: float,
    best_epoch: int,
    bad_epochs: int,
    train_generator: torch.Generator,
    config: Mapping[str, Any],
    basis_path_value: Path,
    basis_metadata: Mapping[str, Any],
    rank_map: Mapping[str, int],
    dataset_fingerprints: Mapping[str, str],
    basis_sha256: str,
    task_stats_path_value: Path,
    task_stats_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "qwen3_fcca_full_resume_epoch_v1",
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "global_step": int(global_step),

        # COMPLETE wrapped model state:
        # frozen pretrained model + frozen FCCA P/Q + trainable cores.
        "full_model_state_dict":
            complete_model_state(model),

        # Retain trainable-only state as a convenient audit view.
        "trainable_state_dict":
            trainable_state(model),

        # Exact basis payload allows recovery even if the external
        # calibration artifact is later replaced/deleted.
        "basis_payload":
            basis_metadata,

        # Full optimiser/scheduler continuation state.
        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        # Exact stochastic continuation at the next epoch boundary.
        "rng_state": capture_rng_state(),
        "train_dataloader_generator_state": train_generator.get_state(),

        # Validation/selection state.
        "history": history,
        "best_validation_wer": float(best_wer),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),

        # Resume contracts. Frozen pretrained weights and P/Q are not duplicated.
        "config": dict(config),
        "base_model": BASE_MODEL,
        "basis_path":
            str(basis_path_value.resolve()),

        "basis_sha256":
            str(basis_sha256),

        "task_stats_path":
            str(task_stats_path_value.resolve()),

        "task_stats_sha256":
            str(task_stats_sha256),

        "basis_payload_method":
            basis_metadata.get("method"),

        "basis_payload_schema":
            basis_metadata.get("schema"),

        "rank_map":
            dict(rank_map),

        "dataset_fingerprints":
            dict(dataset_fingerprints),
        "saved_at_unix": time.time(),
    }


def validate_resume_contract(
    ckpt: Mapping[str, Any],
    config: Mapping[str, Any],
    basis_path_value: Path,
    rank_map: Mapping[str, int],
    dataset_fingerprints: Mapping[str, str],
    basis_sha256: str,
    task_stats_path_value: Path,
    task_stats_sha256: str,
) -> None:
    if ckpt.get("schema") != "qwen3_fcca_full_resume_epoch_v1":
        raise RuntimeError(f"Unsupported checkpoint schema: {ckpt.get('schema')}")

    old = ckpt["config"]
    immutable = [
        "language",
        "method",
        "learning_rate",
        "effective_batch_size",
        "microbatch",
        "gradient_accumulation_steps",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
        "max_epochs",
        "updates_per_epoch",
        "planned_total_optimizer_steps",
        "seed",
    ]
    for key in immutable:
        if old.get(key) != config.get(key):
            raise RuntimeError(
                f"Resume contract mismatch for {key}: "
                f"checkpoint={old.get(key)!r}, current={config.get(key)!r}"
            )

    # The trial-local basis snapshot must be byte-identical to
    # the one stored at the previous epoch.
    if ckpt.get("basis_sha256") != basis_sha256:
        raise RuntimeError(
            "Resume basis SHA mismatch:\n"
            f"checkpoint={ckpt.get('basis_sha256')}\n"
            f"current={basis_sha256}"
        )

    if ckpt.get("task_stats_sha256") != task_stats_sha256:
        raise RuntimeError(
            "Resume task-stat SHA mismatch:\n"
            f"checkpoint={ckpt.get('task_stats_sha256')}\n"
            f"current={task_stats_sha256}"
        )

    if dict(ckpt["rank_map"]) != dict(rank_map):
        raise RuntimeError("Resume rank allocation/basis rank map changed.")

    if dict(ckpt["dataset_fingerprints"]) != dict(dataset_fingerprints):
        raise RuntimeError("Resume dataset fingerprint changed.")


def run_dir_for(language: str, method: str, trial_name: str) -> Path:
    return (
        RESULT_ROOT
        / language
        / METHOD_DIR[method]
        / "training"
        / trial_name
    )


def run_training(args: argparse.Namespace) -> None:
    device = require_cuda(args.device)
    set_seed(SEED)

    language = args.language
    method = args.method

    if args.effective_batch_size % args.microbatch != 0:
        raise ValueError(
            "effective_batch_size must be divisible by microbatch."
        )

    grad_acc = args.effective_batch_size // args.microbatch
    run_dir = run_dir_for(language, method, args.trial_name)

    if args.resume is None:
        try:
            run_dir_nonempty = (
                run_dir.exists()
                and any(run_dir.iterdir())
            )
        except FileNotFoundError:
            run_dir_nonempty = False

        if run_dir_nonempty:
            raise RuntimeError(
                f"Refusing fresh run into non-empty directory: {run_dir}. "
                "Use --resume latest to continue it, or choose a new --trial-name."
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir(run_dir).mkdir(parents=True, exist_ok=True)

    train_ds = load_split(language, "train")
    val_ds = load_split(language, "validation")
    test_ds = load_split(language, "test")

    fingerprints = {
        "train": dataset_fingerprint(train_ds, "train"),
        "validation": dataset_fingerprint(val_ds, "validation"),
        "test": dataset_fingerprint(test_ds, "test"),
    }

    processor = build_processor()

    # -------------------------------------------------------------
    # Resolve resume checkpoint BEFORE constructing the FCCA model.
    # -------------------------------------------------------------
    resume_path = None
    resume_ckpt = None

    if args.resume is not None:
        resume_path = (
            latest_checkpoint(run_dir)
            if args.resume == "latest"
            else Path(args.resume)
        )

        print(
            f"[resume] reading checkpoint metadata {resume_path}",
            flush=True,
        )

        resume_ckpt = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )

    # -------------------------------------------------------------
    # Every trial owns an immutable basis snapshot.
    #
    # Fresh run:
    #     copy canonical precomputed basis -> trial directory
    #
    # Resume:
    #     if snapshot vanished, reconstruct it from basis_payload
    #     embedded in the epoch checkpoint.
    # -------------------------------------------------------------
    snapshot_path = (
        run_dir
        / "frozen_basis_snapshot.pt"
    )

    if resume_ckpt is None:
        source_basis = basis_path(
            language,
            method,
        )

        if not source_basis.exists():
            raise FileNotFoundError(
                f"Missing precomputed basis: {source_basis}"
            )

        source_payload = torch.load(
            source_basis,
            map_location="cpu",
            weights_only=False,
        )

        atomic_torch_save(
            snapshot_path,
            source_payload,
        )

        print(
            f"[basis snapshot] {source_basis} -> {snapshot_path}",
            flush=True,
        )

    elif not snapshot_path.exists():
        embedded = resume_ckpt.get(
            "basis_payload"
        )

        if not isinstance(
            embedded,
            Mapping,
        ):
            raise RuntimeError(
                "Resume checkpoint lacks embedded basis_payload "
                "and frozen_basis_snapshot.pt is missing."
            )

        atomic_torch_save(
            snapshot_path,
            embedded,
        )

        print(
            "[resume] reconstructed frozen_basis_snapshot.pt "
            "from checkpoint.",
            flush=True,
        )

    model = build_model(device)

    targets, basis_payload, used_basis_path, rank_map = (
        load_and_attach_bases(
            model,
            language,
            method,
            path_override=snapshot_path,
        )
    )

    basis_sha256_value = file_sha256(
        used_basis_path
    )

    task_stats_path_value = recorded_task_stats_path(
        language,
        basis_payload,
    )

    # Task stats are provenance-only during training.
    # They are never used to rebuild the basis.
    if task_stats_path_value.exists():
        task_stats_sha256_value = file_sha256(
            task_stats_path_value
        )
    elif resume_ckpt is not None:
        task_stats_sha256_value = str(
            resume_ckpt["task_stats_sha256"]
        )

        print(
            "[resume warning] source task-stat file is absent; "
            "continuation remains exact because the checkpoint "
            "contains the exact basis and complete model state.",
            flush=True,
        )
    else:
        raise FileNotFoundError(
            f"Fresh run source task stats missing: "
            f"{task_stats_path_value}"
        )

    print(
        f"[provenance] basis_sha256={basis_sha256_value}",
        flush=True,
    )

    print(
        f"[provenance] task_stats={task_stats_path_value}",
        flush=True,
    )

    print(
        f"[provenance] task_stats_sha256="
        f"{task_stats_sha256_value}",
        flush=True,
    )

    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("No trainable FCCA cores.")

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = make_loader(
        train_ds,
        processor,
        language,
        batch_size=args.microbatch,
        shuffle=True,
        generator=train_generator,
    )

    updates_per_epoch = math.ceil(len(train_loader) / grad_acc)
    total_steps = updates_per_epoch * args.max_epochs
    warmup_steps = int(round(total_steps * args.warmup_ratio))

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    config = {
        "language": language,
        "language_name": LANGUAGES[language]["name"],
        "qwen_forced_language": QWEN_LANGUAGE_NAME[language],
        "method": method,
        "base_model": BASE_MODEL,
        "learning_rate": float(args.learning_rate),
        "effective_batch_size": int(args.effective_batch_size),
        "microbatch": int(args.microbatch),
        "gradient_accumulation_steps": int(grad_acc),
        "eval_batch": int(args.eval_batch),
        "warmup_ratio": float(args.warmup_ratio),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "seed": SEED,
        "updates_per_epoch": updates_per_epoch,
        "planned_total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "train_examples": len(train_ds),
        "validation_examples": len(val_ds),
        "test_examples": len(test_ds),
        "target_count": len(targets),
        "trainable_parameters": sum(p.numel() for p in params),
        "basis_path":
            str(used_basis_path),

        "basis_sha256":
            basis_sha256_value,

        "task_stats_path":
            str(task_stats_path_value),

        "task_stats_sha256":
            task_stats_sha256_value,

        "rank_min":
            min(rank_map.values()),
        "rank_max": max(rank_map.values()),
        "core_parameter_budget": sum(r * r for r in rank_map.values()),
        "full_split_policy": (
            "Use every row present in week8/data/<language>/{train,validation,test}; "
            "no duration filtering and no subsampling in this runner."
        ),
        "test_policy": (
            "Test is not touched by train. Run final-test separately after all "
            "resume/selection decisions are finished."
        ),
    }

    atomic_json(run_dir / "run_config.json", config)
    atomic_json(run_dir / "dataset_fingerprints.json", fingerprints)

    start_epoch = 1
    global_step = 0
    history: list[dict[str, Any]] = []
    best_wer = float("inf")
    best_epoch = 0
    bad_epochs = 0
    resumed = False

    if args.resume is not None:
        if resume_ckpt is None:
            raise RuntimeError(
                "Internal error: resume checkpoint was not preloaded."
            )

        ckpt = resume_ckpt

        print(
            f"[resume] restoring COMPLETE state from {resume_path}",
            flush=True,
        )

        validate_resume_contract(
            ckpt,
            config,
            used_basis_path,
            rank_map,
            fingerprints,
            basis_sha256_value,
            task_stats_path_value,
            task_stats_sha256_value,
        )

        full_state = ckpt.get(
            "full_model_state_dict"
        )

        if not isinstance(
            full_state,
            Mapping,
        ):
            raise RuntimeError(
                "Checkpoint predates full-model resumability: "
                "missing full_model_state_dict."
            )

        model.load_state_dict(
            full_state,
            strict=True,
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Restore only after all construction/loading that may consume RNG.
        restore_rng_state(ckpt["rng_state"])
        train_generator.set_state(ckpt["train_dataloader_generator_state"])

        start_epoch = int(ckpt["next_epoch"])
        global_step = int(ckpt["global_step"])
        history = list(ckpt["history"])
        best_wer = float(ckpt["best_validation_wer"])
        best_epoch = int(ckpt["best_epoch"])
        bad_epochs = int(ckpt["bad_epochs"])
        resumed = True

        print(
            f"[resume] epoch={ckpt['epoch']} -> next={start_epoch}; "
            f"global_step={global_step}; best={best_wer:.4f}%@{best_epoch}",
            flush=True,
        )

    # Exact uninterrupted-run semantics.
    # best_wer, best_epoch and bad_epochs have already been restored.
    early_stopping_enabled = True

    if resumed:
        print(
            "[resume] early-stopping state restored exactly: "
            f"bad_epochs={bad_epochs}/{args.patience}",
            flush=True,
        )

    if start_epoch > args.max_epochs:
        print(
            f"[train] already at epoch {start_epoch - 1}; "
            f"max_epochs={args.max_epochs}. Nothing to train.",
            flush=True,
        )
        return

    print("=" * 100, flush=True)
    print(
        f"[TRAIN] language={language} method={method} "
        f"epochs={start_epoch}..{args.max_epochs} "
        f"patience={'OFF (resume)' if resumed else args.patience}",
        flush=True,
    )
    print(
        f"[DATA] train={len(train_ds)} validation={len(val_ds)} test={len(test_ds)} "
        "(NO duration filter)",
        flush=True,
    )
    print(
        f"[OPT] lr={args.learning_rate} EBS={args.effective_batch_size} "
        f"microbatch={args.microbatch} grad_acc={grad_acc} "
        f"updates/epoch={updates_per_epoch} total_steps={total_steps} "
        f"warmup_steps={warmup_steps}",
        flush=True,
    )
    print("=" * 100, flush=True)

    started = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            moved = move_batch(batch, device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**moved)
                loss = outputs.loss

            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss epoch={epoch} batch={batch_index}: {loss}"
                )

            running_loss += float(loss.detach().item())
            batches += 1

            group_start = ((batch_index - 1) // grad_acc) * grad_acc
            group_size = min(grad_acc, len(train_loader) - group_start)
            (loss / float(group_size)).backward()

            if batch_index % grad_acc == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if batch_index % 100 == 0 or batch_index == len(train_loader):
                print(
                    f"[train] epoch={epoch:02d} "
                    f"batch={batch_index}/{len(train_loader)} "
                    f"loss={running_loss / batches:.6f} "
                    f"global_step={global_step}",
                    flush=True,
                )

        loss_metrics = evaluate_loss(
            model,
            processor,
            val_ds,
            language,
            device,
            args.eval_batch,
        )

        prediction_path = (
            run_dir / f"validation_predictions_epoch_{epoch:02d}.csv"
        )
        wer_metrics = evaluate_wer(
            model,
            processor,
            val_ds,
            language,
            device,
            prediction_path,
        )

        current_wer = float(wer_metrics["normalised_wer"])
        improved = current_wer < best_wer - 1e-12

        if improved:
            best_wer = current_wer
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(batches, 1),
            "validation_loss": loss_metrics["validation_loss"],
            "validation_normalised_wer": current_wer,
            "validation_substitutions": wer_metrics["normalised_substitutions"],
            "validation_deletions": wer_metrics["normalised_deletions"],
            "validation_insertions": wer_metrics["normalised_insertions"],
            "validation_reference_words": wer_metrics["normalised_reference_words"],
            "learning_rate_after_epoch": float(optimizer.param_groups[0]["lr"]),
            "best_so_far": bool(improved),
            "best_epoch_so_far": best_epoch,
            "best_wer_so_far": best_wer,
            "bad_epochs": bad_epochs,
            "early_stopping_enabled": early_stopping_enabled,
        }
        history.append(row)
        write_csv(run_dir / "training_history.csv", history)

        # Save a FULL RESUME checkpoint after EVERY completed epoch.
        ckpt = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            history=history,
            best_wer=best_wer,
            best_epoch=best_epoch,
            bad_epochs=bad_epochs,
            train_generator=train_generator,
            config=config,
            basis_path_value=used_basis_path,
            basis_metadata=basis_payload,
            rank_map=rank_map,
            dataset_fingerprints=fingerprints,
            basis_sha256=basis_sha256_value,
            task_stats_path_value=task_stats_path_value,
            task_stats_sha256=task_stats_sha256_value,
        )
        latest_path = latest_checkpoint_path(run_dir)

        atomic_torch_save(
            latest_path,
            ckpt,
        )

        if improved:
            atomic_torch_save(
                best_checkpoint_path(run_dir),
                ckpt,
            )

        atomic_json(
            run_dir / "best_validation.json",
            {
                "best_epoch": best_epoch,
                "best_validation_normalised_wer": best_wer,
                "best_checkpoint": str(
                    best_checkpoint_path(run_dir)
                ),
                "latest_checkpoint": str(
                    latest_path
                ),
            },
        )

        print(
            f"[epoch] {epoch:02d} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['validation_loss']:.6f} "
            f"{'CER' if language == 'cmn_hans_cn' else 'WER'}="
            f"{current_wer:.4f}% best={best_wer:.4f}%@{best_epoch} "
            f"latest_checkpoint={latest_path}",
            flush=True,
        )

        if (
            early_stopping_enabled
            and args.patience > 0
            and bad_epochs >= args.patience
        ):
            atomic_json(
                run_dir / "EARLY_STOPPED.json",
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_normalised_wer": best_wer,
                    "bad_epochs": bad_epochs,
                    "resume_instruction": (
                        "Run the identical command with --resume latest. "
                        "Patience will automatically be disabled."
                    ),
                },
            )
            print(
                f"[early-stop] patience={args.patience} reached at epoch {epoch}. "
                "Resume with --resume latest to continue without patience.",
                flush=True,
            )
            break

    completed_epoch = int(history[-1]["epoch"]) if history else start_epoch - 1

    if completed_epoch >= args.max_epochs:
        (run_dir / "TRAIN_COMPLETE").write_text("complete\n", encoding="utf-8")

    atomic_json(
        run_dir / "training_status.json",
        {
            "completed_epoch": completed_epoch,
            "max_epochs": args.max_epochs,
            "best_epoch": best_epoch,
            "best_validation_normalised_wer": best_wer,
            "resumed": resumed,
            "early_stopping_enabled": early_stopping_enabled,
            "runtime_seconds_this_invocation": time.time() - started,
            "test_has_not_been_run_by_train": True,
        },
    )


def run_final_test(args: argparse.Namespace) -> None:
    """
    Evaluate the FULL test split once using the best validation epoch.
    Separated from train so test remains untouched until continuation choices are done.
    """
    device = require_cuda(args.device)
    set_seed(SEED)

    run_dir = run_dir_for(args.language, args.method, args.trial_name)
    final_marker = run_dir / "FINAL_TEST_COMPLETE"

    if final_marker.exists():
        raise RuntimeError(
            f"Final test already completed: {final_marker}. "
            "Refusing to re-run/overwrite."
        )

    best_path = run_dir / "best_validation.json"
    if not best_path.exists():
        raise FileNotFoundError(best_path)

    best = json.loads(best_path.read_text(encoding="utf-8"))
    best_epoch = int(best["best_epoch"])
    ckpt_path = best_checkpoint_path(run_dir)

    processor = build_processor()
    model = build_model(device)
    _, _, used_basis_path, rank_map = load_and_attach_bases(
        model,
        args.language,
        args.method,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_trainable_state(model, ckpt["trainable_state_dict"])

    test_ds = load_split(args.language, "test")
    current_fp = dataset_fingerprint(test_ds, "test")
    if current_fp != ckpt["dataset_fingerprints"]["test"]:
        raise RuntimeError("Test split fingerprint changed since training.")

    metrics = evaluate_wer(
        model,
        processor,
        test_ds,
        args.language,
        device,
        run_dir / "test_predictions.csv",
    )
    metrics.update(
        {
            "selected_by": "minimum validation normalised WER",
            "best_epoch": best_epoch,
            "best_validation_normalised_wer": float(
                best["best_validation_normalised_wer"]
            ),
            "checkpoint": str(ckpt_path),
            "basis_path": str(used_basis_path),
            "core_parameter_budget": sum(r * r for r in rank_map.values()),
        }
    )

    atomic_json(run_dir / "test_metrics.json", metrics)
    final_marker.write_text("complete\n", encoding="utf-8")

    print(
        f"[FINAL TEST] {args.language}/{args.method} "
        f"epoch={best_epoch} "
        f"{'CER' if args.language == 'cmn_hans_cn' else 'WER'}="
        f"{metrics['normalised_wer']:.4f}%",
        flush=True,
    )


def preflight(args: argparse.Namespace) -> None:
    print(f"project_root={PROJECT_ROOT}")
    print(f"data_root={DATA_ROOT}")
    print(f"result_root={RESULT_ROOT}")
    print(f"base_model={BASE_MODEL}")
    print(f"language={args.language}")
    print(f"method={args.method}")

    for split in ("train", "validation", "test"):
        ds = load_split(args.language, split)
        print(
            f"{split:10s} n={len(ds):5d} "
            f"fingerprint={dataset_fingerprint(ds, split)[:16]}..."
        )

    path = basis_path(args.language, args.method)
    print(f"basis={path}")
    if not path.exists():
        raise FileNotFoundError(path)

    device = require_cuda(args.device)
    _ = build_processor()
    model = build_model(device)
    targets, payload, path, rank_map = load_and_attach_bases(
        model,
        args.language,
        args.method,
    )

    print(f"targets={len(targets)}")
    print(f"basis_method={payload.get('method')}")
    print(f"basis_schema={payload.get('schema')}")
    print(f"rank_min={min(rank_map.values())}")
    print(f"rank_max={max(rank_map.values())}")
    print(f"core_budget={sum(r*r for r in rank_map.values()):,}")
    print("preflight=OK")


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--language", choices=sorted(LANGUAGES), required=True)
    p.add_argument("--method", choices=sorted(METHOD_DIR), required=True)
    p.add_argument("--trial-name", default="trial_0000")
    p.add_argument("--device", default="cuda:0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-ASR FCCA-family runner using precomputed Week-8 bases. "
            "Stores a full resumable checkpoint after every epoch."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight")
    add_common(p)

    p = sub.add_parser("train")
    add_common(p)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--effective-batch-size", type=int, required=True)
    p.add_argument("--microbatch", type=int, default=DEFAULT_MICROBATCH)
    p.add_argument("--eval-batch", type=int, default=DEFAULT_EVAL_BATCH)
    p.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    p.add_argument(
        "--resume",
        default=None,
        help=(
            "Use 'latest' or an explicit checkpoint path. "
            "Restores complete model state, optimiser, scheduler, "
            "all RNG state, dataloader RNG, validation-selection state "
            "and early-stopping state exactly."
        ),
    )

    p = sub.add_parser("final-test")
    add_common(p)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "final-test":
        run_final_test(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
