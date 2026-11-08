from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from opencc import OpenCC
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from scipy.signal import resample_poly
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

LANGUAGE_ID = None
LANGUAGE_NAME = None
WHISPER_LANGUAGE = None

DATA_ROOT = None
RESULT_ROOT = None
FCCA_BASIS_PATH = None

BASE_MODEL = "openai/whisper-medium"

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30.0
MAX_AUDIO_SAMPLES = int(
    SAMPLE_RATE * MAX_AUDIO_SECONDS
)

SEED = 2026

# ------------------------------------------------------------
# GLOBAL TRAINING POLICY
# ------------------------------------------------------------

MAX_EPOCHS = 12
PATIENCE = 5

DEFAULT_MICROBATCH = 4
EVAL_BATCH_SIZE = 4

WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

WHISPER_TASK = "transcribe"

# Evaluation-only canonicalisation for Mandarin.
# Training transcripts are NOT modified.
MANDARIN_T2S = OpenCC("t2s")

LANGUAGE_CONFIGS = {
    "af_za": ("Afrikaans", "af"),
    "as_in": ("Assamese", "as"),
    "ast_es": ("Asturian", "es"),
    "bn_in": ("Bengali", "bn"),
    "ca_es": ("Catalan", "ca"),
    "ckb_iq": ("Central Kurdish", "fa"),
    "cs_cz": ("Czech", "cs"),
    "en_us": ("English", "en"),
    "es_419": ("Spanish", "es"),
    "fa_ir": ("Persian", "fa"),
    "kk_kz": ("Kazakh", "kk"),
    "ky_kg": ("Kyrgyz", "kk"),
    "nb_no": ("Norwegian", "no"),
    "nl_nl": ("Dutch", "nl"),
    "or_in": ("Odia", "or"),
    "pl_pl": ("Polish", "pl"),
    "tg_tj": ("Tajik", "tg"),
    "th_th": ("Thai", "th"),
    "uz_uz": ("Uzbek", "uz"),
    "xh_za": ("Xhosa", "xh"),
}


def configure_language(
    language: str,
    whisper_language: str | None = None,
):
    global LANGUAGE_ID
    global LANGUAGE_NAME
    global WHISPER_LANGUAGE
    global DATA_ROOT
    global RESULT_ROOT
    global FCCA_BASIS_PATH
    global SPANISH_SUBSET_PATH

    LANGUAGE_ID = language

    default_name, default_token = (
        LANGUAGE_CONFIGS.get(
            language,
            (
                language,
                language.split("_", 1)[0],
            ),
        )
    )

    LANGUAGE_NAME = default_name

    WHISPER_LANGUAGE = (
        whisper_language
        if whisper_language is not None
        else default_token
    )

    from scripts_fleurs.fleurs_paths import dataset_root
    DATA_ROOT = dataset_root(language)

    RESULT_ROOT = (
        PROJECT_ROOT / "outputs" / "new_results"
        / language
    )

    FCCA_BASIS_PATH = (
        RESULT_ROOT
        / "fcca"
        / "calibration"
        / "calibration_256_all_attention_v2_means.pt"
    )

    SPANISH_SUBSET_PATH = (
        RESULT_ROOT
        / "_data"
        / "train_7.483h_indices.json"
    )


LORA_RANK = 128
LORA_ALPHA = 256.0
LORA_DROPOUT = 0.05

FCCA_RANK = 128
FCCA_SCALE = 1.0


SPANISH_TRAIN_TARGET_HOURS = 7.483

SPANISH_TRAIN_TARGET_SAMPLES = int(
    round(
        SPANISH_TRAIN_TARGET_HOURS
        * 3600.0
        * SAMPLE_RATE
    )
)

SPANISH_SUBSET_PATH = None



def collect_attention_targets(model):
    modules = dict(model.named_modules())
    targets = {}

    leaves = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
    }

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

    if not targets:
        raise RuntimeError(
            "No Whisper attention projection targets found."
        )

    return targets


def get_parent(model, name):
    parts = name.split(".")
    parent = model

    for part in parts[:-1]:
        parent = getattr(parent, part)

    return parent, parts[-1]


class LoRALinear(nn.Module):
    def __init__(
        self,
        base,
        rank=128,
        alpha=256.0,
        dropout=0.05,
    ):
        super().__init__()

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.dropout = (
            nn.Dropout(float(dropout))
            if dropout > 0
            else nn.Identity()
        )

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                dtype=torch.float32,
                device=base.weight.device,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                dtype=torch.float32,
                device=base.weight.device,
            )
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

    def forward(self, x):
        base_out = self.base(x)

        z = F.linear(
            self.dropout(x).float(),
            self.lora_A,
        )

        z = F.linear(
            z,
            self.lora_B,
        )

        return (
            base_out
            + (
                self.scaling * z
            ).to(base_out.dtype)
        )


class FCCALinear(nn.Module):
    def __init__(
        self,
        base,
        P,
        Q,
        scale=1.0,
    ):
        super().__init__()

        self.base = base

        for p in self.base.parameters():
            p.requires_grad = False

        if (
            P.shape[0] != base.out_features
            or Q.shape[0] != base.in_features
            or P.shape[1] != Q.shape[1]
        ):
            raise RuntimeError(
                "FCCA basis shape mismatch."
            )

        self.register_buffer(
            "fcca_P",
            P.to(
                device=base.weight.device,
                dtype=torch.float32,
            ).contiguous(),
        )

        self.register_buffer(
            "fcca_Q",
            Q.to(
                device=base.weight.device,
                dtype=torch.float32,
            ).contiguous(),
        )

        self.fcca_core = nn.Parameter(
            torch.zeros(
                P.shape[1],
                P.shape[1],
                dtype=torch.float32,
                device=base.weight.device,
            )
        )

        self.scale = float(scale)

    def forward(self, x):
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
                self.scale * update
            ).to(base_out.dtype)
        )


def attach_lora(model):
    targets = collect_attention_targets(model)

    for p in model.parameters():
        p.requires_grad = False

    names = sorted(targets)

    for name in names:
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
            LoRALinear(
                base,
                rank=LORA_RANK,
                alpha=LORA_ALPHA,
                dropout=LORA_DROPOUT,
            ),
        )

    return names


def attach_fcca(model):
    if not FCCA_BASIS_PATH.exists():
        raise FileNotFoundError(
            f"Missing FCCA basis: {FCCA_BASIS_PATH}"
        )

    payload = torch.load(
        FCCA_BASIS_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if payload.get("method") != "FCCA":
        raise RuntimeError(
            "Transferred basis is not an FCCA payload."
        )

    if payload.get("base_model") != BASE_MODEL:
        raise RuntimeError(
            "FCCA base-model mismatch: "
            f"{payload.get('base_model')} != {BASE_MODEL}"
        )

    if int(payload.get("calibration_size", -1)) != 256:
        raise RuntimeError(
            "Expected FCCA calibration size 256."
        )

    rank_key = str(FCCA_RANK)

    if rank_key not in payload.get("bases", {}):
        raise RuntimeError(
            f"FCCA basis contains no rank {FCCA_RANK}."
        )

    bases = payload["bases"][rank_key]
    targets = collect_attention_targets(model)

    if set(targets) != set(bases):
        missing = sorted(
            set(targets) - set(bases)
        )

        extra = sorted(
            set(bases) - set(targets)
        )

        raise RuntimeError(
            "FCCA model/basis target mismatch. "
            f"missing={missing[:10]} "
            f"extra={extra[:10]}"
        )

    for p in model.parameters():
        p.requires_grad = False

    names = sorted(targets)

    for name in names:
        parent, leaf = get_parent(
            model,
            name,
        )

        base = getattr(
            parent,
            leaf,
        )

        basis = bases[name]

        setattr(
            parent,
            leaf,
            FCCALinear(
                base,
                basis["P"],
                basis["Q"],
                scale=FCCA_SCALE,
            ),
        )

    return names


def configure_method(model, method):
    if method == "fft":
        for p in model.parameters():
            p.requires_grad = True

        return []

    if method == "lora":
        return attach_lora(model)

    if method == "fcca":
        return attach_fcca(model)

    raise ValueError(method)


def method_output_root(method):
    if method == "fft":
        return RESULT_ROOT / "fft"

    if method == "lora":
        return RESULT_ROOT / f"lora_r{LORA_RANK}"

    if method == "fcca":
        return RESULT_ROOT / "fcca"

    raise ValueError(method)


def distributed_info():
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    distributed = (
        world_size > 1
    )

    return (
        distributed,
        world_size,
        rank,
        local_rank,
    )


DISTRIBUTED, WORLD_SIZE, RANK, LOCAL_RANK = (
    distributed_info()
)


def is_main():
    return RANK == 0


def main_print(
    *args,
    **kwargs,
):
    if is_main():
        print(
            *args,
            **kwargs,
            flush=True,
        )


def barrier():
    if (
        DISTRIBUTED
        and
        dist.is_initialized()
    ):
        dist.barrier()


def setup_distributed():
    if DISTRIBUTED:
        torch.cuda.set_device(
            LOCAL_RANK
        )

        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2),
        )

    else:
        torch.cuda.set_device(
            0
        )


def cleanup_distributed():
    if (
        DISTRIBUTED
        and
        dist.is_initialized()
    ):
        dist.destroy_process_group()


def set_seed(
    seed: int,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )


def json_safe(
    value: Any,
):
    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        torch.dtype,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k):
                json_safe(v)
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def save_json(
    path: Path,
    value: Any,
):
    if not is_main():
        return

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
            json_safe(value),
            ensure_ascii=False,
            indent=2,
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
    rows: Sequence[
        Mapping[str, Any]
    ],
):
    if not is_main():
        return

    rows = list(
        rows
    )

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
                fields.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key:
                        json_safe(
                            row.get(
                                key,
                                "",
                            )
                        )
                    for key
                    in fields
                }
            )


def pcm_to_float(
    audio: np.ndarray,
):
    if np.issubdtype(
        audio.dtype,
        np.floating,
    ):
        return audio.astype(
            np.float32,
            copy=False,
        )

    if audio.dtype == np.uint8:
        return (
            (
                audio.astype(
                    np.float32
                )
                - 128.0
            )
            / 128.0
        )

    info = np.iinfo(
        audio.dtype
    )

    scale = float(
        max(
            abs(info.min),
            abs(info.max),
        )
    )

    return (
        audio.astype(
            np.float32
        )
        / scale
    )


def decode_audio(
    value: Any,
):
    if isinstance(
        value,
        dict,
    ):
        if (
            "array" in value
            and
            value["array"]
            is not None
        ):
            audio = np.asarray(
                value["array"]
            )

            sr = int(
                value.get(
                    "sampling_rate",
                    SAMPLE_RATE,
                )
            )

        elif (
            value.get(
                "bytes"
            )
            is not None
        ):
            audio, sr = sf.read(
                io.BytesIO(
                    value["bytes"]
                ),
                always_2d=False,
            )

        elif value.get(
            "path"
        ):
            audio, sr = sf.read(
                value["path"],
                always_2d=False,
            )

        else:
            raise RuntimeError(
                "Audio row has no "
                "decodable data."
            )

    elif isinstance(
        value,
        (str, Path),
    ):
        audio, sr = sf.read(
            str(value),
            always_2d=False,
        )

    else:
        raise TypeError(
            type(value)
        )

    audio = pcm_to_float(
        np.asarray(
            audio
        )
    )

    if audio.ndim == 2:
        audio = audio.mean(
            axis=1
        )

    if audio.ndim != 1:
        raise RuntimeError(
            f"Bad audio shape "
            f"{audio.shape}"
        )

    if int(sr) != SAMPLE_RATE:
        g = math.gcd(
            int(sr),
            SAMPLE_RATE,
        )

        audio = resample_poly(
            audio,
            SAMPLE_RATE // g,
            int(sr) // g,
        ).astype(
            np.float32
        )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(
        np.float32,
        copy=False,
    )

    if (
        audio.size == 0
        or
        float(
            np.max(
                np.abs(
                    audio
                )
            )
        )
        == 0.0
    ):
        raise RuntimeError(
            "Empty/silent audio."
        )

    return np.ascontiguousarray(
        audio
    )



def fixed_spanish_train_subset(
    ds: Dataset,
):
    if "num_samples" not in ds.column_names:
        raise RuntimeError(
            "7.483-hour Spanish matching requires num_samples."
        )

    durations = [
        int(x)
        for x in ds["num_samples"]
    ]

    if SPANISH_SUBSET_PATH.exists():
        payload = json.loads(
            SPANISH_SUBSET_PATH.read_text(
                encoding="utf-8"
            )
        )

        indices = [
            int(i)
            for i in payload["indices"]
        ]

        if not indices:
            raise RuntimeError(
                "Persisted Spanish subset is empty."
            )

        if max(indices) >= len(ds):
            raise RuntimeError(
                "Persisted Spanish subset no longer matches dataset."
            )

        selected_samples = sum(
            durations[i]
            for i in indices
        )

        if selected_samples != int(
            payload["selected_samples"]
        ):
            raise RuntimeError(
                "Persisted Spanish subset duration changed."
            )

        selected = ds.select(indices)

        main_print(
            "[spanish-fixed-train] "
            f"REUSE n={len(selected)} "
            f"hours={selected_samples / SAMPLE_RATE / 3600.0:.9f} "
            f"error_seconds="
            f"{abs(selected_samples - SPANISH_TRAIN_TARGET_SAMPLES) / SAMPLE_RATE:.6f} "
            f"manifest={SPANISH_SUBSET_PATH}"
        )

        return selected

    rng = random.Random(SEED)

    order = list(
        range(len(ds))
    )

    rng.shuffle(order)

    chosen = []
    chosen_set = set()
    running = 0

    # Deterministic shuffled prefix under the target.
    cursor = 0

    while cursor < len(order):
        idx = order[cursor]
        n = durations[idx]

        if running + n > SPANISH_TRAIN_TARGET_SAMPLES:
            break

        chosen.append(idx)
        chosen_set.add(idx)
        running += n
        cursor += 1

    # Choose the one remaining utterance giving the closest duration.
    best_idx = None
    best_error = abs(
        SPANISH_TRAIN_TARGET_SAMPLES
        - running
    )

    for idx in order:
        if idx in chosen_set:
            continue

        candidate = (
            running
            + durations[idx]
        )

        error = abs(
            SPANISH_TRAIN_TARGET_SAMPLES
            - candidate
        )

        if error < best_error:
            best_error = error
            best_idx = idx

    if best_idx is not None:
        chosen.append(best_idx)
        running += durations[best_idx]

    chosen = sorted(chosen)

    selected = ds.select(
        chosen
    )

    actual_hours = (
        running
        / SAMPLE_RATE
        / 3600.0
    )

    SPANISH_SUBSET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "target_hours":
            SPANISH_TRAIN_TARGET_HOURS,
        "target_samples":
            SPANISH_TRAIN_TARGET_SAMPLES,
        "selected_hours":
            actual_hours,
        "selected_samples":
            running,
        "absolute_error_seconds":
            abs(
                running
                - SPANISH_TRAIN_TARGET_SAMPLES
            )
            / SAMPLE_RATE,
        "seed":
            SEED,
        "source_examples":
            len(ds),
        "selected_examples":
            len(chosen),
        "indices":
            chosen,
    }

    SPANISH_SUBSET_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    main_print(
        "[spanish-fixed-train] "
        f"CREATED n={len(selected)} "
        f"target_hours={SPANISH_TRAIN_TARGET_HOURS:.6f} "
        f"actual_hours={actual_hours:.9f} "
        f"error_seconds={payload['absolute_error_seconds']:.6f} "
        f"manifest={SPANISH_SUBSET_PATH}"
    )

    return selected

def load_split(
    split: str,
):
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            DATA_ROOT
        )

    # Current Week 8 data are generally
    # stored as DatasetDict at the config root.
    if (
        DATA_ROOT
        / "dataset_dict.json"
    ).exists():
        obj = load_from_disk(
            str(
                DATA_ROOT
            )
        )

        if split not in obj:
            raise RuntimeError(
                f"{split} not present in "
                f"{DATA_ROOT}"
            )

        ds = obj[
            split
        ]

    # Also accept an older per-split
    # save_to_disk layout without changing
    # the experiment.
    elif (
        DATA_ROOT
        / split
    ).exists():
        ds = load_from_disk(
            str(
                DATA_ROOT
                / split
            )
        )

    else:
        raise RuntimeError(
            f"Cannot find split "
            f"{split} under "
            f"{DATA_ROOT}"
        )

    required = {
        "audio",
        "transcription",
    }

    missing = (
        required
        - set(
            ds.column_names
        )
    )

    if missing:
        raise RuntimeError(
            f"{split} missing "
            f"columns {missing}"
        )

    raw_n = len(
        ds
    )

    if (
        "num_samples"
        in ds.column_names
    ):
        keep = [
            i
            for i, n
            in enumerate(
                ds[
                    "num_samples"
                ]
            )
            if int(n)
            <= MAX_AUDIO_SAMPLES
        ]

        ds = ds.select(
            keep
        )

        removed = (
            raw_n
            - len(ds)
        )

        main_print(
            f"[duration-filter] "
            f"{split}: "
            f"raw={raw_n} "
            f"kept={len(ds)} "
            f"excluded_over_30s="
            f"{removed}"
        )

    else:
        main_print(
            f"[warning] {split} has "
            "no num_samples; "
            "duration filter will be "
            "checked during decoding."
        )

    try:
        ds = ds.cast_column(
            "audio",
            Audio(
                decode=False
            ),
        )

    except Exception:
        pass

    return ds


def dataset_fingerprint(
    ds: Dataset,
):
    digest = hashlib.sha256()

    for i in range(
        len(ds)
    ):
        row = ds[i]

        digest.update(
            str(
                row.get(
                    "id",
                    i,
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            str(
                row.get(
                    "num_samples",
                    "",
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            str(
                row.get(
                    "transcription",
                    "",
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            b"\n"
        )

    return digest.hexdigest()


def split_hours(
    ds: Dataset,
):
    if (
        "num_samples"
        in ds.column_names
    ):
        return (
            float(
                sum(
                    int(x)
                    for x in ds[
                        "num_samples"
                    ]
                )
            )
            / SAMPLE_RATE
            / 3600.0
        )

    seconds = 0.0

    for row in ds:
        seconds += (
            len(
                decode_audio(
                    row["audio"]
                )
            )
            / SAMPLE_RATE
        )

    return (
        seconds
        / 3600.0
    )


def normalize_text(
    text: str,
):
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    ).casefold()

    if LANGUAGE_ID == "fa_ir":
        text = text.translate(
            str.maketrans(
                {
                    "ي": "ی",
                    "ى": "ی",
                    "ك": "ک",
                }
            )
        )

    out = []

    for char in text:
        category = (
            unicodedata.category(
                char
            )
        )

        if category[0] in {
            "P",
            "S",
        }:
            out.append(
                " "
            )

        elif category in {
            "Cf",
            "Cc",
        }:
            out.append(
                " "
            )

        else:
            try:
                digit = (
                    unicodedata.digit(
                        char
                    )
                )

                out.append(
                    str(
                        int(
                            digit
                        )
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                out.append(
                    char
                )

    return " ".join(
        "".join(
            out
        ).split()
    )


def edit_counts(
    reference: Sequence[str],
    hypothesis: Sequence[str],
):
    n = len(
        reference
    )

    m = len(
        hypothesis
    )

    cost = [
        [0] * (m + 1)
        for _ in range(
            n + 1
        )
    ]

    op = [
        [""] * (m + 1)
        for _ in range(
            n + 1
        )
    ]

    for i in range(
        1,
        n + 1,
    ):
        cost[i][0] = i
        op[i][0] = "D"

    for j in range(
        1,
        m + 1,
    ):
        cost[0][j] = j
        op[0][j] = "I"

    for i in range(
        1,
        n + 1,
    ):
        for j in range(
            1,
            m + 1,
        ):
            same = (
                reference[
                    i - 1
                ]
                ==
                hypothesis[
                    j - 1
                ]
            )

            choices = [
                (
                    cost[
                        i - 1
                    ][
                        j - 1
                    ]
                    + (
                        0
                        if same
                        else 1
                    ),
                    (
                        "C"
                        if same
                        else "S"
                    ),
                ),
                (
                    cost[
                        i - 1
                    ][j]
                    + 1,
                    "D",
                ),
                (
                    cost[i][
                        j - 1
                    ]
                    + 1,
                    "I",
                ),
            ]

            value, action = min(
                choices,
                key=lambda x:
                    x[0],
            )

            cost[i][j] = value
            op[i][j] = action

    s = 0
    d = 0
    ins = 0

    i = n
    j = m

    while (
        i > 0
        or j > 0
    ):
        action = op[i][j]

        if action in {
            "C",
            "S",
        }:
            if action == "S":
                s += 1

            i -= 1
            j -= 1

        elif action == "D":
            d += 1
            i -= 1

        else:
            ins += 1
            j -= 1

    return (
        s,
        d,
        ins,
    )


def corpus_wer(
    references,
    hypotheses,
):
    substitutions = 0
    deletions = 0
    insertions = 0
    reference_words = 0

    for ref, hyp in zip(
        references,
        hypotheses,
        strict=True,
    ):
        ref_normalised = normalize_text(ref)
        hyp_normalised = normalize_text(hyp)

        if LANGUAGE_ID == "cmn_hans_cn":
            # Mandarin evaluation only:
            # 1. canonicalise Traditional -> Simplified
            # 2. remove whitespace
            # 3. compute character error rate
            #
            # Training targets remain the original FLEURS
            # `transcription` field unchanged.
            ref_normalised = MANDARIN_T2S.convert(
                ref_normalised
            )
            hyp_normalised = MANDARIN_T2S.convert(
                hyp_normalised
            )

            rt = [
                char
                for char in ref_normalised
                if not char.isspace()
            ]

            ht = [
                char
                for char in hyp_normalised
                if not char.isspace()
            ]

        else:
            # All other languages retain ordinary whitespace WER.
            rt = ref_normalised.split()
            ht = hyp_normalised.split()

        s, d, i = edit_counts(
            rt,
            ht,
        )

        substitutions += s
        deletions += d
        insertions += i
        reference_words += len(
            rt
        )

    errors = (
        substitutions
        + deletions
        + insertions
    )

    return {
        "normalised_wer":
            (
                100.0
                * errors
                / reference_words
                if reference_words
                else 0.0
            ),

        "normalised_reference_words":
            reference_words,

        "normalised_substitutions":
            substitutions,

        "normalised_deletions":
            deletions,

        "normalised_insertions":
            insertions,
    }


class RepeatedTokenCycleGuard(StoppingCriteria):
    """
    Stop only after an obvious short token cycle has repeated many times.

    Detects cycles of 1..8 decoder tokens, requiring 6 immediately
    consecutive copies of the same cycle.

    Examples caught:
        A A A A A A
        A B A B A B A B A B A B
        A B C A B C A B C ...
    """

    def __init__(
        self,
        max_cycle_length: int = 8,
        repeats_required: int = 6,
    ):
        super().__init__()
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

        sequence_length = int(
            input_ids.shape[1]
        )

        for cycle_length in range(
            1,
            self.max_cycle_length + 1,
        ):
            needed = (
                cycle_length
                * self.repeats_required
            )

            if sequence_length < needed:
                continue

            tail = input_ids[
                :,
                -needed:
            ]

            pattern = tail[
                :,
                -cycle_length:
            ]

            expected = pattern.repeat(
                1,
                self.repeats_required,
            )

            matched = (
                tail == expected
            ).all(
                dim=1
            )

            stop |= matched

        return stop


def build_processor():
    return (
        WhisperProcessor
        .from_pretrained(
            BASE_MODEL,
            language=WHISPER_LANGUAGE,
            task=WHISPER_TASK,
        )
    )


def collate(
    processor,
    rows,
):
    audios = [
        decode_audio(
            row[
                "audio"
            ]
        )
        for row in rows
    ]

    for audio in audios:
        if (
            len(audio)
            > MAX_AUDIO_SAMPLES
        ):
            raise RuntimeError(
                "Audio >30 seconds "
                "reached collator."
            )

    extractor = (
        processor
        .feature_extractor
    )

    audio_batch = extractor(
        audios,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=True,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_AUDIO_SAMPLES,
        truncation=True,
    )

    token_batch = (
        processor.tokenizer(
            [
                str(
                    row[
                        "transcription"
                    ]
                )
                for row in rows
            ],
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
    )

    labels = (
        token_batch
        .input_ids
        .masked_fill(
            token_batch
            .attention_mask
            .ne(1),
            -100,
        )
    )

    decoder_start = int(
        processor
        .tokenizer
        .convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
    )

    if (
        labels.shape[1]
        and
        torch.all(
            labels[:, 0]
            == decoder_start
        )
    ):
        labels = labels[
            :,
            1:,
        ]

    return {
        "input_features":
            audio_batch
            .input_features,

        "attention_mask":
            audio_batch
            .attention_mask,

        "labels":
            labels,
    }


def make_train_loader(
    ds,
    processor,
    epoch,
    microbatch,
):
    if DISTRIBUTED:
        sampler = (
            DistributedSampler(
                ds,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=True,
                seed=SEED,
                drop_last=False,
            )
        )

        sampler.set_epoch(
            epoch
        )

        shuffle = False

    else:
        sampler = None
        shuffle = True

    generator = (
        torch.Generator()
        .manual_seed(
            SEED + epoch
        )
    )

    return DataLoader(
        ds,
        batch_size=microbatch,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x:
            collate(
                processor,
                x,
            ),
    )


def move_batch(
    batch,
    device,
):
    result = {}

    for key, value in (
        batch.items()
    ):
        if torch.is_tensor(
            value
        ):
            value = value.to(
                device,
                non_blocking=True,
            )

            if value.is_floating_point():
                value = value.to(
                    dtype=torch.bfloat16
                )

        result[
            key
        ] = value

    return result


@torch.inference_mode()
def evaluate_validation_loss(
    model,
    processor,
    ds,
    device,
):
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x:
            collate(
                processor,
                x,
            ),
    )

    weighted = 0.0
    tokens = 0
    batches = 0

    for batch in loader:
        batch = move_batch(
            batch,
            device,
        )

        count = int(
            batch[
                "labels"
            ]
            .ne(-100)
            .sum()
            .item()
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            output = model(
                **batch
            )

        loss = output.loss

        if (
            loss is None
            or
            not torch.isfinite(
                loss
            )
        ):
            raise RuntimeError(
                f"Non-finite "
                f"validation loss: "
                f"{loss}"
            )

        weighted += (
            float(
                loss.item()
            )
            * count
        )

        tokens += count
        batches += 1

    return {
        "validation_loss":
            weighted
            / tokens,

        "validation_supervised_tokens":
            tokens,

        "validation_loss_batches":
            batches,
    }


@torch.inference_mode()
def evaluate_wer(
    model,
    processor,
    ds,
    device,
    predictions_path,
):
    model.eval()

    references = []
    hypotheses = []
    rows = []

    for index, row in enumerate(
        ds,
        start=1,
    ):
        audio = decode_audio(
            row[
                "audio"
            ]
        )

        inputs = (
            processor
            .feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_attention_mask=True,
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_AUDIO_SAMPLES,
                truncation=True,
            )
        )

        ids = model.generate(
            input_features=(
                inputs
                .input_features
                .to(
                    device=device,
                    dtype=torch.bfloat16,
                )
            ),
            attention_mask=(
                inputs
                .attention_mask
                .to(
                    device
                )
            ),
            language=(
                WHISPER_LANGUAGE
            ),
            task=(
                WHISPER_TASK
            ),
            num_beams=1,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList(
                [
                    RepeatedTokenCycleGuard(
                        max_cycle_length=8,
                        repeats_required=6,
                    )
                ]
            ),
        )

        hypothesis = (
            processor
            .batch_decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            .strip()
        )

        reference = str(
            row[
                "transcription"
            ]
        )

        references.append(
            reference
        )

        hypotheses.append(
            hypothesis
        )

        rows.append(
            {
                "index":
                    index - 1,

                "id":
                    row.get(
                        "id",
                        index - 1,
                    ),

                "reference_original":
                    reference,

                "hypothesis_original":
                    hypothesis,

                "reference_normalised":
                    normalize_text(
                        reference
                    ),

                "hypothesis_normalised":
                    normalize_text(
                        hypothesis
                    ),
            }
        )

        if (
            index % 50 == 0
            or
            index == len(ds)
        ):
            main_print(
                f"[validation decode] "
                f"{index}/{len(ds)}"
            )

    write_csv(
        predictions_path,
        rows,
    )

    return corpus_wer(
        references,
        hypotheses,
    )


def unwrap_model(
    model,
):
    if isinstance(
        model,
        DDP,
    ):
        return model.module

    return model


def save_checkpoint(
    model,
    path,
):
    if not is_main():
        return

    base = unwrap_model(
        model
    )

    state = {
        key:
            value.detach().cpu()
        for key, value
        in base.state_dict().items()
    }

    tmp = path.with_suffix(
        path.suffix
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        {
            "kind":
                "full_state_dict",

            "base_model":
                BASE_MODEL,

            "language":
                WHISPER_LANGUAGE,

            "task":
                WHISPER_TASK,

            "state_dict":
                state,
        },
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    *,
    method,
    trial_number,
    epoch,
    global_step,
    best_wer,
    best_epoch,
    bad_epochs,
    history,
):
    """
    Full epoch-boundary checkpoint for continuation training.

    This is intentionally separate from save_checkpoint(), which is
    retained for best_checkpoint.pt compatibility.
    """
    if not is_main():
        return

    base = unwrap_model(
        model
    )

    model_state = {
        key:
            value.detach().cpu()
        for key, value
        in base.state_dict().items()
    }

    def cpuify(value):
        if torch.is_tensor(value):
            return (
                value
                .detach()
                .cpu()
            )

        if isinstance(
            value,
            dict,
        ):
            return {
                key:
                    cpuify(item)
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            list,
        ):
            return [
                cpuify(item)
                for item in value
            ]

        if isinstance(
            value,
            tuple,
        ):
            return tuple(
                cpuify(item)
                for item in value
            )

        return value

    payload = {
        "kind":
            "full_training_checkpoint",

        "base_model":
            BASE_MODEL,

        "language_id":
            LANGUAGE_ID,

        "whisper_language":
            WHISPER_LANGUAGE,

        "task":
            WHISPER_TASK,

        "method":
            method,

        "trial_number":
            int(
                trial_number
            ),

        "epoch":
            int(
                epoch
            ),

        "global_step":
            int(
                global_step
            ),

        "best_validation_normalised_wer":
            float(
                best_wer
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "bad_epochs":
            int(
                bad_epochs
            ),

        "history":
            list(
                history
            ),

        "state_dict":
            model_state,

        "optimizer_state_dict":
            cpuify(
                optimizer.state_dict()
            ),

        "scheduler_state_dict":
            cpuify(
                scheduler.state_dict()
            ),

        # RNG state for continuation.
        "python_rng_state":
            random.getstate(),

        "numpy_rng_state":
            np.random.get_state(),

        "torch_rng_state":
            torch.get_rng_state(),

        "cuda_rng_state_all":
            (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
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


def restore_training_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    method,
    trial_number,
    device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        payload.get(
            "kind"
        )
        != "full_training_checkpoint"
    ):
        raise RuntimeError(
            "Checkpoint is not a resumable "
            "full_training_checkpoint: "
            f"{checkpoint_path}"
        )

    expected = {
        "language_id":
            LANGUAGE_ID,

        "method":
            method,

        "trial_number":
            int(
                trial_number
            ),

        "base_model":
            BASE_MODEL,
    }

    for key, expected_value in (
        expected.items()
    ):
        actual = payload.get(
            key
        )

        if actual != expected_value:
            raise RuntimeError(
                "Resume checkpoint mismatch: "
                f"{key}={actual!r}, "
                f"expected={expected_value!r}"
            )

    base = unwrap_model(
        model
    )

    base.load_state_dict(
        payload[
            "state_dict"
        ],
        strict=True,
    )

    optimizer.load_state_dict(
        payload[
            "optimizer_state_dict"
        ]
    )

    # torch.load(..., map_location='cpu') puts Adam tensors on CPU.
    # Put each optimizer state's tensors back beside its parameter.
    for parameter, state in (
        optimizer.state.items()
    ):
        for key, value in list(
            state.items()
        ):
            if torch.is_tensor(
                value
            ):
                state[key] = value.to(
                    parameter.device
                )

    scheduler.load_state_dict(
        payload[
            "scheduler_state_dict"
        ]
    )

    # Restore stochastic state after reconstruction/loading so no
    # initialization work consumes the restored stream.
    random.setstate(
        payload[
            "python_rng_state"
        ]
    )

    np.random.set_state(
        payload[
            "numpy_rng_state"
        ]
    )

    torch.set_rng_state(
        payload[
            "torch_rng_state"
        ]
    )

    cuda_states = payload.get(
        "cuda_rng_state_all"
    )

    if (
        cuda_states is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_states
        )

    epoch = int(
        payload[
            "epoch"
        ]
    )

    global_step = int(
        payload[
            "global_step"
        ]
    )

    best_wer = float(
        payload[
            "best_validation_normalised_wer"
        ]
    )

    best_epoch = int(
        payload[
            "best_epoch"
        ]
    )

    bad_epochs = int(
        payload[
            "bad_epochs"
        ]
    )

    history = list(
        payload.get(
            "history",
            [],
        )
    )

    if len(history) != epoch:
        raise RuntimeError(
            "Resume checkpoint/history mismatch: "
            f"epoch={epoch}, "
            f"history_rows={len(history)}"
        )

    return {
        "epoch":
            epoch,

        "global_step":
            global_step,

        "best_wer":
            best_wer,

        "best_epoch":
            best_epoch,

        "bad_epochs":
            bad_epochs,

        "history":
            history,
    }


def run_trial(
    method: str,
    trial_number: int,
    microbatch: int,
    learning_rate: float,
    effective_batch_size: int,
    warmup_ratio: float,
    resume_checkpoint=None,
):
    if method not in {
        "fft",
        "lora",
        "fcca",
    }:
        raise ValueError(
            f"Unknown method {method}"
        )

    setup_distributed()

    try:
        device = torch.device(
            f"cuda:{LOCAL_RANK}"
            if DISTRIBUTED
            else "cuda:0"
        )

        set_seed(
            SEED
        )

        out = (
            method_output_root(
                method
            )
            / f"trial_{trial_number:04d}"
        )

        complete = (
            out
            / "COMPLETE"
        )

        if (
            complete.exists()
            and resume_checkpoint is None
        ):
            main_print(
                f"[skip] completed "
                f"{out}"
            )
            return

        if is_main():
            if (
                resume_checkpoint is None
                and out.exists()
                and any(
                    out.iterdir()
                )
            ):
                raise RuntimeError(
                    "Refusing to overwrite "
                    "non-empty incomplete "
                    f"trial directory: {out}"
                )

            out.mkdir(
                parents=True,
                exist_ok=True,
            )

        barrier()

        train_ds = load_split(
            "train"
        )

        val_ds = load_split(
            "validation"
        )

        processor = (
            build_processor()
        )

        model = (
            WhisperForConditionalGeneration
            .from_pretrained(
                BASE_MODEL,
                dtype=torch.bfloat16,
            )
        )

        model.config.use_cache = False

        model.to(
            device
        )

        target_modules = (
            configure_method(
                model,
                method,
            )
        )

        if DISTRIBUTED:
            model = DDP(
                model,
                device_ids=[
                    LOCAL_RANK
                ],
                output_device=(
                    LOCAL_RANK
                ),
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        parameters = [
            p
            for p in model.parameters()
            if p.requires_grad
        ]

        if (
            effective_batch_size
            %
            (
                microbatch
                * WORLD_SIZE
            )
            != 0
        ):
            raise RuntimeError(
                "EBS is not divisible "
                "by microbatch*world_size."
            )

        grad_acc = (
            effective_batch_size
            //
            (
                microbatch
                * WORLD_SIZE
            )
        )

        # DistributedSampler changes
        # per-rank loader length.
        first_loader = (
            make_train_loader(
                train_ds,
                processor,
                epoch=1,
                microbatch=microbatch,
            )
        )

        updates_per_epoch = math.ceil(
            len(
                first_loader
            )
            / grad_acc
        )

        total_optimizer_steps = (
            updates_per_epoch
            * MAX_EPOCHS
        )

        warmup_steps = int(
            round(
                total_optimizer_steps
                * warmup_ratio
            )
        )

        optimizer = (
            torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=(
                    WEIGHT_DECAY
                ),
            )
        )

        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=(
                    warmup_steps
                ),
                num_training_steps=(
                    total_optimizer_steps
                ),
            )
        )

        base = unwrap_model(
            model
        )

        total_parameters = sum(
            p.numel()
            for p in base.parameters()
        )

        trainable_parameters = sum(
            p.numel()
            for p in base.parameters()
            if p.requires_grad
        )

        hyperparameters = {
            "language":
                LANGUAGE_ID,

            "language_name":
                LANGUAGE_NAME,

            "dataset_root":
                str(
                    DATA_ROOT
                ),

            "dataset":
                f"FLEURS {LANGUAGE_ID} full train split",

            "base_model":
                BASE_MODEL,

            "method":
                method,

            "rank":
                (
                    LORA_RANK
                    if method == "lora"
                    else (
                        FCCA_RANK
                        if method == "fcca"
                        else None
                    )
                ),

            "target_module_count":
                len(target_modules),

            "target_modules":
                target_modules,

            "lora_alpha":
                (
                    LORA_ALPHA
                    if method == "lora"
                    else None
                ),

            "lora_dropout":
                (
                    LORA_DROPOUT
                    if method == "lora"
                    else None
                ),

            "lora_scaling":
                (
                    LORA_ALPHA / LORA_RANK
                    if method == "lora"
                    else None
                ),

            "fcca_basis_path":
                (
                    str(FCCA_BASIS_PATH)
                    if method == "fcca"
                    else None
                ),

            "fcca_scale":
                (
                    FCCA_SCALE
                    if method == "fcca"
                    else None
                ),

            "fcca_calibration_size":
                (
                    256
                    if method == "fcca"
                    else None
                ),

            "fcca_calibration_seed":
                (
                    2026
                    if method == "fcca"
                    else None
                ),

            "fcca_damping":
                (
                    1.0e-6
                    if method == "fcca"
                    else None
                ),

            "trial_number":
                trial_number,

            "learning_rate":
                learning_rate,

            "effective_batch_size":
                effective_batch_size,

            "per_device_train_batch_size":
                microbatch,

            "world_size":
                WORLD_SIZE,

            "gradient_accumulation_steps":
                grad_acc,

            "effective_batch_size_formula":
                (
                    f"{microbatch}"
                    f"*{WORLD_SIZE}"
                    f"*{grad_acc}"
                    f"={effective_batch_size}"
                ),

            "per_device_eval_batch_size":
                EVAL_BATCH_SIZE,

            "maximum_epochs":
                MAX_EPOCHS,

            "early_stopping_patience":
                PATIENCE,

            "early_stopping_metric":
                "validation_normalised_wer",

            "greater_is_better":
                False,

            "strict_improvement":
                True,

            "equal_wer_counts_as_no_improvement":
                True,

            "optimizer":
                "AdamW",

            "optimizer_betas":
                list(
                    optimizer.defaults[
                        "betas"
                    ]
                ),

            "optimizer_eps":
                optimizer.defaults[
                    "eps"
                ],

            "weight_decay":
                WEIGHT_DECAY,

            "scheduler":
                "linear_with_warmup",

            "warmup_ratio":
                warmup_ratio,

            "warmup_steps":
                warmup_steps,

            "planned_total_optimizer_steps":
                total_optimizer_steps,

            "optimizer_updates_per_epoch":
                updates_per_epoch,

            "gradient_clipping_max_norm":
                MAX_GRAD_NORM,

            "precision":
                "bf16",

            "seed":
                SEED,

            "whisper_language_token":
                WHISPER_LANGUAGE,

            "whisper_task":
                WHISPER_TASK,

            "decoding":
                "greedy",

            "num_beams":
                1,

            "do_sample":
                False,

            "external_language_model":
                False,

            "maximum_audio_seconds":
                MAX_AUDIO_SECONDS,

            "sample_rate_hz":
                SAMPLE_RATE,

            "normaliser":
                (
                    "persian_unicode_variants_word"
                    if LANGUAGE_ID == "fa_ir"
                    else "unicode_word"
                ),

            "normalised_wer_text_normalisation":
                (
                    "NFKC + casefold + "
                    "Unicode punctuation/symbol "
                    "removal + decimal digit "
                    "normalisation + whitespace "
                    "collapse"
                ),

            "train_examples":
                len(
                    train_ds
                ),

            "validation_examples":
                len(
                    val_ds
                ),

            "train_hours":
                split_hours(
                    train_ds
                ),

            "validation_hours":
                split_hours(
                    val_ds
                ),

            "train_membership_sha256":
                dataset_fingerprint(
                    train_ds
                ),

            "validation_membership_sha256":
                dataset_fingerprint(
                    val_ds
                ),

            "total_parameters":
                total_parameters,

            "trainable_parameters":
                trainable_parameters,

            "trainable_percent":
                (
                    100.0
                    * trainable_parameters
                    / total_parameters
                ),

            "torch_version":
                torch.__version__,

            "cuda_runtime_version":
                torch.version.cuda,

            "gpu_name":
                torch.cuda.get_device_name(
                    LOCAL_RANK
                    if DISTRIBUTED
                    else 0
                ),

            "python_version":
                sys.version,

            "distributed_training":
                DISTRIBUTED,
        }

        try:
            import transformers

            hyperparameters[
                "transformers_version"
            ] = (
                transformers
                .__version__
            )

        except Exception:
            pass

        save_json(
            out
            / "hyperparameters.json",
            hyperparameters,
        )

        main_print(
            "=" * 90
        )

        main_print(
            f"{LANGUAGE_NAME.upper()} {method.upper()} "
            f"TRIAL {trial_number:04d}"
        )

        main_print(
            f"lr={learning_rate}"
        )

        main_print(
            f"world_size={WORLD_SIZE}"
        )

        main_print(
            f"microbatch={microbatch}"
        )

        main_print(
            f"grad_acc={grad_acc}"
        )

        main_print(
            f"EBS="
            f"{microbatch}"
            f"*{WORLD_SIZE}"
            f"*{grad_acc}"
            f"={effective_batch_size}"
        )

        main_print(
            f"train_n={len(train_ds)} "
            f"val_n={len(val_ds)}"
        )

        main_print(
            f"train_hours="
            f"{hyperparameters['train_hours']:.3f}"
        )

        main_print(
            f"val_hours="
            f"{hyperparameters['validation_hours']:.3f}"
        )

        main_print(
            f"output={out}"
        )

        main_print(
            "=" * 90
        )

        history = []

        best_wer = float(
            "inf"
        )

        best_epoch = 0
        bad_epochs = 0
        global_step = 0
        start_epoch = 1

        best_checkpoint = (
            out
            / "best_checkpoint.pt"
        )

        best_predictions = (
            out
            / "validation_predictions.csv"
        )

        started = time.time()

        if resume_checkpoint is not None:
            resume_state = (
                restore_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    checkpoint_path=(
                        resume_checkpoint
                    ),
                    method=method,
                    trial_number=(
                        trial_number
                    ),
                    device=device,
                )
            )

            completed_epoch = int(
                resume_state[
                    "epoch"
                ]
            )

            global_step = int(
                resume_state[
                    "global_step"
                ]
            )

            best_wer = float(
                resume_state[
                    "best_wer"
                ]
            )

            best_epoch = int(
                resume_state[
                    "best_epoch"
                ]
            )

            bad_epochs = int(
                resume_state[
                    "bad_epochs"
                ]
            )

            history = list(
                resume_state[
                    "history"
                ]
            )

            start_epoch = (
                completed_epoch
                + 1
            )

            if history:
                previous_runtime = float(
                    history[-1].get(
                        "runtime_seconds_total",
                        0.0,
                    )
                )

                started = (
                    time.time()
                    - previous_runtime
                )

            # If COMPLETE was produced by a run that the user is
            # explicitly choosing to continue from, remove it.
            if (
                is_main()
                and complete.exists()
            ):
                complete.unlink()

            main_print(
                "=" * 90
            )

            main_print(
                "[RESUME PASS] "
                f"checkpoint={resume_checkpoint}"
            )

            main_print(
                f"completed_epoch="
                f"{completed_epoch} "
                f"next_epoch="
                f"{start_epoch}"
            )

            main_print(
                f"global_step="
                f"{global_step} "
                f"best_epoch="
                f"{best_epoch} "
                f"best_WER="
                f"{best_wer:.6f}% "
                f"bad_epochs="
                f"{bad_epochs}"
            )

            main_print(
                f"restored_lr="
                f"{optimizer.param_groups[0]['lr']:.12g}"
            )

            main_print(
                "=" * 90
            )

        if start_epoch > MAX_EPOCHS:
            raise RuntimeError(
                "Checkpoint already completed "
                f"epoch {start_epoch - 1}, "
                f"but MAX_EPOCHS={MAX_EPOCHS}."
            )

        stop_training = False

        for epoch in range(
            start_epoch,
            MAX_EPOCHS + 1,
        ):
            train_loader = (
                make_train_loader(
                    train_ds,
                    processor,
                    epoch,
                    microbatch,
                )
            )

            model.train()

            unwrap_model(
                model
            ).config.use_cache = (
                False
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            local_loss_sum = 0.0
            local_batches = 0

            for (
                batch_index,
                batch,
            ) in enumerate(
                train_loader,
                start=1,
            ):
                batch = move_batch(
                    batch,
                    device,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    output = model(
                        **batch
                    )

                    loss = (
                        output.loss
                    )

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
                        f"batch={batch_index}: "
                        f"{loss}"
                    )

                local_loss_sum += float(
                    loss.detach().item()
                )

                local_batches += 1

                group_start = (
                    (
                        batch_index - 1
                    )
                    // grad_acc
                ) * grad_acc

                group_size = min(
                    grad_acc,
                    len(
                        train_loader
                    )
                    - group_start,
                )

                (
                    loss
                    / group_size
                ).backward()

                if (
                    batch_index
                    % grad_acc
                    == 0
                    or
                    batch_index
                    == len(
                        train_loader
                    )
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

                if (
                    is_main()
                    and
                    (
                        batch_index
                        % 100
                        == 0
                        or
                        batch_index
                        == len(
                            train_loader
                        )
                    )
                ):
                    print(
                        f"[train] "
                        f"T{trial_number:04d} "
                        f"epoch={epoch} "
                        f"batch="
                        f"{batch_index}/"
                        f"{len(train_loader)} "
                        f"loss="
                        f"{local_loss_sum / local_batches:.6f}",
                        flush=True,
                    )

            # Average training loss over
            # all DDP workers.
            stats = torch.tensor(
                [
                    local_loss_sum,
                    float(
                        local_batches
                    ),
                ],
                device=device,
                dtype=torch.float64,
            )

            if DISTRIBUTED:
                dist.all_reduce(
                    stats,
                    op=dist.ReduceOp.SUM,
                )

            epoch_train_loss = (
                float(
                    stats[0].item()
                )
                /
                max(
                    float(
                        stats[1].item()
                    ),
                    1.0,
                )
            )

            # Only rank 0 performs expensive
            # autoregressive validation.
            barrier()

            if is_main():
                eval_model = (
                    unwrap_model(
                        model
                    )
                )

                current_predictions = (
                    out
                    / "_current_validation_predictions.csv"
                )

                loss_metrics = (
                    evaluate_validation_loss(
                        eval_model,
                        processor,
                        val_ds,
                        device,
                    )
                )

                wer_metrics = (
                    evaluate_wer(
                        eval_model,
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
                    best_wer = (
                        current_wer
                    )

                    best_epoch = (
                        epoch
                    )

                    bad_epochs = 0

                    save_checkpoint(
                        model,
                        best_checkpoint,
                    )

                    os.replace(
                        current_predictions,
                        best_predictions,
                    )

                    best_metrics = {
                        **wer_metrics,
                        **loss_metrics,
                        "best_epoch":
                            best_epoch,
                    }

                    save_json(
                        out
                        / "validation_metrics.json",
                        best_metrics,
                    )

                    print(
                        f"[best] "
                        f"epoch={epoch} "
                        f"{'CER' if LANGUAGE_ID == 'cmn_hans_cn' else 'WER'}="
                        f"{best_wer:.4f}%",
                        flush=True,
                    )

                else:
                    current_predictions.unlink(
                        missing_ok=True
                    )

                    bad_epochs += 1

                    print(
                        f"[no improvement] "
                        f"bad_epochs="
                        f"{bad_epochs}/"
                        f"{PATIENCE}",
                        flush=True,
                    )

                row = {
                    "trial_number":
                        trial_number,

                    "epoch":
                        epoch,

                    "global_step":
                        global_step,

                    "learning_rate_initial":
                        learning_rate,

                    "learning_rate_after_epoch":
                        optimizer
                        .param_groups[0][
                            "lr"
                        ],

                    "effective_batch_size":
                        effective_batch_size,

                    "per_device_train_batch_size":
                        microbatch,

                    "gradient_accumulation_steps":
                        grad_acc,

                    "train_loss":
                        epoch_train_loss,

                    "validation_loss":
                        loss_metrics[
                            "validation_loss"
                        ],

                    "validation_normalised_wer":
                        current_wer,

                    "validation_substitutions":
                        wer_metrics[
                            "normalised_substitutions"
                        ],

                    "validation_deletions":
                        wer_metrics[
                            "normalised_deletions"
                        ],

                    "validation_insertions":
                        wer_metrics[
                            "normalised_insertions"
                        ],

                    "validation_reference_words":
                        wer_metrics[
                            "normalised_reference_words"
                        ],

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
                        (
                            time.time()
                            - started
                        ),
                }

                history.append(
                    row
                )

                write_csv(
                    out
                    / "training_history.csv",
                    history,
                )

                # [SAVE-EVERY-EPOCH]
                epoch_checkpoint = (
                    out
                    / f"epoch_{epoch:02d}.pt"
                )

                save_training_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch_checkpoint,
                    method=method,
                    trial_number=(
                        trial_number
                    ),
                    epoch=epoch,
                    global_step=(
                        global_step
                    ),
                    best_wer=(
                        best_wer
                    ),
                    best_epoch=(
                        best_epoch
                    ),
                    bad_epochs=(
                        bad_epochs
                    ),
                    history=history,
                )

                print(
                    f"[epoch checkpoint] {epoch_checkpoint}",
                    flush=True,
                )

                print(
                    f"[epoch] {epoch} "
                    f"train_loss="
                    f"{epoch_train_loss:.6f} "
                    f"val_loss="
                    f"{loss_metrics['validation_loss']:.6f} "
                    f"val_{'CER' if LANGUAGE_ID == 'cmn_hans_cn' else 'WER'}="
                    f"{current_wer:.4f}% "
                    f"best="
                    f"{best_wer:.4f}% "
                    f"bad="
                    f"{bad_epochs}",
                    flush=True,
                )

                stop_training = (
                    bad_epochs
                    >= PATIENCE
                )

            # Broadcast stop decision.
            stop_tensor = (
                torch.tensor(
                    [
                        1
                        if stop_training
                        else 0
                    ],
                    dtype=torch.int64,
                    device=device,
                )
            )

            if DISTRIBUTED:
                dist.broadcast(
                    stop_tensor,
                    src=0,
                )

            stop_training = bool(
                stop_tensor.item()
            )

            barrier()

            if stop_training:
                main_print(
                    f"[early stopping] "
                    f"after epoch={epoch}; "
                    f"best_epoch="
                    f"{best_epoch} "
                    f"best_{'CER' if LANGUAGE_ID == 'cmn_hans_cn' else 'WER'}="
                    f"{best_wer:.4f}%"
                )
                break

        if is_main():
            result = {
                "language":
                    LANGUAGE_ID,

                "method":
                    method,

                "trial_number":
                    trial_number,

                "learning_rate":
                    learning_rate,

                "effective_batch_size":
                    effective_batch_size,

                "best_validation_normalised_wer":
                    best_wer,

                "best_epoch":
                    best_epoch,

                "epochs_ran":
                    len(
                        history
                    ),

                "global_steps":
                    global_step,

                "early_stopped":
                    (
                        len(history)
                        < MAX_EPOCHS
                    ),

                "runtime_seconds":
                    (
                        time.time()
                        - started
                    ),

                "checkpoint":
                    str(
                        best_checkpoint
                    ),

                "training_history":
                    str(
                        out
                        / "training_history.csv"
                    ),

                "validation_predictions":
                    str(
                        best_predictions
                    ),
            }

            save_json(
                out
                / "trial_result.json",
                result,
            )

            complete.write_text(
                "complete\n",
                encoding="utf-8",
            )

            print(
                f"[COMPLETE] {out}",
                flush=True,
            )

        barrier()

    finally:
        cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
        help=(
            "Week8 language folder, e.g. "
            "fa_ir, nl_nl, pl_pl, th_th."
        ),
    )

    parser.add_argument(
        "--whisper-language",
        default=None,
        help=(
            "Optional Whisper language-token override."
        ),
    )

    parser.add_argument(
        "--method",
        choices=[
            "fft",
            "lora",
            "fcca",
        ],
        required=True,
    )

    parser.add_argument(
        "--trial",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--effective-batch-size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--microbatch",
        type=int,
        default=DEFAULT_MICROBATCH,
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=128,
        help="LoRA rank. Default: 128.",
    )

    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help=(
            "LoRA alpha. Default: 2 * --lora-rank, "
            "preserving alpha/r = 2."
        ),
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume from a full epoch_XX.pt "
            "training checkpoint."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    global LORA_RANK
    global LORA_ALPHA

    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be > 0")

    LORA_RANK = int(args.lora_rank)

    LORA_ALPHA = (
        float(args.lora_alpha)
        if args.lora_alpha is not None
        else 2.0 * LORA_RANK
    )

    configure_language(
        args.language,
        args.whisper_language,
    )

    run_trial(
        args.method,
        args.trial,
        args.microbatch,
        args.learning_rate,
        args.effective_batch_size,
        args.warmup_ratio,
        args.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
