from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_from_disk
from scipy.signal import resample_poly
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from scripts_fleurs.fleurs_paths import dataset_root, load_fleurs_split
from scripts_fleurs.shared_frozen_tucker2.calibrate import (
    CTX,
    TaskStat,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    Path("/data/milsrg1/corpora/fleurs-selected-languages")
)

RESULT_ROOT = (
    PROJECT_ROOT / "outputs" / "new_results"
)

BASE_MODEL = "openai/whisper-medium"

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30.0
MAX_AUDIO_SAMPLES = int(
    SAMPLE_RATE
    * MAX_AUDIO_SECONDS
)

CALIBRATION_SIZE = 256
CALIBRATION_SEED = 2026
DAMPING = 1.0e-6


# Supported language gets its own token.
# Unsupported language gets the token of its
# paired supported relative.
LANGUAGE_TOKEN = {
    "es_419": "es",

    "fa_ir": "fa",
    "ckb_iq": "fa",

    "kk_kz": "kk",
    "ky_kg": "kk",
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_attention_targets(model):
    modules = dict(
        model.named_modules()
    )

    targets = {}

    leaves = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
    }

    for name, module in modules.items():
        if not isinstance(
            module,
            torch.nn.Linear,
        ):
            continue

        leaf = name.rsplit(
            ".",
            1,
        )[-1]

        if leaf not in leaves:
            continue

        if (
            ".self_attn." not in name
            and
            ".encoder_attn." not in name
        ):
            continue

        targets[name] = module

    if len(targets) != 288:
        raise RuntimeError(
            f"Expected 288 attention matrices, "
            f"found {len(targets)}"
        )

    return targets


def decode_audio(value):
    if isinstance(value, dict):
        if (
            "array" in value
            and value["array"] is not None
        ):
            x = np.asarray(
                value["array"],
                dtype=np.float32,
            )
            sr = int(
                value.get(
                    "sampling_rate",
                    SAMPLE_RATE,
                )
            )
            return normalise_audio(x, sr)

        # IMPORTANT: save_to_disk may preserve an old/non-local
        # basename in audio["path"] while embedding the actual WAV
        # bytes. Prefer the embedded bytes.
        if (
            "bytes" in value
            and value["bytes"] is not None
        ):
            import io

            x, sr = sf.read(
                io.BytesIO(value["bytes"]),
                always_2d=False,
            )
            return normalise_audio(x, sr)

        if (
            "path" in value
            and value["path"]
            and Path(value["path"]).is_file()
        ):
            x, sr = sf.read(
                value["path"],
                always_2d=False,
            )
            return normalise_audio(x, sr)

        raise RuntimeError(
            f"Audio dict has neither embedded bytes "
            f"nor an existing path: {value.get('path')!r}"
        )

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


def normalise_audio(
    x,
    sr,
):
    x = np.asarray(x)

    if x.ndim == 2:
        # Handle either [C,T] or [T,C].
        if x.shape[0] <= 8:
            x = x.mean(
                axis=0
            )
        else:
            x = x.mean(
                axis=1
            )

    if np.issubdtype(
        x.dtype,
        np.integer,
    ):
        info = np.iinfo(
            x.dtype
        )

        scale = float(
            max(
                abs(info.min),
                abs(info.max),
            )
        )

        x = (
            x.astype(
                np.float32
            )
            / scale
        )
    else:
        x = x.astype(
            np.float32,
            copy=False,
        )

    if sr != SAMPLE_RATE:
        from math import gcd

        g = gcd(
            int(sr),
            SAMPLE_RATE,
        )

        x = resample_poly(
            x,
            SAMPLE_RATE // g,
            int(sr) // g,
        ).astype(
            np.float32,
            copy=False,
        )

    if len(x) > MAX_AUDIO_SAMPLES:
        x = x[
            :MAX_AUDIO_SAMPLES
        ]

    return x


def transcription_of(row):
    for key in (
        "transcription",
        "sentence",
        "text",
    ):
        if (
            key in row
            and
            row[key] is not None
        ):
            text = str(
                row[key]
            ).strip()

            if text:
                return text

    raise RuntimeError(
        f"No transcription field. "
        f"Keys={list(row.keys())}"
    )


def save_atomic(
    payload,
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
        payload,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def maybe_preserve_existing(
    path: Path,
):
    if not path.exists():
        return

    backup = (
        path.parent
        / (
            path.stem
            + "_before_multilang_rebuild"
            + path.suffix
        )
    )

    if backup.exists():
        return

    try:
        os.link(
            path,
            backup,
        )

        print(
            f"[backup hardlink] "
            f"{backup}",
            flush=True,
        )

    except OSError:
        shutil.copy2(
            path,
            backup,
        )

        print(
            f"[backup copy] "
            f"{backup}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
        help="Week8 dataset folder, e.g. en_us, cmn_hans_cn, fa_ir.",
    )

    parser.add_argument(
        "--whisper-language",
        required=True,
        help="Whisper language token/code, e.g. en, zh, fa.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    language = args.language
    whisper_language = (
        args.whisper_language
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required."
        )

    device = torch.device(
        "cuda:0"
    )

    set_seed(
        CALIBRATION_SEED
    )

    data_path = dataset_root(language)

    if not data_path.exists():
        raise FileNotFoundError(
            data_path
        )

    out_path = (
        RESULT_ROOT
        / language
        / "_task_stats"
        / "task_stats_256_all_attention.pt"
    )

    metadata_path = (
        RESULT_ROOT
        / language
        / "_task_stats"
        / "task_stats_256_all_attention.json"
    )

    if (
        out_path.exists()
        and
        not args.force
    ):
        print(
            f"[skip existing] "
            f"{out_path}",
            flush=True,
        )

        return

    if args.force:
        maybe_preserve_existing(
            out_path
        )

    train_ds = load_fleurs_split(language, "train")

    train_ds = train_ds.cast_column("audio", Audio(decode=False))

    if len(train_ds) < CALIBRATION_SIZE:
        raise RuntimeError(
            f"{language}: "
            f"train={len(train_ds)} "
            f"< {CALIBRATION_SIZE}"
        )

    generator = (
        torch.Generator()
        .manual_seed(
            CALIBRATION_SEED
        )
    )

    selected_indices = (
        torch.randperm(
            len(train_ds),
            generator=generator,
        )[
            :CALIBRATION_SIZE
        ]
        .tolist()
    )

    calib_ds = train_ds.select(
        selected_indices
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"[language] {language}",
        flush=True,
    )

    print(
        f"[Whisper token] "
        f"{whisper_language}",
        flush=True,
    )

    print(
        f"[train] {len(train_ds)}",
        flush=True,
    )

    print(
        f"[calibration] "
        f"{len(calib_ds)} "
        f"seed={CALIBRATION_SEED}",
        flush=True,
    )

    print(
        f"[output] {out_path}",
        flush=True,
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            BASE_MODEL,
            language=whisper_language,
            task="transcribe",
        )
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
        )
        .to(device)
    )

    model.eval()

    if hasattr(
        model.config,
        "use_cache",
    ):
        model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    targets = collect_attention_targets(
        model
    )

    stats = {
        name:
            TaskStat(
                name,
                module,
            )
        for name, module
        in targets.items()
    }

    for stat in stats.values():
        stat.install()

    try:
        for index, row in enumerate(
            calib_ds,
            start=1,
        ):
            audio = decode_audio(
                row["audio"]
            )

            transcription = (
                transcription_of(
                    row
                )
            )

            features = (
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

            input_features = (
                features
                .input_features
                .to(
                    device=device,
                    dtype=torch.float16,
                )
            )

            attention_mask = (
                features
                .attention_mask
                .to(
                    device=device
                )
            )

            tokenized = (
                processor
                .tokenizer(
                    transcription,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
            )

            labels = (
                tokenized
                .input_ids
                .to(device)
            )

            # Batch size 1, so no padded label
            # positions. Keep an explicit mask
            # because TaskStat needs decoder time.
            decoder_mask = torch.ones_like(
                labels,
                dtype=torch.bool,
                device=device,
            )

            CTX.encoder_attention_mask = (
                attention_mask
            )

            CTX.decoder_mask = (
                decoder_mask
            )

            model.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                output = model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    labels=labels,
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
                    f"Non-finite loss "
                    f"at calibration example "
                    f"{index}"
                )

            valid_tokens = int(
                labels.numel()
            )

            (
                loss
                * max(
                    valid_tokens,
                    1,
                )
            ).backward()

            model.zero_grad(
                set_to_none=True
            )

            CTX.encoder_attention_mask = None
            CTX.decoder_mask = None

            if (
                index % 16 == 0
                or
                index
                == CALIBRATION_SIZE
            ):
                print(
                    f"[calibration] "
                    f"{index}/"
                    f"{CALIBRATION_SIZE}",
                    flush=True,
                )

    finally:
        CTX.encoder_attention_mask = None
        CTX.decoder_mask = None

        for stat in stats.values():
            stat.remove()

    modules = {}

    for name in sorted(
        stats
    ):
        stat = stats[
            name
        ]

        if stat.positions <= 0:
            raise RuntimeError(
                f"No positions for "
                f"{name}"
            )

        modules[name] = {
            "cross_sum":
                stat.cross
                .detach()
                .cpu(),

            "activation_second_sum":
                stat.activation_second
                .detach()
                .cpu(),

            "gradient_second_sum":
                stat.gradient_second
                .detach()
                .cpu(),

            "positions":
                int(
                    stat.positions
                ),
        }

    payload = {
        "schema":
            "whisper_task_stats_v1",

        "language":
            language,

        "base_model":
            BASE_MODEL,

        "whisper_language":
            whisper_language,

        "task":
            "transcribe",

        "calibration_size":
            CALIBRATION_SIZE,

        "calibration_seed":
            CALIBRATION_SEED,

        "damping":
            DAMPING,

        "selected_indices":
            selected_indices,

        "target_count":
            len(modules),

        "modules":
            modules,
    }

    save_atomic(
        payload,
        out_path,
    )

    metadata = {
        "schema":
            payload["schema"],

        "language":
            language,

        "base_model":
            BASE_MODEL,

        "whisper_language":
            whisper_language,

        "task":
            "transcribe",

        "calibration_size":
            CALIBRATION_SIZE,

        "calibration_seed":
            CALIBRATION_SEED,

        "damping":
            DAMPING,

        "train_examples":
            len(train_ds),

        "selected_indices":
            selected_indices,

        "target_count":
            len(modules),

        "task_stats_path":
            str(out_path),
    }

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[SAVED] {out_path}",
        flush=True,
    )

    print(
        f"[targets] {len(modules)}",
        flush=True,
    )

    del model
    del stats
    del modules

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
