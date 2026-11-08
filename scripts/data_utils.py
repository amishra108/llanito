from __future__ import annotations

from pathlib import Path

import numpy as np
from praatio import textgrid
from scipy.io import wavfile
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000

IGNORED_LABELS = {
    "",
    "sil",
    "sp",
    "spn",
    "<sil>",
    "<unk>",
}


def convert_integer_audio_to_float(
    audio: np.ndarray,
) -> np.ndarray:
    if audio.dtype == np.uint8:
        audio = audio.astype(np.float32)
        return (audio - 128.0) / 128.0

    dtype_info = np.iinfo(audio.dtype)

    scale = float(
        max(
            abs(dtype_info.min),
            abs(dtype_info.max),
        )
    )

    return audio.astype(np.float32) / scale


def load_wav_for_whisper(
    wav_path: Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    print_audio_check: bool = False,
) -> np.ndarray:
    sample_rate, audio = wavfile.read(str(wav_path))

    if audio.size == 0:
        raise ValueError(
            f"Audio file contains no samples: {wav_path}"
        )

    if np.issubdtype(audio.dtype, np.integer):
        audio = convert_integer_audio_to_float(audio)
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(
            f"Unsupported audio shape {audio.shape}: {wav_path}"
        )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if sample_rate != target_sample_rate:
        greatest_common_divisor = np.gcd(
            sample_rate,
            target_sample_rate,
        )

        up_factor = (
            target_sample_rate
            // greatest_common_divisor
        )

        down_factor = (
            sample_rate
            // greatest_common_divisor
        )

        audio = resample_poly(
            audio,
            up_factor,
            down_factor,
        ).astype(np.float32)

    audio = np.ascontiguousarray(
        audio,
        dtype=np.float32,
    )

    peak_amplitude = float(
        np.max(np.abs(audio))
    )

    rms_amplitude = float(
        np.sqrt(np.mean(audio ** 2))
    )

    if print_audio_check:
        print(
            f"Audio check for {wav_path.name}: "
            f"original_sample_rate={sample_rate}, "
            f"output_sample_rate={target_sample_rate}, "
            f"samples={len(audio)}, "
            f"peak={peak_amplitude:.6f}, "
            f"rms={rms_amplitude:.6f}"
        )

    if peak_amplitude == 0.0:
        raise ValueError(
            f"Audio contains only silence: {wav_path}"
        )

    return audio


def get_audio_duration_seconds(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> float:
    return len(audio) / sample_rate

def find_tier_name(
    tier_names,
    requested_name: str,
) -> str | None:
    requested_lower = requested_name.casefold()

    for tier_name in tier_names:
        if tier_name.casefold() == requested_lower:
            return tier_name

    return None


def unpack_interval_entry(
    entry,
) -> tuple[float, float, str]:
    if hasattr(entry, "start"):
        return (
            float(entry.start),
            float(entry.end),
            str(entry.label),
        )

    if len(entry) == 3:
        start, end, label = entry

        return (
            float(start),
            float(end),
            str(label),
        )

    raise ValueError(
        f"Unrecognised TextGrid interval: {entry!r}"
    )


def read_gold_transcript(
    textgrid_path: Path,
    lowercase: bool = True,
) -> str:
    tg = textgrid.openTextgrid(
        str(textgrid_path),
        includeEmptyIntervals=True,
    )

    word_tier_name = find_tier_name(
        tg.tierNames,
        "words",
    )

    if word_tier_name is None:
        raise ValueError(
            f"No words tier found in {textgrid_path}. "
            f"Available tiers: {tuple(tg.tierNames)}"
        )

    word_tier = tg.getTier(word_tier_name)

    words: list[str] = []

    for entry in word_tier.entries:
        _, _, label = unpack_interval_entry(entry)

        label = " ".join(label.strip().split())

        if lowercase:
            label = label.lower()

        if label.casefold() in IGNORED_LABELS:
            continue

        if label:
            words.extend(label.split())

    transcript = " ".join(words)

    if not transcript:
        raise ValueError(
            f"Empty words tier in {textgrid_path}"
        )

    return transcript


def find_matching_textgrid(
    wav_path: Path,
) -> Path | None:
    for path in wav_path.parent.iterdir():
        if (
            path.is_file()
            and path.stem.casefold()
            == wav_path.stem.casefold()
            and path.suffix.casefold()
            == ".textgrid"
        ):
            return path

    return None