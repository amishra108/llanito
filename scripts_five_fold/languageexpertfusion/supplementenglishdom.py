#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile

# Project
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_five_fold.data_utils import load_wav_for_whisper

# Raw Miami-Bangor
CORPUS_ROOT = PROJECT_ROOT / "data" / "raw" / "miami_bangor"
TRANSCRIPT_ROOT = CORPUS_ROOT / "transcripts"
MEDIA_ROOT = CORPUS_ROOT / "media"
DECODED_WAV_ROOT = CORPUS_ROOT / "decoded_wav"

# Existing processed English-dominant dataset
# Uses miamienglishdom if present, otherwise the older englishdom name.
NEW_ROOT = PROJECT_ROOT / "data" / "processed" / "miamienglishdom"
OLD_ROOT = PROJECT_ROOT / "data" / "processed" / "englishdom"
OUTPUT_ROOT = NEW_ROOT if NEW_ROOT.exists() or not OLD_ROOT.exists() else OLD_ROOT

# Existing Miami-larger clips can be reused when available.
MIAMI_LARGER_ROOT = PROJECT_ROOT / "data" / "processed" / "miami_seqstack_larger"

# Supplement with all 70%-100% English utterances.
MIN_ENGLISH_FRACTION = 0.70
MAX_ENGLISH_FRACTION = 1.00
MIN_CLASSIFIED_WORDS = 3
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 20.0

# Audio
TARGET_SAMPLE_RATE = 16_000
OVERWRITE_AUDIO_CLIPS = False

# New speakers keep a deterministic 80/10/10 split.
# Existing speakers always stay in their existing split.
RANDOM_SEED = 2026
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10

# CHAT parsing
CHAT_TIME_RE = re.compile(r"\x15\s*(\d+)[_ ](\d+)\s*\x15")
CHAT_SPEAKER_RE = re.compile(r"^\*([^:]+):\s*(.*)$")
CHAT_PARTICIPANTS_RE = re.compile(r"^@Participants:\s*(.*)$", re.IGNORECASE)
CHAT_MEDIA_RE = re.compile(r"^@Media:\s*([^,\s]+)", re.IGNORECASE)
CHAT_PRECODE_RE = re.compile(r"\[-\s*(eng|spa)\s*\]", re.IGNORECASE)
LANGUAGE_TAG_RE = re.compile(r"@s:([A-Za-z0-9_&+-]+)", re.IGNORECASE)
ANGLE_GROUP_RE = re.compile(r"<([^>]*)>")
BRACKETED_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")
CHAT_CONTROL_RE = re.compile(r"(?<!\S)&[=+\-~]?\S+")
CHAT_REPARANDUM_RE = re.compile(r"<[^>]*>\s*\[(?://|///)\]", re.IGNORECASE)
LEXICAL_PARENTHESES_RE = re.compile(r"\(([^()\s]+)\)")
LEXICAL_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿÑñÁÉÍÓÚÜáéíóúü]")
WHITESPACE_RE = re.compile(r"\s+")

SKIP_TOKENS = {"", "xxx", "yyy", "www", "0", "(.)", "(..)", "(...)"}
CHAT_MARKERS = {"+/.", "+//.", "+...", "+<", "+>", "+^", "+,", '+"', '+"/.'}


@dataclass(frozen=True)
class LanguageCounts:
    english_words: int
    spanish_words: int
    undetermined_words: int
    ignored_words: int

    @property
    def classified_words(self) -> int:
        return self.english_words + self.spanish_words

    @property
    def english_fraction(self) -> float:
        if self.classified_words == 0:
            return 0.0
        return self.english_words / self.classified_words


@dataclass(frozen=True)
class Utterance:
    conversation_id: str
    media_stem: str
    speaker: str
    speaker_code: str
    source_transcript_path: str
    start_ms: int
    end_ms: int
    transcript_raw: str
    transcript: str
    english_words: int
    spanish_words: int
    undetermined_words: int
    ignored_words: int
    english_fraction: float

    @property
    def duration_seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    @property
    def example_id(self) -> str:
        return (
            f"{safe_identifier(self.conversation_id)}_"
            f"{safe_identifier(self.speaker_code)}_"
            f"{self.start_ms:09d}_{self.end_ms:09d}"
        )


# Manifest helpers

def find_manifest(split: str) -> Path:
    for name in [f"{split}.csv", f"{split}_manifest.csv"]:
        path = OUTPUT_ROOT / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find existing {split} manifest under {OUTPUT_ROOT}"
    )


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def key_from_row(row: dict[str, str]) -> tuple[str, str, str, str] | None:
    conversation = row.get("conversation_id", "").strip()
    speaker_code = row.get("speaker_code", "").strip()
    start_ms = row.get("start_ms", "").strip()
    end_ms = row.get("end_ms", "").strip()

    if conversation and speaker_code and start_ms and end_ms:
        return (
            conversation.casefold(),
            speaker_code.casefold(),
            start_ms,
            end_ms,
        )

    return None


def load_existing_state():
    manifests = {}
    fields = {}
    existing_rows = {}
    speaker_split: dict[str, str] = {}
    existing_keys: set[tuple[str, str, str, str]] = set()

    for split in ["train", "validation", "test"]:
        path = find_manifest(split)
        rows, fieldnames = read_csv(path)

        manifests[split] = path
        fields[split] = fieldnames
        existing_rows[split] = rows

        for row in rows:
            speaker = row.get("speaker", "").strip()
            if speaker:
                key = speaker.casefold()
                previous = speaker_split.get(key)
                if previous is not None and previous != split:
                    raise RuntimeError(
                        f"Existing speaker leakage: {speaker!r} is in both "
                        f"{previous} and {split}."
                    )
                speaker_split[key] = split

            row_key = key_from_row(row)
            if row_key is not None:
                existing_keys.add(row_key)

    return manifests, fields, existing_rows, speaker_split, existing_keys


# CHAT parsing

def merge_continuation_lines(lines: list[str]) -> list[str]:
    logical_lines = []
    current = ""

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")

        if line.startswith(("*", "%", "@")):
            if current:
                logical_lines.append(current)
            current = line
        elif current and line[:1].isspace():
            current += " " + line.strip()
        else:
            if current:
                logical_lines.append(current)
                current = ""
            if line.strip():
                logical_lines.append(line.strip())

    if current:
        logical_lines.append(current)

    return logical_lines


def parse_participants(lines: Iterable[str]) -> dict[str, str]:
    names = {}

    for line in lines:
        match = CHAT_PARTICIPANTS_RE.match(line)
        if not match:
            continue

        for entry in match.group(1).split(","):
            parts = entry.strip().split()
            if not parts:
                continue

            code = parts[0].strip()

            if len(parts) >= 3:
                name = "_".join(parts[1:-1]).strip("_")
            elif len(parts) == 2:
                name = parts[1].strip()
            else:
                name = code

            names[code] = name or code

    return names


def find_media_stem(lines: Iterable[str], transcript_path: Path) -> str:
    for line in lines:
        match = CHAT_MEDIA_RE.match(line)
        if match:
            return Path(match.group(1).strip()).stem
    return transcript_path.stem


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.casefold().replace("_", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_chat_transcript(raw_text: str) -> str:
    text = CHAT_TIME_RE.sub(" ", raw_text)
    text = CHAT_REPARANDUM_RE.sub(" ", text)
    text = ANGLE_GROUP_RE.sub(r"\1", text)
    text = BRACKETED_ANNOTATION_RE.sub(" ", text)
    text = CHAT_CONTROL_RE.sub(" ", text)
    text = CHAT_PRECODE_RE.sub(" ", text)
    text = LANGUAGE_TAG_RE.sub("", text)

    for marker in CHAT_MARKERS:
        text = text.replace(marker, " ")

    tokens = []

    for token in text.split():
        token = re.sub(r"\(\d+\)", "", token)
        token = LEXICAL_PARENTHESES_RE.sub(r"\1", token)
        token = token.strip(" \t\r\n.,!?;:\"'()[]{}")

        if not token or token.casefold() in SKIP_TOKENS:
            continue
        if not LEXICAL_RE.search(token):
            continue

        tokens.append(token)

    return normalize_transcript(" ".join(tokens))


def classify_language_tag(tag: str) -> str:
    value = tag.casefold()

    if value in {"eng", "en"}:
        return "english"
    if value in {"spa", "es"}:
        return "spanish"

    # Same rules as the user's existing Miami preparation.
    if value in {"spa+eng", "eng+spa", "es+en", "en+es"}:
        return "spanish"
    if value in {"eng&spa", "spa&eng", "en&es", "es&en"}:
        return "undetermined"

    return "undetermined"


def count_utterance_languages(raw_text: str) -> LanguageCounts:
    spanish_precode = bool(
        re.search(r"\[-\s*spa\s*\]", raw_text, re.IGNORECASE)
    )
    english_precode = bool(
        re.search(r"\[-\s*eng\s*\]", raw_text, re.IGNORECASE)
    )

    text = CHAT_TIME_RE.sub(" ", raw_text)
    text = CHAT_REPARANDUM_RE.sub(" ", text)
    text = ANGLE_GROUP_RE.sub(r"\1", text)
    text = BRACKETED_ANNOTATION_RE.sub(" ", text)
    text = CHAT_PRECODE_RE.sub(" ", text)

    for marker in CHAT_MARKERS:
        text = text.replace(marker, " ")

    english_words = 0
    spanish_words = 0
    undetermined_words = 0
    ignored_words = 0

    for raw_token in text.split():
        tag_match = LANGUAGE_TAG_RE.search(raw_token)

        token = LANGUAGE_TAG_RE.sub("", raw_token)
        token = CHAT_CONTROL_RE.sub("", token)
        token = re.sub(r"\(\d+\)", "", token)
        token = LEXICAL_PARENTHESES_RE.sub(r"\1", token)
        token = token.strip(" \t\r\n.,!?;:\"'()[]{}")

        if (
            not token
            or token.casefold() in SKIP_TOKENS
            or not LEXICAL_RE.search(token)
        ):
            ignored_words += 1
            continue

        if tag_match:
            language = classify_language_tag(tag_match.group(1))
            if language == "english":
                english_words += 1
            elif language == "spanish":
                spanish_words += 1
            else:
                undetermined_words += 1
            continue

        # Preserve the original Miami rule from your preparation code:
        # untagged words count as English unless [- spa] applies.
        if spanish_precode:
            spanish_words += 1
        elif english_precode:
            english_words += 1
        else:
            english_words += 1

    return LanguageCounts(
        english_words=english_words,
        spanish_words=spanish_words,
        undetermined_words=undetermined_words,
        ignored_words=ignored_words,
    )


def parse_raw_english_heavy() -> list[Utterance]:
    chat_files = sorted(
        TRANSCRIPT_ROOT.rglob("*.cha"),
        key=lambda path: str(path).casefold(),
    )

    if not chat_files:
        raise FileNotFoundError(f"No CHAT transcripts under {TRANSCRIPT_ROOT}")

    rows = []

    for index, transcript_path in enumerate(chat_files, start=1):
        lines = merge_continuation_lines(
            transcript_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        )

        speaker_names = parse_participants(lines)
        media_stem = find_media_stem(lines, transcript_path)

        for line in lines:
            speaker_match = CHAT_SPEAKER_RE.match(line)
            if not speaker_match:
                continue

            speaker_code = speaker_match.group(1).strip()
            if speaker_code.casefold() in {"www", "unk", "unknown", "ose"}:
                continue

            raw = speaker_match.group(2).strip()
            timing_matches = list(CHAT_TIME_RE.finditer(raw))
            if not timing_matches:
                continue

            timing = timing_matches[-1]
            start_ms = int(timing.group(1))
            end_ms = int(timing.group(2))

            if end_ms <= start_ms:
                continue

            duration = (end_ms - start_ms) / 1000.0
            if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
                continue

            counts = count_utterance_languages(raw)
            if counts.classified_words < MIN_CLASSIFIED_WORDS:
                continue

            if not (
                MIN_ENGLISH_FRACTION
                <= counts.english_fraction
                <= MAX_ENGLISH_FRACTION
            ):
                continue

            transcript = clean_chat_transcript(raw)
            if not transcript:
                continue

            rows.append(
                Utterance(
                    conversation_id=transcript_path.stem,
                    media_stem=media_stem,
                    speaker=speaker_names.get(speaker_code, speaker_code),
                    speaker_code=speaker_code,
                    source_transcript_path=str(transcript_path.resolve()),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    transcript_raw=raw,
                    transcript=transcript,
                    english_words=counts.english_words,
                    spanish_words=counts.spanish_words,
                    undetermined_words=counts.undetermined_words,
                    ignored_words=counts.ignored_words,
                    english_fraction=round(counts.english_fraction, 6),
                )
            )

        if index % 10 == 0 or index == len(chat_files):
            print(
                f"Parsed {index}/{len(chat_files)} CHAT files; "
                f"{len(rows)} utterances at >=70% English."
            )

    return rows


# Split assignment

def unseen_speaker_split(speaker: str) -> str:
    digest = hashlib.sha256(
        f"{RANDOM_SEED}:{speaker.casefold()}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)

    if value < TRAIN_FRACTION:
        return "train"
    if value < TRAIN_FRACTION + VALIDATION_FRACTION:
        return "validation"
    return "test"


def assign_splits(
    rows: list[Utterance],
    existing_speaker_split: dict[str, str],
) -> dict[str, list[Utterance]]:
    result = {"train": [], "validation": [], "test": []}
    new_assignments: dict[str, str] = {}

    for row in rows:
        speaker_key = row.speaker.casefold()
        split = existing_speaker_split.get(speaker_key)

        if split is None:
            split = new_assignments.get(speaker_key)

        if split is None:
            split = unseen_speaker_split(row.speaker)
            new_assignments[speaker_key] = split

        result[split].append(row)

    (OUTPUT_ROOT / "supplement_unseen_speaker_splits.json").write_text(
        json.dumps(new_assignments, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return result


# Audio helpers

def safe_identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return result or "unknown"


def build_existing_clip_index() -> dict[str, Path]:
    index = {}

    if not MIAMI_LARGER_ROOT.exists():
        return index

    for path in MIAMI_LARGER_ROOT.rglob("*.wav"):
        index.setdefault(path.stem.casefold(), path.resolve())

    print(f"Indexed {len(index)} existing Miami-larger clips for reuse.")
    return index


def build_media_index() -> dict[str, Path]:
    suffixes = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".mp4"}
    index = {}

    # Prefer decoded WAVs over MP3/media originals.
    for root in [DECODED_WAV_ROOT, MEDIA_ROOT]:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in suffixes:
                index.setdefault(path.stem.casefold(), path.resolve())

    if not index:
        raise FileNotFoundError("No Miami media files found.")

    return index


def decode_source_to_wav(source_path: Path) -> Path:
    if source_path.suffix.casefold() == ".wav":
        return source_path

    decoded_path = DECODED_WAV_ROOT / f"{source_path.stem}.wav"

    if decoded_path.exists() and decoded_path.stat().st_size > 0:
        return decoded_path.resolve()

    try:
        import miniaudio
    except Exception as error:
        raise RuntimeError(
            "Need miniaudio for raw MP3s. Install once with:\n"
            "./.venv/bin/pip install miniaudio"
        ) from error

    print(f"Decoding once: {source_path.name}")

    decoded = miniaudio.decode_file(
        str(source_path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
    )

    audio = np.frombuffer(decoded.samples, dtype=np.int16).copy()
    if audio.size == 0:
        raise ValueError(f"Decoded audio is empty: {source_path}")

    DECODED_WAV_ROOT.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(decoded_path), int(decoded.sample_rate), audio)

    return decoded_path.resolve()


def write_clip(
    full_audio: np.ndarray,
    output_path: Path,
    start_ms: int,
    end_ms: int,
) -> bool:
    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not OVERWRITE_AUDIO_CLIPS
    ):
        return True

    start = round(start_ms * TARGET_SAMPLE_RATE / 1000.0)
    end = round(end_ms * TARGET_SAMPLE_RATE / 1000.0)

    start = max(0, start)
    end = min(len(full_audio), end)

    if end <= start:
        return False

    clip = np.ascontiguousarray(full_audio[start:end], dtype=np.float32)

    if (
        clip.size == 0
        or not np.isfinite(clip).all()
        or float(np.max(np.abs(clip))) == 0.0
    ):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip_int16 = (
        np.clip(clip, -1.0, 1.0) * 32767.0
    ).astype(np.int16)

    wavfile.write(str(output_path), TARGET_SAMPLE_RATE, clip_int16)
    return True


def make_manifest_row(
    row: Utterance,
    split: str,
    audio_path: Path,
    source_audio_path: Path,
) -> dict[str, object]:
    return {
        "example_id": row.example_id,
        "conversation_id": row.conversation_id,
        "speaker": row.speaker,
        "speaker_code": row.speaker_code,
        "split": split,
        "audio_path": str(audio_path.resolve()),
        "transcript": row.transcript,
        "transcript_raw": row.transcript_raw,
        "source_audio_path": str(source_audio_path.resolve()),
        "source_transcript_path": row.source_transcript_path,
        "start_ms": row.start_ms,
        "end_ms": row.end_ms,
        "duration_seconds": round(row.duration_seconds, 3),
        "english_words": row.english_words,
        "spanish_words": row.spanish_words,
        "undetermined_words": row.undetermined_words,
        "ignored_words": row.ignored_words,
        "classified_words": row.english_words + row.spanish_words,
        "english_fraction": row.english_fraction,
        "spanish_fraction": round(1.0 - row.english_fraction, 6),
        "is_code_switched": row.english_words > 0 and row.spanish_words > 0,
        "language_domain": "english_dominant",
        "task": "miami",
    }


def append_manifest(
    path: Path,
    existing_rows: list[dict[str, str]],
    fieldnames: list[str],
    new_rows: list[dict[str, object]],
) -> None:
    for row in new_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(new_rows)


# Main

def main() -> None:
    print("=" * 80)
    print("SUPPLEMENT ENGLISH-DOMINANT MIAMI DATA")
    print("=" * 80)
    print(f"Output root: {OUTPUT_ROOT}")
    print("Raw supplement: 70%-100% English")
    print()

    (
        manifests,
        fields,
        existing_rows,
        existing_speaker_split,
        existing_keys,
    ) = load_existing_state()

    print(
        "Existing rows: "
        + ", ".join(
            f"{split}={len(existing_rows[split])}"
            for split in ["train", "validation", "test"]
        )
    )

    raw_candidates = parse_raw_english_heavy()

    new_candidates = []
    duplicate_count = 0

    for row in raw_candidates:
        key = (
            row.conversation_id.casefold(),
            row.speaker_code.casefold(),
            str(row.start_ms),
            str(row.end_ms),
        )

        if key in existing_keys:
            duplicate_count += 1
            continue

        existing_keys.add(key)
        new_candidates.append(row)

    print()
    print(f"Raw >=70% English utterances: {len(raw_candidates)}")
    print(f"Already in English dataset: {duplicate_count}")
    print(f"New utterances to append: {len(new_candidates)}")

    by_split = assign_splits(new_candidates, existing_speaker_split)

    print(
        "New split sizes: "
        + ", ".join(
            f"{split}={len(by_split[split])}"
            for split in ["train", "validation", "test"]
        )
    )

    existing_clip_index = build_existing_clip_index()
    media_index = build_media_index()

    appended = {"train": [], "validation": [], "test": []}
    reused_clips = 0
    newly_cut_clips = 0

    for split in ["train", "validation", "test"]:
        # Group rows requiring raw audio by conversation/media so each long recording
        # is loaded only once.
        needs_raw: dict[str, list[Utterance]] = defaultdict(list)

        for row in by_split[split]:
            existing_clip = existing_clip_index.get(row.example_id.casefold())

            if existing_clip is not None:
                appended[split].append(
                    make_manifest_row(
                        row,
                        split,
                        existing_clip,
                        existing_clip,
                    )
                )
                reused_clips += 1
            else:
                needs_raw[row.media_stem.casefold()].append(row)

        processed = 0
        raw_total = sum(len(values) for values in needs_raw.values())

        for media_key, utterances in sorted(needs_raw.items()):
            source_path = media_index.get(media_key)

            if source_path is None:
                print(
                    f"WARNING: no media for {utterances[0].media_stem}; "
                    f"skipping {len(utterances)} rows."
                )
                continue

            wav_path = decode_source_to_wav(source_path)
            full_audio = load_wav_for_whisper(
                wav_path,
                target_sample_rate=TARGET_SAMPLE_RATE,
                print_audio_check=False,
            )

            for row in utterances:
                output_audio = (
                    OUTPUT_ROOT
                    / "audio"
                    / split
                    / f"{row.example_id}.wav"
                )

                if write_clip(
                    full_audio,
                    output_audio,
                    row.start_ms,
                    row.end_ms,
                ):
                    appended[split].append(
                        make_manifest_row(
                            row,
                            split,
                            output_audio,
                            wav_path,
                        )
                    )
                    newly_cut_clips += 1

                processed += 1

                if processed % 250 == 0 or processed == raw_total:
                    print(
                        f"{split}: {processed}/{raw_total} raw supplemental clips"
                    )

            del full_audio

    for split in ["train", "validation", "test"]:
        append_manifest(
            manifests[split],
            existing_rows[split],
            fields[split],
            appended[split],
        )

    summary = {
        "minimum_english_fraction": MIN_ENGLISH_FRACTION,
        "maximum_english_fraction": MAX_ENGLISH_FRACTION,
        "raw_candidates": len(raw_candidates),
        "already_present": duplicate_count,
        "new_candidates": len(new_candidates),
        "reused_existing_clips": reused_clips,
        "newly_cut_clips": newly_cut_clips,
        "appended": {
            split: len(appended[split])
            for split in ["train", "validation", "test"]
        },
    }

    (OUTPUT_ROOT / "english_supplement_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SUPPLEMENT COMPLETE")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print()
    print("Rerun-safe: rows already present are not appended twice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
