from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from scripts_five_fold.languageexpertfusion.config import (
    ENGLISH_DOM_MAX_SPANISH_FRACTION,
    ENGLISH_DOM_ROOT,
    MIN_CLASSIFIED_WORDS,
    MIAMI_PROCESSED_ROOT,
    SPANISH_DOM_MAX_ENGLISH_FRACTION,
    SPANISH_DOM_ROOT,
)


# Input manifests

INPUT_MANIFESTS = {
    "train": (
        MIAMI_PROCESSED_ROOT
        / "train_manifest.csv"
    ),
    "validation": (
        MIAMI_PROCESSED_ROOT
        / "validation_manifest.csv"
    ),
    "test": (
        MIAMI_PROCESSED_ROOT
        / "test_manifest.csv"
    ),
}


# Raw CHAT files

MIAMI_RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "miami_bangor"
)


# CHAT patterns

LANGUAGES_HEADER_RE = re.compile(
    r"^@Languages:\s*(.+)$",
    re.IGNORECASE,
)

PRECODE_RE = re.compile(
    r"\[-\s*(eng|spa)\s*\]",
    re.IGNORECASE,
)

LANGUAGE_TAG_RE = re.compile(
    r"@s:([A-Za-z0-9_&+\-]+)",
    re.IGNORECASE,
)

TIMING_RE = re.compile(
    r"\x15\s*\d+[_ ]\d+\s*\x15"
)

BRACKET_RE = re.compile(
    r"\[[^\]]*\]"
)

ANGLE_RE = re.compile(
    r"<([^>]*)>"
)

CONTROL_RE = re.compile(
    r"(?<!\S)&[=+\-~]?\S+"
)

LEXICAL_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿÑñÁÉÍÓÚÜáéíóúü]"
)

PARENTHETICAL_SHORTENING_RE = re.compile(
    r"\(([^()\s]+)\)"
)

REPETITION_INDEX_RE = re.compile(
    r"\(\d+\)"
)


# CHAT tokens that are not actual lexical words

SKIP_TOKENS = {
    "",
    "xxx",
    "yyy",
    "www",
    "0",
    "(.)",
    "(..)",
    "(...)",
}


# Helpers

def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input manifest:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    if not rows:
        raise RuntimeError(
            f"Input manifest is empty:\n{path}"
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fallback_columns: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    extra_columns = [
        "language_default",
        "english_words",
        "spanish_words",
        "ambiguous_words",
        "classified_words",
        "english_fraction",
        "spanish_fraction",
        "language_domain",
    ]

    fieldnames: list[str] = []

    for column in fallback_columns + extra_columns:
        if column not in fieldnames:
            fieldnames.append(
                column
            )

    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(
                    column
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


# CHAT source indexing

def build_chat_index() -> dict[str, Path]:
    if not MIAMI_RAW_ROOT.exists():
        raise FileNotFoundError(
            "Miami raw directory does not exist:\n"
            f"{MIAMI_RAW_ROOT}"
        )

    chat_paths = sorted(
        MIAMI_RAW_ROOT.rglob(
            "*.cha"
        )
    )

    if not chat_paths:
        raise FileNotFoundError(
            "No .cha files found under:\n"
            f"{MIAMI_RAW_ROOT}"
        )

    index: dict[str, Path] = {}

    for path in chat_paths:
        key = path.stem.casefold()

        if key not in index:
            index[key] = path.resolve()

    print(
        f"Indexed CHAT files: {len(index)}"
    )

    return index


def normalise_language(
    value: str,
) -> str | None:
    value = (
        value.strip()
        .casefold()
        .replace("-", "")
        .replace("_", "")
    )

    if value in {
        "eng",
        "en",
        "english",
    }:
        return "eng"

    if value in {
        "spa",
        "es",
        "spanish",
    }:
        return "spa"

    return None


def read_chat_default_language(
    path: Path,
) -> str | None:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            match = (
                LANGUAGES_HEADER_RE.match(
                    line.strip()
                )
            )

            if match is None:
                continue

            languages = [
                item.strip()
                for item in re.split(
                    r"[,;\s]+",
                    match.group(1),
                )
                if item.strip()
            ]

            for language in languages:
                normalised = (
                    normalise_language(
                        language
                    )
                )

                if normalised is not None:
                    return normalised

            return None

    return None


def build_default_language_map(
    chat_index: dict[str, Path],
) -> dict[str, str]:
    defaults: dict[str, str] = {}

    counts: Counter[str] = Counter()

    for stem, path in chat_index.items():
        language = (
            read_chat_default_language(
                path
            )
        )

        if language is None:
            counts[
                "missing_default"
            ] += 1
            continue

        defaults[stem] = language
        counts[language] += 1

    print()
    print("CHAT default languages")
    print(
        f"English-default files: "
        f"{counts['eng']}"
    )
    print(
        f"Spanish-default files: "
        f"{counts['spa']}"
    )
    print(
        f"Unknown-default files: "
        f"{counts['missing_default']}"
    )

    return defaults


# Resolve a manifest row to its CHAT conversation

def candidate_conversation_values(
    row: dict[str, str],
) -> list[str]:
    candidates: list[str] = []

    direct_columns = [
        "conversation_id",
        "conversation",
        "chat_file",
        "chat_path",
        "source_transcript_path",
        "transcript_path",
        "source_file",
        "file",
        "recording",
        "recording_id",
        "media_stem",
    ]

    for column in direct_columns:
        value = (
            row.get(
                column,
                ""
            )
            .strip()
        )

        if value:
            candidates.append(
                Path(value).stem
            )

    # Existing audio filenames often preserve
    # the original Miami conversation name.
    audio_path = (
        row.get(
            "audio_path",
            ""
        )
        .strip()
    )

    if audio_path:
        stem = Path(
            audio_path
        ).stem

        candidates.append(
            stem
        )

        # Handles names such as:
        # zeledon6_ABC_000123_000456.wav
        parts = stem.split(
            "_"
        )

        if parts:
            candidates.append(
                parts[0]
            )

    result: list[str] = []

    for value in candidates:
        key = value.casefold()

        if key and key not in result:
            result.append(
                key
            )

    return result


def resolve_conversation_stem(
    row: dict[str, str],
    chat_index: dict[str, Path],
) -> str | None:
    candidates = (
        candidate_conversation_values(
            row
        )
    )

    # Exact match first.
    for candidate in candidates:
        if candidate in chat_index:
            return candidate

    # Then permit filenames with additional
    # suffixes after the conversation identifier.
    for candidate in candidates:
        matches = [
            stem
            for stem in chat_index
            if (
                candidate.startswith(
                    stem + "_"
                )
                or candidate.startswith(
                    stem + "-"
                )
            )
        ]

        if len(matches) == 1:
            return matches[0]

    return None


# Language counting

def utterance_precode(
    transcript_raw: str,
) -> str | None:
    match = PRECODE_RE.search(
        transcript_raw
    )

    if match is None:
        return None

    return normalise_language(
        match.group(1)
    )


def classify_tag(
    tag: str,
) -> str:
    value = (
        tag.casefold()
        .replace("_", "&")
        .replace("+", "&")
    )

    pieces = {
        piece
        for piece in value.split(
            "&"
        )
        if piece
    }

    has_english = bool(
        pieces
        & {
            "eng",
            "en",
            "english",
        }
    )

    has_spanish = bool(
        pieces
        & {
            "spa",
            "es",
            "spanish",
        }
    )

    if (
        has_english
        and has_spanish
    ):
        return "ambiguous"

    if has_english:
        return "eng"

    if has_spanish:
        return "spa"

    return "ambiguous"


def lexical_tokens(
    transcript_raw: str,
) -> list[str]:
    text = TIMING_RE.sub(
        " ",
        transcript_raw,
    )

    text = ANGLE_RE.sub(
        r"\1",
        text,
    )

    # Preserve @s language tags.
    # Remove CHAT bracket annotations such as
    # repetitions, retracing and comments.
    text = BRACKET_RE.sub(
        " ",
        text,
    )

    text = CONTROL_RE.sub(
        " ",
        text,
    )

    # Remove the utterance language precode itself.
    text = PRECODE_RE.sub(
        " ",
        text,
    )

    tokens: list[str] = []

    for raw_token in text.split():
        token = REPETITION_INDEX_RE.sub(
            "",
            raw_token,
        )

        token = (
            PARENTHETICAL_SHORTENING_RE.sub(
                r"\1",
                token,
            )
        )

        token = token.strip(
            " \t\r\n"
            ".,!?;:\"'()[]{}"
            "+<>"
        )

        # Look at the lexical part only when
        # deciding whether this is a word.
        lexical_part = (
            LANGUAGE_TAG_RE.sub(
                "",
                token,
            )
        )

        lexical_part = lexical_part.strip(
            " \t\r\n"
            ".,!?;:\"'()[]{}"
            "+<>"
        )

        if (
            not lexical_part
            or lexical_part.casefold()
            in SKIP_TOKENS
        ):
            continue

        if not LEXICAL_RE.search(
            lexical_part
        ):
            continue

        tokens.append(
            token
        )

    return tokens


def count_languages(
    transcript_raw: str,
    file_default_language: str | None,
) -> dict[str, Any] | None:
    # A [- eng] / [- spa] precode overrides
    # the file's ordinary default language
    # for this utterance.
    default_language = (
        utterance_precode(
            transcript_raw
        )
        or file_default_language
    )

    if default_language not in {
        "eng",
        "spa",
    }:
        return None

    english_words = 0
    spanish_words = 0
    ambiguous_words = 0

    for token in lexical_tokens(
        transcript_raw
    ):
        tag_match = (
            LANGUAGE_TAG_RE.search(
                token
            )
        )

        if tag_match is not None:
            language = classify_tag(
                tag_match.group(1)
            )
        else:
            # THIS is the important correction.
            # Untagged words inherit the current
            # CHAT default language. They are NOT
            # automatically English.
            language = default_language

        if language == "eng":
            english_words += 1

        elif language == "spa":
            spanish_words += 1

        else:
            ambiguous_words += 1

    classified_words = (
        english_words
        + spanish_words
    )

    if (
        classified_words
        < MIN_CLASSIFIED_WORDS
    ):
        return None

    english_fraction = (
        english_words
        / classified_words
    )

    spanish_fraction = (
        spanish_words
        / classified_words
    )

    return {
        "language_default": (
            default_language
        ),
        "english_words": (
            english_words
        ),
        "spanish_words": (
            spanish_words
        ),
        "ambiguous_words": (
            ambiguous_words
        ),
        "classified_words": (
            classified_words
        ),
        "english_fraction": round(
            english_fraction,
            6,
        ),
        "spanish_fraction": round(
            spanish_fraction,
            6,
        ),
    }


# Process one existing manifest split

def process_split(
    split: str,
    input_path: Path,
    chat_index: dict[str, Path],
    default_languages: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows = read_csv(
        input_path
    )

    if "transcript_raw" not in rows[0]:
        raise ValueError(
            "The existing Miami manifest does not "
            "contain a transcript_raw column.\n"
            f"Manifest: {input_path}\n"
            f"Columns: {list(rows[0].keys())}"
        )

    spanish_rows: list[
        dict[str, Any]
    ] = []

    english_rows: list[
        dict[str, Any]
    ] = []

    stats: Counter[str] = Counter()

    for row in rows:
        transcript_raw = (
            row.get(
                "transcript_raw",
                ""
            )
            .strip()
        )

        if not transcript_raw:
            stats[
                "missing_transcript_raw"
            ] += 1
            continue

        precode = utterance_precode(
            transcript_raw
        )

        conversation_stem = (
            resolve_conversation_stem(
                row,
                chat_index,
            )
        )

        file_default: (
            str
            | None
        ) = None

        if conversation_stem is not None:
            file_default = (
                default_languages.get(
                    conversation_stem
                )
            )

        # If the utterance carries its own
        # precode, we do not actually need to
        # know the file default.
        if (
            precode is None
            and file_default is None
        ):
            stats[
                "unresolved_default"
            ] += 1
            continue

        counts = count_languages(
            transcript_raw,
            file_default,
        )

        if counts is None:
            stats[
                "too_few_classified"
            ] += 1
            continue

        enriched = dict(
            row
        )

        enriched.update(
            counts
        )

        english_fraction = float(
            counts[
                "english_fraction"
            ]
        )

        spanish_fraction = float(
            counts[
                "spanish_fraction"
            ]
        )

        # Spanish specialist:
        # no more than 30% English by default.
        if (
            english_fraction
            <= SPANISH_DOM_MAX_ENGLISH_FRACTION
        ):
            spanish_row = dict(
                enriched
            )

            spanish_row[
                "language_domain"
            ] = "spanish_dominant"

            spanish_rows.append(
                spanish_row
            )

            stats[
                "spanish_dominant"
            ] += 1

        # English specialist:
        # e.g. Spanish <= 0.10 means
        # English >= 0.90.
        if (
            spanish_fraction
            <= ENGLISH_DOM_MAX_SPANISH_FRACTION
        ):
            english_row = dict(
                enriched
            )

            english_row[
                "language_domain"
            ] = "english_dominant"

            english_rows.append(
                english_row
            )

            stats[
                "english_dominant"
            ] += 1

        if (
            english_fraction
            > SPANISH_DOM_MAX_ENGLISH_FRACTION
            and spanish_fraction
            > ENGLISH_DOM_MAX_SPANISH_FRACTION
        ):
            stats[
                "middle_mixed"
            ] += 1

    print()
    print("=" * 80)
    print(
        f"SPLIT: {split.upper()}"
    )
    print("=" * 80)
    print(
        f"Input recordings: "
        f"{len(rows)}"
    )
    print(
        f"Spanish-dominant: "
        f"{len(spanish_rows)}"
    )
    print(
        f"English-dominant: "
        f"{len(english_rows)}"
    )
    print(
        f"Mixed / excluded: "
        f"{stats['middle_mixed']}"
    )
    print(
        f"Too few classified words: "
        f"{stats['too_few_classified']}"
    )
    print(
        f"Could not determine CHAT default: "
        f"{stats['unresolved_default']}"
    )
    print(
        f"Missing transcript_raw: "
        f"{stats['missing_transcript_raw']}"
    )

    return (
        spanish_rows,
        english_rows,
    )


# Main

def main() -> None:
    print()
    print("=" * 80)
    print(
        "MIAMI LANGUAGE-DOMINANCE "
        "MANIFEST PREPARATION"
    )
    print("=" * 80)

    print(
        f"Existing Miami data: "
        f"{MIAMI_PROCESSED_ROOT}"
    )

    print(
        "No audio will be copied, "
        "decoded, resampled or rewritten."
    )

    print()
    print(
        "Spanish-dominant threshold:"
    )
    print(
        f"  English fraction <= "
        f"{SPANISH_DOM_MAX_ENGLISH_FRACTION:.2f}"
    )

    print(
        "English-dominant threshold:"
    )
    print(
        f"  Spanish fraction <= "
        f"{ENGLISH_DOM_MAX_SPANISH_FRACTION:.2f}"
    )

    print(
        f"  Equivalent English fraction >= "
        f"{1.0 - ENGLISH_DOM_MAX_SPANISH_FRACTION:.2f}"
    )

    chat_index = (
        build_chat_index()
    )

    default_languages = (
        build_default_language_map(
            chat_index
        )
    )

    total_spanish = 0
    total_english = 0

    for split, input_path in (
        INPUT_MANIFESTS.items()
    ):
        input_rows = read_csv(
            input_path
        )

        original_columns = list(
            input_rows[0].keys()
        )

        (
            spanish_rows,
            english_rows,
        ) = process_split(
            split,
            input_path,
            chat_index,
            default_languages,
        )

        spanish_output = (
            SPANISH_DOM_ROOT
            / f"{split}.csv"
        )

        english_output = (
            ENGLISH_DOM_ROOT
            / f"{split}.csv"
        )

        write_csv(
            spanish_output,
            spanish_rows,
            original_columns,
        )

        write_csv(
            english_output,
            english_rows,
            original_columns,
        )

        total_spanish += len(
            spanish_rows
        )

        total_english += len(
            english_rows
        )

        print(
            f"Saved Spanish manifest: "
            f"{spanish_output}"
        )

        print(
            f"Saved English manifest: "
            f"{english_output}"
        )

    print()
    print("=" * 80)
    print(
        "PREPARATION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Total Spanish-dominant recordings: "
        f"{total_spanish}"
    )

    print(
        f"Total English-dominant recordings: "
        f"{total_english}"
    )

    print()
    print(
        "Spanish manifests:"
    )
    print(
        SPANISH_DOM_ROOT
    )

    print(
        "English manifests:"
    )
    print(
        ENGLISH_DOM_ROOT
    )


if __name__ == "__main__":
    main()