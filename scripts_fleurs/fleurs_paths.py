
from pathlib import Path
from datasets import load_from_disk

CORPUS = Path("/data/milsrg1/corpora/fleurs-selected-languages")

def dataset_root(language):
    candidates = (
        CORPUS / "fleurs" / f"FULL_{language}",
        CORPUS / f"FULL_{language}",
        CORPUS / language,
        CORPUS / "fleurs" / language,
    )
    for path in candidates:
        if (path / "dataset_dict.json").is_file():
            return path
        if (path / "train").is_dir():
            return path
    raise FileNotFoundError(
        f"No saved FLEURS dataset for {language}: {candidates}"
    )

def load_fleurs_split(language, split):
    root = dataset_root(language)
    if (root / "dataset_dict.json").is_file():
        return load_from_disk(str(root))[split]
    return load_from_disk(str(root / split))

def test_manifest_path(language):
    root = dataset_root(language)
    candidates = (
        root / "test_manifest.csv",
        CORPUS / language / "test_manifest.csv",
        CORPUS / "fleurs" / f"FULL_{language}" / "test_manifest.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Fixed-test manifest missing for {language}: {candidates}"
    )
