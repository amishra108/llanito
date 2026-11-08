from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate_loaded_model

from scripts.train import (
    WhisperDataCollator,
    WhisperManifestDataset,
    build_training_arguments,
    configure_whisper_generation,
    load_manifest,
    make_trainer,
    set_seed,
    validate_cuda_environment,
    validate_manifest_files_exist,
    validate_no_audio_overlap,
    validate_no_duplicate_audio,
)

BASE_MODEL = "openai/whisper-medium"
LANGUAGE = "es"
TASK = "transcribe"

RUNS = (
    {
        "name": "lora_r8",
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "learning_rate": 5.0e-4,
    },
    {
        "name": "lora_r128_lr_2p5e4",
        "rank": 128,
        "alpha": 16,
        "dropout": 0.05,
        "learning_rate": 2.5e-4,
    },
)

TARGET_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)

EXPECTED_TARGETS = 288
EXPECTED_TRAINING_EXAMPLES = 904
EXPECTED_TEST_RECORDINGS = 57

EFFECTIVE_BATCH_SIZE = 8
MAX_EPOCHS = 5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 2026

GENERATION_MAX_LENGTH = 225
GENERATION_NUM_BEAMS = 1
DATALOADER_WORKERS = 4
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = MAX_EPOCHS

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "llanito_cv"
TRAIN_MANIFEST = PROCESSED_ROOT / "train_validation_union_manifest.csv"
TEST_MANIFEST = PROCESSED_ROOT / "test_manifest.csv"

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "scripts_fcca" / "lora_union_final_test"

NORMALISATION_FILE = (
    PROJECT_ROOT
    / "scripts"
    / "resources"
    / "LLanito_normalisation_v2.txt"
)

NORMALISATION_MAPPING_COLUMN = "task"
NORMALISATION_SECTION_MARKERS = {
    "passage": "SECTION 1 - PASSAGE RECORDINGS ONLY",
    "sentence": "SECTION 2 - SENTENCE-TASK RECORDINGS ONLY",
}
DATASET_NAME = "llanito_fixed_test"


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def checkpoint_number(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def latest_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    checkpoints = [
        path
        for path in root.iterdir()
        if path.is_dir() and checkpoint_number(path) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_number)


def run_output(name: str) -> Path:
    return OUTPUT_ROOT / name


def checkpoint_root(name: str) -> Path:
    return run_output(name) / "checkpoints"


def evaluation_root(name: str) -> Path:
    return run_output(name) / "test"


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
            f"Unexpected training-set size: expected {EXPECTED_TRAINING_EXAMPLES}, "
            f"found {len(train_rows)}"
        )
    if len(test_rows) != EXPECTED_TEST_RECORDINGS:
        raise RuntimeError(
            f"Unexpected fixed-test size: expected {EXPECTED_TEST_RECORDINGS}, "
            f"found {len(test_rows)}"
        )

    validate_no_duplicate_audio(
        train_rows,
        dataset_name="LoRA train+validation union",
    )
    validate_no_duplicate_audio(
        test_rows,
        dataset_name="Llanito fixed test",
    )
    validate_no_audio_overlap(
        train_rows,
        test_rows,
        first_name="train_validation_union",
        second_name="fixed_test",
    )
    validate_manifest_files_exist(train_rows)
    validate_manifest_files_exist(test_rows)

    return train_rows, test_rows


def attention_target_names(model: nn.Module) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in TARGET_LEAVES:
            continue

        encoder_self = (
            ".encoder.layers." in name
            and ".self_attn." in name
        )
        decoder_self = (
            ".decoder.layers." in name
            and ".self_attn." in name
        )
        decoder_cross = (
            ".decoder.layers." in name
            and ".encoder_attn." in name
        )

        if encoder_self or decoder_self or decoder_cross:
            targets.append(name)

    encoder_count = sum(
        ".encoder.layers." in name and ".self_attn." in name
        for name in targets
    )
    decoder_self_count = sum(
        ".decoder.layers." in name and ".self_attn." in name
        for name in targets
    )
    decoder_cross_count = sum(
        ".decoder.layers." in name and ".encoder_attn." in name
        for name in targets
    )

    actual = (
        encoder_count,
        decoder_self_count,
        decoder_cross_count,
        len(targets),
    )
    expected = (96, 96, 96, EXPECTED_TARGETS)

    if actual != expected:
        raise RuntimeError(
            "Unexpected LoRA placement: "
            f"encoder_self={encoder_count}, "
            f"decoder_self={decoder_self_count}, "
            f"decoder_cross={decoder_cross_count}, "
            f"total={len(targets)}"
        )

    return sorted(targets)


def build_lora_model(*, rank: int, alpha: int, dropout: float):
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    target_modules = attention_target_names(model)

    config_values: dict[str, Any] = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "target_modules": target_modules,
        "task_type": None,
        "inference_mode": False,
    }

    signature = __import__("inspect").signature(LoraConfig.__init__).parameters
    if "use_dora" in signature:
        config_values["use_dora"] = False
    if "use_rslora" in signature:
        config_values["use_rslora"] = False

    model = get_peft_model(
        model,
        LoraConfig(**config_values),
    )

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError("LoRA has no trainable parameters.")

    bad = [
        name
        for name, _ in trainable
        if "lora_" not in name
    ]
    if bad:
        raise RuntimeError(
            "Unexpected non-LoRA trainable parameters: "
            f"{bad[:10]}"
        )

    trainable_parameters = sum(
        parameter.numel()
        for _, parameter in trainable
    )

    if is_rank_zero():
        print(
            "[LoRA] "
            f"targets={len(target_modules)} "
            f"rank={rank} "
            f"alpha={alpha} "
            f"dropout={dropout} "
            f"trainable={trainable_parameters}",
            flush=True,
        )

    return model


def save_run_metadata(
    *,
    run: dict[str, Any],
    final_checkpoint: Path,
) -> None:
    payload = {
        "name": run["name"],
        "base_model": BASE_MODEL,
        "train_manifest": str(TRAIN_MANIFEST),
        "test_manifest": str(TEST_MANIFEST),
        "training_examples": EXPECTED_TRAINING_EXAMPLES,
        "test_recordings": EXPECTED_TEST_RECORDINGS,
        "rank": run["rank"],
        "alpha": run["alpha"],
        "dropout": run["dropout"],
        "learning_rate": run["learning_rate"],
        "epochs": MAX_EPOCHS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": SEED,
        "target_count": EXPECTED_TARGETS,
        "final_checkpoint": str(final_checkpoint),
        "evaluation_path": str(
            evaluation_root(run["name"]) / "evaluation.json"
        ),
    }
    save_json(
        run_output(run["name"]) / "run_metadata.json",
        payload,
    )


def run_one(
    *,
    run: dict[str, Any],
    train_rows,
    test_rows,
    processor,
    train_dataset,
    collator,
    restart: bool,
) -> None:
    name = str(run["name"])
    output = run_output(name)
    checkpoints = checkpoint_root(name)
    done_path = evaluation_root(name) / "evaluation.json"

    if restart and is_rank_zero():
        shutil.rmtree(output, ignore_errors=True)
    distributed_barrier()

    if done_path.exists():
        if is_rank_zero():
            print(
                f"[skip] {name}: fixed-test evaluation already exists",
                flush=True,
            )
        return

    checkpoints.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    validate_cuda_environment(
        use_fp16=True,
        use_bf16=False,
    )

    ws = world_size()
    if EFFECTIVE_BATCH_SIZE % ws != 0:
        raise RuntimeError(
            f"EBS=8 must be divisible by world_size={ws}"
        )
    ga = EFFECTIVE_BATCH_SIZE // ws

    model = build_lora_model(
        rank=int(run["rank"]),
        alpha=int(run["alpha"]),
        dropout=float(run["dropout"]),
    )

    configure_whisper_generation(
        model,
        language=LANGUAGE,
        task=TASK,
        use_cache=False,
    )

    training_args = build_training_arguments(
        checkpoint_directory=checkpoints,
        seed=SEED,
        epochs=float(MAX_EPOCHS),
        learning_rate=float(run["learning_rate"]),
        weight_decay=WEIGHT_DECAY,
        warmup_steps=0,
        warmup_ratio=WARMUP_RATIO,
        train_batch_size=1,
        gradient_accumulation_steps=ga,
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

    if is_rank_zero():
        print()
        print("=" * 80)
        print(f"FINAL UNION LORA | {name}")
        print("=" * 80)
        print(f"train_examples={len(train_rows)}")
        print(f"test_examples={len(test_rows)}")
        print(f"rank={run['rank']}")
        print(f"alpha={run['alpha']}")
        print(f"dropout={run['dropout']}")
        print(f"learning_rate={run['learning_rate']}")
        print(f"epochs={MAX_EPOCHS}")
        print(f"world_size={ws}")
        print(f"GA={ga}")
        print(f"EBS={EFFECTIVE_BATCH_SIZE}")
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
        raise RuntimeError(
            f"No final checkpoint found for {name}"
        )

    checkpoint_dirs = sorted(
        [
            path
            for path in checkpoints.glob("checkpoint-*")
            if path.is_dir()
        ],
        key=checkpoint_number,
    )
    if len(checkpoint_dirs) != MAX_EPOCHS:
        raise RuntimeError(
            f"{name}: expected {MAX_EPOCHS} retained epoch checkpoints, "
            f"found {len(checkpoint_dirs)}"
        )

    if not is_rank_zero():
        distributed_barrier()
        del trainer
        del model
        torch.cuda.empty_cache()
        return

    print()
    print("=" * 80)
    print(
        f"FIXED TEST | {name} | FINAL EPOCH {MAX_EPOCHS}"
    )
    print("=" * 80)
    print(f"final_checkpoint={final_checkpoint}")

    eval_model = trainer.model
    eval_model.eval()
    configure_whisper_generation(
        eval_model,
        language=LANGUAGE,
        task=TASK,
        use_cache=True,
    )

    out = evaluation_root(name)
    out.mkdir(parents=True, exist_ok=True)

    evaluate_loaded_model(
        model_name=f"week9_{name}_union_final",
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
        run=run,
        final_checkpoint=final_checkpoint,
    )

    print(
        f"[complete] {name}: {out / 'evaluation.json'}",
        flush=True,
    )

    distributed_barrier()
    del trainer
    del model
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=("lora_r8", "lora_r128_lr_2p5e4"),
        default=["lora_r8", "lora_r128_lr_2p5e4"],
    )
    parser.add_argument(
        "--restart",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.device_count() < 1:
        raise RuntimeError("No visible CUDA GPU.")

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
        raise RuntimeError("Whisper has no decoder_start_token_id.")
    del probe_model

    train_dataset = WhisperManifestDataset(
        train_rows,
        processor,
    )
    collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
    )

    selected = {
        run["name"]: run
        for run in RUNS
        if run["name"] in args.runs
    }

    if is_rank_zero():
        print()
        print("=" * 80)
        print("LLANITO FINAL UNION LORA")
        print("=" * 80)
        print(f"runs={list(selected)}")
        print(f"train_union={TRAIN_MANIFEST}")
        print(f"train_examples={len(train_rows)}")
        print(f"fixed_test={TEST_MANIFEST}")
        print(f"test_examples={len(test_rows)}")
        print(f"epochs={MAX_EPOCHS}")
        print(f"checkpoints_retained={SAVE_TOTAL_LIMIT}")
        print("=" * 80)

    for name in args.runs:
        run_one(
            run=selected[name],
            train_rows=train_rows,
            test_rows=test_rows,
            processor=processor,
            train_dataset=train_dataset,
            collator=collator,
            restart=args.restart,
        )
        distributed_barrier()

    if is_rank_zero():
        print()
        print("=" * 80)
        print("ALL REQUESTED FINAL UNION LORA RUNS COMPLETE")
        print(f"results={OUTPUT_ROOT}")
        print("=" * 80)


if __name__ == "__main__":
    main()
