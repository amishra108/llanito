from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import WhisperForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "openai/whisper-medium"


# =====================================================================
# Imports
# =====================================================================

def import_file(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def runner_path(language: str) -> Path:
    # Prefer the generic Week-8 trainer because it now supports all
    # FLEURS language folders and standard FFT / LoRA / FCCA.
    generic = (
        ROOT / "scripts_fleurs"
        / "train_whisper_fleurs.py"
    )

    if generic.exists():
        return generic

    fallbacks = {
        "ast_es":
            ROOT / "scripts_fleurs/train_ast_es_fft.py",

        "es_419":
            ROOT / "scripts_fleurs/train_es_419.py",
    }

    if language not in fallbacks:
        raise RuntimeError(
            f"No runner known for {language}"
        )

    return fallbacks[language]


def load_runner(language: str, whisper_language: str | None = None):
    path = runner_path(language)

    runner = import_file(
        f"heldout_runner_{language}",
        path,
    )

    if hasattr(
        runner,
        "configure_language",
    ):
        runner.configure_language(
            language,
            whisper_language,
        )

    required = [
        "load_split",
        "build_processor",
        "evaluate_validation_loss",
        "evaluate_wer",
        "collect_attention_targets",
    ]

    missing = [
        name
        for name in required
        if not hasattr(runner, name)
    ]

    if missing:
        raise RuntimeError(
            f"Runner missing functions: {missing}"
        )

    return runner, path


# =====================================================================
# Checkpoint handling
# =====================================================================

def extract_state(payload):
    """
    Accept old Week-8 checkpoints, newer resumable checkpoints,
    and raw state_dicts.
    """
    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint payload is not a dict"
        )

    for key in (
        "state_dict",
        "model_state_dict",
    ):
        state = payload.get(key)

        if isinstance(
            state,
            dict,
        ):
            return state

    # Historical raw state_dict.
    if (
        payload
        and all(
            torch.is_tensor(v)
            for v in payload.values()
        )
    ):
        return payload

    raise RuntimeError(
        "Could not locate checkpoint state_dict. "
        f"Top-level keys={sorted(payload)}"
    )


def load_state_flexible(
    model,
    checkpoint: Path,
):
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    state = extract_state(
        payload
    )

    # First attempt exact full-model reconstruction.
    try:
        result = model.load_state_dict(
            state,
            strict=True,
        )

        print(
            "[checkpoint] FULL STATE STRICT MATCH",
            flush=True,
        )

        return payload

    except RuntimeError:
        pass

    # Historical PEFT/FCCA checkpoints frequently contain only the
    # trainable parameter state.
    named = dict(
        model.named_parameters()
    )

    missing = [
        name
        for name in state
        if name not in named
    ]

    if missing:
        raise RuntimeError(
            "Checkpoint is neither a strict full state nor "
            "a trainable-only parameter state.\n"
            f"Unknown saved names: {missing[:20]}"
        )

    loaded = 0

    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]

            if (
                tuple(parameter.shape)
                != tuple(value.shape)
            ):
                raise RuntimeError(
                    f"{name}: shape mismatch "
                    f"{tuple(value.shape)} != "
                    f"{tuple(parameter.shape)}"
                )

            parameter.copy_(
                value.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )

            loaded += 1

    print(
        "[checkpoint] TRAINABLE-STATE MATCH "
        f"parameters={loaded}",
        flush=True,
    )

    return payload


# =====================================================================
# Architectures
# =====================================================================

def attach_standard(
    runner,
    model,
    kind: str,
):
    if kind == "fft":
        # No wrapper: full Whisper model checkpoint.
        return

    if kind == "lora8":
        if not hasattr(
            runner,
            "attach_lora",
        ):
            raise RuntimeError(
                "Runner has no attach_lora()"
            )

        # Reconstruct exactly the rank-8 training architecture.
        runner.LORA_RANK = 8
        runner.LORA_ALPHA = 16.0

        runner.attach_lora(
            model
        )

        return

    if kind == "lora128":
        if hasattr(
            runner,
            "attach_lora",
        ):
            try:
                runner.attach_lora(
                    model
                )
            except TypeError:
                runner.attach_lora(
                    model,
                    128,
                )

            return

        raise RuntimeError(
            "Runner has no attach_lora()"
        )

    if kind == "fcca128":
        if hasattr(
            runner,
            "attach_fcca",
        ):
            try:
                runner.attach_fcca(
                    model
                )
            except TypeError:
                # Older generic signatures.
                runner.attach_fcca(
                    model,
                    128,
                )

            return

        raise RuntimeError(
            "Runner has no attach_fcca()"
        )

    raise ValueError(kind)


def attach_asymmetric(
    runner,
    model,
    language: str,
    variant: str,
):
    wrapper_path = (
        ROOT / "scripts_fleurs"
        / "train_asymmetric_one_sided_fcca.py"
    )

    wrapper = import_file(
        "asymmetric_wrapper",
        wrapper_path,
    )

    basis = (
        ROOT / "outputs" / "new_results"
        / language
        / "asymmetric_one_sided_fcca"
        / variant
        / "calibration"
        / "basis_r128.pt"
    )

    if not basis.exists():
        raise FileNotFoundError(
            basis
        )

    attach = wrapper.make_attach(
        runner,
        basis,
        variant,
    )

    attach(
        model
    )


def attach_adaptive(
    runner,
    model,
    language: str,
):
    script = (
        ROOT / "scripts_fleurs"
        / "train_adaptive_budget_fcca.py"
    )

    adaptive = import_file(
        "adaptive_budget_trainer",
        script,
    )

    basis_path = (
        ROOT / "outputs" / "new_results"
        / language
        / "adaptive_budget_fcca"
        / "calibration"
        / "adaptive_budget_basis.pt"
    )

    if not basis_path.exists():
        raise FileNotFoundError(
            basis_path
        )

    basis = torch.load(
        basis_path,
        map_location="cpu",
        weights_only=False,
    )

    adaptive.attach_adaptive_fcca(
        model,
        runner,
        basis,
    )


# =====================================================================
# Evaluation
# =====================================================================

def save_json(
    path: Path,
    payload,
):
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
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    distributed = (
        "LOCAL_RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    )

    if distributed:
        local_rank = int(
            os.environ["LOCAL_RANK"]
        )

        rank = int(
            os.environ["RANK"]
        )

        world_size = int(
            os.environ["WORLD_SIZE"]
        )

        torch.cuda.set_device(
            local_rank
        )

        dist.init_process_group(
            backend="nccl"
        )

        device = torch.device(
            f"cuda:{local_rank}"
        )

    else:
        local_rank = 0
        rank = 0
        world_size = 1

        device = torch.device(
            args.device
        )

        torch.cuda.set_device(
            device
        )

    checkpoint = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    runner, used_runner = load_runner(
        args.language,
        args.whisper_language,
    )

    if not hasattr(
        runner,
        "corpus_wer",
    ):
        raise RuntimeError(
            "Runner lacks corpus_wer(); "
            "needed to merge distributed shards exactly."
        )

    processor = (
        runner.build_processor()
    )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            BASE_MODEL,
            dtype=dtype,
        )
        .to(device)
    )

    model.config.use_cache = True

    if args.kind in {
        "fft",
        "lora8",
        "lora128",
        "fcca128",
    }:
        attach_standard(
            runner,
            model,
            args.kind,
        )

    elif args.kind == "asym_hard":
        attach_asymmetric(
            runner,
            model,
            args.language,
            "hard",
        )

    elif args.kind == "asym_shared_private":
        attach_asymmetric(
            runner,
            model,
            args.language,
            "shared_private",
        )

    elif args.kind == "adaptive_budget":
        attach_adaptive(
            runner,
            model,
            args.language,
        )

    else:
        raise ValueError(
            args.kind
        )

    load_state_flexible(
        model,
        checkpoint,
    )

    model.eval()
    model.config.use_cache = True

    full_ds = runner.load_split(
        "test"
    )

    if distributed:
        indices = list(
            range(
                rank,
                len(full_ds),
                world_size,
            )
        )

        ds = full_ds.select(
            indices
        )

    else:
        ds = full_ds
        indices = list(
            range(
                len(full_ds)
            )
        )

    output_root = (
        ROOT / "outputs" / "new_results"
        / args.language
        / "heldout_test_epoch_audit"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = (
        output_root
        / f"{args.output_name}__predictions.csv"
    )

    metrics_path = (
        output_root
        / f"{args.output_name}__metrics.json"
    )

    shard_predictions = (
        output_root
        / (
            f".{args.output_name}"
            f"__rank{rank}.csv"
        )
    )

    shard_metrics = (
        output_root
        / (
            f".{args.output_name}"
            f"__rank{rank}.json"
        )
    )

    if rank == 0:
        if (
            not args.force
            and (
                predictions.exists()
                or metrics_path.exists()
            )
        ):
            raise RuntimeError(
                "Refusing to overwrite existing held-out result. "
                "Use --force if intentional."
            )

    if distributed:
        dist.barrier()

    print(
        "=" * 88,
        flush=True,
    )

    print(
        f"[rank] {rank}/{world_size}",
        flush=True,
    )

    print(
        f"[device] {device}",
        flush=True,
    )

    print(
        f"[heldout shard] "
        f"{len(ds)}/{len(full_ds)}",
        flush=True,
    )

    print(
        f"[first indices] {indices[:8]}",
        flush=True,
    )

    print(
        "=" * 88,
        flush=True,
    )

    loss_metrics = (
        runner.evaluate_validation_loss(
            model,
            processor,
            ds,
            device,
        )
    )

    # evaluate_wer() comes from the training runner.
    # Its write_csv() intentionally writes only on rank 0 during
    # distributed training. Here every rank has its own unique shard
    # filename, so every rank MUST be allowed to write its predictions.
    original_is_main = runner.is_main

    try:
        runner.is_main = lambda: True

        _ = runner.evaluate_wer(
            model,
            processor,
            ds,
            device,
            shard_predictions,
        )

    finally:
        runner.is_main = original_is_main

    save_json(
        shard_metrics,
        {
            "rank":
                rank,

            "examples":
                len(ds),

            **loss_metrics,
        },
    )

    if distributed:
        dist.barrier()

    if rank == 0:
        all_rows = []

        total_tokens = 0
        weighted_loss = 0.0
        total_loss_batches = 0
        total_examples = 0

        for shard_rank in range(
            world_size
        ):
            pred_path = (
                output_root
                / (
                    f".{args.output_name}"
                    f"__rank{shard_rank}.csv"
                )
            )

            metric_path = (
                output_root
                / (
                    f".{args.output_name}"
                    f"__rank{shard_rank}.json"
                )
            )

            with pred_path.open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(
                    csv.DictReader(
                        handle
                    )
                )

            all_rows.extend(
                rows
            )

            shard = json.loads(
                metric_path.read_text(
                    encoding="utf-8"
                )
            )

            tokens = int(
                shard.get(
                    "validation_supervised_tokens",
                    0,
                )
            )

            loss = float(
                shard[
                    "validation_loss"
                ]
            )

            weighted_loss += (
                loss
                * tokens
            )

            total_tokens += tokens

            total_loss_batches += int(
                shard.get(
                    "validation_loss_batches",
                    0,
                )
            )

            total_examples += int(
                shard[
                    "examples"
                ]
            )

        # Restore original held-out ordering.
        def row_index(row):
            try:
                return int(
                    row["index"]
                )
            except Exception:
                return 0

        all_rows.sort(
            key=row_index
        )

        if not all_rows:
            raise RuntimeError(
                "No prediction rows produced."
            )

        fields = list(
            all_rows[0].keys()
        )

        with predictions.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                all_rows
            )

        references = [
            row[
                "reference_original"
            ]
            for row in all_rows
        ]

        hypotheses = [
            row[
                "hypothesis_original"
            ]
            for row in all_rows
        ]

        wer_metrics = runner.corpus_wer(
            references,
            hypotheses,
        )

        merged_loss = (
            weighted_loss
            / total_tokens
            if total_tokens > 0
            else float("nan")
        )

        result = {
            "language":
                args.language,

            "split":
                "test",

            "kind":
                args.kind,

            "output_name":
                args.output_name,

            "checkpoint":
                str(
                    checkpoint
                ),

            "runner":
                str(
                    used_runner
                ),

            "examples":
                total_examples,

            "world_size":
                world_size,

            "validation_loss":
                merged_loss,

            "validation_supervised_tokens":
                total_tokens,

            "validation_loss_batches":
                total_loss_batches,

            **wer_metrics,
        }

        save_json(
            metrics_path,
            result,
        )

        print()
        print("=" * 88)

        print(
            f"[PASS] {args.output_name}"
        )

        print(
            f"heldout_WER="
            f"{result['normalised_wer']:.8f}%"
        )

        print(
            f"heldout_loss="
            f"{result['validation_loss']:.8f}"
        )

        print(
            f"metrics={metrics_path}"
        )

        print(
            f"predictions={predictions}"
        )

        print("=" * 88)

        for shard_rank in range(
            world_size
        ):
            (
                output_root
                / (
                    f".{args.output_name}"
                    f"__rank{shard_rank}.csv"
                )
            ).unlink(
                missing_ok=True
            )

            (
                output_root
                / (
                    f".{args.output_name}"
                    f"__rank{shard_rank}.json"
                )
            ).unlink(
                missing_ok=True
            )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
    )

    parser.add_argument(
        "--whisper-language",
        default=None,
    )

    parser.add_argument(
        "--kind",
        required=True,
        choices=[
            "fft",
            "lora8",
            "lora128",
            "fcca128",
            "asym_hard",
            "asym_shared_private",
            "adaptive_budget",
        ],
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-name",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(
        parse_args()
    )
