from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

GENERIC_RUNNER = (
    PROJECT_ROOT / "scripts_fleurs"
    / "train_whisper_fleurs.py"
)


def load_runner(
    language: str,
    whisper_language: str,
):
    if not GENERIC_RUNNER.exists():
        raise FileNotFoundError(
            GENERIC_RUNNER
        )

    spec = importlib.util.spec_from_file_location(
        f"generic_fleurs_runner_{language}",
        GENERIC_RUNNER,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {GENERIC_RUNNER}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    configure = getattr(
        module,
        "configure_language",
        None,
    )

    if not callable(configure):
        raise RuntimeError(
            "Generic Week8 runner has no configure_language()"
        )

    # Current generic runner supports a Whisper-language override.
    # Keep compatibility with an older one-argument implementation.
    parameters = inspect.signature(
        configure
    ).parameters

    if len(parameters) >= 2:
        configure(
            language,
            whisper_language,
        )
    else:
        configure(
            language
        )

        module.WHISPER_LANGUAGE = (
            whisper_language
        )

    required = [
        "load_split",
        "build_processor",
        "collect_attention_targets",
        "get_parent",
        "FCCALinear",
        "run_trial",
    ]

    missing = [
        name
        for name in required
        if not hasattr(
            module,
            name,
        )
    ]

    if missing:
        raise RuntimeError(
            f"Generic runner missing required APIs: {missing}"
        )

    print(
        f"[runner] {language} -> {GENERIC_RUNNER}",
        flush=True,
    )

    return module


def basis_path(
    language: str,
    variant: str,
) -> Path:
    return (
        PROJECT_ROOT / "outputs" / "new_results"
        / language
        / "asymmetric_one_sided_fcca"
        / variant
        / "calibration"
        / "basis_r128.pt"
    )


def collect_native_bases(
    payload,
):
    """
    Robustly locate the final native-coordinate P,Q pair for each
    Whisper attention projection.

    The geometry builder contains substantial diagnostic/group data,
    so we recurse rather than assuming one brittle payload layout.
    """
    found = {}

    def visit(
        obj,
        parent_key=None,
    ):
        if isinstance(
            obj,
            dict,
        ):
            if (
                parent_key is not None
                and
                "P" in obj
                and
                "Q" in obj
                and
                torch.is_tensor(
                    obj["P"]
                )
                and
                torch.is_tensor(
                    obj["Q"]
                )
            ):
                P = obj["P"]
                Q = obj["Q"]

                if (
                    P.ndim == 2
                    and
                    Q.ndim == 2
                    and
                    P.shape[1] == 128
                    and
                    Q.shape[1] == 128
                ):
                    name = str(
                        parent_key
                    )

                    if (
                        ".layers."
                        in name
                    ):
                        if name in found:
                            old_P = found[name][
                                "P"
                            ]

                            old_Q = found[name][
                                "Q"
                            ]

                            if not (
                                torch.equal(
                                    old_P,
                                    P,
                                )
                                and
                                torch.equal(
                                    old_Q,
                                    Q,
                                )
                            ):
                                raise RuntimeError(
                                    "Multiple different "
                                    f"P,Q pairs found for {name}"
                                )

                        else:
                            found[name] = {
                                "P":
                                    P.detach().cpu(),

                                "Q":
                                    Q.detach().cpu(),
                            }

            for key, value in obj.items():
                visit(
                    value,
                    key,
                )

        elif isinstance(
            obj,
            (list, tuple),
        ):
            for value in obj:
                visit(
                    value,
                    None,
                )

    visit(
        payload
    )

    return found


def audit_basis(
    runner,
    model,
    bases,
):
    targets = (
        runner
        .collect_attention_targets(
            model
        )
    )

    if len(targets) != 288:
        raise RuntimeError(
            f"Expected 288 Whisper targets; "
            f"found {len(targets)}"
        )

    if set(targets) != set(bases):
        missing = sorted(
            set(targets)
            - set(bases)
        )

        extra = sorted(
            set(bases)
            - set(targets)
        )

        raise RuntimeError(
            "Asymmetric basis/model mismatch. "
            f"missing={missing[:10]} "
            f"extra={extra[:10]}"
        )

    eye = torch.eye(
        128,
        dtype=torch.float32,
    )

    worst_p = 0.0
    worst_q = 0.0

    for name in sorted(
        targets
    ):
        base = targets[
            name
        ]

        P = (
            bases[name]["P"]
            .float()
        )

        Q = (
            bases[name]["Q"]
            .float()
        )

        expected_p = (
            base.out_features,
            128,
        )

        expected_q = (
            base.in_features,
            128,
        )

        if tuple(P.shape) != expected_p:
            raise RuntimeError(
                f"{name}: P shape {tuple(P.shape)} "
                f"!= {expected_p}"
            )

        if tuple(Q.shape) != expected_q:
            raise RuntimeError(
                f"{name}: Q shape {tuple(Q.shape)} "
                f"!= {expected_q}"
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

        worst_p = max(
            worst_p,
            p_err,
        )

        worst_q = max(
            worst_q,
            q_err,
        )

        if (
            p_err > 1.0e-3
            or
            q_err > 1.0e-3
        ):
            raise RuntimeError(
                f"{name}: native P/Q "
                "orthogonality audit failed: "
                f"P={p_err:.6e}, "
                f"Q={q_err:.6e}"
            )

    print(
        "[basis audit] "
        f"targets={len(targets)} "
        f"worst_P_orth={worst_p:.6e} "
        f"worst_Q_orth={worst_q:.6e}",
        flush=True,
    )


def make_attach(
    runner,
    path: Path,
    variant: str,
):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    bases = collect_native_bases(
        payload
    )

    print(
        f"[basis] variant={variant}",
        flush=True,
    )

    print(
        f"[basis] path={path}",
        flush=True,
    )

    print(
        f"[basis] discovered native P,Q "
        f"pairs={len(bases)}",
        flush=True,
    )

    def attach(
        model,
    ):
        audit_basis(
            runner,
            model,
            bases,
        )

        targets = (
            runner
            .collect_attention_targets(
                model
            )
        )

        for parameter in model.parameters():
            parameter.requires_grad = False

        names = sorted(
            targets
        )

        for name in names:
            parent, leaf = (
                runner.get_parent(
                    model,
                    name,
                )
            )

            base = getattr(
                parent,
                leaf,
            )

            item = bases[
                name
            ]

            setattr(
                parent,
                leaf,
                runner.FCCALinear(
                    base,
                    item["P"],
                    item["Q"],
                    scale=1.0,
                ),
            )

        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        expected = (
            288
            * 128
            * 128
        )

        if trainable != expected:
            raise RuntimeError(
                "Trainable budget mismatch: "
                f"{trainable:,} != "
                f"{expected:,}"
            )

        print(
            "[adapter audit] "
            f"trainable={trainable:,} "
            f"expected={expected:,}",
            flush=True,
        )

        return names

    return attach


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
    )

    parser.add_argument(
        "--whisper-language",
        required=True,
        help="Whisper language token, e.g. es, fa, zh, en, kk.",
    )

    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "hard",
            "shared_private",
        ],
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--microbatch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume from a full epoch checkpoint.",
    )

    args = parser.parse_args()

    runner = load_runner(
        args.language,
        args.whisper_language,
    )

    path = basis_path(
        args.language,
        args.variant,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Build asymmetric basis first: {path}"
        )

    if (
        args.effective_batch_size
        % args.microbatch
        != 0
    ):
        raise RuntimeError(
            "effective batch size must be divisible by microbatch"
        )

    # ---------------------------------------------------------
    # Shared matched FCCA protocol.
    # Only LR/EBS/warmup/microbatch are language/run inputs.
    # ---------------------------------------------------------

    runner.MAX_EPOCHS = 12
    runner.PATIENCE = (
        999999
        if args.resume_checkpoint is not None
        else 5
    )

    runner.MICROBATCH = (
        args.microbatch
    )

    runner.EVAL_BATCH_SIZE = 4

    runner.EFFECTIVE_BATCH_SIZE = (
        args.effective_batch_size
    )

    runner.WARMUP_RATIO = (
        args.warmup_ratio
    )

    runner.WEIGHT_DECAY = 0.01
    runner.MAX_GRAD_NORM = 1.0

    runner.SEED = 2026

    runner.FCCA_RANK = 128
    runner.FCCA_SCALE = 1.0

    # Generic Week-8 runner receives the training recipe directly
    # through run_trial().  Do not mutate historical METHOD_TRIALS:
    # the generic runner intentionally has no such table.
    runner.method_output_root = (
        lambda method:
            PROJECT_ROOT / "outputs" / "new_results"
            / args.language
            / "asymmetric_one_sided_fcca"
            / args.variant
            / "training"
    )

    print(
        "[experiment] asymmetric one-sided FCCA",
        flush=True,
    )

    print(
        f"[language] {args.language}",
        flush=True,
    )

    print(
        f"[whisper language] {args.whisper_language}",
        flush=True,
    )

    print(
        f"[variant] {args.variant}",
        flush=True,
    )

    print(
        "[recipe] "
        f"lr={args.learning_rate} "
        f"EBS={args.effective_batch_size} "
        f"microbatch={args.microbatch} "
        f"GA={args.effective_batch_size // args.microbatch} "
        f"warmup={args.warmup_ratio} "
        "epochs=12 "
        f"patience={runner.PATIENCE} "
        "seed=2026",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    runner.run_trial(
        method="fcca",
        trial_number=args.trial,
        microbatch=args.microbatch,
        learning_rate=args.learning_rate,
        effective_batch_size=args.effective_batch_size,
        warmup_ratio=args.warmup_ratio,
        resume_checkpoint=args.resume_checkpoint,
    )

    out = (
        PROJECT_ROOT / "outputs" / "new_results"
        / args.language
        / "asymmetric_one_sided_fcca"
        / args.variant
        / "training"
        / f"trial_{args.trial:04d}"
    )

    for filename in (
        "hyperparameters.json",
        "trial_result.json",
    ):
        file = (
            out
            / filename
        )

        if not file.exists():
            continue

        data = json.loads(
            file.read_text(
                encoding="utf-8"
            )
        )

        data[
            "method"
        ] = (
            "asymmetric_one_sided_fcca"
        )

        data[
            "asymmetric_variant"
        ] = args.variant

        data[
            "language"
        ] = args.language

        data[
            "whisper_language"
        ] = args.whisper_language

        data[
            "basis_path"
        ] = str(
            path
        )

        data[
            "instantaneous_trainable_budget"
        ] = (
            288
            * 128
            * 128
        )

        file.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
