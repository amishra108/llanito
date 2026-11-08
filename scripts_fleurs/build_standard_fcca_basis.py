from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    PROJECT_ROOT / "outputs" / "new_results"
)

BASE_MODEL = "openai/whisper-medium"

CALIBRATION_SIZE = 256
CALIBRATION_SEED = 2026
DAMPING = 1.0e-6
RANK = 128
SCALE = 1.0
EXPECTED_TARGETS = 288


def save_atomic(payload, path: Path):
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    language = args.language

    stats_path = (
        RESULT_ROOT
        / language
        / "_task_stats"
        / "task_stats_256_all_attention.pt"
    )

    out_path = (
        RESULT_ROOT
        / language
        / "fcca"
        / "calibration"
        / "calibration_256_all_attention_v2_means.pt"
    )

    meta_path = out_path.with_suffix(
        ".json"
    )

    if not stats_path.exists():
        raise FileNotFoundError(
            stats_path
        )

    if (
        out_path.exists()
        and not args.force
    ):
        print(
            f"[skip existing] {out_path}",
            flush=True,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required."
        )

    payload = torch.load(
        stats_path,
        map_location="cpu",
        weights_only=False,
    )

    if payload.get(
        "schema"
    ) != "whisper_task_stats_v1":
        raise RuntimeError(
            f"Unexpected task-stat schema: "
            f"{payload.get('schema')}"
        )

    if payload.get(
        "base_model"
    ) != BASE_MODEL:
        raise RuntimeError(
            "Base-model mismatch."
        )

    if int(
        payload.get(
            "calibration_size",
            -1,
        )
    ) != CALIBRATION_SIZE:
        raise RuntimeError(
            "Calibration-size mismatch."
        )

    if int(
        payload.get(
            "calibration_seed",
            -1,
        )
    ) != CALIBRATION_SEED:
        raise RuntimeError(
            "Calibration-seed mismatch."
        )

    modules = payload[
        "modules"
    ]

    if len(modules) != EXPECTED_TARGETS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGETS} modules, "
            f"found {len(modules)}."
        )

    device = torch.device(
        "cuda:0"
    )

    bases = {}
    diagnostics = {}

    names = sorted(
        modules
    )

    for index, name in enumerate(
        names,
        start=1,
    ):
        record = modules[
            name
        ]

        count = float(
            record[
                "positions"
            ]
        )

        if count <= 0:
            raise RuntimeError(
                f"{name}: positions={count}"
            )

        G = (
            record[
                "cross_sum"
            ]
            .double()
            .to(device)
            / count
        )

        a = (
            record[
                "activation_second_sum"
            ]
            .double()
            .to(device)
            / count
        )

        d = (
            record[
                "gradient_second_sum"
            ]
            .double()
            .to(device)
            / count
        )

        inv_a = torch.rsqrt(
            a + DAMPING
        )

        inv_d = torch.rsqrt(
            d + DAMPING
        )

        whitened = (
            inv_d[:, None]
            * G
            * inv_a[None, :]
        )

        U, S, Vh = torch.linalg.svd(
            whitened,
            full_matrices=False,
        )

        if min(
            whitened.shape
        ) < RANK:
            raise RuntimeError(
                f"{name}: matrix too small "
                f"for rank {RANK}: "
                f"{tuple(whitened.shape)}"
            )

        U_r = U[
            :,
            :RANK,
        ]

        V_r = (
            Vh[
                :RANK,
                :,
            ]
            .T
        )

        P0 = (
            inv_d[:, None]
            * U_r
        )

        Q0 = (
            inv_a[:, None]
            * V_r
        )

        # Match the existing FCCA implementations:
        # QR in double, then store float32.
        P, _ = torch.linalg.qr(
            P0.double(),
            mode="reduced",
        )

        Q, _ = torch.linalg.qr(
            Q0.double(),
            mode="reduced",
        )

        P = (
            P[
                :,
                :RANK,
            ]
            .float()
        )

        Q = (
            Q[
                :,
                :RANK,
            ]
            .float()
        )

        eye = torch.eye(
            RANK,
            device=device,
            dtype=torch.float32,
        )

        p_error = float(
            torch.linalg.matrix_norm(
                P.T @ P - eye,
                ord="fro",
            ).item()
        )

        q_error = float(
            torch.linalg.matrix_norm(
                Q.T @ Q - eye,
                ord="fro",
            ).item()
        )

        if (
            p_error > 5.0e-4
            or q_error > 5.0e-4
        ):
            raise RuntimeError(
                f"{name}: QR audit failed "
                f"P={p_error:.6e} "
                f"Q={q_error:.6e}"
            )

        bases[
            name
        ] = {
            "P":
                P.detach().cpu(),

            "Q":
                Q.detach().cpu(),
        }

        diagnostics[
            name
        ] = {
            "rank":
                RANK,

            "sigma_1":
                float(
                    S[0].item()
                ),

            "sigma_r":
                float(
                    S[
                        RANK - 1
                    ].item()
                ),

            "p_orthogonality_error":
                p_error,

            "q_orthogonality_error":
                q_error,
        }

        del (
            G,
            a,
            d,
            inv_a,
            inv_d,
            whitened,
            U,
            S,
            Vh,
            U_r,
            V_r,
            P0,
            Q0,
            P,
            Q,
        )

        torch.cuda.empty_cache()

        if (
            index % 16 == 0
            or index == len(names)
        ):
            print(
                f"[basis] "
                f"{index}/{len(names)}",
                flush=True,
            )

    basis_payload = {
        "method":
            "FCCA",

        "base_model":
            BASE_MODEL,

        "language":
            language,

        "whisper_language":
            payload.get(
                "whisper_language"
            ),

        "task":
            payload.get(
                "task",
                "transcribe",
            ),

        "calibration_size":
            CALIBRATION_SIZE,

        "calibration_seed":
            CALIBRATION_SEED,

        "damping":
            DAMPING,

        "scale":
            SCALE,

        "target_count":
            len(bases),

        "source_task_stats":
            str(
                stats_path
            ),

        "source_task_stat_schema":
            payload.get(
                "schema"
            ),

        "selected_indices":
            payload.get(
                "selected_indices"
            ),

        "diagnostics":
            diagnostics,

        "bases": {
            str(RANK):
                bases,
        },
    }

    save_atomic(
        basis_payload,
        out_path,
    )

    meta = {
        key: value
        for key, value
        in basis_payload.items()
        if key not in {
            "bases",
            "diagnostics",
        }
    }

    meta_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("[FCCA BASIS COMPLETE]")
    print(f"language={language}")
    print(f"rank={RANK}")
    print(f"targets={len(bases)}")
    print(f"output={out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
