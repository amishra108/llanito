from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "openai/whisper-medium"

RANK = 128
SHARED_RANK = 64
PRIVATE_RANK = 64

DAMPING = 1.0e-6

CALIBRATION_SIZE = 256
CALIBRATION_SEED = 2026

SUPPORTED_LANGUAGES = {
    "ast_es",
    "es_419",
    "fa_ir",
}

VARIANTS = {
    "hard",
    "shared_private",
}


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def default_task_stats_path(
    language: str,
) -> Path:
    filename = (
        "task_stats_256_all_attention.pt"
        if language == "es_419"
        else "task_stats_256_all_attention.pt"
    )

    return (
        PROJECT_ROOT / "outputs" / "new_results"
        / language
        / "_task_stats"
        / filename
    )

def output_path(
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


# ---------------------------------------------------------------------
# Task-stat loading
# ---------------------------------------------------------------------

def _first(
    d: dict[str, Any],
    names: tuple[str, ...],
):
    for name in names:
        if name in d:
            return d[name], name

    raise KeyError(
        f"None of {names} found. "
        f"Available keys={sorted(d)}"
    )


def load_task_stats(
    path: Path,
):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "modules" not in payload:
        raise RuntimeError(
            f"{path} has no top-level 'modules'. "
            f"keys={sorted(payload)}"
        )

    modules = payload["modules"]

    if len(modules) != 288:
        raise RuntimeError(
            f"Expected 288 attention matrices, "
            f"found {len(modules)}"
        )

    return payload, modules


def mean_stats(
    stat: dict[str, Any],
    device: torch.device,
):
    count_obj, _ = _first(
        stat,
        (
            "positions",
            "count",
        ),
    )

    count = float(count_obj)

    if count <= 0:
        raise RuntimeError(
            f"Invalid task-stat count={count}"
        )

    G_obj, G_key = _first(
        stat,
        (
            "cross_sum",
            "cross",
            "G_sum",
            "G",
        ),
    )

    a_obj, a_key = _first(
        stat,
        (
            "activation_second_sum",
            "activation_second",
            "a_sum",
            "a",
        ),
    )

    d_obj, d_key = _first(
        stat,
        (
            "gradient_second_sum",
            "gradient_second",
            "d_sum",
            "d",
        ),
    )

    G = G_obj.to(
        device=device,
        dtype=torch.float64,
    )

    a = a_obj.to(
        device=device,
        dtype=torch.float64,
    )

    d = d_obj.to(
        device=device,
        dtype=torch.float64,
    )

    # Current Week-8 task-stat payloads store sums.
    # Explicit *_sum keys therefore divide by positions.
    # Historical non-suffixed caches may also store sums;
    # detect that from the payload convention below.
    if G_key.endswith("_sum"):
        G = G / count

    if a_key.endswith("_sum"):
        a = a / count

    if d_key.endswith("_sum"):
        d = d / count

    # Older shared-frozen-Tucker caches use unsuffixed accumulator
    # names but are also position sums.
    if (
        G_key == "cross"
        and
        "positions" in stat
    ):
        G = G / count

    if (
        a_key in {"activation_second", "a"}
        and
        "positions" in stat
    ):
        a = a / count

    if (
        d_key in {"gradient_second", "d"}
        and
        "positions" in stat
    ):
        d = d / count

    if G.ndim != 2:
        raise RuntimeError(
            f"G must be matrix, got {tuple(G.shape)}"
        )

    if a.ndim != 1 or d.ndim != 1:
        raise RuntimeError(
            "a and d must be vectors"
        )

    if (
        G.shape[0] != d.numel()
        or
        G.shape[1] != a.numel()
    ):
        raise RuntimeError(
            "Task-stat dimensions disagree"
        )

    if not torch.isfinite(G).all():
        raise RuntimeError("Non-finite G")

    if not torch.isfinite(a).all():
        raise RuntimeError("Non-finite a")

    if not torch.isfinite(d).all():
        raise RuntimeError("Non-finite d")

    return G, a, d, int(count)


def whiten(
    stat: dict[str, Any],
    device: torch.device,
):
    G, a, d, count = mean_stats(
        stat,
        device,
    )

    inv_a = torch.rsqrt(
        a + DAMPING
    )

    inv_d = torch.rsqrt(
        d + DAMPING
    )

    Gt = (
        inv_d[:, None]
        * G
        * inv_a[None, :]
    )

    return {
        "G": G,
        "a": a,
        "d": d,
        "inv_a": inv_a,
        "inv_d": inv_d,
        "Gt": Gt,
        "count": count,
    }


# ---------------------------------------------------------------------
# Whisper target parsing
# ---------------------------------------------------------------------

TARGET_RE = re.compile(
    r"^model\."
    r"(encoder|decoder)"
    r"\.layers\.(\d+)\."
    r"(self_attn|encoder_attn)\."
    r"(q_proj|k_proj|v_proj|out_proj|o_proj)$"
)


def parse_target(
    name: str,
):
    m = TARGET_RE.match(name)

    if m is None:
        raise RuntimeError(
            f"Unrecognised target: {name}"
        )

    stack = m.group(1)
    layer = int(m.group(2))
    attn = m.group(3)
    leaf = m.group(4)

    if stack == "encoder":
        if attn != "self_attn":
            raise RuntimeError(name)

        family = "encoder_self"

    elif attn == "self_attn":
        family = "decoder_self"

    else:
        family = "decoder_cross"

    projection = (
        "o"
        if leaf in {"out_proj", "o_proj"}
        else leaf[0]
    )

    return {
        "stack": stack,
        "layer": layer,
        "family": family,
        "projection": projection,
    }


def make_pair_groups(
    names,
):
    groups: dict[
        tuple[str, str, int],
        list[str],
    ] = {}

    for name in names:
        info = parse_target(name)

        layer = info["layer"]

        if not 0 <= layer < 24:
            raise RuntimeError(
                f"Expected layers 0..23: {name}"
            )

        pair_start = (
            layer
            if layer % 2 == 0
            else layer - 1
        )

        key = (
            info["family"],
            info["projection"],
            pair_start,
        )

        groups.setdefault(
            key,
            [],
        ).append(name)

    if len(groups) != 144:
        raise RuntimeError(
            f"Expected 144 adjacent-pair groups, "
            f"found {len(groups)}"
        )

    for key, members in groups.items():
        members.sort(
            key=lambda x: parse_target(x)["layer"]
        )

        if len(members) != 2:
            raise RuntimeError(
                f"{key}: expected 2 matrices, "
                f"got {len(members)}"
            )

        layers = [
            parse_target(x)["layer"]
            for x in members
        ]

        if layers != [
            key[2],
            key[2] + 1,
        ]:
            raise RuntimeError(
                f"Bad pair {key}: {layers}"
            )

    return groups


# ---------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------

def symmetrise(
    A: torch.Tensor,
):
    return 0.5 * (
        A + A.T
    )


def top_eig(
    A: torch.Tensor,
    rank: int,
):
    A = symmetrise(A)

    evals, evecs = torch.linalg.eigh(A)

    if rank > evecs.shape[1]:
        raise RuntimeError(
            f"rank={rank} impossible for "
            f"{tuple(A.shape)}"
        )

    V = evecs[:, -rank:]

    # Descending order for reproducibility/readability.
    V = torch.flip(
        V,
        dims=(1,),
    )

    return V.contiguous()


def orth(
    X: torch.Tensor,
    rank: int,
    name: str,
):
    """
    Return `rank` orthonormal columns whose span contains col(X).

    This uses a full SVD rather than reduced QR.

    If rank(X) < rank, the remaining left-singular vectors give
    a valid orthonormal completion. That is mathematically allowed:
    those extra directions contribute zero to the constrained local
    FCCA objective, while preserving the requested rank-r adapter
    parameterization.
    """
    if X.ndim != 2:
        raise RuntimeError(
            f"{name}: expected a matrix, got {tuple(X.shape)}"
        )

    if rank > X.shape[0]:
        raise RuntimeError(
            f"{name}: rank={rank} exceeds ambient output dimension "
            f"{X.shape[0]}"
        )

    U, _, _ = torch.linalg.svd(
        X,
        full_matrices=True,
    )

    Q = U[:, :rank].contiguous()

    eye = torch.eye(
        rank,
        device=Q.device,
        dtype=Q.dtype,
    )

    err = float(
        torch.linalg.matrix_norm(
            Q.T @ Q - eye,
            ord="fro",
        ).item()
    )

    if err > 1.0e-8:
        raise RuntimeError(
            f"{name}: orthogonality failure {err:.12e}"
        )

    return Q


def column_span_basis(
    X: torch.Tensor,
    rtol: float = 1.0e-10,
):
    """
    Return an orthonormal basis for the ACTUAL numerical column span.

    This is crucial for the shared/private method. Reduced QR on a
    rank-deficient concatenation would fabricate additional orthogonal
    columns and would incorrectly remove directions that are not part
    of any private subspace.
    """
    if X.ndim != 2:
        raise RuntimeError(
            f"Expected a matrix, got {tuple(X.shape)}"
        )

    if X.shape[1] == 0:
        return X[:, :0]

    U, S, _ = torch.linalg.svd(
        X,
        full_matrices=False,
    )

    if S.numel() == 0:
        return X[:, :0]

    threshold = (
        float(S[0].item())
        * float(rtol)
    )

    numerical_rank = int(
        (S > threshold).sum().item()
    )

    return (
        U[:, :numerical_rank]
        .contiguous()
    )


def projector_complement(
    V: torch.Tensor,
):
    n = V.shape[0]

    I = torch.eye(
        n,
        device=V.device,
        dtype=V.dtype,
    )

    return (
        I
        - V @ V.T
    )


def independent_fcca_score(
    Gt: torch.Tensor,
    rank: int,
):
    # Sum of top-r squared singular values.
    s = torch.linalg.svdvals(Gt)

    return float(
        torch.sum(
            s[:rank] ** 2
        ).item()
    )


def side_covariance(
    Gt: torch.Tensor,
    side: str,
):
    if side == "right":
        return (
            Gt.T @ Gt
        )

    if side == "left":
        return (
            Gt @ Gt.T
        )

    raise ValueError(side)


def side_score(
    C: torch.Tensor,
    V: torch.Tensor,
):
    return float(
        torch.trace(
            V.T @ C @ V
        ).item()
    )


# ---------------------------------------------------------------------
# Exact hard one-sided constrained FCCA
# ---------------------------------------------------------------------

def solve_hard_group(
    members: list[str],
    whitened: dict[str, dict[str, torch.Tensor]],
    shared_side: str,
):
    C = {
        name: side_covariance(
            whitened[name]["Gt"],
            shared_side,
        )
        for name in members
    }

    pooled = sum(
        C.values()
    )

    shared = top_eig(
        pooled,
        RANK,
    )

    bases = {}

    shared_objective = 0.0
    independent_objective = 0.0

    for name in members:
        Gt = whitened[name]["Gt"]

        if shared_side == "right":
            V = shared

            U = orth(
                Gt @ V,
                RANK,
                f"{name}: U=orth(GV)",
            )

        else:
            U = shared

            V = orth(
                Gt.T @ U,
                RANK,
                f"{name}: V=orth(G^T U)",
            )

        shared_objective += float(
            torch.sum(
                (
                    U.T
                    @ Gt
                    @ V
                ) ** 2
            ).item()
        )

        independent_objective += (
            independent_fcca_score(
                Gt,
                RANK,
            )
        )

        bases[name] = {
            "U": U,
            "V": V,
        }

    tolerance = (
        1.0e-7
        * max(
            independent_objective,
            1.0,
        )
    )

    if (
        shared_objective
        >
        independent_objective
        + tolerance
    ):
        raise RuntimeError(
            "THEOREM AUDIT FAILED: "
            f"hard shared objective "
            f"{shared_objective:.12e} > "
            f"independent FCCA "
            f"{independent_objective:.12e}"
        )

    return bases, {
        "shared_objective":
            shared_objective,

        "independent_fcca_objective":
            independent_objective,

        "ratio_to_independent":
            (
                shared_objective
                / independent_objective
            ),
    }


# ---------------------------------------------------------------------
# Shared + private 64/64:
#
#   max sum_l tr(Vg^T C_l Vg)
#             + tr(Vlp^T C_l Vlp)
#
# subject to:
#   Vg^T Vg = I_s
#   Vlp^T Vlp = I_p
#   Vg^T Vlp = 0
#
# Exact block update for private spaces.
# Exact block update for shared space given all private spaces.
#
# Therefore every full cycle is monotonic and convergence is to a
# block-stationary/local solution, not claimed global.
# ---------------------------------------------------------------------

def private_update(
    C: torch.Tensor,
    shared: torch.Tensor,
):
    P = projector_complement(
        shared
    )

    projected = (
        P @ C @ P
    )

    private = top_eig(
        projected,
        PRIVATE_RANK,
    )

    # Reproject then QR to suppress numerical leakage.
    private = orth(
        P @ private,
        PRIVATE_RANK,
        "private update",
    )

    return private


def shared_update(
    C_by_name: dict[str, torch.Tensor],
    private_by_name: dict[str, torch.Tensor],
):
    """
    Exact conditional maximizer for the shared block.

    For fixed private subspaces, V_g must lie in the orthogonal
    complement of the UNION of all private spans. Within that feasible
    subspace, Rayleigh-Ritz gives the top-SHARED_RANK eigenspace of the
    projected pooled covariance.
    """
    raw_union = torch.cat(
        [
            private_by_name[name]
            for name in sorted(private_by_name)
        ],
        dim=1,
    )

    union = column_span_basis(
        raw_union
    )

    ambient = int(
        raw_union.shape[0]
    )

    remaining = (
        ambient
        - int(union.shape[1])
    )

    if remaining < SHARED_RANK:
        raise RuntimeError(
            "Private union leaves insufficient room for the "
            f"shared rank: ambient={ambient}, "
            f"private_union_rank={union.shape[1]}, "
            f"remaining={remaining}, "
            f"shared_rank={SHARED_RANK}"
        )

    P = projector_complement(
        union
    )

    pooled = sum(
        C_by_name.values()
    )

    projected = (
        P
        @ pooled
        @ P
    )

    shared = top_eig(
        projected,
        SHARED_RANK,
    )

    # Suppress tiny numerical leakage back into the private union.
    shared = orth(
        P @ shared,
        SHARED_RANK,
        "shared conditional update",
    )

    # Feasibility audit.
    for name, private in private_by_name.items():
        overlap = float(
            torch.linalg.matrix_norm(
                shared.T @ private,
                ord="fro",
            ).item()
        )

        if overlap > 1.0e-7:
            raise RuntimeError(
                f"Shared/private feasibility failure for {name}: "
                f"||Vg^T Vpriv||_F={overlap:.12e}"
            )

    return shared


def shared_private_objective(
    C_by_name,
    shared,
    private_by_name,
):
    total = 0.0

    for name, C in C_by_name.items():
        total += side_score(
            C,
            shared,
        )

        total += side_score(
            C,
            private_by_name[name],
        )

    return total


def make_shared_private_initializations(
    pooled: torch.Tensor,
    device: torch.device,
):
    """
    Four deterministic starts:

      start 0:
        pooled top-64.

      starts 1..3:
        seeded 64-D rotations/subspaces inside the pooled top-128
        eigenspace.

    The rotated starts remain informed by the pooled signal but explore
    different feasible shared/private decompositions. This is more
    defensible than relying on one local stationary point.
    """
    pooled_top128 = top_eig(
        pooled,
        RANK,
    )

    starts = [
        pooled_top128[:, :SHARED_RANK].contiguous()
    ]

    seeds = (
        20261,
        20262,
        20263,
    )

    for seed in seeds:
        generator = torch.Generator(
            device="cpu"
        ).manual_seed(seed)

        random_matrix = torch.randn(
            RANK,
            SHARED_RANK,
            generator=generator,
            dtype=torch.float64,
        ).to(device)

        rotation, _ = torch.linalg.qr(
            random_matrix,
            mode="reduced",
        )

        candidate = (
            pooled_top128
            @ rotation
        )

        candidate = orth(
            candidate,
            SHARED_RANK,
            f"initial shared space seed={seed}",
        )

        starts.append(
            candidate
        )

    return starts


def run_shared_private_start(
    C_by_name: dict[str, torch.Tensor],
    shared_initial: torch.Tensor,
    max_iters: int,
    tol: float,
):
    """
    Monotonic block-coordinate ascent.

    Each private update is the exact conditional optimum given shared.

    Each shared update is the exact conditional optimum given all
    private spaces.

    Therefore the objective must be non-decreasing after every full
    cycle. The method converges to a block-stationary/local solution;
    no global-optimum claim is made.
    """
    shared = shared_initial

    private = {
        name: private_update(
            C_by_name[name],
            shared,
        )
        for name in sorted(C_by_name)
    }

    history = [
        shared_private_objective(
            C_by_name,
            shared,
            private,
        )
    ]

    converged = False

    for iteration in range(
        1,
        max_iters + 1,
    ):
        shared_new = shared_update(
            C_by_name,
            private,
        )

        private_new = {
            name: private_update(
                C_by_name[name],
                shared_new,
            )
            for name in sorted(C_by_name)
        }

        value = (
            shared_private_objective(
                C_by_name,
                shared_new,
                private_new,
            )
        )

        previous = history[-1]

        monotonic_tolerance = (
            1.0e-8
            * max(
                abs(previous),
                1.0,
            )
        )

        if (
            value
            + monotonic_tolerance
            < previous
        ):
            raise RuntimeError(
                "Monotonic block-coordinate ascent audit failed: "
                f"iteration={iteration}, "
                f"new={value:.12e}, "
                f"old={previous:.12e}"
            )

        history.append(
            value
        )

        relative_change = (
            abs(
                value
                - previous
            )
            /
            max(
                abs(previous),
                1.0,
            )
        )

        shared = shared_new
        private = private_new

        if relative_change <= tol:
            converged = True
            break

    return {
        "shared":
            shared,

        "private":
            private,

        "objective":
            history[-1],

        "objective_history":
            history,

        "iterations":
            len(history) - 1,

        "converged":
            converged,
    }


def solve_shared_private_group(
    members: list[str],
    whitened: dict[str, dict[str, torch.Tensor]],
    shared_side: str,
    max_iters: int,
    tol: float,
):
    C = {
        name: side_covariance(
            whitened[name]["Gt"],
            shared_side,
        )
        for name in members
    }

    pooled = sum(
        C.values()
    )

    starts = (
        make_shared_private_initializations(
            pooled,
            pooled.device,
        )
    )

    solutions = []

    for start_index, shared_initial in enumerate(starts):
        solution = run_shared_private_start(
            C_by_name=C,
            shared_initial=shared_initial,
            max_iters=max_iters,
            tol=tol,
        )

        solution["start_index"] = (
            start_index
        )

        solutions.append(
            solution
        )

    # Pick the best stationary/local solution by the ACTUAL
    # shared/private calibration objective.
    best = max(
        solutions,
        key=lambda x: x["objective"],
    )

    shared = best["shared"]
    private = best["private"]

    bases = {}

    constrained_objective = 0.0
    independent_objective = 0.0

    for name in members:
        Gt = whitened[name]["Gt"]

        if shared_side == "right":
            V = torch.cat(
                [
                    shared,
                    private[name],
                ],
                dim=1,
            )

            V = orth(
                V,
                RANK,
                f"{name}: combined right shared/private basis",
            )

            U = orth(
                Gt @ V,
                RANK,
                f"{name}: U=orth(GV)",
            )

        elif shared_side == "left":
            U = torch.cat(
                [
                    shared,
                    private[name],
                ],
                dim=1,
            )

            U = orth(
                U,
                RANK,
                f"{name}: combined left shared/private basis",
            )

            V = orth(
                Gt.T @ U,
                RANK,
                f"{name}: V=orth(G^T U)",
            )

        else:
            raise ValueError(
                shared_side
            )

        matrix_value = float(
            torch.sum(
                (
                    U.T
                    @ Gt
                    @ V
                ) ** 2
            ).item()
        )

        constrained_objective += (
            matrix_value
        )

        independent_objective += (
            independent_fcca_score(
                Gt,
                RANK,
            )
        )

        bases[name] = {
            "U": U,
            "V": V,
        }

    # Since this architecture restricts the independent-FCCA feasible
    # set, its calibration objective cannot exceed independent FCCA.
    theorem_tolerance = (
        1.0e-7
        * max(
            independent_objective,
            1.0,
        )
    )

    if (
        constrained_objective
        >
        independent_objective
        + theorem_tolerance
    ):
        raise RuntimeError(
            "THEOREM AUDIT FAILED: "
            f"shared/private={constrained_objective:.12e} > "
            f"independent_FCCA={independent_objective:.12e}"
        )

    start_summaries = []

    for solution in solutions:
        start_summaries.append({
            "start_index":
                int(solution["start_index"]),

            "final_objective":
                float(solution["objective"]),

            "iterations":
                int(solution["iterations"]),

            "converged":
                bool(solution["converged"]),

            "objective_history":
                [
                    float(x)
                    for x
                    in solution[
                        "objective_history"
                    ]
                ],
        })

    return bases, {
        "shared_private_objective":
            constrained_objective,

        "independent_fcca_objective":
            independent_objective,

        "ratio_to_independent":
            (
                constrained_objective
                /
                independent_objective
            ),

        "number_of_starts":
            len(solutions),

        "selected_start":
            int(
                best["start_index"]
            ),

        "iterations":
            int(
                best["iterations"]
            ),

        "converged":
            bool(
                best["converged"]
            ),

        "objective_history":
            [
                float(x)
                for x
                in best[
                    "objective_history"
                ]
            ],

        "all_starts":
            start_summaries,
    }


# ---------------------------------------------------------------------
# Fisher -> native coordinate mapping
# ---------------------------------------------------------------------

def native_basis(
    U: torch.Tensor,
    V: torch.Tensor,
    stat: dict[str, torch.Tensor],
    name: str,
):
    P0 = (
        stat["inv_d"][:, None]
        * U
    )

    Q0 = (
        stat["inv_a"][:, None]
        * V
    )

    P, RP = torch.linalg.qr(
        P0,
        mode="reduced",
    )

    Q, RQ = torch.linalg.qr(
        Q0,
        mode="reduced",
    )

    P = P[:, :RANK]
    Q = Q[:, :RANK]

    eye = torch.eye(
        RANK,
        device=P.device,
        dtype=P.dtype,
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

    if p_err > 1.0e-8:
        raise RuntimeError(
            f"{name}: native P QR error "
            f"{p_err}"
        )

    if q_err > 1.0e-8:
        raise RuntimeError(
            f"{name}: native Q QR error "
            f"{q_err}"
        )

    return (
        P.float().cpu(),
        Q.float().cpu(),
        RP.float().cpu(),
        RQ.float().cpu(),
    )


# ---------------------------------------------------------------------
# Full build
# ---------------------------------------------------------------------

def build(
    language: str,
    variant: str,
    stats_path: Path,
    device: torch.device,
    max_iters: int,
    tol: float,
):
    task_payload, modules = (
        load_task_stats(
            stats_path
        )
    )

    groups = make_pair_groups(
        modules.keys()
    )

    out = output_path(
        language,
        variant,
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "method":
            "asymmetric_one_sided_fcca",

        "variant":
            variant,

        "base_model":
            BASE_MODEL,

        "language":
            language,

        "rank":
            RANK,

        "shared_rank":
            (
                RANK
                if variant == "hard"
                else SHARED_RANK
            ),

        "private_rank":
            (
                0
                if variant == "hard"
                else PRIVATE_RANK
            ),

        "calibration_size":
            CALIBRATION_SIZE,

        "calibration_seed":
            CALIBRATION_SEED,

        "damping":
            DAMPING,

        "pairing":
            "fixed adjacent non-overlapping pairs "
            "(0,1),(2,3),...,(22,23)",

        "sharing_rule":
            "Q/K/V share Fisher-coordinate right/input side; "
            "O shares Fisher-coordinate left/output side",

        "task_stats_path":
            str(stats_path.resolve()),

        "bases": {
            str(RANK): {}
        },

        "groups": {},

        "target_names":
            sorted(modules),

        "trainable_core_budget":
            288 * RANK * RANK,
    }

    total_constrained = 0.0
    total_independent = 0.0

    ordered_groups = sorted(
        groups.items(),
        key=lambda kv: kv[0],
    )

    for group_index, (
        key,
        members,
    ) in enumerate(
        ordered_groups,
        start=1,
    ):
        family, projection, pair_start = key

        shared_side = (
            "left"
            if projection == "o"
            else "right"
        )

        whitened = {
            name: whiten(
                modules[name],
                device,
            )
            for name in members
        }

        if variant == "hard":
            fisher_bases, diagnostics = (
                solve_hard_group(
                    members,
                    whitened,
                    shared_side,
                )
            )

            constrained = diagnostics[
                "shared_objective"
            ]

        elif variant == "shared_private":
            fisher_bases, diagnostics = (
                solve_shared_private_group(
                    members,
                    whitened,
                    shared_side,
                    max_iters=max_iters,
                    tol=tol,
                )
            )

            constrained = diagnostics[
                "shared_private_objective"
            ]

        else:
            raise ValueError(variant)

        independent = diagnostics[
            "independent_fcca_objective"
        ]

        total_constrained += constrained
        total_independent += independent

        group_name = (
            f"{family}_{projection}_"
            f"{pair_start:02d}_{pair_start + 1:02d}"
        )

        payload["groups"][
            group_name
        ] = {
            "family":
                family,

            "projection":
                projection,

            "layers":
                [
                    pair_start,
                    pair_start + 1,
                ],

            "shared_side":
                shared_side,

            "members":
                members,

            **diagnostics,
        }

        for name in members:
            P, Q, RP, RQ = (
                native_basis(
                    fisher_bases[name]["U"],
                    fisher_bases[name]["V"],
                    whitened[name],
                    name,
                )
            )

            payload["bases"][
                str(RANK)
            ][name] = {
                "P": P,
                "Q": Q,

                # Stored only as audit information.
                # Training starts a fresh zero core in QR coordinates.
                "left_qr_R": RP,
                "right_qr_R": RQ,
            }

        print(
            f"[{group_index:03d}/"
            f"{len(ordered_groups):03d}] "
            f"{group_name} "
            f"side={shared_side} "
            f"ratio="
            f"{constrained / independent:.8f}",
            flush=True,
        )

        del whitened, fisher_bases
        torch.cuda.empty_cache()

    global_ratio = (
        total_constrained
        / total_independent
    )

    if global_ratio > 1.0 + 1.0e-7:
        raise RuntimeError(
            "GLOBAL THEOREM AUDIT FAILED: "
            f"ratio={global_ratio}"
        )

    payload["global_audit"] = {
        "constrained_objective":
            total_constrained,

        "independent_fcca_objective":
            total_independent,

        "ratio_to_independent":
            global_ratio,

        "theorem_check":
            "PASS",

        "statement":
            "Constrained feasible set is a subset of "
            "independent rank-128 FCCA; therefore calibration "
            "objective must not exceed independent FCCA.",
    }

    tmp = out.with_suffix(
        out.suffix
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        payload,
        tmp,
    )

    os.replace(
        tmp,
        out,
    )

    metadata = {
        key: value
        for key, value
        in payload.items()
        if key != "bases"
    }

    out.with_suffix(
        ".json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print(
        f"[PASS] {language} / {variant}"
    )
    print(
        f"[objective ratio] "
        f"{global_ratio:.12f}"
    )
    print(
        f"[basis] {out}"
    )
    print(
        f"[trainable cores] "
        f"{payload['trainable_core_budget']:,}"
    )
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        required=True,
        help="Week8 language folder, e.g. es_419, fa_ir, en_us, cmn_hans_cn.",
    )

    parser.add_argument(
        "--variant",
        choices=sorted(
            VARIANTS
        ),
        required=True,
    )

    parser.add_argument(
        "--task-stats",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-5,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        args.device
    )

    torch.cuda.set_device(
        device
    )

    stats_path = (
        args.task_stats
        if args.task_stats is not None
        else default_task_stats_path(
            args.language
        )
    )

    if not stats_path.exists():
        raise FileNotFoundError(
            stats_path
        )

    print(
        f"[language] {args.language}"
    )
    print(
        f"[variant] {args.variant}"
    )
    print(
        f"[task stats] {stats_path}"
    )
    print(
        f"[device] {device}"
    )

    build(
        language=args.language,
        variant=args.variant,
        stats_path=stats_path,
        device=device,
        max_iters=args.max_iters,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
