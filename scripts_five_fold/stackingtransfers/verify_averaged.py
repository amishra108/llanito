from pathlib import Path
import torch

# Paths

PROJECT_ROOT = Path(
    "/research/milsrg1/user_workspace/am3244/llanito-workspace"
)

SOURCE_ROOT = (
    PROJECT_ROOT
    / "outputs_five_fold" / "models"
    / "llanito_resadapt_encoder_all"
)

AVERAGED_STATE = (
    PROJECT_ROOT
    / "outputs_five_fold" / "models"
    / "llanitoresadaptenc_averaged"
    / "adapter"
    / "adapter_state.pt"
)

SOURCE_STATES = [
    SOURCE_ROOT
    / f"fold_{fold}"
    / "adapter"
    / "adapter_state.pt"
    for fold in range(1, 6)
]

# Load states

states = [
    torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    for path in SOURCE_STATES
]

averaged = torch.load(
    AVERAGED_STATE,
    map_location="cpu",
    weights_only=True,
)

# Check keys

reference_keys = set(
    states[0].keys()
)

for fold_index, state in enumerate(
    states,
    start=1,
):
    assert set(state.keys()) == reference_keys, (
        f"Fold {fold_index} has different keys."
    )

assert set(averaged.keys()) == reference_keys, (
    "Averaged model has different keys."
)

print(
    f"All models contain "
    f"{len(reference_keys)} matching tensors."
)

# Check shapes

for key in sorted(
    reference_keys
):
    shape = states[0][key].shape

    for fold_index, state in enumerate(
        states,
        start=1,
    ):
        assert state[key].shape == shape, (
            f"Shape mismatch for {key} "
            f"in fold {fold_index}."
        )

    assert averaged[key].shape == shape, (
        f"Averaged shape mismatch for {key}."
    )

print(
    "All tensor shapes match."
)

# Verify exact parameter-wise mean

maximum_error = 0.0
worst_key = None

for key in sorted(
    reference_keys
):
    expected = torch.stack(
        [
            state[key].float()
            for state in states
        ],
        dim=0,
    ).mean(
        dim=0
    )

    actual = (
        averaged[key]
        .float()
    )

    error = (
        actual
        - expected
    ).abs().max().item()

    if error > maximum_error:
        maximum_error = error
        worst_key = key

print(
    f"Maximum averaging error: "
    f"{maximum_error}"
)

print(
    f"Worst tensor: "
    f"{worst_key}"
)

if maximum_error < 1e-6:
    print(
        "PASS: averaged weights are "
        "the exact mean of the five folds."
    )

else:
    print(
        "FAIL: averaged weights do not "
        "match the five-fold mean."
    )