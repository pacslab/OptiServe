"""Golden-master for the CORRECTED optimizer preset.

``optimizer_baseline.json`` freezes the *published* behaviour, bugs included, so
a refactor cannot silently move a number that appears in the paper. That leaves
the fixed path unguarded — which is where new work happens. This file freezes it
too, over the same case matrix, so the two presets can never drift into each
other unnoticed.

Regenerate with ``python tests/golden/regenerate.py optimizer_corrected``.
"""

import json
from pathlib import Path

import pytest
from optimizer_cases import compute_all

from optiserve.optimization.compat import OptimizerCompat

BASELINE = json.loads((Path(__file__).parent / "optimizer_baseline_corrected.json").read_text())
PUBLISHED_BASELINE = json.loads((Path(__file__).parent / "optimizer_baseline.json").read_text())


def test_corrected_optimizer_matches_its_baseline():
    result = compute_all(compat=OptimizerCompat.CORRECTED)
    assert result.keys() == BASELINE.keys()
    for case, expected in BASELINE.items():
        assert result[case] == expected, (
            f"Case {case} drifted:\n  got: {result[case]}\n  expected: {expected}"
        )


def test_the_two_presets_are_actually_different():
    """Guards against the compat flags silently becoming a no-op — which would
    make the PUBLISHED baseline pass while testing nothing."""
    differing = [case for case in BASELINE if BASELINE[case] != PUBLISHED_BASELINE[case]]
    assert differing, (
        "CORRECTED and PUBLISHED produce identical results everywhere, so the "
        "compat gates are not wired up"
    )
    # The BAPB memory gate is the flag with a visible effect on this matrix.
    assert any("BAPB" in case for case in differing)


@pytest.mark.parametrize("preset", [OptimizerCompat.PUBLISHED, OptimizerCompat.CORRECTED])
def test_both_presets_respect_the_optimization_boundaries(preset):
    """Whatever the preset, a returned configuration must lie inside the
    feasible cost/latency envelope the optimizer computed for itself."""
    import contextlib
    import io

    from optimizer_cases import _ACCURACY_FORMULA, _acyclic, _optimizer

    optimizer = _optimizer(_acyclic(), compat=preset)
    budget = (optimizer.minimal_cost + optimizer.maximal_cost) / 2
    with contextlib.redirect_stdout(io.StringIO()):
        result = optimizer.BPBA(budget, 0.5, _ACCURACY_FORMULA, BCR=False)

    assert optimizer.minimal_cost <= result.cost <= optimizer.maximal_cost + 1e-9
    assert optimizer.minimal_avg_rt - 1e-9 <= result.response_time_ms
    assert result.response_time_ms <= optimizer.maximal_avg_rt + 1e-9
