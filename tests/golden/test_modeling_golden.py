"""Golden-master regression over a battery of application-model graph shapes
(branch, parallel join, self-loop, mixed) under both delay models. Locks the
graph-reduction refactor as behavior-preserving."""

import json
from pathlib import Path

from modeling_cases import compute_all

from optiserve.modeling.application_model import ApplicationPerformanceModeling

BASELINE = json.loads((Path(__file__).parent / "modeling_baseline.json").read_text())


def test_modeling_battery_matches_baseline():
    result = compute_all(ApplicationPerformanceModeling)
    assert result.keys() == BASELINE.keys()
    for case, expected in BASELINE.items():
        assert result[case] == expected, (
            f"Case {case} drifted:\n  got: {result[case]}\n  expected: {expected}"
        )
