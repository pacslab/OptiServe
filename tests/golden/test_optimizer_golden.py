"""Golden-master regression for the application optimizer — locks the greedy
strategies / BCR variants across acyclic and cyclic graphs so the Stage-7
refactor is provably behavior-preserving."""

import json
from pathlib import Path

from optimizer_cases import compute_all

BASELINE = json.loads((Path(__file__).parent / "optimizer_baseline.json").read_text())


def test_optimizer_battery_matches_baseline():
    result = compute_all()
    assert result.keys() == BASELINE.keys()
    for case, expected in BASELINE.items():
        assert result[case] == expected, (
            f"Case {case} drifted:\n  got: {result[case]}\n  expected: {expected}"
        )
