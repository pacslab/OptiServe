"""Golden-master regression for the analytical application model.

Locks the deterministic outputs (per-node expected executions `ne` and
end-to-end response time) of the App graph from
experiments/application_modeling.ipynb. These values are published in the
notebook (rt = 1739.2857..., see ne dict) and must survive every refactor of
the modeling / graph-reduction code.
"""

import json
from pathlib import Path

from _app_modeling_case import compute

from optiserve.modeling.application_model import (
    ApplicationPerformanceModeling,
)

BASELINE = json.loads((Path(__file__).parent / "app_modeling_baseline.json").read_text())


def test_app_modeling_matches_baseline():
    result = compute(ApplicationPerformanceModeling)
    assert result == BASELINE, (
        "Application-model outputs drifted from the golden baseline.\n"
        f"  got: {result}\n  expected: {BASELINE}"
    )
