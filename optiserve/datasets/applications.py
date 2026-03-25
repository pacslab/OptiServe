"""Benchmark application definitions.

Currently provides App3 — the committed 3-function workflow (``f1 -> resnet ->
yolo`` with a back-edge and a self-loop) used in the evaluation — plus a generic
helper for assembling ML workflows from cached performance models.

NOTE: only App3's graph/formula is defined in the repository. App1/2/4/5/6 were
produced by manually editing the evaluation notebook and their graph, variant,
and accuracy-formula definitions are not committed anywhere; they must be
supplied to reproduce the full experiment set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

from optiserve.modeling.parametric import ParamFunction
from optiserve.workflow import ModelVariant, WorkflowGraph

# App3 configuration (from experiments/test_optimization.ipynb).
_RESNET_VARIANTS = ["resnet-18", "resnet-34", "resnet-50", "resnet-101"]
_YOLO_VARIANTS = ["yolov10n", "yolov10s", "yolov10m", "yolov10l"]
_F1_GRID = list(range(128, 3072, 192))
_RESNET_GRID = list(range(2048, 10000, 512))
_YOLO_GRID = list(range(2048, 10000, 800))

# End-to-end accuracy weights resnet (a1) x2 and yolo (a2); f1 (a0) is non-ML.
APP3_ACCURACY_FORMULA: Callable = lambda a0, a1, a2: 2 * a1 + a2  # noqa: E731


def _load(models_dir: Path, name: str) -> ParamFunction:
    path = models_dir / f"{name}.mdl"
    if not path.exists():
        raise FileNotFoundError(
            f"Cached model {path} not found. App3 requires the modeled_functions/ "
            "cache (produced by profiling on live AWS)."
        )
    return ParamFunction.load(path)


def build_app3(models_dir="modeled_functions") -> Tuple[WorkflowGraph, Callable]:
    """Assemble the App3 workflow from cached per-function models.

    Returns the built :class:`WorkflowGraph` and its end-to-end accuracy formula.
    """
    models_dir = Path(models_dir)

    f1_variants = [ModelVariant("None", _load(models_dir, "f1"))]
    resnet_variants: List[ModelVariant] = [
        ModelVariant(v, _load(models_dir, f"resnet_{v}")) for v in _RESNET_VARIANTS
    ]
    yolo_variants: List[ModelVariant] = [
        ModelVariant(v, _load(models_dir, f"yolo_{v}")) for v in _YOLO_VARIANTS
    ]

    wg = WorkflowGraph()
    wg.add_ml_function(1, f1_variants, _F1_GRID)      # f1
    wg.add_ml_function(2, resnet_variants, _RESNET_GRID)  # resnet
    wg.add_ml_function(3, yolo_variants, _YOLO_GRID)      # yolo
    wg.add_edges([
        ("Start", 1, 1.0), (1, 2, 1.0), (2, 1, 0.3), (2, 3, 0.7),
        (3, 3, 0.2), (3, "End", 0.8),
    ])
    return wg, APP3_ACCURACY_FORMULA
