"""One-shot migration: re-serialize modeled_functions/*.mdl to the current
``optiserve.modeling.parametric`` module path.

The .mdl files are joblib pickles of ParamFunction instances whose class and
``function`` attribute reference whichever module ParamFunction lived in when
they were saved (historically ``src.optimizer.parametric_function``, then
``optiserve.optimizer.parametric_function``). We alias every known legacy path
to the current module in sys.modules so the old pickles load, then re-dump them
under the current path — after which no alias is needed.

Usage:
    python scripts/migrate_mdl.py [--check] [DIR]

    --check   verify every .mdl loads under the CURRENT path (no alias)
    DIR       directory of .mdl files (default: modeled_functions)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

import optiserve
import optiserve.modeling.parametric as _pf

# Legacy fully-qualified module paths ParamFunction has lived under.
_LEGACY_PATHS = [
    "src.optimizer.parametric_function",
    "optiserve.optimizer.parametric_function",
]


def _install_legacy_aliases() -> None:
    sys.modules.setdefault("src", optiserve)
    sys.modules.setdefault("src.optimizer", optiserve)
    for path in _LEGACY_PATHS:
        sys.modules.setdefault(path, _pf)


def _remove_legacy_aliases() -> None:
    for path in _LEGACY_PATHS + ["src.optimizer", "src"]:
        sys.modules.pop(path, None)


def migrate(directory: Path) -> int:
    files = sorted(directory.glob("*.mdl"))
    if not files:
        print(f"No .mdl files found in {directory}")
        return 0
    _install_legacy_aliases()
    for path in files:
        obj = joblib.load(path)
        obj.__class__ = _pf.ParamFunction
        if getattr(obj, "function", None) is not None:
            obj.function = _pf.model_function
        joblib.dump(obj, path)
    _remove_legacy_aliases()
    print(f"Migrated {len(files)} .mdl file(s) in {directory} to optiserve.modeling.parametric.")
    return 0


def check(directory: Path) -> int:
    _remove_legacy_aliases()
    files = sorted(directory.glob("*.mdl"))
    failures = []
    for path in files:
        try:
            obj = joblib.load(path)
            assert type(obj).__module__.startswith("optiserve"), type(obj).__module__
            assert getattr(obj.function, "__module__", "").startswith("optiserve")
        except Exception as exc:  # noqa: BLE001
            failures.append((path.name, repr(exc)))
    if failures:
        print(f"[FAIL] {len(failures)}/{len(files)} .mdl failed:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    print(f"[OK] all {len(files)} .mdl load under optiserve.modeling.parametric.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--check"]
    do_check = "--check" in sys.argv[1:]
    target = Path(args[0]) if args else Path("modeled_functions")
    sys.exit(check(target) if do_check else migrate(target))
