"""One-shot migration: re-serialize modeled_functions/*.mdl from the legacy
`src.optimizer.parametric_function` module path to `optiserve.optimizer.parametric_function`.

The .mdl files are joblib pickles of ParamFunction instances whose class and
`function` attribute reference the old `src.*` module. After the src->optiserve
rename those references no longer resolve. We temporarily alias the old module
names to the new package in sys.modules so the old pickles load, then re-dump
them — the re-dumped files reference the new `optiserve.*` path and no longer
need the alias.

Usage:
    python scripts/migrate_mdl.py [--check] [DIR]

    --check   verify every .mdl loads under the NEW path (no alias); non-zero exit on failure
    DIR       directory of .mdl files (default: modeled_functions)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

import optiserve
import optiserve.optimizer.parametric_function as _pf
import optiserve.utils as _utils
import optiserve.utils.exploration as _exploration


def _install_legacy_aliases() -> None:
    """Make `src.*` resolve to the corresponding `optiserve.*` modules."""
    sys.modules.setdefault("src", optiserve)
    sys.modules.setdefault("src.optimizer", optiserve.optimizer)
    sys.modules.setdefault("src.optimizer.parametric_function", _pf)
    sys.modules.setdefault("src.utils", _utils)
    sys.modules.setdefault("src.utils.exploration", _exploration)


def _remove_legacy_aliases() -> None:
    for name in [
        "src.utils.exploration",
        "src.utils",
        "src.optimizer.parametric_function",
        "src.optimizer",
        "src",
    ]:
        sys.modules.pop(name, None)


def migrate(directory: Path) -> int:
    files = sorted(directory.glob("*.mdl"))
    if not files:
        print(f"No .mdl files found in {directory}")
        return 0
    _install_legacy_aliases()
    migrated = 0
    for path in files:
        obj = joblib.load(path)  # loads via the legacy alias
        # Rebind the class to the new module so the re-dump records the new path.
        obj.__class__ = _pf.ParamFunction
        if getattr(obj, "function", None) is not None:
            obj.function = _pf.model_function
        joblib.dump(obj, path)
        migrated += 1
    _remove_legacy_aliases()
    print(f"Migrated {migrated} .mdl file(s) in {directory} to optiserve.* module path.")
    return 0


def check(directory: Path) -> int:
    """Load every .mdl with NO legacy alias present — proves the new path works."""
    _remove_legacy_aliases()
    files = sorted(directory.glob("*.mdl"))
    failures = []
    for path in files:
        try:
            obj = joblib.load(path)
            cls_mod = type(obj).__module__
            fn_mod = getattr(obj.function, "__module__", "?")
            assert cls_mod.startswith("optiserve"), cls_mod
            assert fn_mod.startswith("optiserve"), fn_mod
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((path.name, repr(exc)))
    if failures:
        print(f"[FAIL] {len(failures)}/{len(files)} .mdl failed to load cleanly:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    print(f"[OK] all {len(files)} .mdl load under optiserve.* with no legacy alias.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    do_check = "--check" in args
    args = [a for a in args if a != "--check"]
    target = Path(args[0]) if args else Path("modeled_functions")
    sys.exit(check(target) if do_check else migrate(target))
