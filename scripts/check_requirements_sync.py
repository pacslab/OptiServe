#!/usr/bin/env python3
"""Assert that the pip requirement files still mirror ``pyproject.toml``.

``pyproject.toml`` is the single source of truth for OptiServe's dependency
sets. ``requirements.txt`` (runtime) and ``requirements-dev.txt`` (runtime +
dev) exist only so Docker layers and CI caches can be keyed on a small file that
changes rarely. This script fails the build the moment the two drift, which is
the failure mode that otherwise shows up as "works locally, wrong versions in
the image".

    python scripts/check_requirements_sync.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_requirements(path: Path) -> set[str]:
    """Requirement specifiers from a pip requirements file.

    Comments, blank lines and ``-r``/``-c`` includes are skipped; includes are
    resolved separately by the caller so each file is checked against the set it
    is supposed to mirror.
    """
    specs: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        specs.add(line.replace(" ", ""))
    return specs


def _project_dependencies() -> tuple[set[str], set[str]]:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = meta["project"]
    runtime = {dep.replace(" ", "") for dep in project["dependencies"]}
    dev = {dep.replace(" ", "") for dep in project["optional-dependencies"]["dev"]}
    return runtime, dev


def _report(name: str, expected: set[str], actual: set[str]) -> bool:
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        print(f"OK   {name}: {len(actual)} requirements match pyproject.toml")
        return True
    print(f"FAIL {name} has drifted from pyproject.toml", file=sys.stderr)
    for spec in sorted(missing):
        print(f"       missing: {spec}", file=sys.stderr)
    for spec in sorted(extra):
        print(f"       unexpected: {spec}", file=sys.stderr)
    return False


def main() -> int:
    runtime_expected, dev_expected = _project_dependencies()

    runtime_actual = _read_requirements(ROOT / "requirements.txt")
    # requirements-dev.txt starts with `-r requirements.txt`, so it only lists
    # the dev additions; compare it against the dev extra alone.
    dev_actual = _read_requirements(ROOT / "requirements-dev.txt")

    ok = _report("requirements.txt", runtime_expected, runtime_actual)
    ok &= _report("requirements-dev.txt", dev_expected, dev_actual)

    if not ok:
        print(
            "\nFix: edit pyproject.toml, then mirror the change into the "
            "requirements files (they are a build-cache convenience, not a "
            "second source of truth).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
