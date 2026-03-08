"""Every module in the optiserve package must import without error."""
import importlib
import pkgutil

import optiserve


def test_all_modules_import():
    failures = []
    for module in pkgutil.walk_packages(optiserve.__path__, "optiserve."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # noqa: BLE001 - collect all failures
            failures.append(f"{module.name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)
