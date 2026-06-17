"""``python -m optiserve`` entry point (mirrors the ``optiserve`` console script)."""

from optiserve.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
