# syntax=docker/dockerfile:1.7
###############################################################################
# OptiServe — multi-stage image.
#
#   base     shared interpreter + unprivileged user (nothing project-specific)
#   builder  build toolchain; resolves deps and builds a wheel from the source
#   runtime  minimal production image: venv + wheel, no compilers, no source
#   dev      builder + dev extras + the test suite; used by CI and compose
#
#   docker build --target runtime -t optiserve:latest .
#   docker build --target dev     -t optiserve:dev    .
###############################################################################

ARG PYTHON_VERSION=3.11

###############################################################################
# base
###############################################################################
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # matplotlib writes a font cache to $HOME on first import; point it at a
    # writable path so the image works with a read-only rootfs or a random uid.
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg

# Unprivileged user created in `base` so every downstream stage can chown to it.
RUN groupadd --gid 10001 optiserve \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin optiserve

WORKDIR /app

###############################################################################
# builder — owns the compilers. The runtime image never inherits from this.
###############################################################################
FROM base AS builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
 && apt-get install --no-install-recommends -y build-essential gfortran \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel build

# Dependency layer first: requirements.txt changes far less often than source,
# so every source-only edit reuses this (expensive) layer. CI asserts that this
# file stays in sync with pyproject.toml's [project.dependencies].
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Source last, then the wheel built from it.
COPY pyproject.toml README.md LICENSE ./
COPY optiserve/ ./optiserve/
RUN python -m build --wheel --outdir /tmp/dist \
 && pip install --no-deps /tmp/dist/*.whl

###############################################################################
# runtime — production image: no compilers, no source tree, no root.
###############################################################################
FROM base AS runtime

COPY --from=builder --chown=optiserve:optiserve /opt/venv /opt/venv

# Cached models and evaluation outputs are mounted at run time, not baked in.
RUN install -d -o optiserve -g optiserve \
      /app/modeled_functions /app/output /tmp/matplotlib

USER optiserve

# Import check inside the final image: a broken wheel fails the build here
# rather than in production.
RUN python -c "import optiserve; print('optiserve', optiserve.__version__)"

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import optiserve"]

ENTRYPOINT ["optiserve"]
CMD ["--help"]

###############################################################################
# dev — the test/CI image: source tree, dev extras, editable install.
###############################################################################
FROM builder AS dev

COPY requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

COPY . .
RUN pip install --no-deps -e . \
 && install -d -o optiserve -g optiserve /tmp/matplotlib \
 && chown -R optiserve:optiserve /app /opt/venv

USER optiserve

ENTRYPOINT []
CMD ["pytest", "-q"]
