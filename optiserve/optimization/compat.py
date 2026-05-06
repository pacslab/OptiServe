"""Behaviour presets for the application optimizer.

OptiServe has a tension no single default resolves: the greedy strategies
contain confirmed defects, and the same code produced published results that a
golden-master battery locks byte-for-byte. Silently fixing a bug rewrites
history; silently keeping it ships a known-wrong optimizer.

The resolution is an explicit preset. Every divergence between "what was
published" and "what is correct" is one named flag, guarded by one ``if`` at the
defect site, and covered by a golden baseline on *both* sides.

    OptimizerCompat.CORRECTED   the default: the defects are fixed
    OptimizerCompat.PUBLISHED   bug-for-bug reproduction of the published runs

Reproducing a figure from the paper therefore reads as an explicit statement of
intent::

    ApplicationOptimizer(app, config=OptimizationConfig(
        compat=OptimizerCompat.PUBLISHED))

and any result can be traced to the preset that produced it.
"""

from __future__ import annotations

from enum import Flag, auto

__all__ = ["OptimizerCompat"]


class OptimizerCompat(Flag):
    """Named deviations from corrected optimizer behaviour."""

    NONE = 0

    #: BPBA phase 1 never rebases ``current_cost`` after committing a model
    #: upgrade, while still decrementing ``surplus`` by the same amount — so
    #: from the second upgrade onward every prior upgrade is charged to the
    #: budget twice. The strategy then abandons accuracy constraints that are
    #: comfortably affordable. (BCPA's equivalent loop *does* rebase, which is
    #: what makes this a defect rather than a design choice.)
    LEGACY_BPBA_SURPLUS = auto()

    #: BAPB's candidate scan breaks on ``mem <= current_memory``, so a variant
    #: upgrade can only ever be evaluated *bundled with* a memory increase. The
    #: pure "switch to a better model, keep the memory" move — usually the
    #: cheapest way to buy accuracy — is never scored, so the strategy spends
    #: budget on memory it does not need. The exclusion is correct in BPBA
    #: phase 2, where staying at the same memory yields no latency reduction;
    #: it was copied to BAPB, where the variant dimension also changes.
    LEGACY_BAPB_MEM_GATE = auto()

    #: BPBA and BCPA use different spellings for the same BCR modes
    #: (``RT/M`` vs ``M/RT``, ``ERT/C`` vs ``C/ERT``), and an unrecognised
    #: spelling silently disables BCR pruning instead of failing. The
    #: evaluation harness relies on that: ``run_opt_curve`` passes BPBA's
    #: spellings to BCPA, so the ``BCR_M/RT`` and ``BCR_C/ERT`` columns of every
    #: published curve are BCR-*disabled* reruns. Under ``CORRECTED`` an
    #: unknown BCR type raises instead.
    LEGACY_BCR_ALIASES = auto()

    #: Everything as published. Use to reproduce the paper's numbers exactly.
    PUBLISHED = LEGACY_BPBA_SURPLUS | LEGACY_BAPB_MEM_GATE | LEGACY_BCR_ALIASES

    #: Every known defect fixed. The default.
    CORRECTED = NONE

    def enabled(self, flag: OptimizerCompat) -> bool:
        """Whether ``flag`` is set. Reads better than ``bool(self & flag)``."""
        return bool(self & flag)
