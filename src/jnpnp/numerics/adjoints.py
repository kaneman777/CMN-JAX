"""Adjoint / checkpoint configuration helpers (stub).

Prefer Diffrax RecursiveCheckpointAdjoint; avoid BacksolveAdjoint for stiff HH.
"""

from __future__ import annotations


def default_adjoint(checkpoints: int | None = None):
    raise NotImplementedError("Return RecursiveCheckpointAdjoint(...) in Phase 2.")
