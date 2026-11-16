"""Surrogate gradient for spike readouts (stub)."""

from __future__ import annotations


def spike_surrogate(v, threshold, sharpness: float = 1.0):
    """Forward: hard threshold (or identity for HH); backward: smooth surrogate.

    Implement with jax.lax.custom_vjp in Phase 2.
    """
    raise NotImplementedError("custom_vjp surrogate spike in Phase 2.")
