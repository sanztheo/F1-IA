"""Backward‑compat shim for older imports.

Re‑exports the MLP policy helpers from rl.models.mlp_policy.
"""

from .models.mlp_policy import init_mlp, forward, mutate  # noqa: F401
