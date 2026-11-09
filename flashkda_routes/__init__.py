"""Reproducible experiments on FlashKDA's SM80-MMA design and SM100 alternatives.

The package is deliberately split from the upstream kernels: it generates an
isolated copy of the pinned FlashKDA sources, compiles it as a separate Torch
extension, and never modifies or shadows the installed baseline.
"""

__version__ = "1.0.0"
