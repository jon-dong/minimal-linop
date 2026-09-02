"""Helpers to check operators: the dot-product test and dense materialisation."""

import math

import torch

from ._utils import as_shape

__all__ = ["adjoint_error", "to_matrix"]


def adjoint_error(op, x, y=None):
    """Relative discrepancy of the dot-product test,
    ``|Re<A x, y> - Re<x, A^H y>| / (|A x| |y|)``.

    Should be at round-off level (about 1e-6 in single precision, 1e-14 in
    double) when ``applyT`` is the adjoint of ``apply``.  The real part of the
    inner product is used so that mixed real/complex operators such as
    ``LinOpReal`` are covered as well.  ``y`` defaults to a random tensor
    shaped like ``A x``.
    """
    Ax = op.apply(x)
    if y is None:
        y = torch.randn_like(Ax)
    ATy = op.applyT(y)
    lhs = (Ax.conj() * y).sum().real
    rhs = (x.conj() * ATy).sum().real
    scale = Ax.norm() * y.norm()
    return float(abs(lhs - rhs) / max(float(scale), torch.finfo(Ax.real.dtype).tiny))


def to_matrix(op, in_shape=None, dtype=torch.complex64, device=None):
    """The dense matrix of ``op``, column ``i`` being ``op`` applied to the
    ``i``-th basis vector of ``in_shape`` (default ``op.in_shape``), flattened
    in row-major order.  Uses one batched call, so ``op`` must be
    batch-transparent.  For inspection, debugging and small problems."""
    shape = as_shape(op.in_shape if in_shape is None else in_shape)
    n = math.prod(shape)
    basis = torch.eye(n, dtype=dtype, device=device).reshape(n, *shape)
    return op.apply(basis).reshape(n, -1).T
