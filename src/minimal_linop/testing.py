"""Helpers to check operators: the dot-product test, the spectral norm and
dense materialisation."""

import math

import torch

from ._utils import as_index, as_shape

__all__ = ["adjoint_error", "operator_norm", "to_matrix"]


def adjoint_error(op, x, y=None):
    """Relative discrepancy of the dot-product test,
    ``|Re<A x, y> - Re<x, A^H y>| / (|A x| |y|)``.

    Should be at round-off level (about 1e-6 in single precision, 1e-14 in
    double) when ``applyT`` is the adjoint of ``apply``.  The real part of the
    inner product is used so that mixed real/complex operators such as
    ``LinOpReal`` are covered as well.  ``y`` defaults to a random tensor
    shaped like ``A x``.

    If ``A x`` or ``y`` vanishes the denominator falls back to ``|x| |A^H y|``,
    which bounds the same discrepancy; if both do, the test is vacuous and the
    result is 0.

    The operator is applied in the caller's grad mode, so an ``applyT``
    written with ``torch.autograd.grad`` is checked like any other.  Only the
    inner products and norms stay out of autograd: they are taken on detached
    tensors, so ``x`` may require grad and the operator may hold learnable
    tensors.
    """
    Ax = op.apply(x)
    if y is None:
        y = torch.randn_like(Ax)
    ATy = op.applyT(y)
    x, y, Ax, ATy = x.detach(), y.detach(), Ax.detach(), ATy.detach()
    lhs = (Ax.conj() * y).sum().real
    rhs = (x.conj() * ATy).sum().real
    scale = float(Ax.norm()) * float(y.norm())
    if scale == 0.0:
        scale = float(x.norm()) * float(ATy.norm())
    if scale == 0.0:
        return 0.0
    return float(abs(lhs - rhs)) / scale


def operator_norm(op, x0, n_iter=50):
    """Estimate the spectral norm ``||A||_2`` by power iteration on ``A^H A``.

    ``x0`` fixes the shape, dtype and device of the iterate; ``n_iter``
    products ``A^H A x`` are formed and the largest singular value is the
    square root of the limiting Rayleigh quotient.  The estimate approaches
    ``||A||_2`` from below (it is exact only in the limit, and slower the
    closer the two largest singular values are), so a gradient step ``1 /
    ||A||^2`` derived from it should keep a small margin.  "From below" holds
    up to the round-off of the norms: in single precision the estimate can
    end slightly above.  Returns 0.0 for the zero operator.

    As in ``adjoint_error``, the operator is applied in the caller's grad
    mode.  The iterate is detached at each step, so no graph accumulates over
    the iterations and ``x0`` may require grad.
    """
    n_iter = as_index(n_iter, "n_iter")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    x0 = x0.detach()
    scale = float(x0.norm())
    if scale == 0.0:
        raise ValueError("x0 must be non-zero")
    x, value = x0 / scale, 0.0
    for _ in range(n_iter):
        x = op.applyT(op.apply(x)).detach()
        value = float(x.norm())
        if value == 0.0:
            return 0.0
        x = x / value
    return math.sqrt(value)


def to_matrix(op, in_shape=None, dtype=torch.complex64, device=None):
    """The dense matrix of ``op``, column ``i`` being ``op`` applied to the
    ``i``-th basis vector of ``in_shape`` (default ``op.in_shape``), flattened
    in row-major order.  Uses one batched call, so ``op`` must be
    batch-transparent.  For inspection, debugging and small problems.

    The basis vectors are created with ``dtype`` on ``device`` (the default
    device when None): name the device of the operator's own tensors when
    they live elsewhere.  The matrix describes a complex-linear operator;
    ``LinOpReal`` and ``LinOpImag``, linear over the reals only, have none."""
    shape = op.in_shape if in_shape is None else in_shape
    if shape is None:
        raise ValueError(f"{type(op).__name__} declares no in_shape; pass in_shape=...")
    shape = as_shape(shape)
    n = math.prod(shape)
    basis = torch.eye(n, dtype=dtype, device=device).reshape(n, *shape)
    return op.apply(basis).reshape(n, -1).T
