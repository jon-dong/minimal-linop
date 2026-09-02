"""Elementary operators: identity, diagonal, real/imaginary parts, reductions,
dense matrices, function wrappers and concatenation."""

import torch

from ._utils import as_shape, complex_dtype
from .base import LinOp

__all__ = [
    "LinOpIdentity", "LinOpMul", "LinOpReal", "LinOpImag", "LinOpSumReduce",
    "LinOpMatrix", "LinOpFunction", "LinOpCat",
]


class LinOpIdentity(LinOp):
    """``A x = x``."""

    def apply(self, x):
        return x

    def applyT(self, y):
        return y


class LinOpMul(LinOp):
    """Element-wise multiplication by fixed coefficients, ``A x = c * x``
    (a diagonal operator).  ``c`` may broadcast against ``x``; the adjoint
    multiplies by ``conj(c)``."""

    def __init__(self, coefficients: torch.Tensor):
        self.coefficients = coefficients
        self.in_shape = self.out_shape = tuple(coefficients.shape)

    def apply(self, x):
        return self.coefficients * x

    def applyT(self, y):
        return self.coefficients.conj() * y


class LinOpReal(LinOp):
    """``A x = Re(x)``, complex -> real.  The adjoint embeds a real tensor as
    the real part of a complex one (``<Re x, y> = Re <x, y + 0j>``)."""

    def apply(self, x):
        return torch.real(x)

    def applyT(self, y):
        y = torch.real(y)                       # no-op for real input
        return y.to(complex_dtype(y.dtype))


class LinOpImag(LinOp):
    """``A x = Im(x)``, complex -> real.  The adjoint is ``y -> 1j * y``."""

    def apply(self, x):
        return torch.imag(x)

    def applyT(self, y):
        return 1j * torch.real(y)


class LinOpSumReduce(LinOp):
    """Sum along one axis (keeping it with size 1); the adjoint broadcasts
    back to ``size`` along that axis.

    Example: ``LinOpSumReduce(dim=-3, size=N)`` maps ``(..., N, H, W)`` to
    ``(..., 1, H, W)``, e.g. to add up N incoherent intensity images.
    """

    def __init__(self, dim: int, size: int):
        self.dim, self.size = dim, size

    def apply(self, x):
        return x.sum(dim=self.dim, keepdim=True)

    def applyT(self, y):
        shape = list(y.shape)
        shape[self.dim] = self.size
        return y.expand(shape)


class LinOpMatrix(LinOp):
    """A dense matrix ``M`` of shape ``(m, n)`` acting on the last axis:
    ``A x = M x``; adjoint ``M^H y``."""

    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        self.in_shape, self.out_shape = (matrix.shape[1],), (matrix.shape[0],)

    def apply(self, x):
        return torch.einsum("ij,...j->...i", self.matrix, x)

    def applyT(self, y):
        return torch.einsum("ij,...j->...i", self.matrix.conj().T, y)


class LinOpFunction(LinOp):
    """Wrap two callables ``apply`` and ``applyT`` into an operator.

    Handy for a one-off operator without writing a class::

        F = LinOpFunction(lambda x: torch.fft.fft(x, norm="ortho"),
                          lambda y: torch.fft.ifft(y, norm="ortho"))
    """

    def __init__(self, apply, applyT, in_shape=None, out_shape=None):
        self._apply, self._applyT = apply, applyT
        self.in_shape = None if in_shape is None else as_shape(in_shape)
        self.out_shape = None if out_shape is None else as_shape(out_shape)

    def apply(self, x):
        return self._apply(x)

    def applyT(self, y):
        return self._applyT(y)


class LinOpCat(LinOp):
    """Stack the outputs of several operators along the last axis.

    Given operators ``A_0, ..., A_{K-1}`` acting on the same input::

        apply(x)  = cat([A_0 x, ..., A_{K-1} x], dim=-1)
        applyT(y) = sum_k A_k^H y_k

    where ``y_k`` is the slice of ``y`` matching the width of ``A_k x``.  The
    per-operator widths come from the sub-operators' ``out_shape`` (or their
    common ``in_shape`` when none declares one and they are shape-preserving),
    and are refreshed by every ``apply``.  The sub-operators must agree on
    every output axis except the last.
    """

    def __init__(self, ops):
        self.ops = list(ops)
        ins = {tuple(op.in_shape) for op in self.ops if op.in_shape is not None}
        if len(ins) > 1:
            raise ValueError(f"sub-operators must share the same in_shape; got {sorted(ins)}")
        self.in_shape = next(iter(ins)) if ins else None

        outs = [op.out_shape for op in self.ops]
        if all(o is not None for o in outs):
            leads = {tuple(o[:-1]) for o in outs}
            if len(leads) > 1:
                raise ValueError(
                    "sub-operators must agree on every output axis except the last "
                    f"(the concatenation axis); got {sorted(leads)}"
                )
            self._widths = [o[-1] for o in outs]
            self.out_shape = next(iter(leads)) + (sum(self._widths),)
        elif all(o is None for o in outs) and self.in_shape is not None:
            self._widths = [self.in_shape[-1]] * len(self.ops)
            self.out_shape = self.in_shape[:-1] + (sum(self._widths),)
        else:
            self._widths = None
            self.out_shape = None

    def apply(self, x):
        outs = [op.apply(x) for op in self.ops]
        self._widths = [o.shape[-1] for o in outs]
        return torch.cat(outs, dim=-1)

    def applyT(self, y):
        if self._widths is None:
            raise RuntimeError(
                "LinOpCat cannot split its adjoint input: no sub-operator declares "
                "out_shape and apply() has not been called yet"
            )
        chunks = torch.split(y, self._widths, dim=-1)
        return sum(op.applyT(c) for op, c in zip(self.ops, chunks))
