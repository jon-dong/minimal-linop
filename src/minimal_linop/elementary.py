"""Elementary operators: identity, diagonal, real/imaginary parts, reductions,
dense matrices, function wrappers and concatenation."""

import torch

from ._utils import as_dims, as_shape, complex_dtype
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
    multiplies by ``conj(c)``.

    The declared shape is ``c.shape``, except when ``c`` is a scalar or has a
    size-1 axis: it then broadcasts, the shape it acts on is not determined by
    ``c``, and the operator declares none rather than a wrong one.
    """

    def __init__(self, coefficients: torch.Tensor):
        self.coefficients = coefficients
        shape = tuple(coefficients.shape)
        if shape and 1 not in shape:
            self.in_shape = self.out_shape = shape

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

    preserves_shape = False

    def __init__(self, dim: int, size: int):
        dims = as_dims(dim)                      # one trailing (negative) axis
        if len(dims) != 1:
            raise ValueError("LinOpSumReduce reduces exactly one axis")
        self.dim, self.size = dims[0], int(size)

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
        return torch.einsum("ij,...i->...j", self.matrix.conj(), y)


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
    widths come from the sub-operators' ``out_shape``; a sub-operator that
    declares none must preserve shape, and the width left over by the
    declared ones is shared equally among those.  The sub-operators must
    agree on every output axis except the last.
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
            self.out_shape = next(iter(leads)) + (sum(o[-1] for o in outs),)
        elif (all(o is None for o in outs) and self.in_shape is not None
              and all(op.preserves_shape for op in self.ops)):
            self.out_shape = self.in_shape[:-1] + (len(self.ops) * self.in_shape[-1],)
        else:
            self.out_shape = None
        self.preserves_shape = len(self.ops) == 1 and self.ops[0].preserves_shape

    def _widths(self, y):
        """How the last axis of ``y`` splits among the sub-operators."""
        widths = [None if op.out_shape is None else op.out_shape[-1] for op in self.ops]
        free = [k for k, w in enumerate(widths) if w is None]
        if not all(self.ops[k].preserves_shape for k in free):
            raise ValueError(
                "LinOpCat cannot split its adjoint input: a sub-operator declares no "
                "out_shape and does not preserve shape; declare it, e.g. with "
                "LinOpFunction(apply, applyT, out_shape=...)"
            )
        left = y.shape[-1] - sum(w for w in widths if w is not None)
        share, rem = divmod(left, len(free)) if free else (0, left)
        if rem or left < 0:
            raise ValueError(
                f"LinOpCat cannot split {y.shape[-1]} columns into widths {widths} "
                f"(None: an equal share of what is left)"
            )
        return [share if w is None else w for w in widths]

    def apply(self, x):
        return torch.cat([op.apply(x) for op in self.ops], dim=-1)

    def applyT(self, y):
        chunks = torch.split(y, self._widths(y), dim=-1)
        return sum(op.applyT(c) for op, c in zip(self.ops, chunks))
