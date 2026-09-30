"""The ``LinOp`` base class and the combinators behind ``@``, ``+``, ``-``,
``*`` and ``.H``."""

from abc import ABC, abstractmethod
import numbers

import torch

__all__ = ["LinOp", "LinOpComposition", "LinOpSum", "LinOpScalarMul", "LinOpAdjoint"]


class LinOp(ABC):
    """A linear operator ``A``, given by its action and its adjoint.

    Subclasses implement ``apply(x) = A x`` and ``applyT(y) = A^H y``, where
    ``A^H`` is the Hermitian adjoint: ``<A x, y> = <x, A^H y>`` for the inner
    product ``<u, v> = sum(conj(u) * v)``.  For a real operator this is the
    transpose.  Operators act on the trailing axes of ``x`` and leave leading
    (batch) axes alone.

    ``in_shape`` and ``out_shape`` describe those trailing axes.  ``None``
    means shape-agnostic (an FFT works on any length) and is compatible with
    everything in the shape checks.  A composition fills a missing shape from
    the operator next to it, which assumes the shape-agnostic one preserves
    shape; an operator that does not (``LinOpGrad``) sets the class attribute
    ``preserves_shape = False`` and the composition leaves the shape unknown.

    Algebra (every expression returns a new ``LinOp``)::

        A @ B      composition:  (A @ B)(x) = A(B(x))
        A @ x      A.apply(x) when x is a tensor; so is A(x)
        A + B      sum                    A - B      difference
        c * A      scaling by a scalar (adjoint scales by conj(c));  -A
        A.H        adjoint: A.H.apply == A.applyT.  A.H.H is A itself.
    """

    in_shape = None
    out_shape = None
    preserves_shape = True      # False if shape-agnostic but shape-changing

    @abstractmethod
    def apply(self, x):
        """Forward action ``A x``."""

    @abstractmethod
    def applyT(self, y):
        """Adjoint action ``A^H y``."""

    def __call__(self, x):
        return self.apply(x)

    def __matmul__(self, other):
        if isinstance(other, LinOp):
            return LinOpComposition(self, other)
        return self.apply(other)

    def __add__(self, other):
        if isinstance(other, LinOp):
            return LinOpSum(self, other)
        raise TypeError("only a LinOp can be added to a LinOp")

    def __radd__(self, other):
        if isinstance(other, numbers.Number) and other == 0:
            return self                      # lets sum([A, B, C]) work
        raise TypeError("only a LinOp can be added to a LinOp")

    def __sub__(self, other):
        if isinstance(other, LinOp):
            return LinOpSum(self, LinOpScalarMul(other, -1))
        raise TypeError("only a LinOp can be subtracted from a LinOp")

    def __rsub__(self, other):
        raise TypeError("only a LinOp can be subtracted from a LinOp")

    def __neg__(self):
        return LinOpScalarMul(self, -1)

    def __mul__(self, other):
        if isinstance(other, LinOp):
            raise TypeError("use @ to compose operators; * is for scalars")
        if isinstance(other, torch.Tensor) and other.numel() != 1:
            raise TypeError("* is for scalars; use LinOpMul for element-wise coefficients")
        return LinOpScalarMul(self, other)

    __rmul__ = __mul__

    @property
    def H(self):
        """The adjoint operator."""
        return LinOpAdjoint(self)

    def __repr__(self):
        return f"{type(self).__name__}(in_shape={self.in_shape}, out_shape={self.out_shape})"


def _check_same(a, b, what):
    if a is not None and b is not None and tuple(a) != tuple(b):
        raise ValueError(f"incompatible {what}: {tuple(a)} vs {tuple(b)}")


class LinOpComposition(LinOp):
    """``(A @ B)(x) = A(B(x))``; adjoint ``B^H A^H``."""

    def __init__(self, A: LinOp, B: LinOp):
        _check_same(B.out_shape, A.in_shape, "shapes in A @ B (B.out_shape vs A.in_shape)")
        self.A, self.B = A, B
        self.in_shape = B.in_shape if B.in_shape is not None else (
            A.in_shape if B.preserves_shape else None)
        self.out_shape = A.out_shape if A.out_shape is not None else (
            B.out_shape if A.preserves_shape else None)
        self.preserves_shape = A.preserves_shape and B.preserves_shape

    def apply(self, x):
        return self.A.apply(self.B.apply(x))

    def applyT(self, y):
        return self.B.applyT(self.A.applyT(y))


class LinOpSum(LinOp):
    """``(A + B)(x) = A(x) + B(x)``; adjoint ``A^H + B^H``."""

    def __init__(self, A: LinOp, B: LinOp):
        _check_same(A.in_shape, B.in_shape, "input shapes in A + B")
        _check_same(A.out_shape, B.out_shape, "output shapes in A + B")
        self.A, self.B = A, B
        self.in_shape = A.in_shape if A.in_shape is not None else B.in_shape
        self.out_shape = A.out_shape if A.out_shape is not None else B.out_shape
        self.preserves_shape = A.preserves_shape and B.preserves_shape

    def apply(self, x):
        return self.A.apply(x) + self.B.apply(x)

    def applyT(self, y):
        return self.A.applyT(y) + self.B.applyT(y)


class LinOpScalarMul(LinOp):
    """``(c A)(x) = c A(x)``; adjoint ``y -> A^H(conj(c) y)``.

    That is ``conj(c) A^H y`` whenever ``A^H`` is complex-linear.  The scalar
    goes in before ``A^H`` so that the adjoint is also right for the operators
    that are only real-linear (``LinOpReal``, ``LinOpImag`` and what is built
    on them), where the two differ for a complex ``c``.
    """

    def __init__(self, A: LinOp, scalar):
        self.A, self.scalar = A, scalar
        self.in_shape, self.out_shape = A.in_shape, A.out_shape
        self.preserves_shape = A.preserves_shape

    def apply(self, x):
        return self.A.apply(x) * self.scalar

    def applyT(self, y):
        c = self.scalar
        c = c.conj() if isinstance(c, torch.Tensor) else c.conjugate()
        return self.A.applyT(y * c)


class LinOpAdjoint(LinOp):
    """``A.H``: swaps ``apply`` and ``applyT``.  ``A.H.H`` is ``A`` itself."""

    def __init__(self, A: LinOp):
        self.A = A
        self.in_shape, self.out_shape = A.out_shape, A.in_shape
        self.preserves_shape = A.preserves_shape

    def apply(self, y):
        return self.A.applyT(y)

    def applyT(self, x):
        return self.A.apply(x)

    @property
    def H(self):
        return self.A
