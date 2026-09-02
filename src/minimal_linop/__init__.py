"""minimal-linop: linear operators with exact adjoints in PyTorch.

Every operator implements ``apply(x) = A x`` and ``applyT(y) = A^H y`` and
composes with ``@``, ``+``, ``-``, ``*`` and ``.H``.  Operators act on the
trailing axes of a tensor and leave leading (batch) axes alone.
"""

from .base import LinOp, LinOpComposition, LinOpSum, LinOpScalarMul, LinOpAdjoint
from .elementary import (
    LinOpIdentity,
    LinOpMul,
    LinOpReal,
    LinOpImag,
    LinOpSumReduce,
    LinOpMatrix,
    LinOpFunction,
    LinOpCat,
)
from .fourier import LinOpFft, LinOpIfft, LinOpFftShift, LinOpZoomFft
from .spatial import (
    LinOpRoll,
    LinOpCrop,
    LinOpPatch,
    LinOpFlip,
    LinOpGrad,
    LinOpDownsample,
    LinOpUpsample,
)
from .testing import adjoint_error, to_matrix

__version__ = "0.1.0"

__all__ = [
    # base + combinators
    "LinOp", "LinOpComposition", "LinOpSum", "LinOpScalarMul", "LinOpAdjoint",
    # elementary
    "LinOpIdentity", "LinOpMul", "LinOpReal", "LinOpImag", "LinOpSumReduce",
    "LinOpMatrix", "LinOpFunction", "LinOpCat",
    # fourier
    "LinOpFft", "LinOpIfft", "LinOpFftShift", "LinOpZoomFft",
    # spatial
    "LinOpRoll", "LinOpCrop", "LinOpPatch", "LinOpFlip", "LinOpGrad",
    "LinOpDownsample", "LinOpUpsample",
    # helpers
    "adjoint_error", "to_matrix",
]
