"""adjoint_error, to_matrix and the public export list."""
import torch

import minimal_linop
from minimal_linop import LinOp, LinOpMatrix, LinOpFft, LinOpCrop, LinOpReal, adjoint_error, to_matrix

torch.manual_seed(0)
C64, C128 = torch.complex64, torch.complex128


class WrongAdjoint(LinOp):
    def apply(self, x):
        return 2.0 * x

    def applyT(self, y):
        return 3.0 * y


class TestAdjointError:

    def test_correct_adjoint_is_tiny(self):
        assert adjoint_error(LinOpMatrix(torch.randn(3, 4, dtype=C64)), torch.randn(4, dtype=C64)) < 1e-5

    def test_wrong_adjoint_is_detected(self):
        # With y = A x / 2 the discrepancy is |<x, x>| / (2 |x|^2) = 1/2, whatever
        # x is; taking y at random would make the error |cos(x, y)| / 2, which is
        # below any fixed threshold for a fair share of the draws.
        x = torch.randn(8)
        assert adjoint_error(WrongAdjoint(), x, x) > 0.1

    def test_mixed_real_complex(self):
        assert adjoint_error(LinOpReal(), torch.randn(8, dtype=C64), torch.randn(8)) < 1e-5


class TestToMatrix:

    def test_matrix_operator(self):
        M = torch.randn(3, 4, dtype=C64)
        assert torch.allclose(to_matrix(LinOpMatrix(M)), M, atol=1e-6)

    def test_fft_is_unitary(self):
        F = to_matrix(LinOpFft(), in_shape=8, dtype=C128)
        assert torch.allclose(F.conj().T @ F, torch.eye(8, dtype=C128), atol=1e-12)

    def test_adjoint_matrix_is_conjugate_transpose(self):
        A = LinOpCrop((4, 6), (2, 3))
        M, MH = to_matrix(A, dtype=C128), to_matrix(A.H, dtype=C128)
        assert M.shape == (6, 24) and torch.equal(MH, M.conj().T)


def test_every_public_operator_is_exported():
    missing = [n for n in dir(minimal_linop) if n.startswith("LinOp") and n not in minimal_linop.__all__]
    assert missing == []
