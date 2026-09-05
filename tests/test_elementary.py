"""Identity, Mul, Real, Imag, SumReduce, Matrix, Function, Cat."""
import pytest
import torch

from minimal_linop import (
    LinOp, LinOpIdentity, LinOpMul, LinOpReal, LinOpImag, LinOpSumReduce,
    LinOpMatrix, LinOpFunction, LinOpCat, LinOpFft, LinOpCrop, LinOpRoll,
    LinOpGrad, adjoint_error,
)

torch.manual_seed(0)
C64 = torch.complex64


class TestIdentity:
    def test_apply_and_adjoint(self):
        x = torch.randn(8)
        assert torch.equal(LinOpIdentity().apply(x), x)
        assert torch.equal(LinOpIdentity().applyT(x), x)


class TestMul:
    def test_values_and_shapes(self):
        c = torch.tensor([1.0, 2.0, 3.0])
        A = LinOpMul(c)
        assert A.in_shape == A.out_shape == (3,)
        assert torch.equal(A.apply(torch.ones(3)), c)

    def test_adjoint_conjugates(self):
        c = torch.randn(8, dtype=C64)
        A = LinOpMul(c)
        assert torch.equal(A.applyT(torch.ones(8, dtype=C64)), c.conj())
        assert adjoint_error(A, torch.randn(8, dtype=C64), torch.randn(8, dtype=C64)) < 1e-5

    def test_broadcasts_over_batch(self):
        A = LinOpMul(torch.randn(4, 4, dtype=C64))
        assert A.apply(torch.randn(3, 4, 4, dtype=C64)).shape == (3, 4, 4)

    def test_broadcasting_coefficients_declare_no_shape(self):
        """A size-1 axis broadcasts, so c.shape is not the shape acted on."""
        A = LinOpMul(torch.randn(1, 8))
        assert A.in_shape is None and A.out_shape is None
        assert A.apply(torch.randn(4, 8)).shape == (4, 8)
        B = A @ LinOpCrop((16, 16), (4, 8))          # would have raised on (1, 8)
        assert B.out_shape == (4, 8)
        assert adjoint_error(B, torch.randn(16, 16), torch.randn(4, 8)) < 1e-6
        assert LinOpMul(torch.tensor(2.0)).in_shape is None

    def test_scalar_free_shape_is_declared(self):
        assert LinOpMul(torch.randn(4, 6)).in_shape == (4, 6)


class TestRealImag:
    def test_real(self):
        A = LinOpReal()
        x = torch.tensor([1 + 2j, 3 + 4j], dtype=C64)
        assert torch.equal(A.apply(x), torch.tensor([1.0, 3.0]))
        y = A.applyT(torch.tensor([1.0, 2.0]))
        assert y.is_complex() and torch.equal(y.real, torch.tensor([1.0, 2.0])) and (y.imag == 0).all()
        assert adjoint_error(A, torch.randn(8, dtype=C64), torch.randn(8)) < 1e-5

    def test_imag(self):
        A = LinOpImag()
        x = torch.tensor([1 + 2j, 3 + 4j], dtype=C64)
        assert torch.equal(A.apply(x), torch.tensor([2.0, 4.0]))
        assert torch.equal(A.applyT(torch.tensor([1.0, 2.0])), torch.tensor([1j, 2j]))
        assert adjoint_error(A, torch.randn(8, dtype=C64), torch.randn(8)) < 1e-5


class TestSumReduce:
    def test_shapes_and_values(self):
        A = LinOpSumReduce(dim=-3, size=3)
        x = torch.ones(2, 3, 4, 5)
        assert torch.equal(A.apply(x), 3.0 * torch.ones(2, 1, 4, 5))
        y = A.applyT(torch.randn(2, 1, 4, 5))
        assert y.shape == (2, 3, 4, 5) and torch.equal(y[:, 0], y[:, 2])

    def test_adjoint(self):
        A = LinOpSumReduce(dim=-3, size=4)
        assert adjoint_error(A, torch.randn(3, 4, 5, 6), torch.randn(3, 1, 5, 6)) < 1e-5

    def test_positive_dim_rejected(self):
        """dim=1 would reduce a batch axis, and the operator would not be
        batch-transparent."""
        with pytest.raises(ValueError, match="trailing"):
            LinOpSumReduce(dim=1, size=3)

    def test_is_not_shape_preserving(self):
        A = LinOpSumReduce(dim=-3, size=4) @ LinOpCrop((4, 8, 8), (4, 4, 4))
        assert A.in_shape == (4, 8, 8) and A.out_shape is None


class TestMatrix:
    def test_values(self):
        M = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        A = LinOpMatrix(M)
        assert A.in_shape == (2,) and A.out_shape == (3,)
        assert torch.equal(A.apply(torch.tensor([1.0, 1.0])), M @ torch.tensor([1.0, 1.0]))
        assert torch.equal(A.applyT(torch.ones(3)), M.T @ torch.ones(3))

    def test_complex_adjoint_is_conjugate_transpose(self):
        M = torch.randn(3, 4, dtype=C64)
        y = torch.randn(3, dtype=C64)
        assert torch.allclose(LinOpMatrix(M).applyT(y), M.conj().T @ y, atol=1e-6)
        assert adjoint_error(LinOpMatrix(M), torch.randn(4, dtype=C64), y) < 1e-5

    def test_batch(self):
        A = LinOpMatrix(torch.randn(3, 4))
        assert A.apply(torch.randn(5, 2, 4)).shape == (5, 2, 3)


class TestFunction:
    def test_wraps_callables(self):
        F = LinOpFunction(lambda x: torch.fft.fft(x, norm="ortho"),
                          lambda y: torch.fft.ifft(y, norm="ortho"), in_shape=8, out_shape=8)
        assert F.in_shape == (8,)
        x, y = torch.randn(8, dtype=C64), torch.randn(8, dtype=C64)
        assert torch.equal(F.apply(x), torch.fft.fft(x, norm="ortho"))
        assert adjoint_error(F, x, y) < 1e-5
        assert isinstance(F, LinOp)


class TestCat:

    def test_shape_agnostic_sub_operators(self):
        A = LinOpCat([LinOpFft(), LinOpFft(), LinOpFft()])
        x = torch.randn(2, 16, dtype=C64)
        assert A.apply(x).shape == (2, 48)
        assert adjoint_error(A, x, torch.randn(2, 48, dtype=C64)) < 1e-5

    def test_unequal_widths_from_out_shape(self):
        A0, A1 = LinOpMatrix(torch.randn(4, 8, dtype=C64)), LinOpMatrix(torch.randn(6, 8, dtype=C64))
        A = LinOpCat([A0, A1])
        assert A.in_shape == (8,) and A.out_shape == (10,)
        x, y = torch.randn(8, dtype=C64), torch.randn(10, dtype=C64)
        assert torch.allclose(A.applyT(y), A0.applyT(y[:4]) + A1.applyT(y[4:]), atol=1e-6)
        assert adjoint_error(A, x, y) < 1e-5

    def test_2d_ptychography_like_chain(self):
        in_shape, probe = (16, 16), (8, 8)
        F, P, crop = LinOpFft(dim=(-2, -1)), LinOpMul(torch.randn(8, 8, dtype=C64)), LinOpCrop(in_shape, probe)
        ops = [F @ P @ crop @ LinOpRoll(s, dim=(-2, -1)) for s in [(0, 0), (2, 0), (0, 3)]]
        A = LinOpCat(ops)
        x = torch.randn(*in_shape, dtype=C64)
        assert A.out_shape == tuple(A.apply(x).shape) == (8, 24)
        assert adjoint_error(A, x, torch.randn(8, 24, dtype=C64)) < 1e-5

    def test_in_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="in_shape"):
            LinOpCat([LinOpMatrix(torch.randn(4, 8)), LinOpMatrix(torch.randn(4, 6))])

    def test_leading_axes_mismatch_raises(self):
        with pytest.raises(ValueError, match="concatenation axis"):
            LinOpCat([LinOpCrop((16, 16), (8, 4)), LinOpCrop((16, 16), (4, 4))])

    def test_mixed_declarations_leave_out_shape_unknown(self):
        A = LinOpCat([LinOpCrop((16, 16), (8, 8)), LinOpFft(dim=(-2, -1))])
        assert A.out_shape is None

    def test_declared_widths_fix_the_split(self):
        A = LinOpCat([LinOpCrop((8, 8), (4, 4)), LinOpCrop((8, 8), (4, 2))])
        assert A.out_shape == (4, 6)
        with pytest.raises(ValueError, match="split"):
            A.applyT(torch.randn(4, 7))
        assert adjoint_error(A, torch.randn(3, 8, 8), torch.randn(3, 4, 6)) < 1e-6

    def test_undeclared_shape_preserving_sub_operators_share_the_rest(self):
        """Crop declares 4 columns, the two FFTs declare nothing and preserve
        shape, so they get (12 - 4) / 2 = 4 columns each."""
        A = LinOpCat([LinOpCrop(4, 4), LinOpFft(), LinOpFft()])
        x = torch.randn(3, 4, dtype=C64)
        assert A.out_shape is None and A.apply(x).shape == (3, 12)
        assert adjoint_error(A, x, torch.randn(3, 12, dtype=C64)) < 1e-5
        with pytest.raises(ValueError, match="split"):
            A.applyT(torch.randn(3, 11, dtype=C64))

    def test_undeclared_shape_changing_sub_operator_raises(self):
        """LinOpGrad declares no out_shape and changes the shape: the adjoint
        input cannot be split, and apply() leaves no state behind that would
        allow it."""
        A = LinOpCat([LinOpGrad(2), LinOpGrad(2)])
        x = torch.randn(4, 5)
        assert A.out_shape is None and A.apply(x).shape == (2, 4, 10)
        with pytest.raises(ValueError, match="does not preserve shape"):
            A.applyT(torch.randn(2, 4, 10))
        # Declaring the widths through LinOpFunction fixes it.
        G = LinOpGrad(2)
        A = LinOpCat([LinOpFunction(G.apply, G.applyT, in_shape=(4, 5), out_shape=(2, 4, 5))] * 2)
        assert A.out_shape == (2, 4, 10)
        assert adjoint_error(A, x, torch.randn(2, 4, 10)) < 1e-6
