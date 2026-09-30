"""LinOp base class, operator algebra and combinators."""
import pytest
import torch

from minimal_linop import (
    LinOp, LinOpComposition, LinOpSum, LinOpScalarMul, LinOpAdjoint,
    LinOpMatrix, LinOpFft, LinOpIdentity, LinOpGrad, LinOpCrop, LinOpMul,
    LinOpReal, LinOpImag, adjoint_error, to_matrix,
)

torch.manual_seed(0)
C64, C128 = torch.complex64, torch.complex128


def cmat(m, n):
    return LinOpMatrix(torch.randn(m, n, dtype=C64))


class TestAbstractInterface:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LinOp()

    def test_missing_method_raises(self):
        class OnlyApply(LinOp):
            def apply(self, x):
                return x
        with pytest.raises(TypeError):
            OnlyApply()

    def test_call_is_apply(self):
        A = cmat(3, 3)
        x = torch.randn(3, dtype=C64)
        assert torch.equal(A(x), A.apply(x))
        assert torch.equal(A @ x, A.apply(x))

    def test_repr(self):
        assert repr(cmat(2, 3)) == "LinOpMatrix(in_shape=(3,), out_shape=(2,))"


class TestComposition:

    def test_apply_and_adjoint_order(self):
        A, B = cmat(2, 3), cmat(3, 4)
        x, y = torch.randn(4, dtype=C64), torch.randn(2, dtype=C64)
        AB = A @ B
        assert isinstance(AB, LinOpComposition)
        assert torch.allclose(AB.apply(x), A.apply(B.apply(x)))
        assert torch.allclose(AB.applyT(y), B.applyT(A.applyT(y)))
        assert adjoint_error(AB, x, y) < 1e-5

    def test_shapes(self):
        AB = cmat(2, 3) @ cmat(3, 4)
        assert AB.in_shape == (4,) and AB.out_shape == (2,)

    def test_shape_agnostic_operator_inherits_shapes(self):
        A = cmat(4, 4) @ LinOpFft()
        assert A.in_shape == (4,) and A.out_shape == (4,)
        B = LinOpFft() @ cmat(4, 4)
        assert B.in_shape == (4,) and B.out_shape == (4,)

    def test_shape_changing_agnostic_operator_leaves_the_shape_unknown(self):
        """LinOpGrad is shape-agnostic but adds a channel axis, so a
        composition must not claim the other operator's shape."""
        A = LinOpGrad(2) @ LinOpCrop((8, 8), (4, 4))
        assert A.in_shape == (4 * 2, 4 * 2) and A.out_shape is None
        assert not A.preserves_shape
        assert A.apply(torch.randn(8, 8)).shape == (2, 4, 4)
        B = LinOpMul(torch.randn(2, 4, 4)) @ A        # would have raised on (4, 4)
        assert B.in_shape == (8, 8) and B.out_shape == (2, 4, 4)
        assert adjoint_error(B, torch.randn(8, 8), torch.randn(2, 4, 4)) < 1e-6

    def test_incompatible_shapes_raise(self):
        with pytest.raises(ValueError, match="A @ B"):
            cmat(2, 3) @ cmat(5, 4)

    def test_nested_adjoint(self):
        A, B, C, D = (cmat(4, 4) for _ in range(4))
        x, y = torch.randn(4, dtype=C64), torch.randn(4, dtype=C64)
        assert adjoint_error(A @ B @ C @ D, x, y) < 1e-5
        assert adjoint_error((A + B) @ (C - D), x, y) < 1e-5


class TestAdjoint:

    def test_H_swaps_apply_and_applyT(self):
        A = cmat(2, 3)
        x, y = torch.randn(3, dtype=C64), torch.randn(2, dtype=C64)
        assert isinstance(A.H, LinOpAdjoint)
        assert torch.equal(A.H.apply(y), A.applyT(y))
        assert torch.equal(A.H.applyT(x), A.apply(x))
        assert A.H.in_shape == (2,) and A.H.out_shape == (3,)

    def test_no_T_alias(self):
        """`.T` would read as a transpose, which is not the adjoint of a
        complex operator; only `.H` exists."""
        A = cmat(2, 3)
        for op in (A, A.H, A @ A.H, 2.0 * A):
            with pytest.raises(AttributeError):
                op.T

    def test_double_adjoint_is_the_operator(self):
        A = cmat(2, 3)
        assert A.H.H is A

    def test_adjoint_of_composition_and_sum(self):
        A, B = cmat(4, 4), cmat(4, 4)
        x = torch.randn(4, dtype=C64)
        assert torch.allclose((A @ B).H.apply(x), (B.H @ A.H).apply(x), atol=1e-5)
        assert torch.allclose((A + B).H.apply(x), (A.H + B.H).apply(x), atol=1e-5)


class TestSumAndDifference:

    def test_sum(self):
        A, B = cmat(3, 3), cmat(3, 3)
        x, y = torch.randn(3, dtype=C64), torch.randn(3, dtype=C64)
        S = A + B
        assert isinstance(S, LinOpSum)
        assert torch.allclose(S.apply(x), A.apply(x) + B.apply(x))
        assert adjoint_error(S, x, y) < 1e-5

    def test_difference_has_the_right_sign(self):
        A, B = LinOpMatrix(5.0 * torch.eye(3)), LinOpMatrix(2.0 * torch.eye(3))
        x = torch.ones(3)
        assert torch.allclose((B - A).apply(x), -3.0 * x)
        assert adjoint_error(B - A, x) < 1e-6

    def test_negation(self):
        A = cmat(3, 3)
        x = torch.randn(3, dtype=C64)
        assert torch.allclose((-A).apply(x), -A.apply(x))

    def test_builtin_sum(self):
        ops = [cmat(3, 3) for _ in range(3)]
        x = torch.randn(3, dtype=C64)
        assert torch.allclose(sum(ops).apply(x), sum(op.apply(x) for op in ops), atol=1e-5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="input shapes"):
            cmat(4, 3) + cmat(4, 5)
        with pytest.raises(ValueError, match="output shapes"):
            cmat(3, 4) - cmat(5, 4)

    def test_shape_agnostic_is_compatible(self):
        S = cmat(4, 4) + LinOpFft()
        assert S.in_shape == (4,)
        x, y = torch.randn(4, dtype=C64), torch.randn(4, dtype=C64)
        assert adjoint_error(S, x, y) < 1e-5


class TestScalarMul:

    def test_real_scalar(self):
        A = cmat(3, 3)
        x = torch.randn(3, dtype=C64)
        assert isinstance(3.0 * A, LinOpScalarMul)
        assert torch.allclose((3.0 * A).apply(x), 3.0 * A.apply(x))
        assert torch.allclose((A * 3.0).apply(x), 3.0 * A.apply(x))

    def test_complex_scalar_adjoint_conjugates(self):
        c = 2.0 + 1j
        A = c * LinOpIdentity()
        y = torch.randn(4, dtype=C64)
        assert torch.allclose(A.applyT(y), (2.0 - 1j) * y)
        assert adjoint_error(A, torch.randn(4, dtype=C64), y) < 1e-5

    def test_tensor_scalar(self):
        A = cmat(4, 4) * torch.tensor(1.0 - 2.0j)
        x, y = torch.randn(4, dtype=C64), torch.randn(4, dtype=C64)
        assert adjoint_error(A, x, y) < 1e-5

    @pytest.mark.parametrize("c", [1j, 2.0 + 1j, -1.5, torch.tensor(0.5 - 2.0j, dtype=C128)])
    def test_scalar_times_a_real_linear_operator(self, c):
        """(c A)^H y = A^H (conj(c) y).  That is conj(c) A^H y only when A^H is
        complex-linear, which the adjoints of LinOpReal and LinOpImag are not."""
        x, y = torch.randn(3, 8, dtype=C128), torch.randn(3, 8, dtype=C128)
        F = LinOpFft()
        for A in (LinOpReal(), LinOpImag(), LinOpReal() @ F, F @ LinOpImag().H @ LinOpReal()):
            assert adjoint_error(c * A, x, y) < 1e-12
            assert adjoint_error(A * c, x, y) < 1e-12
        # A real domain: the adjoint of x -> c x, x real, is y -> Re(conj(c) y).
        x_real = torch.randn(3, 8, dtype=torch.float64)
        assert adjoint_error(c * LinOpReal().H, x_real, y) < 1e-12
        cbar = complex(c).conjugate()
        assert torch.allclose((c * LinOpReal()).applyT(y), (cbar * y).real.to(C128), atol=1e-12)
        assert torch.allclose((c * LinOpReal().H).applyT(y), (cbar * y).real, atol=1e-12)

    @pytest.mark.parametrize("c", [1j, 2.0 + 1j, -1.5, torch.tensor(0.5 - 2.0j, dtype=C128)])
    def test_scalar_times_a_complex_linear_operator(self, c):
        """There the adjoint is conj(c) A^H, on the dense matrices as well."""
        M = torch.randn(3, 4, dtype=C128)
        A = LinOpMatrix(M)
        x, y = torch.randn(2, 4, dtype=C128), torch.randn(2, 3, dtype=C128)
        cbar = complex(c).conjugate()
        assert adjoint_error(c * A, x, y) < 1e-12
        assert torch.allclose((c * A).applyT(y), cbar * A.applyT(y), atol=1e-12)
        assert torch.allclose(to_matrix((c * A).H, dtype=C128), cbar * M.conj().T, atol=1e-12)
        assert adjoint_error(c * (LinOpFft() @ A), x, y) < 1e-12


class TestErrors:

    def test_scalar_add_and_sub_raise(self):
        A = LinOpIdentity()
        for bad in (lambda: A + 3, lambda: 3 + A, lambda: A - 2.0, lambda: 2.0 - A):
            with pytest.raises(TypeError):
                bad()

    def test_linop_times_linop_raises(self):
        with pytest.raises(TypeError, match="@"):
            LinOpIdentity() * LinOpIdentity()

    def test_non_scalar_tensor_times_linop_raises(self):
        with pytest.raises(TypeError, match="LinOpMul"):
            LinOpIdentity() * torch.ones(4)
