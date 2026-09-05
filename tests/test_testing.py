"""adjoint_error, operator_norm, to_matrix and the public export list."""
import math

import pytest
import torch

import minimal_linop
from minimal_linop import (
    LinOp, LinOpIdentity, LinOpMul, LinOpReal, LinOpImag, LinOpSumReduce,
    LinOpMatrix, LinOpFunction, LinOpCat, LinOpFft, LinOpIfft, LinOpFftShift,
    LinOpRoll, LinOpConv, LinOpCrop, LinOpPatch, LinOpFlip, LinOpGrad, LinOpDownsample,
    LinOpUpsample, adjoint_error, operator_norm, to_matrix,
)

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

    def test_vacuous_check_is_zero_not_infinite(self):
        """x = 0 makes both sides vanish: no information, and no division by
        an underflowing scale."""
        A = LinOpMatrix(torch.randn(3, 3))
        assert adjoint_error(A, torch.zeros(3), torch.randn(3)) == 0.0

    def test_null_space_input_still_catches_a_wrong_adjoint(self):
        """A x = 0 but A^H y != 0: the error is normalised by |x| |A^H y|."""
        class Zero(LinOp):
            def apply(self, x):
                return torch.zeros_like(x)

            def applyT(self, y):
                return y                          # not the adjoint of 0

        e = adjoint_error(Zero(), torch.ones(8), torch.ones(8))
        assert 0.1 < e <= 1.0 + 1e-6              # bounded by Cauchy-Schwarz


class TestOperatorNorm:
    """The power-iteration estimate against the singular values of the dense
    matrix."""

    @pytest.mark.parametrize("name,op,shape,dtype", [
        ("matrix", LinOpMatrix(torch.randn(6, 5, dtype=C64)), (5,), C64),
        ("fft", LinOpFft(), (8,), C64),
        ("crop", LinOpCrop((8, 8), (4, 4)), (8, 8), C64),
        ("conv", LinOpConv(torch.randn(4, 6, dtype=C64)), (4, 6), C64),
        ("grad", LinOpGrad(2), (4, 4), torch.float64),
        ("upsample", LinOpUpsample((3, 4), 2), (3, 4), torch.float64),
        ("composition", LinOpFft(dim=(-2, -1)) @ LinOpMul(torch.randn(4, 4, dtype=C128))
                        @ LinOpCrop((6, 6), (4, 4)), (6, 6), C128),
    ])
    def test_matches_the_spectral_norm_of_the_dense_matrix(self, name, op, shape, dtype):
        M = to_matrix(op, in_shape=shape, dtype=dtype)
        exact = float(torch.linalg.matrix_norm(M.to(C128), 2))
        est = operator_norm(op, torch.randn(*shape, dtype=dtype), n_iter=100)
        assert est == pytest.approx(exact, rel=1e-4)

    def test_estimate_rises_from_below(self):
        A = LinOpMatrix(torch.randn(8, 8, dtype=C128))
        exact = float(torch.linalg.matrix_norm(to_matrix(A, dtype=C128), 2))
        x0 = torch.randn(8, dtype=C128)
        estimates = [operator_norm(A, x0, n_iter=k) for k in (1, 2, 5, 50)]
        assert all(a <= b * (1 + 1e-12) for a, b in zip(estimates, estimates[1:]))
        assert all(e <= exact * (1 + 1e-12) for e in estimates)
        assert estimates[-1] == pytest.approx(exact, rel=1e-6)

    def test_batch_axes_do_not_change_the_norm(self):
        A = LinOpMatrix(torch.randn(5, 5, dtype=C64))
        x0 = torch.randn(5, dtype=C64)
        assert operator_norm(A, x0) == pytest.approx(
            operator_norm(A, x0.expand(3, 5).clone()), rel=1e-5)

    def test_zero_operator(self):
        assert operator_norm(0.0 * LinOpIdentity(), torch.randn(4)) == 0.0

    def test_bad_arguments(self):
        with pytest.raises(ValueError, match="n_iter"):
            operator_norm(LinOpIdentity(), torch.randn(4), n_iter=0)
        with pytest.raises(ValueError, match="x0"):
            operator_norm(LinOpIdentity(), torch.zeros(4))


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

    def test_shape_changing_operator(self):
        M = to_matrix(LinOpGrad(1), in_shape=4, dtype=C128)
        assert M.shape == (4, 4)                       # (1, 4) outputs, 4 inputs
        assert torch.equal(M, to_matrix(LinOpGrad(1).H, in_shape=(1, 4), dtype=C128).conj().T)

    def test_missing_in_shape_is_named(self):
        with pytest.raises(ValueError, match="in_shape"):
            to_matrix(LinOpFft())


# --- every operator of the catalogue, checked with the dot-product test -----

def _catalogue():
    """One instance of every operator in the README catalogue, with the input
    and output vectors of its dot-product test."""
    z = lambda *s: torch.randn(*s, dtype=C128)
    items = [
        ("LinOpIdentity", LinOpIdentity(), z(3, 8), z(3, 8)),
        ("LinOpMul", LinOpMul(z(8)), z(3, 8), z(3, 8)),
        ("LinOpReal", LinOpReal(), z(3, 8), torch.randn(3, 8, dtype=torch.float64)),
        ("LinOpImag", LinOpImag(), z(3, 8), torch.randn(3, 8, dtype=torch.float64)),
        ("LinOpSumReduce", LinOpSumReduce(-2, 4), z(3, 4, 8), z(3, 1, 8)),
        ("LinOpMatrix", LinOpMatrix(z(5, 8)), z(3, 8), z(3, 5)),
        ("LinOpFunction", LinOpFunction(lambda x: torch.fft.fft(x, norm="ortho"),
                                        lambda y: torch.fft.ifft(y, norm="ortho")),
         z(3, 8), z(3, 8)),
        ("LinOpCat", LinOpCat([LinOpCrop(8, 4), LinOpCrop(8, 2)]), z(3, 8), z(3, 6)),
        ("LinOpFft", LinOpFft(dim=(-2, -1), norm="backward"), z(3, 4, 8), z(3, 4, 8)),
        ("LinOpIfft", LinOpIfft(norm="forward"), z(3, 8), z(3, 8)),
        ("LinOpFftShift", LinOpFftShift(dim=(-2, -1)), z(3, 5, 8), z(3, 5, 8)),
        ("LinOpRoll", LinOpRoll((2, -3), dim=(-2, -1), pad_zeros=True), z(3, 5, 8), z(3, 5, 8)),
        ("LinOpConv", LinOpConv(z(5, 8)), z(3, 5, 8), z(3, 5, 8)),
        ("LinOpCrop", LinOpCrop((5, 8), (3, 4), fourier_origin=True), z(3, 5, 8), z(3, 3, 4)),
        ("LinOpPatch", LinOpPatch((5, 8), (3, 4), shifts=(2, -3)), z(3, 5, 8), z(3, 3, 4)),
        ("LinOpFlip", LinOpFlip(dim=(-2, -1)), z(3, 5, 8), z(3, 5, 8)),
        ("LinOpGrad", LinOpGrad(2), z(3, 5, 8), z(3, 2, 5, 8)),
        ("LinOpDownsample", LinOpDownsample((5, 8), 3), z(3, 5, 8), z(3, 2, 3)),
        ("LinOpUpsample", LinOpUpsample((5, 8), 3), z(3, 5, 8), z(3, 15, 24)),
    ]
    try:                                              # optional [fft] extra
        import minimal_fft                            # noqa: F401
        from minimal_linop import LinOpZoomFft
        items.append(("LinOpZoomFft", LinOpZoomFft((5, 8), (4, 6), k_start=-0.4, k_end=0.9),
                      z(3, 5, 8), z(3, 4, 6)))
    except ImportError:                               # pragma: no cover
        pass
    A, B = LinOpMatrix(z(4, 4)), LinOpMatrix(z(4, 4))
    items += [
        ("LinOpComposition", A @ B, z(3, 4), z(3, 4)),
        ("LinOpSum", A + B, z(3, 4), z(3, 4)),
        ("LinOpScalarMul", (1.0 - 2.0j) * A, z(3, 4), z(3, 4)),
        ("LinOpAdjoint", A.H, z(3, 4), z(3, 4)),
    ]
    return items


CATALOGUE = _catalogue()


@pytest.mark.parametrize("name,op,x,y", CATALOGUE, ids=[c[0] for c in CATALOGUE])
def test_every_operator_passes_the_dot_product_test(name, op, x, y):
    assert adjoint_error(op, x, y) < 1e-12


@pytest.mark.parametrize("name,op,x,y", CATALOGUE, ids=[c[0] for c in CATALOGUE])
def test_every_operator_is_batch_transparent(name, op, x, y):
    """Leading axes are left alone: the batch result is the stack of the
    single results, and vmap agrees."""
    close = lambda a, b: torch.allclose(a, b, rtol=1e-12, atol=1e-12)
    assert close(op.apply(x)[0], op.apply(x[0]))
    assert close(op.applyT(y)[0], op.applyT(y[0]))
    assert close(torch.func.vmap(op.apply)(x), op.apply(x))


@pytest.mark.parametrize("name,op,x,y", CATALOGUE, ids=[c[0] for c in CATALOGUE])
def test_every_operator_declares_the_shapes_it_produces(name, op, x, y):
    if op.in_shape is not None:
        assert tuple(x.shape[-len(op.in_shape):]) == op.in_shape
    if op.out_shape is not None:
        assert tuple(op.apply(x).shape[-len(op.out_shape):]) == op.out_shape


def test_every_public_operator_is_exported():
    missing = [n for n in dir(minimal_linop) if n.startswith("LinOp") and n not in minimal_linop.__all__]
    assert missing == []


def test_the_catalogue_test_covers_every_exported_operator():
    covered = {name for name, *_ in CATALOGUE}
    exported = {n for n in minimal_linop.__all__ if n.startswith("LinOp") and n != "LinOp"}
    assert exported - covered <= {"LinOpZoomFft"}      # skipped without minimal-fft
