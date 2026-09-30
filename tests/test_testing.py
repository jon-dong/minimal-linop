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


class GradSpy(LinOp):
    """2 I, noting at each call whether autograd is recording."""

    def __init__(self):
        self.recording = []

    def apply(self, x):
        self.recording.append(torch.is_grad_enabled())
        return 2.0 * x

    def applyT(self, y):
        self.recording.append(torch.is_grad_enabled())
        return 2.0 * y


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

    def test_runs_without_recording_a_graph(self):
        """The check is not differentiated: a tensor that requires grad, or an
        operator with learnable coefficients, is compared under no_grad."""
        spy, x = GradSpy(), torch.randn(8, requires_grad=True)
        assert adjoint_error(spy, x) < 1e-6
        assert spy.recording == [False, False] and torch.is_grad_enabled()
        probe = torch.randn(8, dtype=C64, requires_grad=True)
        assert adjoint_error(LinOpMul(probe), torch.randn(8, dtype=C64)) < 1e-5


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
        with pytest.raises(TypeError, match="n_iter"):
            operator_norm(LinOpIdentity(), torch.randn(4), n_iter=2.5)
        with pytest.raises(ValueError, match="x0"):
            operator_norm(LinOpIdentity(), torch.zeros(4))

    def test_runs_without_recording_a_graph(self):
        spy, x0 = GradSpy(), torch.randn(8, requires_grad=True)
        assert operator_norm(spy, x0, n_iter=3) == pytest.approx(2.0, rel=1e-6)
        assert spy.recording == [False] * 6 and torch.is_grad_enabled()


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
        import minimal_zoom_fft                            # noqa: F401
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


# --- integer arguments: sizes, factors, axes ---------------------------------

class Index:
    """An integer that is not a Python int, the way a NumPy integer is: it
    only has ``__index__``."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class TestIntegerArguments:
    """Whatever is an integer counts as one (a NumPy integer, a 0-d integer
    tensor); a float does not, even a whole one; a size is not negative."""

    def test_integers_that_are_not_python_ints_are_accepted(self):
        i = Index
        assert LinOpCrop(i(8), i(4)).out_shape == (4,)
        assert LinOpCrop((i(8), torch.tensor(6)), (4, i(3))).in_shape == (8, 6)
        assert LinOpPatch((i(8), 6), (i(4), 3), shifts=(1, 2)).out_shape == (4, 3)
        assert LinOpDownsample(i(9), i(2)).out_shape == (5,)
        assert LinOpUpsample(i(3), torch.tensor(2)).out_shape == (6,)
        assert LinOpSumReduce(i(-2), i(4)).size == 4
        assert LinOpGrad(i(2)).ndim == 2
        assert LinOpFft(dim=i(-1)).dim == (-1,) and LinOpFlip(dim=(i(-2), -1)).dim == (-2, -1)
        assert LinOpFunction(lambda x: x, lambda y: y, in_shape=i(8)).in_shape == (8,)
        assert to_matrix(LinOpFft(), in_shape=i(4)).shape == (4, 4)
        assert operator_norm(LinOpIdentity(), torch.randn(4), n_iter=i(3)) == pytest.approx(1.0)
        assert all(type(s) is int for s in LinOpCrop(i(8), torch.tensor(4)).out_shape)

    def test_numpy_integers_are_accepted(self):
        np = pytest.importorskip("numpy")
        assert LinOpCrop(np.int64(8), np.int64(4)).out_shape == (4,)
        assert LinOpCrop(np.array([8, 6]), np.array([4, 3])).in_shape == (8, 6)
        assert LinOpDownsample(np.int32(9), np.int64(2)).out_shape == (5,)
        assert LinOpFft(dim=np.int64(-1)).dim == (-1,)
        assert to_matrix(LinOpFft(), in_shape=np.int64(4)).shape == (4, 4)

    @pytest.mark.parametrize("make", [
        lambda: LinOpCrop(8.0, 4), lambda: LinOpCrop(8, 3.7), lambda: LinOpCrop((8, 8.0), (4, 4)),
        lambda: LinOpPatch(8, 4.0), lambda: LinOpDownsample(8, 2.5), lambda: LinOpUpsample(4, 2.0),
        lambda: LinOpSumReduce(-1, 3.5), lambda: LinOpGrad(2.5),
        lambda: LinOpFft(dim=-1.0), lambda: LinOpFft(dim=None), lambda: LinOpRoll(1, dim=(-2.0,)),
        lambda: LinOpFunction(lambda x: x, lambda y: y, in_shape=8.0),
        lambda: to_matrix(LinOpFft(), in_shape=4.0),
    ])
    def test_a_float_is_not_an_integer(self, make):
        with pytest.raises(TypeError, match="integer"):
            make()

    @pytest.mark.parametrize("make", [
        lambda: LinOpCrop(8, -2), lambda: LinOpCrop(-8, -8), lambda: LinOpPatch(8, -2),
        lambda: LinOpDownsample(-8), lambda: LinOpDownsample(8, -2), lambda: LinOpDownsample(8, 0),
        lambda: LinOpUpsample(-4), lambda: LinOpUpsample(4, 0), lambda: LinOpSumReduce(-1, -3),
    ])
    def test_a_size_is_not_negative_and_a_factor_is_at_least_one(self, make):
        with pytest.raises(ValueError):
            make()


def test_every_public_operator_is_exported():
    missing = [n for n in dir(minimal_linop) if n.startswith("LinOp") and n not in minimal_linop.__all__]
    assert missing == []


def test_the_catalogue_test_covers_every_exported_operator():
    covered = {name for name, *_ in CATALOGUE}
    exported = {n for n in minimal_linop.__all__ if n.startswith("LinOp") and n != "LinOp"}
    assert exported - covered <= {"LinOpZoomFft"}      # skipped without minimal-zoom-fft


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_operators_work_when_mps_is_the_default_device():
    """Index tables and masks follow the input, wherever torch creates
    tensors by default: on an MPS input and on a CPU one."""
    torch.set_default_device("mps")
    try:
        ops = [
            LinOpPatch((6, 8), (3, 4), shifts=(1, -2), pad_zeros=True),
            LinOpRoll((1, -2), dim=(-2, -1), pad_zeros=True),
            LinOpCrop((6, 8), (3, 4), fourier_origin=True),
            LinOpGrad(2),
            LinOpDownsample((6, 8), 3),
        ]
        for device in ("mps", "cpu"):
            x = torch.randn(2, 6, 8, dtype=C64, device=device)
            for op in ops:
                y = op.apply(x)
                assert y.device.type == device and op.applyT(y).device.type == device
                assert adjoint_error(op, x) < 1e-5
        assert to_matrix(LinOpFft(), in_shape=4).device.type == "mps"
    finally:
        torch.set_default_device("cpu")
