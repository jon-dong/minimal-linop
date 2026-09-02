"""Roll, Crop, Patch, Flip, Grad, Downsample, Upsample."""
import pytest
import torch

from minimal_linop import (
    LinOpRoll, LinOpCrop, LinOpPatch, LinOpFlip, LinOpGrad,
    LinOpDownsample, LinOpUpsample, adjoint_error,
)

torch.manual_seed(0)
C64 = torch.complex64


class TestRoll:

    def test_values(self):
        x = torch.tensor([0.0, 1.0, 2.0])
        assert torch.equal(LinOpRoll(1).apply(x), torch.tensor([2.0, 0.0, 1.0]))
        assert torch.equal(LinOpRoll(-1).apply(x), torch.tensor([1.0, 2.0, 0.0]))
        assert torch.equal(LinOpRoll(1, pad_zeros=True).apply(x), torch.tensor([0.0, 0.0, 1.0]))
        assert torch.equal(LinOpRoll(1, pad_zeros=True).applyT(x), torch.tensor([1.0, 2.0, 0.0]))

    def test_2d_values(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(LinOpRoll((1, 1), dim=(-2, -1)).apply(x), torch.tensor([[4.0, 3.0], [2.0, 1.0]]))
        assert torch.equal(LinOpRoll((1, 1), dim=(-2, -1), pad_zeros=True).apply(x),
                           torch.tensor([[0.0, 0.0], [0.0, 1.0]]))

    @pytest.mark.parametrize("pad_zeros", [False, True])
    @pytest.mark.parametrize("shifts", [(2, 3), (-2, 3), (0, -5), (7, -9)])
    def test_adjoint(self, shifts, pad_zeros):
        A = LinOpRoll(shifts, dim=(-2, -1), pad_zeros=pad_zeros)
        assert adjoint_error(A, torch.randn(2, 6, 8), torch.randn(2, 6, 8)) < 1e-6

    def test_tensor_and_float_shifts(self):
        assert LinOpRoll(torch.tensor([1.6, -0.4]), dim=(-2, -1)).shifts == (2, 0)
        assert LinOpRoll(torch.tensor(3)).shifts == (3,)
        assert LinOpRoll([1, 2.0], dim=(-2, -1)).shifts == (1, 2)

    @pytest.mark.parametrize("pad_zeros", [False, True])
    def test_torch_func_transforms(self, pad_zeros):
        """Out-of-place masking keeps the operator usable under vmap / jacrev,
        even when the shifts are read from a tensor inside the traced function."""
        shift = torch.tensor([2, -1])

        def f(phase):
            op = LinOpRoll((shift[0], shift[1]), dim=(-2, -1), pad_zeros=pad_zeros)
            return op.apply(torch.exp(1j * phase.to(C64))).abs() ** 2

        jac = torch.func.jacrev(f)(0.2 * torch.randn(5, 5))
        assert jac.shape == (5, 5, 5, 5) and torch.isfinite(jac).all()
        A = LinOpRoll((1, -2), dim=(-2, -1), pad_zeros=pad_zeros)
        batch = torch.randn(4, 5, 5)
        assert torch.equal(torch.func.vmap(A.apply)(batch), A.apply(batch))

    def test_shift_dim_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            LinOpRoll((1, 2), dim=-1)

    @pytest.mark.parametrize("make", [lambda d: LinOpRoll(1, dim=d), lambda d: LinOpFlip(dim=d)])
    def test_positive_dim_rejected(self, make):
        """Operators act on trailing axes; a positive dim would eat a batch axis."""
        with pytest.raises(ValueError, match="trailing"):
            make(0)
        with pytest.raises(ValueError, match="trailing"):
            make((0, 1))


class TestCrop:

    def test_values(self):
        A = LinOpCrop((3, 3), (2, 2))
        x = torch.arange(9.0).reshape(3, 3)
        assert torch.equal(A.apply(x), torch.tensor([[0.0, 1.0], [3.0, 4.0]]))

    @pytest.mark.parametrize("in_s,out_s", [(8, 4), (7, 5), (6, 2), (5, 3), (4, 3), ((7, 8), (3, 6)), ((6, 6, 5), (4, 2, 3))])
    def test_shapes_projection_adjoint(self, in_s, out_s):
        A = LinOpCrop(in_s, out_s)
        x = torch.randn(*A.in_shape)
        y = torch.randn(*A.out_shape)
        assert A.apply(x).shape == A.out_shape and A.applyT(y).shape == A.in_shape
        assert torch.equal(A.apply(A.applyT(y)), y)          # crop(pad(y)) == y
        assert adjoint_error(A, x, y) < 1e-6

    @pytest.mark.parametrize("in_s,out_s", [(8, 4), (12, 6), (10, 4), ((8, 8), (4, 4)), ((6, 8), (4, 4))])
    def test_fourier_origin(self, in_s, out_s):
        A = LinOpCrop(in_s, out_s, fourier_origin=True)
        x, y = torch.randn(*A.in_shape, dtype=C64), torch.randn(*A.out_shape, dtype=C64)
        assert A.apply(x).shape == A.out_shape and A.applyT(y).shape == A.in_shape
        assert torch.equal(A.apply(A.applyT(y)), y)
        assert adjoint_error(A, x, y) < 1e-6

    def test_fourier_origin_keeps_low_frequencies(self):
        X = torch.fft.fft(torch.randn(16, dtype=C64))
        Y = LinOpCrop(16, 6, fourier_origin=True).apply(X)
        assert torch.equal(Y, torch.cat([X[:3], X[-3:]]))

    def test_batch(self):
        assert LinOpCrop((6, 6), (4, 4)).apply(torch.randn(3, 6, 6)).shape == (3, 4, 4)

    def test_too_large_output_raises(self):
        with pytest.raises(ValueError):
            LinOpCrop(4, 6)


class TestPatch:
    """LinOpPatch must equal LinOpCrop @ LinOpRoll, only faster."""

    @pytest.mark.parametrize("pad_zeros", [False, True])
    def test_equals_crop_roll_1d_all_shifts(self, pad_zeros):
        x = torch.randn(11)
        for s in range(-12, 13):
            ref = LinOpCrop(11, 5) @ LinOpRoll(s, pad_zeros=pad_zeros)
            A = LinOpPatch(11, 5, shifts=s, pad_zeros=pad_zeros)
            assert torch.equal(A.apply(x), ref.apply(x)), s
            y = torch.randn(5)
            assert torch.equal(A.applyT(y), ref.applyT(y)), s

    @pytest.mark.parametrize("pad_zeros", [False, True])
    @pytest.mark.parametrize("shifts", [(0, 0), (2, -3), (-4, 5), (8, 11), (-9, -12), (3, 0)])
    def test_equals_crop_roll_2d(self, shifts, pad_zeros):
        x = torch.randn(9, 12)
        ref = LinOpCrop((9, 12), (4, 6)) @ LinOpRoll(shifts, dim=(-2, -1), pad_zeros=pad_zeros)
        A = LinOpPatch((9, 12), (4, 6), shifts=shifts, pad_zeros=pad_zeros)
        assert torch.equal(A.apply(x), ref.apply(x))
        y = torch.randn(4, 6)
        assert torch.equal(A.applyT(y), ref.applyT(y))

    @pytest.mark.parametrize("shifts", [(0, 0), (3, -2), (-5, 4), (12, 10)])
    def test_equals_crop_roll_fourier_origin(self, shifts):
        x = torch.randn(12, 10, dtype=C64)
        ref = LinOpCrop((12, 10), (6, 4), fourier_origin=True) @ LinOpRoll(shifts, dim=(-2, -1))
        A = LinOpPatch((12, 10), (6, 4), shifts=shifts, fourier_origin=True)
        assert torch.equal(A.apply(x), ref.apply(x))
        y = torch.randn(6, 4, dtype=C64)
        assert torch.equal(A.applyT(y), ref.applyT(y))

    @pytest.mark.parametrize("pad_zeros", [False, True])
    def test_adjoint_batch_and_vmap(self, pad_zeros):
        A = LinOpPatch((9, 12), (4, 6), shifts=(2, -3), pad_zeros=pad_zeros)
        x, y = torch.randn(3, 9, 12), torch.randn(3, 4, 6)
        assert A.apply(x).shape == (3, 4, 6) and A.applyT(y).shape == (3, 9, 12)
        assert adjoint_error(A, x, y) < 1e-6
        assert torch.equal(torch.func.vmap(A.apply)(x), A.apply(x))
        assert torch.equal(torch.func.vmap(A.applyT)(y), A.applyT(y))


class TestFlip:

    def test_values_involution_self_adjoint(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        A = LinOpFlip()
        assert torch.equal(A.apply(x), torch.tensor([4.0, 3.0, 2.0, 1.0]))
        assert torch.equal(A.apply(A.apply(x)), x)
        assert torch.equal(A.apply(x), A.applyT(x))
        B = LinOpFlip(dim=(-2, -1))
        assert adjoint_error(B, torch.randn(4, 5), torch.randn(4, 5)) < 1e-6


class TestGrad:

    def test_is_not_shape_preserving(self):
        assert LinOpGrad(2).preserves_shape is False
        assert LinOpGrad(2).in_shape is None and LinOpGrad(2).out_shape is None

    def test_size_one_axis(self):
        A = LinOpGrad(2)
        x = torch.randn(1, 5)
        assert torch.equal(A.apply(x)[0], torch.zeros(1, 5))     # nothing to difference
        assert adjoint_error(A, x, torch.randn(2, 1, 5)) < 1e-6

    def test_shapes_and_values_2d(self):
        A = LinOpGrad()
        x = torch.arange(12.0).reshape(3, 4)
        g = A.apply(x)
        assert g.shape == (2, 3, 4)
        assert torch.equal(g[0, :-1, :], 4.0 * torch.ones(2, 4)) and torch.equal(g[0, -1], torch.zeros(4))
        assert torch.equal(g[1, :, :-1], torch.ones(3, 3)) and torch.equal(g[1, :, -1], torch.zeros(3))
        assert A.apply(torch.randn(5, 3, 4)).shape == (5, 2, 3, 4)
        assert torch.equal(A.apply(torch.ones(5, 6)), torch.zeros(2, 5, 6))

    @pytest.mark.parametrize("ndim,shape", [(1, (7,)), (2, (4, 5)), (3, (3, 4, 5))])
    def test_adjoint(self, ndim, shape):
        A = LinOpGrad(ndim)
        x, y = torch.randn(2, *shape), torch.randn(2, ndim, *shape)
        assert A.applyT(y).shape == (2, *shape)
        assert adjoint_error(A, x, y) < 1e-6

    def test_complex_and_vmap(self):
        A = LinOpGrad()
        x = torch.randn(3, 4, 5, dtype=C64)
        assert adjoint_error(A, x, torch.randn(3, 2, 4, 5, dtype=C64)) < 1e-6
        assert torch.equal(torch.func.vmap(A.apply)(x), A.apply(x))


class TestDownsampleUpsample:

    def test_downsample_values(self):
        x = torch.arange(16.0).reshape(4, 4)
        A = LinOpDownsample((4, 4))
        assert A.out_shape == (2, 2)
        assert torch.equal(A.apply(x), x[0::2, 0::2])
        xt = A.applyT(torch.ones(2, 2))
        assert torch.equal(xt[0::2, 0::2], torch.ones(2, 2)) and xt.sum() == 4

    def test_upsample_values(self):
        A = LinOpUpsample((2, 2))
        y = A.apply(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert torch.equal(y, torch.tensor([[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                                            [3.0, 3.0, 4.0, 4.0], [3.0, 3.0, 4.0, 4.0]]))
        assert torch.equal(A.applyT(torch.ones(4, 4)), 4.0 * torch.ones(2, 2))

    @pytest.mark.parametrize("factor", [2, 3])
    @pytest.mark.parametrize("shape", [(8,), (7, 6), (5, 4, 6)])
    def test_adjoints(self, shape, factor):
        D, U = LinOpDownsample(shape, factor), LinOpUpsample(shape, factor)
        x = torch.randn(2, *shape, dtype=C64)
        assert D.apply(x).shape == (2, *D.out_shape)
        assert adjoint_error(D, x, torch.randn(2, *D.out_shape, dtype=C64)) < 1e-6
        assert adjoint_error(U, x, torch.randn(2, *U.out_shape, dtype=C64)) < 1e-6
        assert D.applyT(D.apply(x)).shape == x.shape

    def test_downsample_of_upsample_is_identity(self):
        x = torch.randn(4, 5)
        U, D = LinOpUpsample((4, 5), 3), LinOpDownsample((12, 15), 3)
        assert torch.equal(D.apply(U.apply(x)), x)

    def test_vmap(self):
        D = LinOpDownsample((7, 6))
        y = torch.randn(3, 4, 3)
        assert torch.equal(torch.func.vmap(D.applyT)(y), D.applyT(y))
