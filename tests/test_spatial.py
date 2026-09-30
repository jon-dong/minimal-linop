"""Roll, Conv, Crop, Patch, Flip, Grad, Downsample, Upsample."""
import pytest
import torch

from minimal_linop import (
    LinOpRoll, LinOpConv, LinOpCrop, patch_by_crop_and_roll, LinOpPatch, LinOpFlip,
    LinOpGrad, LinOpDownsample, LinOpUpsample, LinOpFftShift, adjoint_error, to_matrix,
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


def circular_convolution_2d(h, x):
    """Brute force: y[i, j] = sum_{k, l} h[k, l] x[(i - k) % m, (j - l) % n]."""
    m, n = h.shape
    y = torch.zeros_like(x)
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    y[i, j] += h[k, l] * x[(i - k) % m, (j - l) % n]
    return y


class TestConv:

    def test_against_brute_force_2d(self):
        h, x = torch.randn(5, 7, dtype=torch.complex128), torch.randn(5, 7, dtype=torch.complex128)
        assert torch.allclose(LinOpConv(h).apply(x), circular_convolution_2d(h, x), atol=1e-12)

    def test_delta_kernels_are_rolls(self):
        x = torch.randn(3, 6, 8)
        for shift in [(0, 0), (0, 1), (2, 0), (5, 7)]:
            h = torch.zeros(6, 8)
            h[shift] = 1.0
            ref = LinOpRoll(shift, dim=(-2, -1)).apply(x)
            assert torch.allclose(LinOpConv(h).apply(x), ref, atol=1e-6), shift

    def test_adjoint_is_the_correlation(self):
        """A^H is convolution with conj(h[-m]), i.e. the flipped conjugate
        kernel with index 0 kept in place."""
        h = torch.randn(6, 8, dtype=C64)
        h_adj = torch.roll(torch.flip(h, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)).conj()
        y = torch.randn(2, 6, 8, dtype=C64)
        assert torch.allclose(LinOpConv(h).applyT(y), LinOpConv(h_adj).apply(y), atol=1e-5)
        assert adjoint_error(LinOpConv(h), torch.randn(2, 6, 8, dtype=C64), y) < 1e-5

    def test_real_in_real_out(self):
        h, x = torch.randn(4, 5), torch.randn(3, 4, 5)
        A = LinOpConv(h)
        assert A.apply(x).dtype == torch.float32 and A.applyT(x).dtype == torch.float32
        assert A.apply(x.to(C64)).dtype == C64
        assert LinOpConv(h.to(C64)).apply(x).dtype == C64
        assert adjoint_error(A, x, torch.randn(3, 4, 5)) < 1e-6

    @pytest.mark.parametrize("kernel_dtype,input_dtype,out_dtype", [
        (torch.float32, torch.float64, torch.float64),
        (torch.float64, torch.float32, torch.float64),
        (C64, torch.float64, torch.complex128),
        (torch.float32, torch.complex128, torch.complex128),
        (torch.float64, C64, torch.complex128),
    ])
    def test_mixed_precision_is_computed_in_the_common_precision(self, kernel_dtype, input_dtype, out_dtype):
        """A float32 kernel on a float64 input is the float64 convolution of
        the same numbers, as for LinOpMul and LinOpMatrix: the output has the
        promoted dtype and the accuracy that goes with it, in both directions."""
        h, x = torch.randn(5, 7, dtype=kernel_dtype), torch.randn(5, 7, dtype=input_dtype)
        A = LinOpConv(h)
        y = A.apply(x)
        assert y.dtype == out_dtype
        assert (y - circular_convolution_2d(h.to(out_dtype), x.to(out_dtype))).abs().max() < 1e-12
        h_adj = torch.roll(torch.flip(h, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1)).conj()
        z = A.applyT(x)
        assert z.dtype == out_dtype
        assert (z - circular_convolution_2d(h_adj.to(out_dtype), x.to(out_dtype))).abs().max() < 1e-12

    def test_shapes_and_matrix_is_circulant(self):
        h = torch.randn(6)
        A = LinOpConv(h)
        assert A.in_shape == A.out_shape == (6,)
        M = to_matrix(A, dtype=torch.float32)
        assert torch.allclose(M[:, 0], h, atol=1e-6)
        assert torch.allclose(M[:, 1], torch.roll(h, 1), atol=1e-6)

    def test_vmap_and_autograd(self):
        h = torch.randn(4, 5, dtype=C64)
        x = torch.randn(3, 4, 5, dtype=C64)
        A = LinOpConv(h)
        assert torch.allclose(torch.func.vmap(A.apply)(x), A.apply(x), atol=1e-6)
        x64 = torch.randn(4, 5, dtype=torch.complex128, requires_grad=True)
        assert torch.autograd.gradcheck(LinOpConv(h.to(torch.complex128)).apply, (x64,))


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

    @pytest.mark.parametrize("in_s,out_s", [(8, 4), (9, 5), (9, 4), (12, 3), ((11, 7), (6, 4)),
                                            ((12, 10), (3, 5))])
    def test_fourier_origin_is_ifftshift_crop_fftshift(self, in_s, out_s):
        """The identity the docstring states, in both directions."""
        A = LinOpCrop(in_s, out_s, fourier_origin=True)
        shift = LinOpFftShift(dim=A.dim)
        ref = shift.H @ LinOpCrop(in_s, out_s) @ shift
        x, y = torch.randn(2, *A.in_shape, dtype=C64), torch.randn(2, *A.out_shape, dtype=C64)
        assert torch.equal(A.apply(x), ref.apply(x))
        assert torch.equal(A.applyT(y), ref.applyT(y))

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
    """LinOpPatch must equal patch_by_crop_and_roll, only faster."""

    @pytest.mark.parametrize("pad_zeros", [False, True])
    @pytest.mark.parametrize("fourier_origin", [False, True])
    def test_equals_its_definition(self, pad_zeros, fourier_origin):
        kw = dict(shifts=(2, -3), pad_zeros=pad_zeros, fourier_origin=fourier_origin)
        A, ref = LinOpPatch((9, 12), (4, 6), **kw), patch_by_crop_and_roll((9, 12), (4, 6), **kw)
        x, y = torch.randn(3, 9, 12, dtype=C64), torch.randn(3, 4, 6, dtype=C64)
        assert torch.equal(A.apply(x), ref.apply(x))
        assert torch.equal(A.applyT(y), ref.applyT(y))
        assert ref.in_shape == (9, 12) and ref.out_shape == (4, 6)

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

    @pytest.mark.parametrize("kwargs", [
        dict(in_shape=(8, 8), out_shape=(16, 4)),                    # window larger than the signal
        dict(in_shape=(8, 8), out_shape=(4,)),                       # a different number of axes
        dict(in_shape=(8, 8), out_shape=(4, 4), shifts=(1, 2, 3)),   # one shift too many
    ])
    def test_refuses_what_its_definition_refuses(self, kwargs):
        """The twins agree on what they reject, not only on what they compute."""
        with pytest.raises(ValueError):
            LinOpPatch(**kwargs)
        with pytest.raises(ValueError):
            patch_by_crop_and_roll(**kwargs)

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

    def test_needs_at_least_one_axis(self):
        for ndim in (0, -1):
            with pytest.raises(ValueError, match="ndim"):
                LinOpGrad(ndim)

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
