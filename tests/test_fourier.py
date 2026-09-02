"""Fft, Ifft, FftShift and ZoomFft."""
import math

import pytest
import torch

from minimal_linop import LinOpFft, LinOpIfft, LinOpFftShift, LinOpZoomFft, adjoint_error, to_matrix

torch.manual_seed(0)
C64, C128 = torch.complex64, torch.complex128
NORMS = ["ortho", "backward", "forward", None]


class TestFft:

    def test_matches_torch(self):
        x = torch.randn(3, 8, 8, dtype=C64)
        assert torch.equal(LinOpFft().apply(x), torch.fft.fftn(x, dim=(-1,), norm="ortho"))
        assert torch.equal(LinOpFft(dim=(-2, -1)).apply(x), torch.fft.fft2(x, norm="ortho"))

    def test_ortho_is_unitary(self):
        x = torch.randn(16, dtype=C64)
        A = LinOpFft()
        assert torch.allclose(A.applyT(A.apply(x)), x, atol=1e-6)

    @pytest.mark.parametrize("norm", NORMS)
    def test_adjoint_every_norm(self, norm):
        A = LinOpFft(dim=(-2, -1), norm=norm)
        assert adjoint_error(A, torch.randn(4, 5, dtype=C128), torch.randn(4, 5, dtype=C128)) < 1e-12

    @pytest.mark.parametrize("norm", ["ortho", "backward", "forward"])
    def test_matrix_of_adjoint_is_conjugate_transpose(self, norm):
        A = LinOpFft(norm=norm)
        M = to_matrix(A, in_shape=6, dtype=C128)
        MH = to_matrix(A.H, in_shape=6, dtype=C128)
        assert torch.allclose(MH, M.conj().T, atol=1e-12)

    def test_invalid_norm_rejected(self):
        with pytest.raises(ValueError, match="norm"):
            LinOpFft(norm="orthogonal")

    def test_positive_dim_rejected(self):
        with pytest.raises(ValueError, match="trailing"):
            LinOpFft(dim=0)
        with pytest.raises(ValueError, match="trailing"):
            LinOpFftShift(dim=(0, 1))


class TestIfft:

    def test_is_adjoint_of_fft(self):
        x = torch.randn(8, dtype=C64)
        assert torch.allclose(LinOpIfft().apply(x), LinOpFft().applyT(x))
        assert torch.allclose(LinOpIfft().applyT(x), LinOpFft().apply(x))

    @pytest.mark.parametrize("norm", NORMS)
    def test_adjoint_every_norm(self, norm):
        A = LinOpIfft(norm=norm)
        assert adjoint_error(A, torch.randn(9, dtype=C128), torch.randn(9, dtype=C128)) < 1e-12


class TestFftShift:

    @pytest.mark.parametrize("shape", [(7,), (8,), (7, 8), (9, 9)])
    def test_roundtrip_and_adjoint(self, shape):
        A = LinOpFftShift(dim=tuple(range(-len(shape), 0)))
        x = torch.randn(*shape, dtype=C64)
        assert torch.equal(A.apply(x), torch.fft.fftshift(x, dim=A.dim))
        assert torch.equal(A.applyT(A.apply(x)), x)
        assert adjoint_error(A, x, torch.randn(*shape, dtype=C64)) < 1e-6

    def test_only_shifts_requested_axes(self):
        x = torch.randn(3, 8)
        out = LinOpFftShift().apply(x)
        assert torch.equal(out[1], torch.fft.fftshift(x[1]))


class TestZoomFft:
    minimal_fft = pytest.importorskip("minimal_fft")

    def test_defaults_are_the_fft(self):
        A, F = LinOpZoomFft((8, 8)), LinOpFft(dim=(-2, -1))
        x = torch.randn(2, 8, 8, dtype=C128)
        assert torch.allclose(A.apply(x), F.apply(x), atol=1e-12)
        assert torch.allclose(A.applyT(x), F.applyT(x), atol=1e-12)

    @pytest.mark.parametrize("center", [False, True, 3.5])
    @pytest.mark.parametrize("norm", ["ortho", "backward", "forward"])
    def test_adjoint_on_zoom_band(self, norm, center):
        A = LinOpZoomFft((9, 12), (5, 7), k_start=-0.4, k_end=0.9, norm=norm,
                         center=center, include_end=True)
        assert A.in_shape == (9, 12) and A.out_shape == (5, 7)
        x, y = torch.randn(9, 12, dtype=C128), torch.randn(5, 7, dtype=C128)
        assert A.apply(x).shape == (5, 7) and A.applyT(y).shape == (9, 12)
        assert adjoint_error(A, x, y) < 1e-12

    def test_1d_and_batch(self):
        A = LinOpZoomFft(16, 32, k_start=-math.pi / 2, k_end=math.pi / 2)
        x = torch.randn(3, 16, dtype=C64)
        assert A.apply(x).shape == (3, 32)
        assert adjoint_error(A, x, torch.randn(3, 32, dtype=C64)) < 1e-5

    def test_composes(self):
        from minimal_linop import LinOpMul
        pupil = LinOpMul(torch.randn(16, 16, dtype=C128))
        A = LinOpZoomFft((16, 16), (40, 40), k_start=-1.0, k_end=1.0, center=True) @ pupil
        assert adjoint_error(A, torch.randn(16, 16, dtype=C128), torch.randn(40, 40, dtype=C128)) < 1e-12
