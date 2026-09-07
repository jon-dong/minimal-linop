"""Fourier-domain operators: FFT, inverse FFT, fftshift and the zoomed FFT."""

import math

import torch

from ._utils import as_dims, as_shape
from .base import LinOp

__all__ = ["LinOpFft", "LinOpIfft", "LinOpFftShift", "LinOpZoomFft"]

# With W the unnormalised DFT matrix over the transformed axes and N the
# product of their lengths, torch's normalisations are
#     fft(norm="backward") = W        ifft(norm="backward") = W^H / N
#     fft(norm="ortho")    = W / sqrt(N)   ifft(norm="ortho") = W^H / sqrt(N)
#     fft(norm="forward")  = W / N    ifft(norm="forward")  = W^H
# so the adjoint of a transform is the opposite transform with the conjugate
# norm.  Only "ortho" is unitary (adjoint == inverse).
_CONJUGATE_NORM = {None: "forward", "backward": "forward", "ortho": "ortho", "forward": "backward"}


def _conjugate_norm(norm):
    try:
        return _CONJUGATE_NORM[norm]
    except (KeyError, TypeError):
        raise ValueError(
            f"invalid FFT norm {norm!r}; expected 'backward', 'ortho', 'forward' or None"
        ) from None


class LinOpFft(LinOp):
    """FFT along ``dim`` (default: last axis; ``dim=(-2, -1)`` for images).

    ``applyT`` is the Hermitian adjoint for every ``norm``; it equals the
    inverse FFT only for the default ``norm="ortho"``, where the FFT is unitary.
    """

    def __init__(self, dim=-1, norm="ortho"):
        self.dim, self.norm = as_dims(dim), norm
        self.adjoint_norm = _conjugate_norm(norm)

    def apply(self, x):
        return torch.fft.fftn(x, dim=self.dim, norm=self.norm)

    def applyT(self, y):
        return torch.fft.ifftn(y, dim=self.dim, norm=self.adjoint_norm)


class LinOpIfft(LinOpFft):
    """Inverse FFT along ``dim``; the adjoint is the forward FFT with the conjugate norm."""

    def apply(self, x):
        return torch.fft.ifftn(x, dim=self.dim, norm=self.norm)

    def applyT(self, y):
        return torch.fft.fftn(y, dim=self.dim, norm=self.adjoint_norm)


class LinOpFftShift(LinOp):
    """``torch.fft.fftshift`` along ``dim``; the adjoint is ``ifftshift``."""

    def __init__(self, dim=-1):
        self.dim = as_dims(dim)

    def apply(self, x):
        return torch.fft.fftshift(x, dim=self.dim)

    def applyT(self, y):
        return torch.fft.ifftshift(y, dim=self.dim)


class LinOpZoomFft(LinOp):
    """Zoomed FFT over the band ``[k_start, k_end]`` (radians per sample),
    acting on the last ``len(in_shape)`` axes.  Needs the ``minimal-zoom-fft``
    package (``pip install minimal-linop[fft]``).

    ``apply`` is ``minimal_zoom_fft.zoom_fft`` and ``applyT`` is
    ``minimal_zoom_fft.zoom_ifft`` with the conjugate norm, which is the exact
    adjoint on any band.  With the defaults this is ``LinOpFft`` with
    ``dim=(-len(in_shape), ..., -1)``.

    Parameters
    ----------
    in_shape : int or tuple of int
        Spatial shape of the input (trailing axes).
    out_shape : int or tuple of int, optional
        Number of band samples per axis.  Default: ``in_shape``.
    k_start, k_end, norm, center, include_end
        As in ``minimal_zoom_fft.zoom_fft`` (scalar or one value per axis).
    """

    def __init__(self, in_shape, out_shape=None, k_start=0.0, k_end=2 * math.pi,
                 norm="ortho", center=False, include_end=False):
        try:
            import minimal_zoom_fft
        except ImportError as e:
            raise ImportError(
                "LinOpZoomFft needs the minimal-zoom-fft package: pip install minimal-zoom-fft"
            ) from e
        self._fft = minimal_zoom_fft
        self.in_shape = as_shape(in_shape)
        self.out_shape = self.in_shape if out_shape is None else as_shape(out_shape)
        if len(self.out_shape) != len(self.in_shape):
            raise ValueError("in_shape and out_shape must have the same length")
        self.dim = tuple(range(-len(self.in_shape), 0))
        self.k_start, self.k_end, self.norm = k_start, k_end, norm
        self.center, self.include_end = center, include_end
        self.adjoint_norm = _conjugate_norm(norm)

    def apply(self, x):
        return self._fft.zoom_fft(
            x, self.out_shape, self.k_start, self.k_end, dim=self.dim, norm=self.norm,
            center=self.center, include_end=self.include_end)

    def applyT(self, y):
        return self._fft.zoom_ifft(
            y, self.in_shape, self.k_start, self.k_end, dim=self.dim,
            norm=self.adjoint_norm, center=self.center, include_end=self.include_end)
