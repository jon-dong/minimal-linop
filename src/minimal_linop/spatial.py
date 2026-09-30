"""Spatial operators: shifts, convolutions, crops, patches, flips, finite
differences and integer-factor resampling.  All are N-D: they act on ``dim``
or on the last ``len(in_shape)`` axes."""

import torch

from ._utils import as_dims, as_index, as_ints, as_shape, pad_axis
from .base import LinOp

__all__ = [
    "LinOpRoll", "LinOpConv", "LinOpCrop", "patch_by_crop_and_roll", "LinOpPatch",
    "LinOpFlip", "LinOpGrad", "LinOpDownsample", "LinOpUpsample",
]


class LinOpRoll(LinOp):
    """Circular shift by ``shifts`` along ``dim``; the adjoint shifts back.

    With ``pad_zeros=True`` the samples that wrapped around are zeroed
    instead, i.e. a shift with zero boundary.  The masking is out-of-place,
    so the operator works under ``torch.func`` transforms.
    """

    def __init__(self, shifts, dim=-1, pad_zeros=False):
        self.shifts, self.dim = as_ints(shifts), as_dims(dim)
        if len(self.shifts) != len(self.dim):
            raise ValueError("shifts and dim must have the same length")
        self.pad_zeros = pad_zeros

    def _roll(self, x, shifts):
        x = torch.roll(x, shifts=shifts, dims=self.dim)
        if self.pad_zeros:
            for s, d in zip(shifts, self.dim):
                if s == 0:
                    continue
                n = x.shape[d]
                pos = torch.arange(n, device=x.device)
                keep = (pos >= s) if s > 0 else (pos < n + s)   # wrapped region -> False
                shape = [1] * x.ndim
                shape[d] = n
                x = torch.where(keep.reshape(shape), x, x.new_zeros(()))
        return x

    def apply(self, x):
        return self._roll(x, self.shifts)

    def applyT(self, y):
        return self._roll(y, tuple(-s for s in self.shifts))


class LinOpConv(LinOp):
    """Circular convolution with a fixed kernel ``h`` on the last ``h.ndim``
    axes, ``(A x)[n] = sum_m h[m] x[n - m]`` with indices taken modulo the
    axis lengths, computed as ``ifftn(fftn(x) * fftn(h))``.  The adjoint is
    the circular correlation with ``h``: convolution with ``conj(h[-m])``,
    whose transfer function is ``conj(fftn(h))``.

    The origin of the kernel is index 0, as for ``torch.roll`` and the FFT: a
    delta at index 0 is the identity and a delta at index 1 is ``LinOpRoll(1)``.
    A point-spread function stored with its centre in the middle of the array
    goes through ``torch.fft.ifftshift`` first.  The input's trailing shape
    must be ``h.shape``; the output is real when both ``x`` and ``h`` are.

    Kernel and input meet in their common precision before the transforms, as
    the two factors of a product do, so a float32 kernel on a float64 input
    gives the float64 convolution of the same numbers.  The transfer function
    is computed once, in the precision of the kernel; an input more precise
    than the kernel costs one more transform of the kernel per application.
    """

    def __init__(self, kernel: torch.Tensor):
        self.kernel = kernel
        self.in_shape = self.out_shape = tuple(kernel.shape)
        self.dim = tuple(range(-kernel.ndim, 0))
        self.transfer = torch.fft.fftn(kernel, dim=self.dim)

    def _in_common_precision(self, x):
        """The transfer function and ``x`` in the precision of the more
        precise of the kernel and ``x``, each staying real or complex as it
        is (both untouched when the precisions agree)."""
        real = torch.promote_types(self.kernel.real.dtype, x.real.dtype)
        kernel = self.kernel.to(torch.promote_types(self.kernel.dtype, real))
        transfer = self.transfer if kernel is self.kernel else torch.fft.fftn(kernel, dim=self.dim)
        return transfer, x.to(torch.promote_types(x.dtype, real))

    def _filter(self, x, transfer):
        y = torch.fft.ifftn(torch.fft.fftn(x, dim=self.dim) * transfer, dim=self.dim)
        return y if x.is_complex() or self.kernel.is_complex() else y.real

    def apply(self, x):
        transfer, x = self._in_common_precision(x)
        return self._filter(x, transfer)

    def applyT(self, y):
        transfer, y = self._in_common_precision(y)
        return self._filter(y, transfer.conj())


class LinOpCrop(LinOp):
    """Central crop from ``in_shape`` to ``out_shape`` on the last
    ``len(in_shape)`` axes; the adjoint zero-pads back.

    ``fourier_origin=True`` crops around the Fourier origin instead: it keeps
    the corner blocks of a DC-in-the-corner spectrum and discards the middle
    (high) frequencies.  It is exactly ``ifftshift . crop . fftshift``, the
    central crop taken with the origin moved to the middle first, which is
    where the ``(o + 1) // 2`` split of the kept block comes from.
    """

    def __init__(self, in_shape, out_shape, fourier_origin=False):
        self.in_shape, self.out_shape = as_shape(in_shape), as_shape(out_shape)
        if len(self.in_shape) != len(self.out_shape):
            raise ValueError("in_shape and out_shape must have the same length")
        if any(o > i for i, o in zip(self.in_shape, self.out_shape)):
            raise ValueError("out_shape must fit inside in_shape")
        self.ndim = len(self.in_shape)
        self.dim = tuple(range(-self.ndim, 0))
        self.fourier_origin = fourier_origin
        self._starts = tuple(i // 2 - o // 2 for i, o in zip(self.in_shape, self.out_shape))
        self._fc_starts = tuple((o + 1) // 2 for o in self.out_shape)   # positive-frequency block

    def apply(self, x):
        if not self.fourier_origin:
            idx = tuple(slice(s, s + o) for s, o in zip(self._starts, self.out_shape))
            return x[(...,) + idx]
        for d, lo, i, o in zip(self.dim, self._fc_starts, self.in_shape, self.out_shape):
            x = torch.cat([x.narrow(d, 0, lo), x.narrow(d, i - (o - lo), o - lo)], dim=d)
        return x

    def applyT(self, y):
        if not self.fourier_origin:
            for d, s, i, o in zip(self.dim, self._starts, self.in_shape, self.out_shape):
                y = pad_axis(y, d, s, i - s - o)
            return y
        for d, lo, i, o in zip(self.dim, self._fc_starts, self.in_shape, self.out_shape):
            gap = list(y.shape)
            gap[d] = i - o                       # the discarded middle, as zeros
            y = torch.cat([y.narrow(d, 0, lo), y.new_zeros(gap),
                           y.narrow(d, lo, o - lo)], dim=d)
        return y


def patch_by_crop_and_roll(in_shape, out_shape, shifts=None, pad_zeros=False, fourier_origin=False):
    """The shifted window as a composition: ``LinOpCrop @ LinOpRoll``.

    This is the *definition* of ``LinOpPatch`` and the readable version of
    it: roll the signal by ``shifts`` (zeroing what wrapped when
    ``pad_zeros``), then crop the centre (or the Fourier origin) to
    ``out_shape``.  It costs a full copy of the signal per application;
    ``LinOpPatch`` computes the same numbers touching only the window.
    """
    in_shape, out_shape = as_shape(in_shape), as_shape(out_shape)
    dim = tuple(range(-len(in_shape), 0))
    shifts = (0,) * len(in_shape) if shifts is None else shifts
    return LinOpCrop(in_shape, out_shape, fourier_origin) @ LinOpRoll(shifts, dim, pad_zeros)


class LinOpPatch(LinOp):
    """Extract a shifted ``out_shape`` window from an ``in_shape`` signal.

    Equal to ``patch_by_crop_and_roll`` with the same arguments, but touches
    only the window samples: ``apply`` gathers them, so it is O(patch)
    whatever the signal size, and ``applyT`` scatter-adds them into the
    O(signal) zero tensor it has to return, instead of rolling the whole
    signal twice.  This is the operator for ptychography-like models where a
    small probe scans a large object.  Both directions are out-of-place and
    work under ``torch.func``.
    """

    def __init__(self, in_shape, out_shape, shifts=None, pad_zeros=False, fourier_origin=False):
        self.in_shape, self.out_shape = as_shape(in_shape), as_shape(out_shape)
        self.ndim = len(self.in_shape)
        if len(self.out_shape) != self.ndim:
            raise ValueError("in_shape and out_shape must have the same length")
        if any(o > i for i, o in zip(self.in_shape, self.out_shape)):
            raise ValueError("out_shape must fit inside in_shape")
        self.shifts = (0,) * self.ndim if shifts is None else as_ints(shifts)
        if len(self.shifts) != self.ndim:
            raise ValueError("shifts must have one entry per axis of in_shape")
        self.pad_zeros, self.fourier_origin = pad_zeros, fourier_origin

        # Per axis, the source index of each window sample: the indices the
        # crop keeps, walked back through the roll (rolled[i] = x[i - s]).
        # With pad_zeros a source outside the signal is masked out instead
        # of wrapping around.
        idxs, masks = [], []
        for n, o, s in zip(self.in_shape, self.out_shape, self.shifts):
            if fourier_origin:
                lo = (o + 1) // 2
                kept = torch.cat([torch.arange(lo), torch.arange(n - (o - lo), n)])
            else:
                start = n // 2 - o // 2
                kept = torch.arange(start, start + o)
            source = kept - s
            masks.append((source >= 0) & (source < n))
            idxs.append(source.clamp(0, n - 1) if pad_zeros else torch.remainder(source, n))
        # Open grid: axis k's indices are shaped (o_k, 1, ..., 1) so that the
        # tuple broadcasts to out_shape in x[..., i0, i1, ...].
        self._idxs = tuple(t.reshape((-1,) + (1,) * (self.ndim - 1 - k)) for k, t in enumerate(idxs))
        self._mask = None
        if pad_zeros:
            mask = torch.ones(self.out_shape, dtype=torch.bool)
            for k, m in enumerate(masks):
                mask = mask & m.reshape((-1,) + (1,) * (self.ndim - 1 - k))
            self._mask = mask
        self._on_device = {}

    def _tables(self, device):
        """The gather indices and the mask, moved to ``device`` once."""
        if device not in self._on_device:
            self._on_device[device] = (
                tuple(i.to(device) for i in self._idxs),
                None if self._mask is None else self._mask.to(device))
        return self._on_device[device]

    def apply(self, x):
        idxs, mask = self._tables(x.device)
        y = x[(...,) + idxs]
        return y if mask is None else torch.where(mask, y, y.new_zeros(()))

    def applyT(self, y):
        idxs, mask = self._tables(y.device)
        if mask is not None:
            y = torch.where(mask, y, y.new_zeros(()))
        # index_put addresses leading axes, so the window axes go first and
        # the batch axes last, and back again afterwards.
        window, front = tuple(range(-self.ndim, 0)), tuple(range(self.ndim))
        y = y.movedim(window, front)
        z = y.new_zeros(self.in_shape + y.shape[self.ndim:])
        z = z.index_put(idxs, y, accumulate=True)
        return z.movedim(front, window)


class LinOpFlip(LinOp):
    """Reverse the axes in ``dim``.  Self-adjoint."""

    def __init__(self, dim=-1):
        self.dim = as_dims(dim)

    def apply(self, x):
        return torch.flip(x, dims=self.dim)

    def applyT(self, y):
        return torch.flip(y, dims=self.dim)


class LinOpGrad(LinOp):
    """Forward-difference gradient on the last ``ndim`` axes, with zero
    boundary (the last difference along each axis is 0).

    ``apply``:  ``(..., *spatial) -> (..., ndim, *spatial)``, channel ``k``
    holding the difference along spatial axis ``k``.
    ``applyT``: the negative divergence, exact adjoint of the above.
    Used for total-variation regularisation.

    Shape-agnostic but *not* shape-preserving (it adds the channel axis), so
    ``preserves_shape`` is False and a composition with it leaves the shape it
    cannot know undeclared rather than guessing.
    """

    preserves_shape = False

    def __init__(self, ndim=2):
        self.ndim = as_index(ndim, "ndim")

    def apply(self, x):
        grads = []
        for d in range(-self.ndim, 0):
            n = x.shape[d]
            g = x.narrow(d, 1, n - 1) - x.narrow(d, 0, n - 1)
            grads.append(pad_axis(g, d, 0, 1))
        return torch.stack(grads, dim=-self.ndim - 1)

    def applyT(self, y):
        out = 0
        for k, d in enumerate(range(-self.ndim, 0)):
            g = y.select(-self.ndim - 1, k)
            g = g.narrow(d, 0, g.shape[d] - 1)
            out = out + pad_axis(g, d, 1, 0) - pad_axis(g, d, 0, 1)
        return out


def _as_factor(factor) -> int:
    """A resampling factor as a Python int, at least 1."""
    factor = as_index(factor, "factor")
    if factor < 1:
        raise ValueError(f"factor must be at least 1; got {factor}")
    return factor


class LinOpDownsample(LinOp):
    """Keep every ``factor``-th sample along the last ``len(in_shape)`` axes
    (starting at index 0); the adjoint puts them back and fills with zeros."""

    def __init__(self, in_shape, factor=2):
        self.in_shape, self.factor = as_shape(in_shape), _as_factor(factor)
        self.out_shape = tuple(-(-n // self.factor) for n in self.in_shape)
        self.dim = tuple(range(-len(self.in_shape), 0))

    def apply(self, x):
        return x[(...,) + (slice(None, None, self.factor),) * len(self.dim)]

    def applyT(self, y):
        f = self.factor
        for d, n in zip(self.dim, self.in_shape):
            p = d % y.ndim
            y = y.unsqueeze(p + 1)                              # (..., m, 1, ...)
            y = pad_axis(y, p + 1, 0, f - 1).flatten(p, p + 1)  # (..., m * f, ...)
            y = y.narrow(p, 0, n)
        return y


class LinOpUpsample(LinOp):
    """Replicate every sample ``factor`` times along the last
    ``len(in_shape)`` axes (nearest-neighbour upsampling); the adjoint sums
    each ``factor``-block."""

    def __init__(self, in_shape, factor=2):
        self.in_shape, self.factor = as_shape(in_shape), _as_factor(factor)
        self.out_shape = tuple(n * self.factor for n in self.in_shape)
        self.dim = tuple(range(-len(self.in_shape), 0))

    def apply(self, x):
        for d in self.dim:
            x = x.repeat_interleave(self.factor, dim=d)
        return x

    def applyT(self, y):
        for d, n in zip(self.dim, self.in_shape):
            y = y.unflatten(d, (n, self.factor)).sum(dim=d)
        return y
