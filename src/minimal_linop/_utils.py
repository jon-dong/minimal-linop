"""Private helpers shared by the operator modules."""

import torch

_COMPLEX_DTYPE = {
    torch.float32: torch.complex64,
    torch.complex64: torch.complex64,
    torch.float64: torch.complex128,
    torch.complex128: torch.complex128,
}


def as_dims(dim) -> tuple:
    """``dim`` as a tuple of negative ints (int -> 1-tuple).

    Only negative axes are accepted: operators act on the trailing axes and
    leave the leading (batch) axes alone, which a positive axis, counted
    from the front, would silently break.
    """
    dims = (int(dim),) if isinstance(dim, int) else tuple(int(d) for d in dim)
    if any(d >= 0 for d in dims):
        raise ValueError(
            f"dim must be negative, so that the operator acts on trailing axes; got {dim!r}"
        )
    return dims


def as_shape(shape) -> tuple:
    """A shape as a tuple of ints (int -> 1-tuple)."""
    return (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)


def _to_int(v) -> int:
    if isinstance(v, torch.Tensor):
        return int(torch.round(v)) if v.is_floating_point() else int(v)
    return int(round(v))


def as_ints(values) -> tuple:
    """Shifts as a tuple of Python ints (scalar -> 1-tuple).

    Tensor entries are read with ``int()`` rather than through numpy so an
    operator can be built inside ``torch.func`` transforms (``vmap``,
    ``jacrev``), where tensors may have no storage.
    """
    if isinstance(values, torch.Tensor) and values.ndim == 0:
        return (_to_int(values),)
    if isinstance(values, (torch.Tensor, list, tuple)):
        return tuple(_to_int(v) for v in values)
    return (_to_int(values),)


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Complex counterpart of a real or complex dtype (single stays single)."""
    try:
        return _COMPLEX_DTYPE[dtype]
    except KeyError:
        raise TypeError(f"unsupported dtype {dtype}") from None


def pad_axis(x: torch.Tensor, dim: int, before: int, after: int) -> torch.Tensor:
    """Zero-pad one axis (out-of-place, so it works under ``torch.func``)."""
    dim = dim % x.ndim
    pad = [0, 0] * (x.ndim - 1 - dim) + [before, after]
    return torch.nn.functional.pad(x, pad)
