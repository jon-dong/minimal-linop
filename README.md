# minimal-linop

Linear operators with exact adjoints in PyTorch, composable with `@`, `+`, `*` and `.H`. One dependency, batch-transparent, device-agnostic.

Most of computational imaging is `y = A x` for a linear `A` that is far too large to store: FFTs, masks, crops, shifts, their compositions. Reconstruction needs `A` and its adjoint `Aᴴ` (that is where every gradient comes from), and it needs them to be *exactly* adjoint or solvers drift. This library gives you a small catalogue of such operators, each with a tested adjoint, and the algebra to combine them into forward models.

## Install

```bash
pip install minimal-linop                                     # once published
pip install git+https://github.com/jon-dong/minimal-linop     # from GitHub
pip install "minimal-linop[fft]"                              # adds minimal-fft for LinOpZoomFft
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Quick start

```python
import torch
from minimal_linop import LinOpFft, LinOpMul, LinOpCrop, adjoint_error, operator_norm

# A coherent imaging model: crop a window from the object, multiply by a
# probe, take the 2-D FFT.  Every operator acts on the last two axes.
crop  = LinOpCrop(in_shape=(256, 256), out_shape=(64, 64))
probe = LinOpMul(torch.randn(64, 64, dtype=torch.complex64))
F     = LinOpFft(dim=(-2, -1))
A     = F @ probe @ crop                       # (A @ B)(x) = A(B(x))

x_true = torch.randn(8, 256, 256, dtype=torch.complex64)   # a batch of 8 objects
b = A @ x_true                                              # measurements, shape (8, 64, 64)

# The adjoint is what a gradient needs: for L(x) = ½‖A x − b‖², ∇L = Aᴴ(A x − b).
x = torch.zeros_like(x_true)
grad = A.H @ (A @ x - b)

# Check any operator with the dot-product test (should be ~1e-6 in single precision).
print(adjoint_error(A, x_true))

# The spectral norm, e.g. for a gradient step 1 / ‖A‖²: power iteration on AᴴA.
print(operator_norm(A, x_true))
```

## Algebra

Every expression below returns a new operator; nothing is materialised.

| Expression | Meaning |
|---|---|
| `A.apply(x)`, `A(x)`, `A @ x` | forward action `A x` |
| `A.applyT(y)`, `A.H @ y` | adjoint action `Aᴴ y` |
| `A @ B` | composition, `(A @ B)(x) = A(B(x))` |
| `A + B`, `A - B`, `-A` | sum and difference |
| `c * A` | scaling by a scalar; the adjoint scales by `conj(c)` |
| `sum([A, B, C])` | the sum of a list (`0 + A` returns `A`) |
| `A.H.H` | `A` itself |

`Aᴴ` is the Hermitian adjoint for the inner product `⟨u, v⟩ = Σ conj(u) v`. For a real operator that is the transpose; for a complex one it is not, which is why there is no `.T` alias.

Shapes: `in_shape` / `out_shape` describe the trailing axes an operator acts on; `None` means shape-agnostic (an FFT accepts any length) and is compatible with everything. Sums and compositions check declared shapes at construction. A composition fills a shape the shape-agnostic side leaves undeclared from the other side, which assumes that side preserves shape; an operator that does not says so with `preserves_shape = False` (`LinOpGrad`, `LinOpSumReduce`, a `LinOpCat` of several operators) and the composition leaves the shape unknown rather than guessing it wrong.

## Catalogue

| Operator | `apply` | Adjoint |
|---|---|---|
| `LinOpIdentity()` | `x` | `x` |
| `LinOpMul(c)` | `c * x` (diagonal; `c` may broadcast) | `conj(c) * y` |
| `LinOpReal()`, `LinOpImag()` | `Re x`, `Im x` (complex → real) | embed as real / imaginary part |
| `LinOpSumReduce(dim, size)` | sum along one axis | broadcast back |
| `LinOpMatrix(M)` | `M x` on the last axis | `Mᴴ y` |
| `LinOpFunction(f, fT)` | wrap two callables | |
| `LinOpCat([A_k])` | `cat([A_k x], dim=-1)` | `Σ A_kᴴ y_k` |
| `LinOpFft(dim, norm)`, `LinOpIfft` | `fftn` / `ifftn` (`norm="ortho"` by default, unitary) | the opposite transform with the conjugate norm |
| `LinOpFftShift(dim)` | `fftshift` | `ifftshift` |
| `LinOpZoomFft(in_shape, out_shape, k_start, k_end, ...)` | zoomed FFT on a band (needs `minimal-fft`) | `zoom_ifft` |
| `LinOpRoll(shifts, dim, pad_zeros)` | circular shift, optionally zeroing the wrap | shift back |
| `LinOpCrop(in_shape, out_shape, fourier_origin)` | central crop (or around the Fourier origin) | zero-pad |
| `LinOpPatch(in_shape, out_shape, shifts, ...)` | shifted window, `Crop @ Roll` gathering only the window | scatter-add into zeros |
| `LinOpFlip(dim)` | reverse axes | itself |
| `LinOpGrad(ndim)` | forward differences, `(..., *s) → (..., ndim, *s)` | negative divergence |
| `LinOpDownsample(in_shape, factor)` | keep every `factor`-th sample | zero-interleave |
| `LinOpUpsample(in_shape, factor)` | nearest-neighbour replication | block sum |

All operators act on trailing axes (`dim` defaults to the last axis; use `dim=(-2, -1)` or a 2-tuple `in_shape` for images), leave leading batch axes alone, work on CPU/CUDA/MPS, support autograd, and are out-of-place so they run under `torch.func.vmap` / `jacrev`.

## Write your own

Subclass `LinOp` and implement the two methods; the algebra comes for free.

```python
import torch
from minimal_linop import LinOp, adjoint_error

class Conv1d(LinOp):
    """Circular convolution with a kernel h."""
    def __init__(self, h):
        self.kernel = torch.fft.fft(h)
    def apply(self, x):
        return torch.fft.ifft(torch.fft.fft(x) * self.kernel)
    def applyT(self, y):
        return torch.fft.ifft(torch.fft.fft(y) * self.kernel.conj())

C = Conv1d(torch.randn(64, dtype=torch.complex64))
assert adjoint_error(C, torch.randn(64, dtype=torch.complex64)) < 1e-5
```

Or wrap two functions: `LinOpFunction(apply, applyT, in_shape, out_shape)`. Declare `in_shape` / `out_shape` when they are fixed so compositions can check them, and set `preserves_shape = False` if the operator leaves them undeclared but changes the shape. The name `H` is taken by the adjoint property, so do not use it for an attribute.

## Checking operators

- `adjoint_error(A, x, y=None)` returns `|Re⟨A x, y⟩ − Re⟨x, Aᴴ y⟩| / (‖A x‖ ‖y‖)`, with the denominator falling back to `‖x‖ ‖Aᴴ y‖` when `A x` or `y` vanishes. Every operator in this library is tested this way.
- `operator_norm(A, x0, n_iter=50)` estimates `‖A‖₂` by power iteration on `AᴴA`; `x0` fixes the shape, dtype and device. The estimate approaches `‖A‖₂` from below, so a step size `1/‖A‖²` taken from it deserves a margin.
- `to_matrix(A, in_shape=None, dtype=torch.complex64)` materialises the dense matrix for small problems (`to_matrix(A.H) == to_matrix(A).conj().T`).

## Conventions worth knowing

- Every `dim` is negative. Operators act on the trailing axes, so an axis counted from the front — which would consume a batch axis — is rejected at construction.
- `norm="ortho"` is the default for the FFT operators, so `Aᴴ = A⁻¹` for them. With `"backward"` or `"forward"` the adjoint is still exact, but it is the opposite transform with the *other* norm, not the inverse.
- `LinOpCrop` centres like `torch`: it keeps indices `in//2 - out//2` onward. `fourier_origin=True` keeps the low frequencies of a DC-in-the-corner spectrum.
- `LinOpMul` declares `in_shape = out_shape = c.shape`, unless `c` is a scalar or has a size-1 axis: it then broadcasts, the shape it acts on is not determined by `c`, and it declares none.
- `LinOpRoll` shifts given as tensors are rounded to the nearest integer (ties to even) and read with `int()`, so operators can be built inside `torch.func` transforms.
- `LinOpCat` splits its adjoint input according to the sub-operators' `out_shape`; if none is declared, call `apply` once first.

## Relation to other libraries

The same idea as `scipy.sparse.linalg.LinearOperator`, [PyLops](https://pylops.readthedocs.io), [GlobalBioIm](https://biomedical-imaging-group.github.io/GlobalBioIm/) and the `physics` classes of [deepinv](https://deepinv.github.io), reduced to what you need to write and verify forward models in PyTorch, and small enough to read in one sitting. Extracted from the `ciel` computational-imaging library.

## Tests

```bash
pip install -e ".[test]" && pip install minimal-fft   # or pip install -e ../minimal-fft
pytest
```

## License

MIT
