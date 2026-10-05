# minimal-linop

Linear operators with exact adjoints in PyTorch, composable with `@`, `+`, `*` and `.H`.

Imaging models are `y = A x` with an `A` too large to store: FFTs, masks, crops, convolutions and their compositions. Reconstruction needs `A` and its adjoint `Aᴴ`, exactly, or the solver drifts. This is a small catalogue of such operators, each with a tested adjoint, and the algebra to combine them into forward models.

## Install

```bash
pip install minimal-linop
pip install "minimal-linop[fft]"                              # adds minimal-zoom-fft for LinOpZoomFft
pip install git+https://github.com/jon-dong/minimal-linop     # the development version
```

Python ≥ 3.10, PyTorch ≥ 2.0.

## Quick start

```python
import torch
from minimal_linop import LinOpFft, LinOpMul, LinOpCrop, LinOpConv, adjoint_error, operator_norm

# A coherent imaging model: crop a window, multiply by a probe, 2-D FFT.
crop  = LinOpCrop(in_shape=(256, 256), out_shape=(64, 64))
probe = LinOpMul(torch.randn(64, 64, dtype=torch.complex64))
F     = LinOpFft(dim=(-2, -1))
A     = F @ probe @ crop                       # (A @ B)(x) = A(B(x))

x_true = torch.randn(8, 256, 256, dtype=torch.complex64)   # a batch of 8 objects
b = A @ x_true                                              # shape (8, 64, 64)

# The gradient of ½‖A x − b‖² is Aᴴ(A x − b).
x = torch.zeros_like(x_true)
grad = A.H @ (A @ x - b)

print(adjoint_error(A, x_true))   # dot-product test, < 1e-5 in single precision
print(operator_norm(A, x_true))   # ‖A‖₂ by power iteration, for a step 1 / ‖A‖²

# An incoherent model: blur by a PSF stored with its centre in the middle.
psf = torch.rand(256, 256)
blur = LinOpConv(torch.fft.ifftshift(psf / psf.sum()))
```

## Algebra

Every expression returns a new operator; nothing is materialised.

| Expression | Meaning |
|---|---|
| `A.apply(x)`, `A(x)`, `A @ x` | forward action `A x` |
| `A.applyT(y)`, `A.H @ y` | adjoint action `Aᴴ y` |
| `A @ B` | composition, `(A @ B)(x) = A(B(x))` |
| `A + B`, `A - B`, `-A` | sum and difference |
| `c * A` | scaling; the adjoint scales by `conj(c)` |
| `sum([A, B, C])` | the sum of a list |
| `A.H.H` | `A` itself |

`Aᴴ` is the Hermitian adjoint for `⟨u, v⟩ = Σ conj(u) v`; for a complex operator it is not the transpose, so there is no `.T`.

`in_shape` and `out_shape` are the trailing axes an operator acts on; `None` means any. Sums and compositions compare declared shapes at construction, not when applied, so the check on an input's shape is yours. An operator whose declared shapes get in the way of a composition can be wrapped as `LinOpFunction(A.apply, A.applyT)`, which declares nothing.

## Catalogue

Formulas are written along one axis, `n` indexing the output and `N` the axis length; N-D operators apply them on every axis they act on. Indices are modulo `N` where the formula says circular.

| Operator | `A x` | `Aᴴ y` |
|---|---|---|
| `LinOpIdentity()` | `x` | `y` |
| `LinOpMul(c)` | `c ⊙ x` (`c` may broadcast) | `conj(c) ⊙ y` |
| `LinOpReal()`, `LinOpImag()` | `Re x`, `Im x` (complex → real) | `y + 0i`, `i y` |
| `LinOpSumReduce(dim, size)` | `Σₙ x[n]` over one axis, kept with length 1 | `y` broadcast back to `size` |
| `LinOpMatrix(M)` | `M x` on the last axis | `Mᴴ y` |
| `LinOpFunction(f, fT)` | `f(x)` | `fT(y)` |
| `LinOpCat([A_k])` | `cat([A_k x], dim=-1)` | `Σₖ A_kᴴ yₖ`, `yₖ` the columns of `A_k x` |
| `LinOpFft(dim, norm)`, `LinOpIfft` | `fftn` / `ifftn` (`norm="ortho"` by default) | the opposite transform with the conjugate norm |
| `LinOpFftShift(dim)` | `fftshift` | `ifftshift` |
| `LinOpZoomFft(in_shape, out_shape, k_start, k_end, ...)` | `zoom_fft` on a band (needs `minimal-zoom-fft`) | `zoom_ifft` with the conjugate norm |
| `LinOpRoll(shifts, dim, pad_zeros)` | `x[n − s]`, circular; with `pad_zeros`, `0` outside `[0, N)` | `y[n + s]`, same rule |
| `LinOpConv(h)` | `Σₘ h[m] x[n − m]`, circular, on the last `h.ndim` axes | `Σₘ conj(h[m]) y[n + m]` |
| `LinOpCrop(in_shape, out_shape, fourier_origin)` | `x[c + n]`, `c = in//2 − out//2`; with `fourier_origin`, the first `⌈out/2⌉` and the last `⌊out/2⌋` samples | zero-pad back where it was taken from |
| `LinOpPatch(in_shape, out_shape, shifts, pad_zeros, fourier_origin)` | `Crop(Roll(x))`, defined by `patch_by_crop_and_roll` | scatter-add of `y` into zeros |
| `LinOpFlip(dim)` | `x[N − 1 − n]` | itself |
| `LinOpGrad(ndim)` | `(∇x)ₖ[n] = x[n + eₖ] − x[n]`, `0` at the last index; `(..., *s) → (..., ndim, *s)` | `Σₖ (yₖ[n − eₖ] − yₖ[n])`, `yₖ = 0` at the last index and outside the grid |
| `LinOpDownsample(in_shape, factor)` | `x[f n]` | `y[n / f]` where `f` divides `n`, `0` elsewhere |
| `LinOpUpsample(in_shape, factor)` | `x[⌊n / f⌋]` | `Σⱼ y[f n + j]`, `j < f` |

All operators act on trailing axes (`dim=(-2, -1)` or a 2-tuple `in_shape` for images), leave batch axes alone, run on CPU, CUDA or MPS, support autograd and are out-of-place, so they work under `torch.func`.

## Write your own

Subclass `LinOp` and implement the two methods; the algebra comes for free.

```python
import torch
from minimal_linop import LinOp, adjoint_error

class CumSum(LinOp):
    """Running sum along the last axis; the adjoint is the running sum from the end."""
    def apply(self, x):
        return torch.cumsum(x, dim=-1)
    def applyT(self, y):
        return torch.cumsum(y.flip(-1), dim=-1).flip(-1)

S = CumSum()
assert adjoint_error(S, torch.randn(64, dtype=torch.complex64)) < 1e-5
```

Or wrap two functions: `LinOpFunction(apply, applyT, in_shape, out_shape)`.

## Checking operators

- `adjoint_error(A, x, y=None)`: the dot-product test, `|Re⟨A x, y⟩ − Re⟨x, Aᴴ y⟩| / (‖A x‖ ‖y‖)`.
- `operator_norm(A, x0, n_iter=50)`: `‖A‖₂` by power iteration on `AᴴA`, from below, so give a step `1/‖A‖²` a margin.
- `to_matrix(A, in_shape=None, dtype=torch.complex64, device=None)`: the dense matrix, for small problems.

## Conventions

- Every `dim` is negative; an axis counted from the front is refused.
- The FFT operators default to `norm="ortho"`, so `Aᴴ = A⁻¹`. With another norm the adjoint is the opposite transform with the other norm, not the inverse.
- `LinOpCrop` keeps indices `in//2 − out//2` onward, like `torch`; `fourier_origin=True` keeps the low frequencies of a DC-in-the-corner spectrum.
- `LinOpConv` puts the origin of the kernel at index 0, like the FFT; a PSF centred in the array goes through `ifftshift` first.
- An operator built from a tensor lives on that tensor's device, there is no `.to()`. Output dtype follows the input, promoted as the corresponding product would be.
- `LinOpIdentity`, the central `LinOpCrop`, `LinOpDownsample`, `LinOpReal` and `LinOpImag` return views; clone before modifying in place.
- `LinOpPatch` is defined by `patch_by_crop_and_roll`, written first in the source; the tests hold the fast version to it.

Same idea as `scipy.sparse.linalg.LinearOperator`, [PyLops](https://pylops.readthedocs.io), [GlobalBioIm](https://biomedical-imaging-group.github.io/GlobalBioIm/) and the `physics` classes of [deepinv](https://deepinv.github.io), reduced to what forward models in PyTorch need. The operators come from the `ciel` library at EPFL.

## Tutorials

Three notebooks in [`notebooks/`](notebooks/), after `pip install -e ".[notebooks]"`:

1. [Why](notebooks/01_why_linear_operators.ipynb): a forward model by composition, its gradient, and what a sloppy adjoint does to a solver.
2. [What it computes](notebooks/02_what_it_computes.ipynb): every identity, adjoint and convention above, checked against brute force in float64.
3. [Benchmark](notebooks/03_benchmark.ipynb): overhead against hand-written torch, batching, precision, when not to use it.

## Tests

```bash
pip install -e ".[test]" && pip install minimal-zoom-fft   # or pip install -e ../minimal-zoom-fft
pytest
```

Every adjoint is checked with the dot-product test and, for the complex-linear operators, against the dense matrix; `LinOpConv` against a double sum, `LinOpPatch` against its definition. The README's Python blocks run as a test.

## License

MIT

## Manifest

- Purpose: linear operators with exact adjoints, and the algebra to compose them, in PyTorch.
- Dependencies: `torch`. Optional: `minimal-zoom-fft` for `LinOpZoomFft`.
- Size: about 1100 lines of implementation in 7 modules, 29 public names (the base class, 24 operators, the patch definition and 3 helpers); about 1350 lines of tests; 3 tutorial notebooks.
- Origin: the `ciel` computational-imaging library, EPFL.
- Provenance: written with Claude (Anthropic) from a brief; read and checked in full by Jonathan Dong.
- Version: 0.1.0, MIT.
