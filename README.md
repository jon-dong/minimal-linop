# minimal-linop

Linear operators with exact adjoints in PyTorch, composable with `@`, `+`, `*` and `.H`. One dependency, batch-transparent, device-agnostic.

Most of computational imaging is `y = A x` for a linear `A` that is far too large to store: FFTs, masks, crops, shifts, convolutions, their compositions. Reconstruction needs `A` and its adjoint `Aᴴ` (that is where every gradient comes from), and it needs them to be *exactly* adjoint or solvers drift. This library gives you a small catalogue of such operators, each with a tested adjoint, and the algebra to combine them into forward models.

## Install

```bash
pip install minimal-linop                                     # once published
pip install git+https://github.com/jon-dong/minimal-linop     # from GitHub
pip install "minimal-linop[fft]"                              # adds minimal-zoom-fft for LinOpZoomFft
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Quick start

```python
import torch
from minimal_linop import LinOpFft, LinOpMul, LinOpCrop, LinOpConv, adjoint_error, operator_norm

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

# Check any operator with the dot-product test (round-off: < 1e-5 in single precision).
print(adjoint_error(A, x_true))

# The spectral norm, e.g. for a gradient step 1 / ‖A‖²: power iteration on AᴴA.
print(operator_norm(A, x_true))

# An incoherent model: blur by a point-spread function stored with its
# centre in the middle of the array, hence the ifftshift.
psf = torch.rand(256, 256)
blur = LinOpConv(torch.fft.ifftshift(psf / psf.sum()))
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

Formulas are written along one axis, `n` indexing the output and `N` the axis length; N-D operators apply them on every axis they act on. Indices are taken modulo `N` where the formula says *circular*.

| Operator | `A x` | `Aᴴ y` |
|---|---|---|
| `LinOpIdentity()` | `x` | `y` |
| `LinOpMul(c)` | `c ⊙ x` (diagonal; `c` may broadcast) | `conj(c) ⊙ y` |
| `LinOpReal()`, `LinOpImag()` | `Re x`, `Im x` (complex → real) | `y + 0i`, `i y` |
| `LinOpSumReduce(dim, size)` | `Σₙ x[n]` over one axis, kept with length 1 | `y` broadcast back to `size` |
| `LinOpMatrix(M)` | `M x` on the last axis | `Mᴴ y` |
| `LinOpFunction(f, fT)` | `f(x)` | `fT(y)` |
| `LinOpCat([A_k])` | `cat([A_k x], dim=-1)` | `Σₖ A_kᴴ yₖ`, `yₖ` the columns of `A_k x` |
| `LinOpFft(dim, norm)`, `LinOpIfft` | `fftn` / `ifftn` (`norm="ortho"` by default, unitary) | the opposite transform with the conjugate norm |
| `LinOpFftShift(dim)` | `fftshift` | `ifftshift` |
| `LinOpZoomFft(in_shape, out_shape, k_start, k_end, ...)` | `zoom_fft` on a band (needs `minimal-zoom-fft`) | `zoom_ifft` with the conjugate norm |
| `LinOpRoll(shifts, dim, pad_zeros)` | `x[n − s]`, circular; with `pad_zeros`, `0` where `n − s` falls outside `[0, N)` | `y[n + s]`, same rule |
| `LinOpConv(h)` | `Σₘ h[m] x[n − m]`, circular, on the last `h.ndim` axes | `Σₘ conj(h[m]) y[n + m]` (correlation) |
| `LinOpCrop(in_shape, out_shape, fourier_origin)` | `x[c + n]`, `n < out`, `c = in//2 − out//2`; with `fourier_origin`, the first `⌈out/2⌉` and the last `⌊out/2⌋` samples | zero-pad: `y` back where it was taken from, `0` elsewhere |
| `LinOpPatch(in_shape, out_shape, shifts, pad_zeros, fourier_origin)` | `Crop(Roll(x))`, defined by `patch_by_crop_and_roll`, gathering only the window | scatter-add of `y` into zeros |
| `LinOpFlip(dim)` | `x[N − 1 − n]` | itself |
| `LinOpGrad(ndim)` | `(∇x)ₖ[n] = x[n + eₖ] − x[n]`, `0` at the last index; `(..., *s) → (..., ndim, *s)` | `Σₖ (yₖ[n − eₖ] − yₖ[n])` with `yₖ = 0` at the last index and outside the grid (negative divergence) |
| `LinOpDownsample(in_shape, factor)` | `x[f n]` | `y[n / f]` where `f` divides `n`, `0` elsewhere |
| `LinOpUpsample(in_shape, factor)` | `x[⌊n / f⌋]` | `Σⱼ y[f n + j]`, `j < f`, the sum of each block |

All operators act on trailing axes (`dim` defaults to the last axis; use `dim=(-2, -1)` or a 2-tuple `in_shape` for images), leave leading batch axes alone, work on CPU/CUDA/MPS, support autograd, and are out-of-place so they run under `torch.func.vmap` / `jacrev`.

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

Or wrap two functions: `LinOpFunction(apply, applyT, in_shape, out_shape)`. Declare `in_shape` / `out_shape` when they are fixed so compositions can check them, and set `preserves_shape = False` if the operator leaves them undeclared but changes the shape. The name `H` is taken by the adjoint property, so do not use it for an attribute.

## Checking operators

- `adjoint_error(A, x, y=None)` returns `|Re⟨A x, y⟩ − Re⟨x, Aᴴ y⟩| / (‖A x‖ ‖y‖)`, with the denominator falling back to `‖x‖ ‖Aᴴ y‖` when `A x` or `y` vanishes. Every operator in this library is tested this way.
- `operator_norm(A, x0, n_iter=50)` estimates `‖A‖₂` by power iteration on `AᴴA`; `x0` fixes the shape, dtype and device. The estimate approaches `‖A‖₂` from below, so a step size `1/‖A‖²` taken from it deserves a margin.
- `to_matrix(A, in_shape=None, dtype=torch.complex64)` materialises the dense matrix for small problems (`to_matrix(A.H) == to_matrix(A).conj().T`).

## Conventions worth knowing

- Every `dim` is negative. Operators act on the trailing axes, so an axis counted from the front, which would consume a batch axis, is rejected at construction.
- `norm="ortho"` is the default for the FFT operators, so `Aᴴ = A⁻¹` for them. With `"backward"` or `"forward"` the adjoint is still exact, but it is the opposite transform with the *other* norm, not the inverse.
- `LinOpCrop` centres like `torch`: it keeps indices `in//2 - out//2` onward. `fourier_origin=True` keeps the low frequencies of a DC-in-the-corner spectrum.
- `LinOpConv` puts the origin of the kernel at index 0, like `torch.roll` and the FFT: a delta at index 0 is the identity, a delta at index 1 is `LinOpRoll(1)`. A point-spread function with its centre in the middle of the array goes through `torch.fft.ifftshift` first. Real kernel and real input give a real output.
- `LinOpMul` declares `in_shape = out_shape = c.shape`, unless `c` is a scalar or has a size-1 axis: it then broadcasts, the shape it acts on is not determined by `c`, and it declares none.
- An operator built from a tensor (`LinOpMul`, `LinOpConv`, `LinOpMatrix`) lives on that tensor's device and torch raises on an input that is somewhere else, so build it where you will use it; there is no `.to()` to move one afterwards. `LinOpPatch` is the exception: its index tables follow the input's device.
- Output dtype equals input dtype as long as the operator's own tensor has the input's precision. `LinOpMul`, `LinOpConv` and `LinOpMatrix` promote the two dtypes the way the corresponding product does, so a float64 kernel on a float32 input gives float64, and a real matrix on a complex input gives complex.
- `LinOpRoll` shifts given as tensors are rounded to the nearest integer (ties to even) and read with `int()`, so operators can be built inside `torch.func` transforms.
- `LinOpCat` splits its adjoint input according to the sub-operators' `out_shape`. A sub-operator that declares none must preserve shape, and the columns left over by the declared ones are shared equally among those; a shape-changing sub-operator without a declared `out_shape` is refused rather than guessed.
- `LinOpPatch` is defined by `patch_by_crop_and_roll`, the same window as a `LinOpCrop @ LinOpRoll` composition. The definition is written first in the source and the tests hold the fast version to it, bit for bit.

## Relation to other libraries

The same idea as `scipy.sparse.linalg.LinearOperator`, [PyLops](https://pylops.readthedocs.io), [GlobalBioIm](https://biomedical-imaging-group.github.io/GlobalBioIm/) and the `physics` classes of [deepinv](https://deepinv.github.io), reduced to what you need to write and verify forward models in PyTorch, readable in full.

The operators come from the `LinOp` framework of the `ciel` computational-imaging library (EPFL), where they drive phase-retrieval and ptychography models; the classes were reduced, re-derived and re-tested for this package.

## Tutorials

Three notebooks in [`notebooks/`](notebooks/), runnable after `pip install -e ".[notebooks]"`:

1. [Why linear operators](notebooks/01_why_linear_operators.ipynb): reconstruction is `min ½‖Ax − b‖²`, its gradient is `Aᴴ(Ax − b)`; a coherent forward model built by composition, and what a sloppy adjoint does to the solver.
2. [What it computes](notebooks/02_what_it_computes.ipynb): every identity of the algebra, every adjoint of the catalogue and every convention, checked against brute force in float64.
3. [Benchmark](notebooks/03_benchmark.ipynb): the overhead against hand-written torch, `LinOpPatch` against its definition, batching, `to_matrix`, precision, and when not to use it.

## Tests

```bash
pip install -e ".[test]" && pip install minimal-zoom-fft   # or pip install -e ../minimal-zoom-fft
pytest                     # add ".[notebooks]" to run the tutorials as tests too
```

Every adjoint in the catalogue is checked with the dot-product test and against the dense matrix, `LinOpConv` against a brute-force double sum, `LinOpPatch` against its definition, and the README's Python blocks are executed as a test.

## License

MIT

## Manifest

- Purpose: linear operators with exact adjoints, and the algebra to compose them, in PyTorch.
- Dependencies: `torch`. Optional: `minimal-zoom-fft` for `LinOpZoomFft`.
- Size: about 1000 lines of implementation in 7 modules, 29 public names (the base class, 24 operators, the patch definition and 3 helpers); about 1100 lines of tests; 3 tutorial notebooks.
- Origin: the `ciel` computational-imaging library, EPFL.
- Provenance: written with Claude (Anthropic) from a brief; read and checked in full by Jonathan Dong.
- Version: 0.1.0, MIT.
