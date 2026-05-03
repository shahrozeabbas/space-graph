# space-graph

[![PyPI version](https://img.shields.io/badge/pypi-v2.0.0-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/space-graph/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19421985.svg)](https://doi.org/10.5281/zenodo.19421985)

Discover which variables in your dataset are directly related to each other, even after accounting for all other variables. Given a data matrix, SPACE estimates a sparse network of **partial correlations** -- connections that remain after removing indirect effects. Designed for settings where the number of variables can far exceed the number of samples (e.g. genomics).

Python implementation of **SPACE** (Sparse Partial Correlation Estimation) from Peng et al. (2009), with no R or C dependencies.

Paper: [Sparse Partial Correlation Estimation for High-Dimensional Data](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.0126)

## Install

```bash
pip install space-graph
```

From source, the package builds with [maturin](https://www.maturin.rs/); with Rust installed, `space_graph._rust` provides `jsrm_solve` and `JsrmWorkspace` (reused buffers for repeated same-shape solves). They back `backend='rust'` and are preferred for `backend='auto'`. Without Rust, `pip install -e .` still installs pure Python (NumPy loop).

```bash
pip install maturin
maturin develop   # editable install with extension
```

From GitHub:

```bash
pip install git+https://github.com/shahrozeabbas/space-graph.git
```

## Usage

```python
import numpy as np
from space_graph import SPACE

X = np.random.randn(20, 5)

model = SPACE(
    alpha=0.7,
    max_outer_iter=2,
    max_inner_iter=500,
    tol=1e-6,
    weight='uniform',
)

model.fit(X)

print(model.partial_correlation_)
```

## Choosing `alpha`

Use `SPACE.select_alpha(X, alphas)` to pick the regularization strength by BIC (Peng et al. 2009, Sec. 2.4). It fits across the grid and returns the `alpha` minimizing `sum_i [n * log(RSS_i) + log(n) * k_i]`, where `k_i` is the number of neighbors of variable `i`.

```python
import numpy as np
from space_graph import SPACE

template = SPACE(max_outer_iter=3)
alphas = np.geomspace(0.005, 1.0, 15)
best_alpha = template.select_alpha(X, alphas)

model = SPACE(alpha=best_alpha, max_outer_iter=3).fit(X)
```

Pass `return_curve=True` to also get the per-alpha BIC vector for plotting. The template is not mutated — reuse or discard it.

## Penalty

- **`alpha`**: overall regularization strength (like sklearn).
- **`gamma`** (γ) in `[0, 1]`: mix between L1 and L2 terms.
    - `lam1 = alpha * gamma` (L1 / sparsity)
    - `lam2 = alpha * (1 - gamma)` (L2 / ridge)
    - Default `gamma=1` matches R `space::space.joint` default `lam2 = 0`.

## Options

- **`tol`** (default `1e-6`): inner solver convergence + active-set threshold, same scale as the reference C code.
- **`weight`**: default **`uniform`** (equivalently **`equal`**, unit weights). Other modes:
    - **`sig`** — reweight each variable by its residual variance (updates each outer iteration).
    - **`degree`** — reweight by node connectivity (updates each outer iteration).
    - custom ndarray of length `p` — internally normalized to mean 1, so `alpha`'s scale stays comparable to the other modes.
- **`backend`** (default **`auto`**): inner joint sparse regression (JSRM) solver. **`numpy`** always uses pure NumPy. **`rust`** requires the compiled `space_graph._rust` extension. **`auto`** uses Rust when available, otherwise NumPy.

## Tests

```bash
pytest
```

Optional: build `libjsrm_test.so` from `../space/src/JSRM.c` to run the ctypes cross-check in `tests/test_space.py`.

## License

GPL-3.0-or-later (same family as the original `space` R package).
