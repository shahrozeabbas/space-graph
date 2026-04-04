# space-graph

[![PyPI version](https://img.shields.io/pypi/v/space-graph)](https://pypi.org/project/space-graph/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19421985.svg)](https://doi.org/10.5281/zenodo.19421985)

Discover which variables in your dataset are directly related to each other, even after accounting for all other variables. Given a data matrix, SPACE estimates a sparse network of **partial correlations** -- connections that remain after removing indirect effects. Designed for settings where the number of variables can far exceed the number of samples (e.g. genomics).

Pure Python implementation of **SPACE** (Sparse Partial Correlation Estimation) from Peng et al. (2009), with no R or C dependencies.

Paper: [Sparse Partial Correlation Estimation for High-Dimensional Data](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.0126)

## Install

```bash
pip install space-graph
```

Optional Numba (faster inner `jsrm` loop):

```bash
pip install 'space-graph[numba]'
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

## Penalty

`alpha` is overall regularization strength (like sklearn). `gamma` (γ) in `[0, 1]` sets the mix: `lam1 = alpha * gamma`, `lam2 = alpha * (1 - gamma)`. Default `gamma=1` matches R `space::space.joint` default `lam2 = 0`.

## Options

- **`gamma`** (default `1`): mix γ in `[0, 1]` for `lam1` / `lam2`; `1` matches R default `lam2 = 0`.
- **`tol`** (default `1e-6`): inner coordinate-descent stopping tolerance (and active-set threshold), same scale as the reference C code.
- **`weight`**: default **`uniform`** (unit weights). Use **`equal`** as an alias. Other modes: **`sig`**, **`degree`**, or a custom positive vector of length `p`.
- **`backend`** (default **`auto`**): inner JSRM shooting loop. **`numpy`** always uses pure NumPy (no Numba import). **`auto`** uses Numba when installed (lazy on first `fit`), otherwise NumPy. **`numba`** requires Numba and raises `ImportError` if it is missing. The first `fit` with **`auto`** or **`numba`** may include JIT compilation time.

## Tests

```bash
pytest
```

For a quick manual run after an editable install (or with `PYTHONPATH=.` from the repo root), see [`test_space_graph.py`](test_space_graph.py).

Optional: build `libjsrm_test.so` from `../space/src/JSRM.c` to run the ctypes cross-check in `tests/test_space.py`.

## License

GPL-3.0-or-later (same family as the original `space` R package).
