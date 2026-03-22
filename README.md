# space-graph

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

The public parameter `alpha` in `[0, 1]` maps to inner penalties as `lam1 = alpha` and `lam2 = 1 - alpha`, matching the reference elastic-net-style JSRM solver.

## Options

- **`tol`** (default `1e-6`): inner coordinate-descent stopping tolerance (and active-set threshold), same scale as the reference C code.
- **`weight`**: default **`uniform`** (unit weights). Use **`equal`** as an alias. Other modes: **`sig`**, **`degree`**, or a custom positive vector of length `p`.

## Tests

```bash
pytest
```

Optional: build `libjsrm_test.so` from `../space/src/JSRM.c` to run the ctypes cross-check in `tests/test_space.py`.

## License

GPL-3.0-or-later (same family as the original `space` R package).
