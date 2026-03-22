# space-graph

[![PyPI version](https://img.shields.io/pypi/v/space-graph)](https://pypi.org/project/space-graph/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

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

`alpha` is overall regularization strength (like sklearn). Inner JSRM penalties are `lam1 = alpha * λ` and `lam2 = alpha * (1 - λ)` with a fixed mix `λ` that matches R `space::space.joint` defaults (`lam2 = 0`; see `space_graph.penalties.LAMBDA`).

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
