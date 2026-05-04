# space-graph

[![PyPI version](https://img.shields.io/badge/pypi-v2.0.1-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/space-graph/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Build wheels](https://github.com/shahrozeabbas/space-graph/actions/workflows/wheels.yml/badge.svg)](https://github.com/shahrozeabbas/space-graph/actions/workflows/wheels.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

`space-graph` estimates sparse partial-correlation networks from tabular data.

Given a data matrix, it finds pairs of variables that remain related after accounting for all other variables. This is useful for network discovery in high-dimensional settings such as genomics, where the number of variables may be large relative to the number of samples.

The package implements **SPACE** (Sparse Partial Correlation Estimation) from Peng et al. (2009): [Sparse Partial Correlation Estimation for High-Dimensional Data](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.0126).

## Install

```bash
pip install space-graph
```

## Quick Start

```python
import numpy as np
from space_graph import SPACE

X = np.random.randn(100, 20)

model = SPACE(alpha=0.5).fit(X)

partial_correlations = model.partial_correlation_
```

`partial_correlations` is a square matrix. Entry `(i, j)` is the estimated partial correlation between variables `i` and `j`. Values close to zero indicate no direct relationship after conditioning on the other variables.

## Choosing `alpha`

`alpha` controls sparsity:

- larger `alpha` -> fewer edges
- smaller `alpha` -> more edges

You can choose `alpha` from a grid using BIC or AIC:

```python
alphas = np.geomspace(0.01, 1.0, 20)

template = SPACE(max_outer_iter=3)
best_alpha = template.select_alpha(X, alphas, criterion='bic')

model = SPACE(alpha=best_alpha, max_outer_iter=3).fit(X)
```

BIC is the default and is usually more conservative. AIC often selects a denser network:

```python
best_alpha = template.select_alpha(X, alphas, criterion='aic')
```

To inspect the scores:

```python
best_alpha, scores = template.select_alpha(
    X,
    alphas,
    criterion='bic',
    return_curve=True,
)
```

## Important Options

```python
SPACE(
    alpha=0.5,
    gamma=1.0,
    weight='uniform',
    standardize=True,
    backend='auto',
)
```

- `alpha`: regularization strength. Higher values produce sparser networks.
- `gamma`: mix between L1 sparsity and L2 shrinkage. The default `1.0` is pure L1.
- `weight`: node weighting scheme. Use `'uniform'` for most cases.
- `standardize`: centers and scales columns before fitting.
- `backend`: `'auto'` uses the Rust backend when available and falls back to NumPy.

## Output

After fitting:

```python
model.partial_correlation_
model.sig_
model.weight_
```

The main result is `partial_correlation_`, a symmetric matrix with unit diagonal.

## Citation

If you use this package, please cite:

Peng, J., Wang, P., Zhou, N., & Zhu, J. (2009). Sparse partial correlation estimation by joint sparse regression models. *Journal of the American Statistical Association*, 104(486), 735-746.

## License

GPL-3.0-or-later.
