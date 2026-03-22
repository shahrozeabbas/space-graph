import numpy as np
from space_graph import SPACE

rng = np.random.default_rng(0)

n, p = 50, 8

X = rng.standard_normal((n, p))

model = SPACE(
    alpha=0.7,
    max_outer_iter=5,
    max_inner_iter=2000,
    tol=1e-6,
    weight='uniform',
)

model.fit(X)
print('partial_correlation_:\n', model.partial_correlation_)