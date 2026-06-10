'''Type stubs for optional ``space_graph._rust`` extension (maturin build).'''

import numpy as np
import numpy.typing as npt

class JsrmWorkspace:
    def __init__(self, n: int, p: int) -> None: ...
    def solve(
        self,
        y_data: npt.NDArray[np.float64],
        sigma_sr: npt.NDArray[np.float64],
        lambda1: float,
        lambda2: float,
        n_iter: int,
        tol: float,
        init_beta: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        ...

def jsrm_solve(
    y_data: npt.NDArray[np.float64],
    sigma_sr: npt.NDArray[np.float64],
    lambda1: float,
    lambda2: float,
    n_iter: int,
    tol: float,
    init_beta: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    ...
