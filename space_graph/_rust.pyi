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

def jsrm_shooting_loop(
    y_m: npt.NDArray[np.float64],
    e_m: npt.NDArray[np.float64],
    beta_new: npt.NDArray[np.float64],
    beta_old: npt.NDArray[np.float64],
    beta_last: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    b_s: npt.NDArray[np.float64],
    lambda1: float,
    lambda2: float,
    n: int,
    p: int,
    n_iter: int,
    change_i: int,
    change_j: int,
    beta_change: float,
    tol: float,
) -> None:
    ...
