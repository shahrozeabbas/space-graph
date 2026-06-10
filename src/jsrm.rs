//! JSRM: full inner solve, reusable workspace, and optional low-level shooting loop.

use ndarray::{Array2, ArrayView2, ArrayViewMut2, ShapeBuilder};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Use dense GEMM when nonzero fraction exceeds this (matches Python ``solver``).
const MATMUL_DENSE_NNZ_FRACTION: f64 = 0.2;

#[inline]
fn sign_like_numpy(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Column-major index for (n, p) sample-by-variable buffers: ``j*n + k``.
#[inline(always)]
fn idx_ye(k: usize, j: usize, n: usize) -> usize {
    j * n + k
}

#[inline(always)]
fn idx_pp(i: usize, j: usize, p: usize) -> usize {
    i * p + j
}

/// Upper triangle ``(i < j)`` in JSRM.c scan order.
fn upper_tri_ij_jsrm_order(p: usize) -> (Vec<usize>, Vec<usize>) {
    let mut ir = Vec::new();
    let mut jr = Vec::new();
    if p < 2 {
        return (ir, jr);
    }
    for j in (1..p).rev() {
        for i in (0..j).rev() {
            ir.push(i);
            jr.push(j);
        }
    }
    (ir, jr)
}

fn count_nonzero_beta_slice(beta: &[f64], p: usize) -> usize {
    let pp = p * p;
    debug_assert_eq!(beta.len(), pp);
    beta.iter().take(pp).filter(|x| **x != 0.0).count()
}

#[inline]
fn elastic_net_shrink(beta_next: f64, b_s: f64, lambda1: f64, lambda2: f64) -> f64 {
    let temp1 = beta_next;
    let mut temp = if beta_next > 0.0 {
        beta_next - lambda1 / b_s
    } else {
        -beta_next - lambda1 / b_s
    };
    if temp < 0.0 {
        return 0.0;
    }
    temp /= 1.0 + lambda2;
    if temp1 < 0.0 {
        temp = -temp;
    }
    temp
}

fn apply_residual_slice(
    e: &mut [f64],
    y: &[f64],
    change_i: usize,
    change_j: usize,
    beta_change: f64,
    b: &[f64],
    n: usize,
    p: usize,
) {
    let c1 = beta_change * b[idx_pp(change_j, change_i, p)];
    let c2 = beta_change * b[idx_pp(change_i, change_j, p)];
    if c1 != 0.0 {
        for kk in 0..n {
            e[idx_ye(kk, change_i, n)] += y[idx_ye(kk, change_j, n)] * c1;
        }
    }
    if c2 != 0.0 {
        for kk in 0..n {
            e[idx_ye(kk, change_j, n)] += y[idx_ye(kk, change_i, n)] * c2;
        }
    }
}

fn aij_aji_slice(
    e: &[f64],
    y: &[f64],
    cur_i: usize,
    cur_j: usize,
    b: &[f64],
    n: usize,
    p: usize,
) -> (f64, f64) {
    let mut aij = 0.0;
    let mut aji = 0.0;
    for k in 0..n {
        aij += e[idx_ye(k, cur_j, n)] * y[idx_ye(k, cur_i, n)];
        aji += e[idx_ye(k, cur_i, n)] * y[idx_ye(k, cur_j, n)];
    }
    aij *= b[idx_pp(cur_i, cur_j, p)];
    aji *= b[idx_pp(cur_j, cur_i, p)];
    (aij, aji)
}

/// ``f_fit = Y @ (beta * B)``; ``g_scratch`` holds ``beta * B`` in the dense path only.
fn ym_times_elementwise_into_flat(
    y: &[f64],
    beta: &[f64],
    b: &[f64],
    f_fit: &mut [f64],
    g_scratch: &mut [f64],
    n: usize,
    p: usize,
) {
    let pp = p * p;
    debug_assert_eq!(y.len(), n * p);
    debug_assert_eq!(beta.len(), pp);
    debug_assert_eq!(b.len(), pp);
    debug_assert_eq!(f_fit.len(), n * p);
    debug_assert_eq!(g_scratch.len(), pp);

    if p == 0 {
        return;
    }
    let nnz = count_nonzero_beta_slice(beta, p);
    if (nnz as f64) > MATMUL_DENSE_NNZ_FRACTION * (pp as f64) {
        for i in 0..pp {
            g_scratch[i] = beta[i] * b[i];
        }
        let yv = ArrayView2::from_shape((n, p).f(), y).expect("y shape");
        let sv = ArrayView2::from_shape((p, p), g_scratch).expect("scaled shape");
        let mut outv = ArrayViewMut2::from_shape((n, p).f(), f_fit).expect("out shape");
        outv.assign(&yv.dot(&sv));
        return;
    }
    f_fit.fill(0.0);
    for j in 0..p {
        for i in 0..p {
            let bij = beta[idx_pp(i, j, p)];
            if bij != 0.0 {
                let s = bij * b[idx_pp(i, j, p)];
                for k in 0..n {
                    f_fit[idx_ye(k, j, n)] += y[idx_ye(k, i, n)] * s;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn jsrm_one_step_slice(
    y: &[f64],
    e: &mut [f64],
    beta_new: &mut [f64],
    beta_old: &mut [f64],
    b: &[f64],
    b_s: &[f64],
    cur_i: usize,
    cur_j: usize,
    change_i: usize,
    change_j: usize,
    beta_change: f64,
    lambda1: f64,
    lambda2: f64,
    n: usize,
    p: usize,
    eps1: f64,
    gate_residual: bool,
) -> (f64, usize, usize) {
    beta_old[idx_pp(change_i, change_j, p)] = beta_new[idx_pp(change_i, change_j, p)];
    beta_old[idx_pp(change_j, change_i, p)] = beta_new[idx_pp(change_j, change_i, p)];

    if !gate_residual || (beta_change < -eps1 || beta_change > eps1) {
        apply_residual_slice(e, y, change_i, change_j, beta_change, b, n, p);
    }

    let (aij, aji) = aij_aji_slice(e, y, cur_i, cur_j, b, n, p);

    let bs = b_s[idx_pp(cur_i, cur_j, p)];
    let beta_next = (aij + aji) / bs + beta_old[idx_pp(cur_i, cur_j, p)];
    let temp = elastic_net_shrink(beta_next, bs, lambda1, lambda2);

    beta_new[idx_pp(cur_i, cur_j, p)] = temp;
    beta_new[idx_pp(cur_j, cur_i, p)] = temp;

    let new_change = beta_old[idx_pp(cur_i, cur_j, p)] - temp;
    (new_change, cur_i, cur_j)
}

#[allow(clippy::too_many_arguments)]
fn jsrm_shooting_loop_slices(
    y: &[f64],
    e: &mut [f64],
    beta_new: &mut [f64],
    beta_old: &mut [f64],
    b: &[f64],
    b_s: &[f64],
    lambda1: f64,
    lambda2: f64,
    n: usize,
    p: usize,
    n_iter: usize,
    mut change_i: usize,
    mut change_j: usize,
    mut beta_change: f64,
    tol: f64,
) {
    let eps1 = tol;
    let maxdif_tol = tol;
    let pp = p * p;
    debug_assert_eq!(y.len(), n * p);
    debug_assert_eq!(e.len(), n * p);
    debug_assert_eq!(beta_new.len(), pp);
    debug_assert_eq!(beta_old.len(), pp);
    debug_assert_eq!(b.len(), pp);
    debug_assert_eq!(b_s.len(), pp);

    for _ in 0..n_iter {
        let mut nrow_pick = 0usize;
        let mut active_max = -100.0f64;
        for j in (1..p).rev() {
            for i in (0..j).rev() {
                let b_ij = beta_new[idx_pp(i, j, p)];
                if !(b_ij > eps1 || b_ij < -eps1) {
                    continue;
                }
                nrow_pick += 1;
                let (nc, ci, cj) = jsrm_one_step_slice(
                    y,
                    e,
                    beta_new,
                    beta_old,
                    b,
                    b_s,
                    i,
                    j,
                    change_i,
                    change_j,
                    beta_change,
                    lambda1,
                    lambda2,
                    n,
                    p,
                    eps1,
                    false,
                );
                beta_change = nc;
                change_i = ci;
                change_j = cj;
                let d = nc.abs();
                if d > active_max {
                    active_max = d;
                }
            }
        }

        if active_max < maxdif_tol || nrow_pick < 1 {
            let mut full_max = -100.0f64;
            for cur_i in 0..(p - 1) {
                for cur_j in (cur_i + 1)..p {
                    let (nc, ci, cj) = jsrm_one_step_slice(
                        y,
                        e,
                        beta_new,
                        beta_old,
                        b,
                        b_s,
                        cur_i,
                        cur_j,
                        change_i,
                        change_j,
                        beta_change,
                        lambda1,
                        lambda2,
                        n,
                        p,
                        eps1,
                        true,
                    );
                    beta_change = nc;
                    change_i = ci;
                    change_j = cj;
                    let d = nc.abs();
                    if d > full_max {
                        full_max = d;
                    }
                }
            }

            if full_max < maxdif_tol {
                return;
            }
        }
    }
}

// --- reusable buffers -----------------------------------------------------

pub struct JsrmBuffers {
    n: usize,
    p: usize,
    y_m: Vec<f64>,
    normx: Vec<f64>,
    b: Vec<f64>,
    b_sq: Vec<f64>,
    b_s: Vec<f64>,
    g: Vec<f64>,
    f_fit: Vec<f64>,
    e_m: Vec<f64>,
    beta_new: Vec<f64>,
    beta_old: Vec<f64>,
    i_ut: Vec<usize>,
    j_ut: Vec<usize>,
}

impl JsrmBuffers {
    pub fn new(n: usize, p: usize) -> Self {
        let np = n * p;
        let pp = p * p;
        let (i_ut, j_ut) = upper_tri_ij_jsrm_order(p);
        Self {
            n,
            p,
            y_m: vec![0.0; np],
            normx: vec![0.0; p],
            b: vec![0.0; pp],
            b_sq: vec![0.0; pp],
            b_s: vec![0.0; pp],
            g: vec![0.0; pp],
            f_fit: vec![0.0; np],
            e_m: vec![0.0; np],
            beta_new: vec![0.0; pp],
            beta_old: vec![0.0; pp],
            i_ut,
            j_ut,
        }
    }

    fn solve(
        &mut self,
        y_view: ArrayView2<f64>,
        sigma_sr: &[f64],
        lambda1: f64,
        lambda2: f64,
        n_iter: usize,
        tol: f64,
        init_beta: Option<ArrayView2<f64>>,
    ) -> PyResult<()> {
        let n = self.n;
        let p = self.p;
        if y_view.nrows() != n || y_view.ncols() != p {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Y_data shape must match workspace (n, p)",
            ));
        }
        if sigma_sr.len() != p {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sigma_sr must have length p",
            ));
        }

        // copy + center columns (column-major y_m)
        for j in 0..p {
            for i in 0..n {
                self.y_m[idx_ye(i, j, n)] = y_view[[i, j]];
            }
        }
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += self.y_m[idx_ye(i, j, n)];
            }
            let m = s / n as f64;
            for i in 0..n {
                self.y_m[idx_ye(i, j, n)] -= m;
            }
        }

        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                let v = self.y_m[idx_ye(i, j, n)];
                s += v * v;
            }
            self.normx[j] = s;
        }

        for i in 0..p {
            for j in 0..p {
                self.b[idx_pp(i, j, p)] = sigma_sr[i] / sigma_sr[j];
            }
        }
        let pp = p * p;
        for i in 0..pp {
            let v = self.b[i];
            self.b_sq[i] = v * v;
        }
        for i in 0..p {
            for j in 0..p {
                self.b_s[idx_pp(i, j, p)] = self.b_sq[idx_pp(i, j, p)] * self.normx[i]
                    + self.b_sq[idx_pp(j, i, p)] * self.normx[j];
            }
        }

        self.beta_new.fill(0.0);
        match init_beta {
            None => {
                // column-major (n,p) memory is C-order (p,n) view of Y^T
                let yt = ArrayView2::from_shape((p, n), &self.y_m[..]).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("internal y_m")
                })?;
                let g_mat = yt.dot(&yt.t());
                self.g.copy_from_slice(
                    g_mat
                        .as_slice()
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Gramian not contiguous",
                            )
                        })?,
                );

                for j in 1..p {
                    for i in 0..j {
                        let temp1 =
                            self.g[idx_pp(i, j, p)] * (self.b[idx_pp(j, i, p)] + self.b[idx_pp(i, j, p)]);
                        let tt = temp1.abs() - lambda1;
                        if tt >= 0.0 {
                            let b_s_ij = self.b_s[idx_pp(i, j, p)] * (1.0 + lambda2);
                            let bet = sign_like_numpy(temp1) * tt / b_s_ij;
                            self.beta_new[idx_pp(i, j, p)] = bet;
                            self.beta_new[idx_pp(j, i, p)] = bet;
                        }
                    }
                }
            }
            Some(ib) => {
                for i in 0..p {
                    for j in 0..p {
                        self.beta_new[idx_pp(i, j, p)] =
                            0.5 * (ib[[i, j]] + ib[[j, i]]);
                    }
                }
                for i in 0..p {
                    self.beta_new[idx_pp(i, i, p)] = 0.0;
                }
            }
        }

        ym_times_elementwise_into_flat(
            &self.y_m[..],
            &self.beta_new[..],
            &self.b[..],
            &mut self.f_fit[..],
            &mut self.g[..],
            n,
            p,
        );

        for idx in 0..n * p {
            self.e_m[idx] = self.y_m[idx] - self.f_fit[idx];
        }

        self.beta_old.copy_from_slice(&self.beta_new[..]);

        let eps1 = tol;
        if self.i_ut.is_empty() {
            return Ok(());
        }

        let mut first_t: Option<usize> = None;
        for t in 0..self.i_ut.len() {
            let v = self.beta_new[idx_pp(self.i_ut[t], self.j_ut[t], p)];
            if v > eps1 || v < -eps1 {
                first_t = Some(t);
                break;
            }
        }
        let Some(t0) = first_t else {
            return Ok(());
        };

        let cur_i = self.i_ut[t0];
        let cur_j = self.j_ut[t0];
        let (aij, aji) = aij_aji_slice(
            &self.e_m[..],
            &self.y_m[..],
            cur_i,
            cur_j,
            &self.b[..],
            n,
            p,
        );
        let b_s_ij = self.b_s[idx_pp(cur_i, cur_j, p)];
        let beta_next = (aij + aji) / b_s_ij + self.beta_old[idx_pp(cur_i, cur_j, p)];
        let temp = elastic_net_shrink(beta_next, b_s_ij, lambda1, lambda2);
        let beta_change = self.beta_old[idx_pp(cur_i, cur_j, p)] - temp;
        self.beta_new[idx_pp(cur_i, cur_j, p)] = temp;
        self.beta_new[idx_pp(cur_j, cur_i, p)] = temp;
        let change_i = cur_i;
        let change_j = cur_j;

        jsrm_shooting_loop_slices(
            &self.y_m[..],
            &mut self.e_m[..],
            &mut self.beta_new[..],
            &mut self.beta_old[..],
            &self.b[..],
            &self.b_s[..],
            lambda1,
            lambda2,
            n,
            p,
            n_iter,
            change_i,
            change_j,
            beta_change,
            tol,
        );
        Ok(())
    }

    fn beta_to_pyarray(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let p = self.p;
        let arr =
            Array2::from_shape_vec((p, p), self.beta_new.to_vec()).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("beta layout")
            })?;
        Ok(arr.into_pyarray(py).unbind())
    }
}

#[pyclass]
pub struct JsrmWorkspace {
    buf: JsrmBuffers,
}

#[pymethods]
impl JsrmWorkspace {
    #[new]
    fn new(n: usize, p: usize) -> Self {
        Self {
            buf: JsrmBuffers::new(n, p),
        }
    }

    #[pyo3(signature = (y_data, sigma_sr, lambda1, lambda2, n_iter, tol, init_beta=None))]
    fn solve(
        &mut self,
        py: Python<'_>,
        y_data: PyReadonlyArray2<f64>,
        sigma_sr: PyReadonlyArray1<f64>,
        lambda1: f64,
        lambda2: f64,
        n_iter: usize,
        tol: f64,
        init_beta: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        if tol <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tol must be positive",
            ));
        }
        let y = y_data.as_array();
        let p = self.buf.p;
        if y.nrows() != self.buf.n || y.ncols() != p {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Y_data shape must match workspace (n, p)",
            ));
        }
        let sig = sigma_sr.as_array();
        if sig.len() != p {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sigma_sr must have length p",
            ));
        }
        let sigma_vec: Vec<f64> = sig.iter().copied().collect();
        let init_view = if let Some(ref ib) = init_beta {
            let v = ib.as_array();
            if v.nrows() != p || v.ncols() != p {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "init_beta must have shape (p, p)",
                ));
            }
            Some(v)
        } else {
            None
        };

        self.buf.solve(
            y.view(),
            &sigma_vec,
            lambda1,
            lambda2,
            n_iter,
            tol,
            init_view,
        )?;
        self.buf.beta_to_pyarray(py)
    }
}

#[pyfunction]
#[pyo3(signature = (y_data, sigma_sr, lambda1, lambda2, n_iter, tol, init_beta=None))]
pub fn jsrm_solve(
    py: Python<'_>,
    y_data: PyReadonlyArray2<f64>,
    sigma_sr: PyReadonlyArray1<f64>,
    lambda1: f64,
    lambda2: f64,
    n_iter: usize,
    tol: f64,
    init_beta: Option<PyReadonlyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    if tol <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "tol must be positive",
        ));
    }
    let y = y_data.as_array();
    let n = y.nrows();
    let p = y.ncols();
    let sig = sigma_sr.as_array();
    if sig.len() != p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sigma_sr must have length p",
        ));
    }
    let sigma_vec: Vec<f64> = sig.iter().copied().collect();
    let init_view = if let Some(ref ib) = init_beta {
        let v = ib.as_array();
        if v.nrows() != p || v.ncols() != p {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "init_beta must have shape (p, p)",
            ));
        }
        Some(v)
    } else {
        None
    };

    let mut buf = JsrmBuffers::new(n, p);
    buf.solve(y.view(), &sigma_vec, lambda1, lambda2, n_iter, tol, init_view)?;
    buf.beta_to_pyarray(py)
}
