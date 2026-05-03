//! JSRM: full inner solve, reusable workspace, and optional low-level shooting loop.

use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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

#[inline(always)]
fn idx_np(k: usize, j: usize, p: usize) -> usize {
    k * p + j
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
            e[idx_np(kk, change_i, p)] += y[idx_np(kk, change_j, p)] * c1;
        }
    }
    if c2 != 0.0 {
        for kk in 0..n {
            e[idx_np(kk, change_j, p)] += y[idx_np(kk, change_i, p)] * c2;
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
        aij += e[idx_np(k, cur_j, p)] * y[idx_np(k, cur_i, p)];
        aji += e[idx_np(k, cur_i, p)] * y[idx_np(k, cur_j, p)];
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
    if (nnz as f64) > 0.2 * (pp as f64) {
        for i in 0..pp {
            g_scratch[i] = beta[i] * b[i];
        }
        let yv = ArrayView2::from_shape((n, p), y).expect("y shape");
        let sv = ArrayView2::from_shape((p, p), g_scratch).expect("scaled shape");
        let mut outv = ArrayViewMut2::from_shape((n, p), f_fit).expect("out shape");
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
                    f_fit[idx_np(k, j, p)] += y[idx_np(k, i, p)] * s;
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
    beta_last: &mut [f64],
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
    debug_assert_eq!(beta_last.len(), pp);
    debug_assert_eq!(b.len(), pp);
    debug_assert_eq!(b_s.len(), pp);

    for _ in 0..n_iter {
        for ii in 0..p {
            let base = ii * p;
            let bl_base = ii * p;
            for jj in 0..p {
                beta_last[bl_base + jj] = beta_new[base + jj];
            }
        }

        let mut nrow_pick = 0usize;
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
            }
        }

        let mut maxdif = -100.0f64;
        if nrow_pick > 0 {
            for ii in 0..p {
                for jj in 0..p {
                    let mut d = beta_last[idx_pp(ii, jj, p)] - beta_new[idx_pp(ii, jj, p)];
                    if d < 0.0 {
                        d = -d;
                    }
                    if d > maxdif {
                        maxdif = d;
                    }
                }
            }
        }

        if maxdif < maxdif_tol || nrow_pick < 1 {
            for ii in 0..p {
                for jj in 0..p {
                    beta_last[idx_pp(ii, jj, p)] = beta_new[idx_pp(ii, jj, p)];
                }
            }

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
                }
            }

            maxdif = -100.0;
            for ii in 0..p {
                for jj in 0..p {
                    let mut d = beta_last[idx_pp(ii, jj, p)] - beta_new[idx_pp(ii, jj, p)];
                    if d < 0.0 {
                        d = -d;
                    }
                    if d > maxdif {
                        maxdif = d;
                    }
                }
            }

            if maxdif < maxdif_tol {
                return;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn jsrm_shooting_loop_inner(
    y_m: ArrayView2<f64>,
    e_m: &mut ArrayViewMut2<f64>,
    beta_new: &mut ArrayViewMut2<f64>,
    beta_old: &mut ArrayViewMut2<f64>,
    beta_last: &mut ArrayViewMut2<f64>,
    b: ArrayView2<f64>,
    b_s: ArrayView2<f64>,
    lambda1: f64,
    lambda2: f64,
    n: usize,
    p: usize,
    n_iter: usize,
    change_i: usize,
    change_j: usize,
    beta_change: f64,
    tol: f64,
) {
    let y_sl = y_m.as_slice().expect("y contiguous");
    let e_sl = e_m.as_slice_mut().expect("e contiguous");
    let bn_sl = beta_new.as_slice_mut().expect("beta_new contiguous");
    let bo_sl = beta_old.as_slice_mut().expect("beta_old contiguous");
    let bl_sl = beta_last.as_slice_mut().expect("beta_last contiguous");
    let b_sl = b.as_slice().expect("b contiguous");
    let bs_sl = b_s.as_slice().expect("b_s contiguous");
    jsrm_shooting_loop_slices(
        y_sl,
        e_sl,
        bn_sl,
        bo_sl,
        bl_sl,
        b_sl,
        bs_sl,
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
    beta_last: Vec<f64>,
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
            beta_last: vec![0.0; pp],
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

        // copy + center columns
        for j in 0..p {
            for i in 0..n {
                self.y_m[idx_np(i, j, p)] = y_view[[i, j]];
            }
        }
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += self.y_m[idx_np(i, j, p)];
            }
            let m = s / n as f64;
            for i in 0..n {
                self.y_m[idx_np(i, j, p)] -= m;
            }
        }

        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                let v = self.y_m[idx_np(i, j, p)];
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
                let y_arr = ArrayView2::from_shape((n, p), &self.y_m[..])
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("internal y_m"))?;
                let g_mat = y_arr.t().dot(&y_arr);
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

        self.beta_last.fill(0.0);
        jsrm_shooting_loop_slices(
            &self.y_m[..],
            &mut self.e_m[..],
            &mut self.beta_new[..],
            &mut self.beta_old[..],
            &mut self.beta_last[..],
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
        Ok(arr.into_pyarray_bound(py).unbind())
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn jsrm_shooting_loop(
    y_m: PyReadonlyArray2<f64>,
    e_m: &Bound<'_, PyArray2<f64>>,
    beta_new: &Bound<'_, PyArray2<f64>>,
    beta_old: &Bound<'_, PyArray2<f64>>,
    beta_last: &Bound<'_, PyArray2<f64>>,
    b: PyReadonlyArray2<f64>,
    b_s: PyReadonlyArray2<f64>,
    lambda1: f64,
    lambda2: f64,
    n: usize,
    p: usize,
    n_iter: usize,
    change_i: usize,
    change_j: usize,
    beta_change: f64,
    tol: f64,
) -> PyResult<()> {
    let y = y_m.as_array();
    if y.nrows() != n || y.ncols() != p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "y_m shape must be (n, p)",
        ));
    }
    let b_v = b.as_array();
    let bs_v = b_s.as_array();
    if b_v.nrows() != p || b_v.ncols() != p || bs_v.nrows() != p || bs_v.ncols() != p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "B and B_s must be (p, p)",
        ));
    }

    unsafe {
        let mut e = e_m.as_array_mut();
        let mut bn = beta_new.as_array_mut();
        let mut bo = beta_old.as_array_mut();
        let mut bl = beta_last.as_array_mut();
        if e.shape() != [n, p] || bn.shape() != [p, p] || bo.shape() != [p, p] || bl.shape() != [p, p]
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "array shape mismatch",
            ));
        }
        jsrm_shooting_loop_inner(
            y, &mut e, &mut bn, &mut bo, &mut bl, b_v, bs_v, lambda1, lambda2, n, p, n_iter,
            change_i, change_j, beta_change, tol,
        );
    }
    Ok(())
}
