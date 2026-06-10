//! PyO3 extension: JSRM shooting loop only.

mod jsrm;

use pyo3::prelude::*;

/// Compiled extension imported as ``space_graph._rust``.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<jsrm::JsrmWorkspace>()?;
    m.add_function(wrap_pyfunction!(jsrm::jsrm_solve, m)?)?;
    Ok(())
}
