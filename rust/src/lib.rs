//! Optional PyO3 coordinate core for topojson (BSD-3-Clause).
//!
//! Copyright (c) 2026, Matthew J Whittle
//!
//! Quantized join/cut on packed integer buffers and Chaikin on float arcs.
//! No GEOS. No CartAGen / Smoothify source.

mod chaikin;
mod cut;
mod hash;
mod join;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn as_xy_i64(arr: PyReadonlyArray2<'_, i64>) -> PyResult<Vec<[i64; 2]>> {
    let view = arr.as_array();
    if view.ndim() != 2 || view.shape()[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "expected (n, 2) int64, got shape {:?}",
            view.shape()
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect())
}

fn as_xy_f64(arr: PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 2]>> {
    let view = arr.as_array();
    if view.ndim() != 2 || view.shape()[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "expected (n, 2) float64, got shape {:?}",
            view.shape()
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect())
}

fn xy_to_py<'py>(py: Python<'py>, xy: Vec<[i64; 2]>) -> Bound<'py, PyArray2<i64>> {
    let n = xy.len();
    let flat: Vec<i64> = xy.into_iter().flat_map(|p| [p[0], p[1]]).collect();
    PyArray2::from_vec2(py, &flat.chunks(2).map(|c| c.to_vec()).collect::<Vec<_>>())
        .unwrap_or_else(|_| PyArray2::zeros(py, [n, 2], false))
}

fn xy_f_to_py<'py>(py: Python<'py>, xy: Vec<[f64; 2]>) -> Bound<'py, PyArray2<f64>> {
    let rows: Vec<Vec<f64>> = xy.into_iter().map(|p| vec![p[0], p[1]]).collect();
    PyArray2::from_vec2(py, &rows).expect("chaikin output shape")
}

/// Junctions of quantized lines on the integer grid.
///
/// Parameters
/// ----------
/// xy : (V, 2) int64
/// li : (V,) int32 line index per vertex
/// closed : (n_lines,) bool
///
/// Returns
/// -------
/// points : (J, 2) int64
/// pairs : (P, 2) int32 line index pairs for the pairwise fallback
#[pyfunction]
fn shared_path_ends_on_grid<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, i64>,
    li: PyReadonlyArray1<'py, i32>,
    closed: PyReadonlyArray1<'py, bool>,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<i32>>)> {
    let xy = as_xy_i64(xy)?;
    let li: Vec<i32> = li.as_array().iter().copied().collect();
    let closed: Vec<bool> = closed.as_array().iter().copied().collect();
    if xy.len() != li.len() {
        return Err(PyValueError::new_err("xy and li length mismatch"));
    }
    let (points, pairs) =
        py.allow_threads(|| join::shared_path_ends_on_grid(&xy, &li, &closed));
    let points_py = xy_to_py(py, points);
    let pair_rows: Vec<Vec<i32>> = pairs.into_iter().map(|p| vec![p[0], p[1]]).collect();
    let pairs_py = if pair_rows.is_empty() {
        PyArray2::zeros(py, [0, 2], false)
    } else {
        PyArray2::from_vec2(py, &pair_rows).unwrap()
    };
    Ok((points_py, pairs_py))
}

/// Cut quantized lines at junctions (packed arrays; no STRtree).
#[pyfunction]
fn cut_lines_on_grid<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, i64>,
    li: PyReadonlyArray1<'py, i32>,
    jxy: PyReadonlyArray2<'py, i64>,
    is_ring: PyReadonlyArray1<'py, bool>,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
)> {
    let xy = as_xy_i64(xy)?;
    let li: Vec<i32> = li.as_array().iter().copied().collect();
    let jxy = as_xy_i64(jxy)?;
    let is_ring: Vec<bool> = is_ring.as_array().iter().copied().collect();
    if xy.len() != li.len() {
        return Err(PyValueError::new_err("xy and li length mismatch"));
    }
    let (parts, offsets, n_parts) =
        py.allow_threads(|| cut::cut_lines_on_grid(&xy, &li, &jxy, &is_ring));
    let parts_py = xy_to_py(py, parts);
    let offsets_i64: Vec<i64> = offsets.into_iter().map(|v| v as i64).collect();
    Ok((
        parts_py,
        offsets_i64.into_pyarray(py).to_owned(),
        n_parts.into_pyarray(py).to_owned(),
    ))
}

/// Chaikin corner-cutting; endpoints fixed. Matches ``topojson.ops.chaikin``.
#[pyfunction]
#[pyo3(name = "chaikin")]
fn chaikin_py<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    iterations: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xy = as_xy_f64(xy)?;
    let out = py.allow_threads(|| chaikin::chaikin(&xy, iterations));
    Ok(xy_f_to_py(py, out))
}

/// Packed multi-arc Chaikin. Returns ``(xy_out, offsets_out)``.
#[pyfunction]
#[pyo3(name = "chaikin_many")]
fn chaikin_many_py<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    offsets: PyReadonlyArray1<'py, i64>,
    iterations: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
    let xy = as_xy_f64(xy)?;
    let offsets: Vec<usize> = offsets.as_array().iter().map(|&v| v as usize).collect();
    let (out, new_off) = py.allow_threads(|| chaikin::chaikin_many(&xy, &offsets, iterations));
    let new_off_i64: Vec<i64> = new_off.into_iter().map(|v| v as i64).collect();
    Ok((xy_f_to_py(py, out), new_off_i64.into_pyarray(py).to_owned()))
}

/// Build status / version string for benches.
#[pyfunction]
fn core_info() -> String {
    format!("topojson._core {} (join/cut/chaikin)", env!("CARGO_PKG_VERSION"))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shared_path_ends_on_grid, m)?)?;
    m.add_function(wrap_pyfunction!(cut_lines_on_grid, m)?)?;
    m.add_function(wrap_pyfunction!(chaikin_py, m)?)?;
    m.add_function(wrap_pyfunction!(chaikin_many_py, m)?)?;
    m.add_function(wrap_pyfunction!(core_info, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
