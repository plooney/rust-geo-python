use crate::core::functions::{array2_to_polygon, point_poly_distance, polygons_to_array2};
use crate::error::{validate_non_empty, validate_point_dimensions, validate_xy_dimensions};
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::parallel::prelude::ParallelIterator;

use geo::{BooleanOps, Distance, Euclidean, LineString, Point, Polygon, unary_union};
use numpy::ndarray::{Array1, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{Bound, PyResult, Python};
use pyo3::prelude::*;

#[pyfunction(name = "point_polygon_distance")]
pub fn point_poly_distance_py<'py>(
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<f64> {
    validate_point_dimensions(x.shape())?;
    validate_xy_dimensions(y.shape())?;
    validate_non_empty(y.shape(), "polygon array")?;

    let x = x.as_array();
    let y = y.as_array();
    let distance = point_poly_distance(x, y);
    Ok(distance)
}

#[pyfunction(name = "points_polygon_distance")]
pub fn points_poly_distance_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_xy_dimensions(x.shape())?;
    validate_xy_dimensions(y.shape())?;
    validate_non_empty(x.shape(), "points array")?;
    validate_non_empty(y.shape(), "polygon array")?;

    let x = x.as_array();
    let y = y.as_array();
    let distances = x
        .axis_iter(Axis(0))
        .map(|p| point_poly_distance(p, y))
        .collect::<Array1<f64>>();
    Ok(distances.into_pyarray(py))
}

#[pyfunction(name = "polygon_polygon_distance")]
pub fn poly_poly_distance_py<'py>(
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<f64> {
    validate_xy_dimensions(x.shape())?;
    validate_xy_dimensions(y.shape())?;
    validate_non_empty(x.shape(), "first polygon array")?;
    validate_non_empty(y.shape(), "second polygon array")?;

    let path_x = x
        .as_array()
        .axis_iter(Axis(0))
        .map(|x| Point::new(x[0], x[1]))
        .collect::<LineString>();
    let path_y = y
        .as_array()
        .axis_iter(Axis(0))
        .map(|x| Point::new(x[0], x[1]))
        .collect::<LineString>();
    let distance = Euclidean.distance(&path_x, &path_y);
    Ok(distance)
}

#[pyfunction(name = "points_polygon_dist_mut")]
pub fn points_poly_distance_mut_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_xy_dimensions(x.shape())?;
    validate_xy_dimensions(y.shape())?;
    validate_non_empty(x.shape(), "points array")?;
    validate_non_empty(y.shape(), "polygon array")?;

    let x = x.as_array();
    let y = y.as_array();
    let distances_vec = x
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|p| point_poly_distance(p, y))
        .collect::<Vec<f64>>();
    Ok(distances_vec.into_pyarray(py))
}

#[pyfunction]
pub fn union_set_shapes<'py>(
    py: Python<'py>,
    pyarrays: Vec<(PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>)>,
) -> PyResult<Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)>> {
    if pyarrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "shape collection must be non-empty",
        ));
    }

    let polygons = pyarrays
        .iter()
        .map(|(x, ys)| array2_to_polygon(x, ys))
        .collect::<PyResult<Vec<Polygon>>>()?;
    let union = unary_union(&polygons);
    Ok(polygons_to_array2(
        py,
        union.iter().collect::<Vec<&Polygon>>(),
    ))
}

#[pyfunction]
pub fn intersection_shapes<'py>(
    py: Python<'py>,
    pyarray_x: (PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>),
    pyarray_y: (PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>),
) -> PyResult<Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)>> {
    let polygon_x = array2_to_polygon(&pyarray_x.0, &pyarray_x.1)?;
    let polygon_y = array2_to_polygon(&pyarray_y.0, &pyarray_y.1)?;
    let intersection = polygon_x.intersection(&polygon_y);
    Ok(polygons_to_array2(
        py,
        intersection.iter().collect::<Vec<&Polygon>>(),
    ))
}

#[pyfunction]
pub fn difference_shapes<'py>(
    py: Python<'py>,
    pyarray_x: (PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>),
    pyarray_y: (PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>),
) -> PyResult<Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)>> {
    let polygon_x = array2_to_polygon(&pyarray_x.0, &pyarray_x.1)?;
    let polygon_y = array2_to_polygon(&pyarray_y.0, &pyarray_y.1)?;
    let intersection = polygon_x.difference(&polygon_y);
    Ok(polygons_to_array2(
        py,
        intersection.iter().collect::<Vec<&Polygon>>(),
    ))
}
