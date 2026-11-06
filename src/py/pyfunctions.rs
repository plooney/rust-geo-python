use crate::core::circle_intersection::{BoundaryRegion, circle_areas_inside_regions};
use crate::core::functions::{array2_to_polygon, point_poly_distance, polygons_to_array2};
use crate::error::{validate_non_empty, validate_point_dimensions, validate_xy_dimensions};
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::parallel::prelude::ParallelIterator;

use geo::{BooleanOps, Distance, Euclidean, LineString, Point, Polygon, unary_union};
use numpy::ndarray::{Array1, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{Bound, PyResult, Python};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyIterator, PyList, PyListMethods};

fn extract_coordinate_ring(ring: &Bound<'_, PyAny>, label: &str) -> PyResult<Vec<[f64; 2]>> {
    let coordinates = ring
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!(
            "{label} must be a float64 array with shape (n, 2)"
        )))?;
    validate_xy_dimensions(coordinates.shape())?;
    validate_non_empty(coordinates.shape(), label)?;

    let points = coordinates
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1]])
        .collect::<Vec<_>>();
    if points.len() < 3 || points.iter().flatten().any(|coordinate| !coordinate.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} must contain at least three finite coordinates"
        )));
    }
    Ok(points)
}

fn extract_boundary_regions(boundaries: &Bound<'_, PyAny>) -> PyResult<Vec<BoundaryRegion>> {
    let boundary_list = boundaries.cast::<PyList>().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("boundaries_as_shapes must be a list of dicts")
    })?;
    let mut regions = Vec::with_capacity(boundary_list.len());
    let iterator = PyIterator::from_object(boundary_list.as_any())?;

    for (region_index, item) in iterator.enumerate() {
        let item = item?;
        let boundary = item.cast::<PyDict>().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "boundary {region_index} must be a dict"
            ))
        })?;
        let exterior_value = boundary
            .get_item("outer boundary")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!(
                "boundary {region_index} is missing 'outer boundary'"
            )))?;
        let exterior = extract_coordinate_ring(&exterior_value, "outer boundary")?;

        let mut interiors = Vec::new();
        if let Some(interiors_value) = boundary.get_item("interior boundary")? {
            if !interiors_value.is_none() {
                let holes = PyIterator::from_object(interiors_value.as_any()).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "boundary {region_index} 'interior boundary' must be a sequence of rings"
                    ))
                })?;
                for hole in holes {
                    interiors.push(extract_coordinate_ring(&hole?, "interior boundary ring")?);
                }
            }
        }

        regions.push(BoundaryRegion {
            exterior,
            interiors,
        });
    }

    if regions.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boundaries_as_shapes must contain at least one region",
        ));
    }
    Ok(regions)
}

/// Update each circle's observable area after clipping it to included regions
/// and subtracting their interior holes.
#[pyfunction]
pub fn update_circle_areas_using_boundaries<'py>(
    point_locations: PyReadonlyArray2<'py, f64>,
    areas: Bound<'py, PyArray1<f64>>,
    distance_to_boundary: PyReadonlyArray2<'py, f64>,
    radius: f64,
    remain_within_connected_component: bool,
    boundaries_as_shapes: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_xy_dimensions(point_locations.shape())?;
    if distance_to_boundary.shape().len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "distance_to_boundary must be a 2D array",
        ));
    }
    if !radius.is_finite() || radius < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "radius must be finite and non-negative",
        ));
    }

    let centers = point_locations
        .as_array()
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1]])
        .collect::<Vec<_>>();
    if centers.iter().flatten().any(|coordinate| !coordinate.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "point_locations must contain only finite coordinates",
        ));
    }
    if distance_to_boundary.shape()[0] != centers.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "distance_to_boundary must have one row per point",
        ));
    }
    if areas.shape().len() != 1 || areas.shape()[0] != centers.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "areas must have one value per point",
        ));
    }

    let regions = extract_boundary_regions(&boundaries_as_shapes)?;
    let updated = circle_areas_inside_regions(
        &centers,
        radius,
        &regions,
        remain_within_connected_component,
    )?;

    let mut writable_areas = areas.try_readwrite().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("areas must be writable and not already borrowed")
    })?;
    for (index, area) in updated.into_iter().enumerate() {
        writable_areas.as_array_mut()[index] = area;
    }
    drop(writable_areas);

    Ok(areas)
}

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
