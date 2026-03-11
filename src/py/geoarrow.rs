#[cfg(feature = "geoarrow")]
use arrow_array::ArrayRef;
#[cfg(feature = "geoarrow")]
use geoarrow::array::{GeometryBuilder, PolygonBuilder};
#[cfg(feature = "geoarrow")]
use geoarrow::datatypes::{CoordType, Dimension, GeometryType, Metadata, PolygonType};
#[cfg(feature = "geoarrow")]
use geoarrow_array::GeoArrowArray;
#[cfg(feature = "geoarrow")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "geoarrow")]
use pyo3::{Py, PyResult, Python};
#[cfg(feature = "geoarrow")]
use pyo3_arrow::PyArray;

#[cfg(feature = "geoarrow")]
use crate::py::enums::{RustGeomVecCollection, RustPolygon};

#[cfg(feature = "geoarrow")]
fn geoarrow_array_to_py<'py>(py: Python<'py>, array: ArrayRef) -> PyResult<Py<PyArray>> {
    Py::new(py, PyArray::from_array_ref(array))
}

#[cfg(feature = "geoarrow")]
#[pyo3::pyfunction]
pub fn geoarray_from_polygons<'py>(
    py: Python<'py>,
    rust_polygons: Vec<RustPolygon>,
) -> PyResult<Py<PyArray>> {
    if rust_polygons.is_empty() {
        return Err(PyValueError::new_err("polygon list must be non-empty"));
    }

    let typ = PolygonType::new(Dimension::XY, Metadata::default().into())
        .with_coord_type(CoordType::Interleaved);
    let polygons = rust_polygons
        .iter()
        .map(|p| p.polygon_ref())
        .collect::<Vec<_>>();
    let array = PolygonBuilder::from_polygons(&polygons, typ).finish();
    geoarrow_array_to_py(py, array.to_array_ref())
}

#[cfg(feature = "geoarrow")]
#[pyo3::pyfunction]
pub fn geoarray_from_geom_vec_collection<'py>(
    py: Python<'py>,
    geoms: &RustGeomVecCollection,
) -> PyResult<Py<PyArray>> {
    if geoms.geoms_ref().is_empty() {
        return Err(PyValueError::new_err(
            "geometry collection must be non-empty",
        ));
    }

    let typ = GeometryType::new(Metadata::default().into()).with_coord_type(CoordType::Interleaved);
    let mut builder = GeometryBuilder::new(typ);
    for geom in geoms.geoms_ref().iter() {
        builder
            .push_geometry(Some(geom))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    let array = builder.finish();
    geoarrow_array_to_py(py, array.to_array_ref())
}
