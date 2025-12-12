mod enums;
mod functions;
mod pyfunctions;
pub use enums::{RustLineString, RustMultiPoint, RustMultiPolygon, RustPoint, RustPolygon, Shape};

#[pyo3::pymodule]
mod rust_geo_python {
    use ndarray::parallel::prelude::ParallelIterator;
    use numpy::ndarray::{Array1, Array2, Axis};
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
    };

    use geo::orient::{Direction, Orient};
    use geo::{
        Area, BooleanOps, Buffer, Contains, ContainsProperly, Distance, Euclidean,
        HausdorffDistance, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
        Simplify, unary_union,
    };
    use ndarray::parallel::prelude::IntoParallelIterator;
    use ndarray::{ArrayView1, ArrayView2};
    use pyo3::{Bound, PyResult, Python};
    use pyo3::{IntoPyObjectExt, prelude::*};

    #[pymodule_export]
    use super::{RustLineString, RustMultiPoint, RustMultiPolygon, RustPoint, RustPolygon, Shape};

    #[pymodule_export]
    use super::pyfunctions::union;
}
