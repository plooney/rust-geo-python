#[macro_use]
mod macros;
mod core;
mod error;
mod py;

#[pyo3::pymodule]
mod rust_geo_python {

    #[pymodule_export]
    use crate::py::enums::{
        RustGeomVecCollection, RustGeometryCollection, RustLine, RustLineString,
        RustMultiLineString, RustMultiPoint, RustMultiPolygon, RustPoint, RustPolygon, RustRect,
        RustShape, RustTriangle, from_wkt, intersect_tile, point_in_polygon, union,
        union_with_adapter,
    };

    #[pymodule_export]
    use crate::py::pyfunctions::{
        difference_shapes, intersection_shapes, point_poly_distance_py,
        points_poly_distance_mut_py, points_poly_distance_py, poly_poly_distance_py,
        union_set_shapes,
    };
}
