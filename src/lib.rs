#[macro_use]
mod macros;
mod enums;
mod functions;
mod pyfunctions;
mod tiling;

#[pyo3::pymodule]
mod rust_geo_python {

    #[pymodule_export]
    use crate::enums::{
        RustGeomVecCollection, RustLineString, RustMultiPoint, RustMultiPolygon, RustPoint,
        RustPolygon, RustShape, intersect_tile, point_in_polygon, union, union_with_adapter,
    };

    #[pymodule_export]
    use crate::pyfunctions::{
        difference_shapes, intersection_shapes, point_poly_distance_py,
        points_poly_distance_mut_py, points_poly_distance_py, poly_poly_distance_py,
        union_set_shapes,
    };
}
