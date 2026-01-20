mod enums;
mod functions;
mod pyfunctions;

#[pyo3::pymodule]
mod rust_geo_python {

    #[pymodule_export]
    use crate::enums::{
        RustLineString, RustMultiPoint, RustMultiPolygon, RustPoint, RustPointCollection,
        RustPolygon, Shape, point_in_polygon, union,
    };

    #[pymodule_export]
    use crate::pyfunctions::{
        difference_shapes, intersection_shapes, point_poly_distance_py,
        points_poly_distance_mut_py, points_poly_distance_py, poly_poly_distance_py,
        union_set_shapes,
    };
}
