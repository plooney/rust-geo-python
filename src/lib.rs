mod enums;
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

    fn point_poly_distance(x: ArrayView1<f64>, y: ArrayView2<f64>) -> f64 {
        let path = y
            .axis_iter(Axis(0))
            .map(|x| Point::new(x[0], x[1]))
            .collect::<LineString>();
        let point = Point::new(x[0], x[1]);
        let distance = Euclidean.distance(&point, &path);
        distance
    }

    #[pyfunction(name = "point_polygon_distance")]
    fn point_poly_distance_py<'py>(
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let distance = point_poly_distance(x, y);
        Ok(distance)
    }

    #[pyfunction(name = "points_polygon_distance")]
    fn points_poly_distance_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let distances = x
            .axis_iter(Axis(0))
            .map(|p| point_poly_distance(p, y))
            .collect::<Array1<f64>>();
        distances.into_pyarray(py)
    }

    #[pyfunction(name = "polygon_polygon_distance")]
    fn poly_poly_distance_py<'py>(
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> f64 {
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
        distance
    }

    #[pyfunction(name = "points_polygon_dist_mut")]
    fn points_poly_distance_mut_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let distances_vec = x
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|p| point_poly_distance(p, y))
            .collect::<Vec<f64>>();
        distances_vec.into_pyarray(py)
    }

    fn array2_to_linestring<'py>(x: &PyReadonlyArray2<'py, f64>) -> LineString {
        assert_eq!(x.shape()[1], 2, "Y dimension not equal to 2");
        let path = x
            .as_array()
            .axis_iter(Axis(0))
            .map(|y| Point::new(y[0], y[1]))
            .collect::<LineString>();
        path
    }

    fn array2_to_polygon<'py>(
        x: &PyReadonlyArray2<'py, f64>,
        ys: &Vec<PyReadonlyArray2<'py, f64>>,
    ) -> Polygon {
        let exterior = array2_to_linestring(&x);
        let interiors = ys
            .iter()
            .map(|y| array2_to_linestring(y))
            .collect::<Vec<LineString>>();
        Polygon::new(exterior, interiors)
    }

    fn linestring_to_pyarray2<'py>(py: Python<'py>, ls: &LineString) -> Bound<'py, PyArray2<f64>> {
        let arr = linestring_to_array(ls);
        let pyarray = PyArray2::from_owned_array(py, arr);
        pyarray
    }

    fn linestring_to_array<'py>(ls: &LineString) -> Array2<f64> {
        let n_points = ls.points().len();
        let mut arr = Array2::zeros((n_points, 2));
        let mut i = 0;
        ls.points().for_each(|p| {
            let (x, y) = p.x_y();
            arr[[i, 0]] = x;
            arr[[i, 1]] = y;
            i += 1;
        });
        arr
    }

    fn polygons_to_array2<'py>(
        py: Python<'py>,
        polygons: Vec<&Polygon>,
    ) -> Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)> {
        polygons
            .iter()
            .map(|p| {
                let ext = p.exterior();
                let ext_array = linestring_to_pyarray2(py, ext);
                let int_arrays = p
                    .interiors()
                    .iter()
                    .map(|ls| linestring_to_pyarray2(py, ls))
                    .collect::<Vec<Bound<'py, PyArray2<f64>>>>();
                (ext_array, int_arrays)
            })
            .collect::<Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)>>()
    }

    #[pyfunction]
    fn union_set_shapes<'py>(
        py: Python<'py>,
        pyarrays: Vec<(PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>)>,
    ) -> Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)> {
        let polygons = pyarrays
            .iter()
            .map(|(x, ys)| array2_to_polygon(x, ys))
            .collect::<Vec<Polygon>>();
        let union = unary_union(&polygons);
        polygons_to_array2(py, union.iter().collect::<Vec<&Polygon>>())
    }

    #[pyfunction]
    fn intersection_shapes<'py>(
        py: Python<'py>,
        pyarray_x: (PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>),
        pyarray_y: (PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>),
    ) -> Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)> {
        let polygon_x = array2_to_polygon(&pyarray_x.0, &pyarray_x.1);
        let polygon_y = array2_to_polygon(&pyarray_y.0, &pyarray_y.1);
        let intersection = polygon_x.intersection(&polygon_y);
        polygons_to_array2(py, intersection.iter().collect::<Vec<&Polygon>>())
    }

    #[pyfunction]
    fn difference_shapes<'py>(
        py: Python<'py>,
        pyarray_x: (PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>),
        pyarray_y: (PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>),
    ) -> Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)> {
        let polygon_x = array2_to_polygon(&pyarray_x.0, &pyarray_x.1);
        let polygon_y = array2_to_polygon(&pyarray_y.0, &pyarray_y.1);
        let intersection = polygon_x.difference(&polygon_y);
        polygons_to_array2(py, intersection.iter().collect::<Vec<&Polygon>>())
    }
}
