use ndarray::parallel::prelude::ParallelIterator;
use numpy::ndarray::{Array1, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use geo::{Distance, Euclidean, LineString, Point, Polygon, unary_union};
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::{ArrayView1, ArrayView2};
use pyo3::{
    Bound, PyResult, Python, pymodule,
    types::{PyDict, PyList, PyModule},
};

#[pymodule]
#[pyo3(name = "rust_geo_python")]
fn polygon<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn point_poly_distance(x: ArrayView1<f64>, y: ArrayView2<f64>) -> f64 {
        let path = y
            .axis_iter(Axis(0))
            .map(|x| Point::new(x[0], x[1]))
            .collect::<LineString>();
        let point = Point::new(x[0], x[1]);
        let distance = Euclidean.distance(&point, &path);
        distance
    }

    #[pyfn(m)]
    #[pyo3(name = "point_polygon_distance")]
    fn point_poly_distance_py<'py>(
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let distance = point_poly_distance(x, y);
        Ok(distance)
    }

    #[pyfn(m)]
    #[pyo3(name = "points_polygon_distance")]
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
    #[pyfn(m)]
    #[pyo3(name = "polygon_polygon_distance")]
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

    #[pyfn(m)]
    #[pyo3(name = "points_polygon_dist_mut")]
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

    fn polygon_to_array2<'py>(polygons: &Vec<Polygon>) -> () {}

    #[pyfn(m)]
    #[pyo3(name = "union_set_shapes")]
    fn union_set_shapes<'py>(
        py: Python<'py>,
        pyarrays: Vec<(PyReadonlyArray2<'py, f64>, Vec<PyReadonlyArray2<'py, f64>>)>,
        //) -> PyList<(PyArray2, PyList<PyArray2>)> {
    ) -> () {
        let polygons = pyarrays
            .iter()
            .map(|(x, ys)| array2_to_polygon(x, ys))
            .collect::<Vec<Polygon>>();
        let union = unary_union(&polygons);
    }

    Ok(())
}
