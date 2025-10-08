use ndarray::parallel::prelude::ParallelIterator;
use numpy::ndarray::{Axis, Array1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};

use geo::{Point, LineString, Distance, Euclidean};
use ndarray::{ArrayView1, ArrayView2};
use ndarray::parallel::prelude::IntoParallelIterator;
use pyo3::{pymodule, types::{PyModule}, Bound, PyResult, Python};

#[pymodule]
#[pyo3(name = "rust_geo_python")]
fn polygon<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {

    fn point_poly_distance(x: ArrayView1<f64>, y: ArrayView2<f64>) -> f64 {
        let path = y.axis_iter(Axis(0))
            .map(|x| Point::new(x[0], x[1]))
            .collect::<LineString>();
        //let polygon = Polygon::new(path, vec![]);
        let point = Point::new(x[0], x[1]);
        let distance = Euclidean.distance(&point,&path);
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
        let distances = x.axis_iter(Axis(0))
            .map(|p| point_poly_distance(p,y))
            .collect::<Array1<f64>>();
        distances.into_pyarray(py)
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
        let distances_vec = x.axis_iter(Axis(0)).into_par_iter()
            .map(|p| point_poly_distance(p,y))
            .collect::<Vec<f64>>();
        distances_vec.into_pyarray(py)
    }

    Ok(())
}