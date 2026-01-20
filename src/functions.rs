use geo::{Distance, Euclidean, LineString, Point, Polygon};
use ndarray::{ArrayView1, ArrayView2};
use numpy::ndarray::{Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};

use pyo3::{Bound, Python};

pub fn point_poly_distance(x: ArrayView1<f64>, y: ArrayView2<f64>) -> f64 {
    let path = y
        .axis_iter(Axis(0))
        .map(|x| Point::new(x[0], x[1]))
        .collect::<LineString>();
    let point = Point::new(x[0], x[1]);
    let distance = Euclidean.distance(&point, &path);
    distance
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

pub fn array2_to_polygon<'py>(
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

pub fn polygons_to_array2<'py>(
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
