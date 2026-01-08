use ndarray::parallel::prelude::ParallelIterator;
use numpy::ToPyArray;
use numpy::ndarray::{Array1, Array2, Axis};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
};

use geo::orient::{Direction, Orient};
use geo::{
    Area, BooleanOps, Buffer, Contains, ContainsProperly, Distance, Euclidean, HausdorffDistance,
    Intersects, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, Simplify,
    unary_union,
};
use pyo3::exceptions::PyTypeError;
use pyo3::{Bound, PyResult, Python};
use pyo3::{IntoPyObjectExt, prelude::*};
use std::sync::Arc;
use wkt::ToWkt;

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

fn multipoint_to_array<'py>(mp: &MultiPoint) -> Array2<f64> {
    let n_points = mp.len();
    let mut arr = Array2::zeros((n_points, 2));
    let mut i = 0;
    mp.iter().for_each(|p| {
        let (x, y) = p.x_y();
        arr[[i, 0]] = x;
        arr[[i, 1]] = y;
        i += 1;
    });
    arr
}

fn polygon_to_array2<'py>(
    py: Python<'py>,
    polygon: &Polygon,
) -> (Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>) {
    let ext = polygon.exterior();
    let ext_array = linestring_to_pyarray2(py, ext);
    let int_arrays = polygon
        .interiors()
        .iter()
        .map(|ls| linestring_to_pyarray2(py, ls))
        .collect::<Vec<Bound<'py, PyArray2<f64>>>>();
    (ext_array, int_arrays)
}

#[derive(Clone)]
pub enum Shapes {
    Point(Arc<Point>),
    MultiPoint(Arc<MultiPoint>),
    LineString(Arc<LineString>),
    MultiLineString(Arc<MultiLineString>),
    Polygon(Arc<Polygon>),
    MultiPolygon(Arc<MultiPolygon>),
}

#[pyclass(subclass)]
#[derive(Clone)]
pub struct Shape {
    inner: Shapes,
}

#[pyclass(extends=Shape)]
#[derive(Clone)]
pub struct RustPoint {
    point: Arc<Point>,
}
#[pyclass(extends=Shape)]
#[derive(Clone)]
pub struct RustMultiPoint {
    multipoint: Arc<MultiPoint>,
}
#[pyclass(extends=Shape)]
pub struct RustLineString {
    linestring: Arc<LineString>,
}
#[pyclass(extends=Shape)]
#[derive(Clone)]
pub struct RustPolygon {
    polygon: Arc<Polygon>,
}
#[pyclass(extends=Shape)]
pub struct RustMultiLineString {
    multilinestring: Arc<MultiLineString>,
}

#[pyclass(extends=Shape)]
pub struct RustMultiPolygon {
    multipolygon: Arc<MultiPolygon>,
}

#[pymethods]
impl RustLineString {
    #[new]
    fn new(x: PyReadonlyArray2<f64>) -> (Self, Shape) {
        let ls = array2_to_linestring(&x);
        let ls_arc = Arc::new(ls);
        (
            RustLineString {
                linestring: ls_arc.clone(),
            },
            Shape {
                inner: Shapes::LineString(ls_arc),
            },
        )
    }

    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = linestring_to_array(&self.linestring);
        let pyarray = PyArray2::from_owned_array(py, arr);
        Ok(pyarray)
    }
}

#[pymethods]
impl RustMultiPoint {
    #[new]
    fn new(x: PyReadonlyArray2<f64>) -> (Self, Shape) {
        let ls = array2_to_linestring(&x);

        let multipoint = ls.points().collect::<MultiPoint>();
        let multipoint_arc = Arc::new(multipoint);

        (
            RustMultiPoint {
                multipoint: multipoint_arc.clone(),
            },
            Shape {
                inner: Shapes::MultiPoint(multipoint_arc),
            },
        )
    }

    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = multipoint_to_array(&self.multipoint);
        let pyarray = PyArray2::from_owned_array(py, arr);
        Ok(pyarray)
    }
}

#[pymethods]
impl RustPoint {
    #[new]
    fn new(x: f64, y: f64) -> (Self, Shape) {
        let point = Point::new(x, y);
        let point_arc = Arc::new(point);
        (
            RustPoint {
                point: point_arc.clone(),
            },
            Shape {
                inner: Shapes::Point(point_arc),
            },
        )
    }

    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let xy = self.point.x_y();
        xy.into_bound_py_any(py)
    }
}

#[pymethods]
impl RustPolygon {
    #[new]
    fn new(x: PyReadonlyArray2<f64>, ys: Vec<PyReadonlyArray2<f64>>) -> (Self, Shape) {
        let polygon = array2_to_polygon(&x, &ys).orient(Direction::Default);
        let polygon_arc = Arc::new(polygon);
        (
            RustPolygon {
                polygon: polygon_arc.clone(),
            },
            Shape {
                inner: Shapes::Polygon(polygon_arc),
            },
        )
    }

    fn xy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)> {
        Ok(polygon_to_array2(py, self.polygon.as_ref()))
    }

    fn simplify<'py>(&self, py: Python<'py>, epsilon: f64) -> PyResult<Py<PyAny>> {
        let simple_polygon = self.polygon.simplify(epsilon);
        let polygon_arc = Arc::new(simple_polygon);
        let initializer: PyClassInitializer<RustPolygon> = PyClassInitializer::from((
            RustPolygon {
                polygon: polygon_arc.clone(),
            },
            Shape {
                inner: Shapes::Polygon(polygon_arc),
            },
        ));
        Ok(Py::new(py, initializer)?.into_any())
    }

    fn area(&self) -> f64 {
        self.polygon.signed_area()
    }
}

#[pymethods]
impl RustMultiLineString {
    #[new]
    fn new(ys: Vec<PyReadonlyArray2<f64>>) -> (Self, Shape) {
        let lss = ys
            .iter()
            .map(|x| array2_to_linestring(x))
            .collect::<MultiLineString>();
        let lss_arc = Arc::new(lss);
        (
            RustMultiLineString {
                multilinestring: lss_arc.clone(),
            },
            Shape {
                inner: Shapes::MultiLineString(lss_arc),
            },
        )
    }

    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let pyarrays = self
            .multilinestring
            .iter()
            .map(|x| linestring_to_pyarray2(py, x))
            .collect::<Vec<Bound<'py, PyArray2<f64>>>>();
        Ok(pyarrays)
    }
}

#[pymethods]
impl RustMultiPolygon {
    #[new]
    fn new(pyarrays: Vec<(PyReadonlyArray2<f64>, Vec<PyReadonlyArray2<f64>>)>) -> (Self, Shape) {
        let polygons = pyarrays
            .iter()
            .map(|(x, ys)| array2_to_polygon(&x, &ys).orient(Direction::Default))
            .collect::<Vec<Polygon>>();
        let multipolygon = MultiPolygon(polygons);
        let multipolygon_arc = Arc::new(multipolygon);
        (
            RustMultiPolygon {
                multipolygon: multipolygon_arc.clone(),
            },
            Shape {
                inner: Shapes::MultiPolygon(multipolygon_arc),
            },
        )
    }

    fn xy<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)>> {
        let result_vec = self
            .multipolygon
            .iter()
            .map(|x| polygon_to_array2(py, x))
            .collect::<Vec<(Bound<'py, PyArray2<f64>>, Vec<Bound<'py, PyArray2<f64>>>)>>();
        Ok(result_vec)
    }

    fn simplify<'py>(&self, py: Python<'py>, epsilon: f64) -> PyResult<Py<PyAny>> {
        let simple_polygon = self.multipolygon.simplify(epsilon);
        let multipolygon_arc = Arc::new(simple_polygon);
        let initializer: PyClassInitializer<RustMultiPolygon> = PyClassInitializer::from((
            RustMultiPolygon {
                multipolygon: multipolygon_arc.clone(),
            },
            Shape {
                inner: Shapes::MultiPolygon(multipolygon_arc),
            },
        ));
        Ok(Py::new(py, initializer)?.into_any())
    }

    fn area(&self) -> f64 {
        self.multipolygon.signed_area()
    }
}

#[pymethods]
impl Shape {
    fn distance(&self, rhs: &Shape) -> f64 {
        match (&self.inner, &rhs.inner) {
            (Shapes::Point(p), Shapes::Point(q)) => Euclidean.distance(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::Point(q)) => Euclidean.distance(p.as_ref(), q.as_ref()),
            (Shapes::Point(p), Shapes::LineString(q)) => Euclidean.distance(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::LineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::Point(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::LineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiLineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiLineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::LineString(p), Shapes::MultiLineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Point(q)) => Euclidean.distance(p.as_ref(), q.as_ref()),
            (Shapes::Polygon(p), Shapes::LineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiLineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Polygon(q)) => Euclidean.distance(p.as_ref(), q.as_ref()),
            (Shapes::Point(p), Shapes::Polygon(q)) => Euclidean.distance(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::Polygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::Polygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Point(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::LineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiLineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiPolygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::LineString(p), Shapes::MultiPolygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::Point(q)) => Euclidean.distance(p.as_ref(), q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::LineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::Polygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiPoint(q)) => Euclidean.distance(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPoint(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPoint(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => {
                Euclidean.distance(p.as_ref(), q.as_ref())
            }
        }
    }

    fn hausdorff_distance(&self, rhs: &Shape) -> f64 {
        match (&self.inner, &rhs.inner) {
            (Shapes::Point(p), Shapes::Point(q)) => p.as_ref().hausdorff_distance(q.as_ref()),
            (Shapes::LineString(p), Shapes::Point(q)) => p.as_ref().hausdorff_distance(q.as_ref()),
            (Shapes::Point(p), Shapes::LineString(q)) => p.as_ref().hausdorff_distance(q.as_ref()),
            (Shapes::LineString(p), Shapes::LineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::Point(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::LineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiLineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiLineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::LineString(p), Shapes::MultiLineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Point(q)) => p.as_ref().hausdorff_distance(q.as_ref()),
            (Shapes::Polygon(p), Shapes::LineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiLineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Polygon(q)) => p.as_ref().hausdorff_distance(q.as_ref()),
            (Shapes::Point(p), Shapes::Polygon(q)) => p.as_ref().hausdorff_distance(q.as_ref()),
            (Shapes::LineString(p), Shapes::Polygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::Polygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Point(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::LineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiLineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::LineString(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::Point(q)) => p.as_ref().hausdorff_distance(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::LineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::Polygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiPoint(q)) => p.as_ref().hausdorff_distance(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPoint(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPoint(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => {
                p.as_ref().hausdorff_distance(q.as_ref())
            }
        }
    }

    fn contains(&self, rhs: &Shape) -> bool {
        match (&self.inner, &rhs.inner) {
            (Shapes::Point(p), Shapes::Point(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::LineString(p), Shapes::Point(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Point(p), Shapes::LineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::LineString(p), Shapes::LineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Point(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::LineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiLineString(q)) => {
                p.as_ref().contains(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiLineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiLineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Polygon(p), Shapes::Point(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Polygon(p), Shapes::LineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Polygon(p), Shapes::MultiLineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Polygon(p), Shapes::Polygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Point(p), Shapes::Polygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::LineString(p), Shapes::Polygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Polygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Point(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::LineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiLineString(q)) => {
                p.as_ref().contains(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Point(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().contains(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Point(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::LineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Polygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Point(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::Polygon(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
        }
    }

    fn contains_properly(&self, rhs: &Shape) -> bool {
        match (&self.inner, &rhs.inner) {
            (Shapes::Point(p), Shapes::Point(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::LineString(p), Shapes::Point(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::Point(p), Shapes::LineString(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::LineString(p), Shapes::LineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::Point(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::LineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiLineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiLineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::LineString(p), Shapes::MultiLineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Point(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::Polygon(p), Shapes::LineString(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::Polygon(p), Shapes::MultiLineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Polygon(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::Point(p), Shapes::Polygon(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::LineString(p), Shapes::Polygon(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Polygon(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Point(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::LineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiLineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiPolygon(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::Point(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::LineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::Polygon(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiPoint(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPoint(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPoint(q)) => p.as_ref().contains_properly(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => {
                p.as_ref().contains_properly(q.as_ref())
            }
        }
    }

    fn intersects(&self, rhs: &Shape) -> bool {
        match (&self.inner, &rhs.inner) {
            (Shapes::Point(p), Shapes::Point(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::LineString(p), Shapes::Point(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::Point(p), Shapes::LineString(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::LineString(p), Shapes::LineString(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Point(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::LineString(q)) => {
                p.as_ref().intersects(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiLineString(q)) => {
                p.as_ref().intersects(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiLineString(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiLineString(q)) => {
                p.as_ref().intersects(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Point(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::Polygon(p), Shapes::LineString(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::Polygon(p), Shapes::MultiLineString(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::Polygon(p), Shapes::Polygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::Point(p), Shapes::Polygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::LineString(p), Shapes::Polygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Polygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Point(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::LineString(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiLineString(q)) => {
                p.as_ref().intersects(q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::Point(p), Shapes::MultiPolygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPolygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().intersects(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Point(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::LineString(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => {
                p.as_ref().intersects(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::Polygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::Point(p), Shapes::MultiPoint(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPoint(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => {
                p.as_ref().intersects(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPoint(q)) => p.as_ref().intersects(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => p.as_ref().intersects(q.as_ref()),
        }
    }

    fn to_wkt(&self) -> String {
        match &self.inner {
            Shapes::Point(p) => p.as_ref().wkt_string(),
            Shapes::MultiPoint(p) => p.as_ref().wkt_string(),
            Shapes::LineString(p) => p.wkt_string(),
            Shapes::MultiLineString(p) => p.wkt_string(),
            Shapes::MultiPolygon(p) => p.wkt_string(),
            Shapes::Polygon(p) => p.wkt_string(),
        }
    }

    fn buffer<'py>(&self, py: Python<'py>, radius: f64) -> PyResult<Py<PyAny>> {
        let polygons = match &self.inner {
            Shapes::Point(p) => p.buffer(radius),
            Shapes::MultiPoint(p) => p.buffer(radius),
            Shapes::LineString(p) => p.buffer(radius),
            Shapes::MultiLineString(p) => p.buffer(radius),
            Shapes::MultiPolygon(p) => p.buffer(radius),
            Shapes::Polygon(p) => p.buffer(radius),
        };
        let multipolygon_arc = Arc::new(polygons);
        let initializer: PyClassInitializer<RustMultiPolygon> = PyClassInitializer::from((
            RustMultiPolygon {
                multipolygon: multipolygon_arc.clone(),
            },
            Shape {
                inner: Shapes::MultiPolygon(multipolygon_arc),
            },
        ));
        Ok(Py::new(py, initializer)?.into_any())
    }

    fn intersection<'py>(&self, py: Python<'py>, rhs: &Shape) -> PyResult<Py<PyAny>> {
        match (&self.inner, &rhs.inner) {
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => {
                mpg_to_pyany(py, p.as_ref().intersection(q.as_ref()))
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => {
                mpg_to_pyany(py, p.as_ref().intersection(q.as_ref()))
            }
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => {
                mpg_to_pyany(py, p.as_ref().intersection(q.as_ref()))
            }
            (_, _) => Err(PyTypeError::new_err("Not implemented yet")),
        }
    }

    fn union<'py>(&self, py: Python<'py>, rhs: &Shape) -> PyResult<Py<PyAny>> {
        match (&self.inner, &rhs.inner) {
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => {
                mpg_to_pyany(py, p.as_ref().union(q.as_ref()))
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => {
                mpg_to_pyany(py, p.as_ref().union(q.as_ref()))
            }
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => {
                mpg_to_pyany(py, p.as_ref().union(q.as_ref()))
            }
            (_, _) => Err(PyTypeError::new_err("Not implemented yet")),
        }
    }

    fn difference<'py>(&self, py: Python<'py>, rhs: &Shape) -> PyResult<Py<PyAny>> {
        match (&self.inner, &rhs.inner) {
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => {
                mpg_to_pyany(py, p.as_ref().difference(q.as_ref()))
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => {
                mpg_to_pyany(py, p.as_ref().difference(q.as_ref()))
            }
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => {
                mpg_to_pyany(py, p.as_ref().difference(q.as_ref()))
            }
            (_, _) => Err(PyTypeError::new_err("Not implemented yet")),
        }
    }

    fn boundary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            Shapes::Point(_) => Ok(py.None()),
            Shapes::MultiPoint(_) => Ok(py.None()),
            Shapes::LineString(p) => {
                let multipoint = p.points().collect::<MultiPoint>();
                let multipoint_arc = Arc::new(multipoint);
                let initializer: PyClassInitializer<RustMultiPoint> = PyClassInitializer::from((
                    RustMultiPoint {
                        multipoint: multipoint_arc.clone(),
                    },
                    Shape {
                        inner: Shapes::MultiPoint(multipoint_arc),
                    },
                ));
                Ok(Py::new(py, initializer)?.into_any())
            }
            Shapes::MultiLineString(p) => {
                let points: Vec<Point<f64>> = Vec::new();

                let multipoint = MultiPoint::new(p.iter().fold(points, |mut points, x| {
                    points.extend(&x.clone().into_points());
                    points
                }));

                let multipoint_arc = Arc::new(multipoint);
                let initializer: PyClassInitializer<RustMultiPoint> = PyClassInitializer::from((
                    RustMultiPoint {
                        multipoint: multipoint_arc.clone(),
                    },
                    Shape {
                        inner: Shapes::MultiPoint(multipoint_arc),
                    },
                ));
                Ok(Py::new(py, initializer)?.into_any())
            }
            Shapes::MultiPolygon(p) => {
                let lss: Vec<LineString<f64>> = Vec::new();

                let multilinestring = MultiLineString::new(p.iter().fold(lss, |mut lss, x| {
                    lss.push(x.exterior().clone());
                    lss.extend(x.interiors().to_vec());
                    lss
                }));
                let multilinestring_arc = Arc::new(multilinestring);

                let initializer: PyClassInitializer<RustMultiLineString> =
                    PyClassInitializer::from((
                        RustMultiLineString {
                            multilinestring: multilinestring_arc.clone(),
                        },
                        Shape {
                            inner: Shapes::MultiLineString(multilinestring_arc),
                        },
                    ));
                Ok(Py::new(py, initializer)?.into_any())
            }
            Shapes::Polygon(p) => {
                let mut lss: Vec<LineString<f64>> = Vec::new();
                lss.push(p.exterior().clone());
                lss.extend(p.interiors().to_vec());

                let multilinestring = MultiLineString::new(lss);

                let multilinestring_arc = Arc::new(multilinestring);

                let initializer: PyClassInitializer<RustMultiLineString> =
                    PyClassInitializer::from((
                        RustMultiLineString {
                            multilinestring: multilinestring_arc.clone(),
                        },
                        Shape {
                            inner: Shapes::MultiLineString(multilinestring_arc),
                        },
                    ));
                Ok(Py::new(py, initializer)?.into_any())
            }
        }
    }
}

#[pyfunction(name = "intersection")]
pub fn intersection<'py>(
    py: Python<'py>,
    polygon_lhs: &RustPolygon,
    polygon_rhs: &RustPolygon,
) -> PyResult<Py<PyAny>> {
    let intersection = polygon_lhs
        .polygon
        .intersection(polygon_rhs.polygon.as_ref());
    let multipolygon_arc = Arc::new(intersection);
    let initializer: PyClassInitializer<RustMultiPolygon> = PyClassInitializer::from((
        RustMultiPolygon {
            multipolygon: multipolygon_arc.clone(),
        },
        Shape {
            inner: Shapes::MultiPolygon(multipolygon_arc),
        },
    ));
    Ok(Py::new(py, initializer)?.into_any())
}

pub fn mpg_to_pyany<'py>(py: Python<'py>, mpg: MultiPolygon) -> PyResult<Py<PyAny>> {
    let multipolygon_arc = Arc::new(mpg);
    let initializer: PyClassInitializer<RustMultiPolygon> = PyClassInitializer::from((
        RustMultiPolygon {
            multipolygon: multipolygon_arc.clone(),
        },
        Shape {
            inner: Shapes::MultiPolygon(multipolygon_arc),
        },
    ));
    Ok(Py::new(py, initializer)?.into_any())
}

#[pyfunction]
pub fn union<'py>(py: Python<'py>, rust_polygons: Vec<RustPolygon>) -> PyResult<Py<PyAny>> {
    let polygons = rust_polygons
        .iter()
        .map(|x| x.polygon.as_ref())
        .collect::<Vec<&Polygon>>();
    let union = unary_union(polygons);
    let multipolygon_arc = Arc::new(union);
    let initializer: PyClassInitializer<RustMultiPolygon> = PyClassInitializer::from((
        RustMultiPolygon {
            multipolygon: multipolygon_arc.clone(),
        },
        Shape {
            inner: Shapes::MultiPolygon(multipolygon_arc),
        },
    ));
    Ok(Py::new(py, initializer)?.into_any())
}

#[pyfunction]
pub fn point_in_polygon<'py>(rust_point: RustPoint, rust_polygon: RustPolygon) -> PyResult<bool> {
    let point = rust_point.point.as_ref();
    let polygon = rust_polygon.polygon;
    let is_in = polygon.as_ref().contains(point);
    Ok(is_in)
}

#[pyclass(subclass)]
#[derive(Clone)]
pub struct RustPointCollection {
    points: Vec<Point>,
}

fn array2_to_points<'py>(x: &PyReadonlyArray2<'py, f64>) -> Vec<Point<f64>> {
    assert_eq!(x.shape()[1], 2, "Y dimension not equal to 2");
    let points = x
        .as_array()
        .axis_iter(Axis(0))
        .map(|y| Point::new(y[0], y[1]))
        .collect::<Vec<Point<f64>>>();
    points
}

fn points_to_array<'py>(points: &Vec<Point<f64>>) -> Array2<f64> {
    let n_points = points.len();
    let mut arr = Array2::zeros((n_points, 2));
    let mut i = 0;
    points.iter().for_each(|p| {
        let (x, y) = p.x_y();
        arr[[i, 0]] = x;
        arr[[i, 1]] = y;
        i += 1;
    });
    arr
}

#[pymethods]
impl RustPointCollection {
    #[new]
    fn new(x: PyReadonlyArray2<f64>) -> Self {
        let points = array2_to_points(&x);
        RustPointCollection { points: points }
    }

    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = points_to_array(&self.points);
        let pyarray = PyArray2::from_owned_array(py, arr);
        Ok(pyarray)
    }

    fn distance_ls<'py>(
        &self,
        py: Python<'py>,
        ls: &RustLineString,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let arr = self
            .points
            .iter()
            .map(|p| Euclidean.distance(p, ls.linestring.as_ref()))
            .collect::<Array1<f64>>();
        let pyarray = PyArray1::from_owned_array(py, arr);
        Ok(pyarray)
    }

    fn distance_points<'py>(
        &self,
        py: Python<'py>,
        other: &RustPointCollection,
    ) -> Bound<'py, PyArray2<f64>> {
        let n_points = self.points.len();
        let n_points_other = other.points.len();
        let index = (0..n_points)
            //.into_par_iter()
            .flat_map(|i| (0..n_points_other).map(move |j| (i, j)));

        let shape = (n_points, n_points_other);
        let mut arr = Array2::zeros(shape);
        //let pyarray2 = PyArray2::<f64>::zeros(py, shape, false);
        //

        index.for_each(|(i, j)| {
            let a = self.points.get(i);
            let b = other.points.get(j);
            if let (Some(p), Some(q)) = (a, b) {
                let d = Euclidean.distance(p, q);
                arr[[i, j]] = d;
            }
        });

        arr.to_pyarray(py)
    }

    //fn distance_points<'py>(
    //    &self,
    //    py: Python<'py>,
    //    other: &RustPointCollection,
    //) -> Bound<'py, PyArray2<f64>> {
    //    let n = self.points.len();
    //    let m = other.points.len();

    //    let mut ds = Vec::with_capacity(n * m);

    //    ds.extend(self.points.iter().flat_map(|p| {
    //        other.points.iter().map(move |q| {
    //            let d = (((p.x() - q.x()) * (p.x() - q.x())) + ((p.y() - q.y()) * (p.y() - q.y())))
    //                .sqrt();
    //            d
    //        })
    //    }));

    //    let arr = Array2::from_shape_vec((n, m), ds).unwrap();
    //    arr.into_pyarray(py)
    //}
}

//#[pyclass(subclass)]
//#[derive(Clone)]
//pub struct RustLineStringCollection {
//    lss: Vec<LineString>,
//}
//
//#[pymethods]
//impl RustLineStringCollection {
//    #[new]
//    fn new(x: Vec<PyReadonlyArray2<f64>>) -> Self {
//        let lss = vec_array2_to_lss(&x);
//        RustLineStringCollection { lss: lss }
//    }
//
//    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
//        let vec_arr = linestrings_to_vec_array(&self.lss);
//        let pyarray = PyArray2::from_owned_array(py, arr);
//        Ok(pyarray)
//    }
//
//    fn distance_ls<'py>(
//        &self,
//        py: Python<'py>,
//        ls_other: &RustLineString,
//    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
//        let arr = self
//            .lss
//            .iter()
//            .map(|ls| Euclidean.distance(ls, ls_other.linestring.as_ref()))
//            .collect::<Array1<f64>>();
//        //    into_iter()
//
//        let pyarray = PyArray1::from_owned_array(py, arr);
//        Ok(pyarray)
//    }
//
//    fn distance_points<'py>(
//        &self,
//        py: Python<'py>,
//        other: &RustPointCollection,
//    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
//        let n_points = self.points.len();
//        let n_points_other = other.points.len();
//        let mut arr = Array2::zeros((n_points, n_points_other));
//        let mut i = 0;
//        self.points.iter().for_each(|p| {
//            let mut j = 0;
//            other.points.iter().for_each(|q| {
//                arr[[i, j]] = Euclidean.distance(p, q);
//                j += 1;
//            });
//            i += 1;
//        });
//
//        let pyarray = PyArray2::from_owned_array(py, arr);
//        Ok(pyarray)
//    }
//}
