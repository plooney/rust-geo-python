use numpy::ToPyArray;
use numpy::ndarray::{Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};

use crate::tiling::{intersect_tile_using_buffered_adapter_mpg, unary_union_with_adapter};
use geo::algorithm::Relate;
use geo::algorithm::relate::IntersectionMatrix;
use geo::orient::{Direction, Orient};
use geo::{
    Area, BooleanOps, BoundingRect, Buffer, Contains, ContainsProperly, Distance, Euclidean,
    Geometry, HausdorffDistance, Intersects, Line, LineString, MultiLineString, MultiPoint,
    MultiPolygon, Point, Polygon, Rect, Simplify, Triangle, Validation, coord, unary_union,
};
use pyo3::exceptions::PyTypeError;
use pyo3::{Bound, PyResult, Python};
use pyo3::{IntoPyObjectExt, prelude::*};
use std::fmt::Debug;
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
    Line(Arc<Line>),
    MultiPoint(Arc<MultiPoint>),
    LineString(Arc<LineString>),
    MultiLineString(Arc<MultiLineString>),
    Polygon(Arc<Polygon>),
    MultiPolygon(Arc<MultiPolygon>),
    Triangle(Arc<Triangle>),
    Rect(Arc<Rect>),
}

#[pyclass(subclass)]
#[derive(Clone)]
pub struct RustShape {
    inner: Shapes,
}

#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustPoint {
    point: Arc<Point>,
}
#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustLine {
    line: Arc<Line>,
}
#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustMultiPoint {
    multipoint: Arc<MultiPoint>,
}
#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustLineString {
    linestring: Arc<LineString>,
}
#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustPolygon {
    polygon: Arc<Polygon>,
}
#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustMultiLineString {
    multilinestring: Arc<MultiLineString>,
}

#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustMultiPolygon {
    multipolygon: Arc<MultiPolygon>,
}

#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustTriangle {
    triangle: Arc<Triangle>,
}

#[pyclass(extends=RustShape)]
#[derive(Clone)]
pub struct RustRect {
    rect: Arc<Rect>,
}

#[pyclass]
#[derive(Clone)]
pub struct RustIntersectionMatrix {
    intersection_matrix: IntersectionMatrix,
}

#[pymethods]
impl RustIntersectionMatrix {
    fn is_overlaps(&self) -> bool {
        self.intersection_matrix.is_overlaps()
    }
    fn is_intersects(&self) -> bool {
        self.intersection_matrix.is_intersects()
    }
    fn is_contains(&self) -> bool {
        self.intersection_matrix.is_contains()
    }
    fn is_within(&self) -> bool {
        self.intersection_matrix.is_within()
    }
    fn is_equal_topo(&self) -> bool {
        self.intersection_matrix.is_equal_topo()
    }
    fn is_contains_properly(&self) -> bool {
        self.intersection_matrix.is_contains_properly()
    }
    fn is_crosses(&self) -> bool {
        self.intersection_matrix.is_crosses()
    }
    fn is_coveredby(&self) -> bool {
        self.intersection_matrix.is_coveredby()
    }
    fn is_covers(&self) -> bool {
        self.intersection_matrix.is_covers()
    }
    fn is_touches(&self) -> bool {
        self.intersection_matrix.is_touches()
    }
    fn is_disjoint(&self) -> bool {
        self.intersection_matrix.is_disjoint()
    }
}

#[pymethods]
impl RustLineString {
    #[new]
    fn new(x: PyReadonlyArray2<f64>) -> (Self, RustShape) {
        let ls = array2_to_linestring(&x);
        let ls_arc = Arc::new(ls);
        (
            RustLineString {
                linestring: ls_arc.clone(),
            },
            RustShape {
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
    fn new(x: PyReadonlyArray2<f64>) -> (Self, RustShape) {
        let ls = array2_to_linestring(&x);

        let multipoint = ls.points().collect::<MultiPoint>();
        let multipoint_arc = Arc::new(multipoint);

        (
            RustMultiPoint {
                multipoint: multipoint_arc.clone(),
            },
            RustShape {
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
impl RustLine {
    #[new]
    fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> (Self, RustShape) {
        let line = Line::new(coord! { x: x0, y: y0 }, coord! { x: x1, y: y1 });
        let line_arc = Arc::new(line);
        (
            RustLine {
                line: line_arc.clone(),
            },
            RustShape {
                inner: Shapes::Line(line_arc),
            },
        )
    }

    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ps = self.line.points();
        (ps.0.x_y(), ps.1.x_y()).into_bound_py_any(py)
    }
}

#[pymethods]
impl RustTriangle {
    #[new]
    fn new(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> (Self, RustShape) {
        let tri = Triangle::new(
            coord! { x: x0, y: y0 },
            coord! { x: x1, y: y1 },
            coord! { x: x2, y: y2 },
        );
        let tri_arc = Arc::new(tri);
        (
            RustTriangle {
                triangle: tri_arc.clone(),
            },
            RustShape {
                inner: Shapes::Triangle(tri_arc),
            },
        )
    }

    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ps = &self.triangle;
        (ps.0.x_y(), ps.1.x_y(), ps.2.x_y()).into_bound_py_any(py)
    }
}

#[pymethods]
impl RustRect {
    #[new]
    fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> (Self, RustShape) {
        let rect = Rect::new(coord! { x: x0, y: y0 }, coord! { x: x1, y: y1 });
        let rect_arc = Arc::new(rect);
        (
            RustRect {
                rect: rect_arc.clone(),
            },
            RustShape {
                inner: Shapes::Rect(rect_arc),
            },
        )
    }

    fn xy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr = linestring_to_array(&self.rect.to_polygon().exterior());
        let pyarray = PyArray2::from_owned_array(py, arr);
        Ok(pyarray)
    }

    fn to_polygon<'py>(&self, py: Python<'py>) -> PyResult<Py<RustPolygon>> {
        let polygon = self.rect.to_polygon();
        let polygon_arc = Arc::new(polygon);
        let initializer: PyClassInitializer<RustPolygon> = PyClassInitializer::from((
            RustPolygon {
                polygon: polygon_arc.clone(),
            },
            RustShape {
                inner: Shapes::Polygon(polygon_arc),
            },
        ));
        Ok(Py::new(py, initializer)?)
    }
}

#[pymethods]
impl RustPoint {
    #[new]
    fn new(x: f64, y: f64) -> (Self, RustShape) {
        let point = Point::new(x, y);
        let point_arc = Arc::new(point);
        (
            RustPoint {
                point: point_arc.clone(),
            },
            RustShape {
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
    fn new(x: PyReadonlyArray2<f64>, ys: Vec<PyReadonlyArray2<f64>>) -> (Self, RustShape) {
        let polygon = array2_to_polygon(&x, &ys).orient(Direction::Default);
        let polygon_arc = Arc::new(polygon);
        (
            RustPolygon {
                polygon: polygon_arc.clone(),
            },
            RustShape {
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
            RustShape {
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
    fn new(ys: Vec<PyReadonlyArray2<f64>>) -> (Self, RustShape) {
        let lss = ys
            .iter()
            .map(|x| array2_to_linestring(x))
            .collect::<MultiLineString>();
        let lss_arc = Arc::new(lss);
        (
            RustMultiLineString {
                multilinestring: lss_arc.clone(),
            },
            RustShape {
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
    fn new(
        pyarrays: Vec<(PyReadonlyArray2<f64>, Vec<PyReadonlyArray2<f64>>)>,
    ) -> (Self, RustShape) {
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
            RustShape {
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
            RustShape {
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
impl RustShape {
    //fn scale(&self, rhs: &RustShape) -> PyResult<Py<PyAny>> {}
    fn distance(&self, rhs: &RustShape) -> f64 {
        match_shapes_algo!(self, rhs, Euclidean, distance)
    }

    fn unsigned_area(&self) -> f64 {
        match_shape!(self, unsigned_area)
    }

    fn hausdorff_distance(&self, rhs: &RustShape) -> f64 {
        match_shapes_method!(self, rhs, hausdorff_distance)
    }

    fn bounding_rect<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let rect_option = match &self.inner {
            Shapes::Point(p) => Some(p.bounding_rect()),
            Shapes::MultiPoint(p) => p.bounding_rect(),
            Shapes::LineString(p) => p.bounding_rect(),
            Shapes::MultiLineString(p) => p.bounding_rect(),
            Shapes::MultiPolygon(p) => p.bounding_rect(),
            Shapes::Polygon(p) => p.bounding_rect(),
            Shapes::Line(p) => Some(p.bounding_rect()),
            Shapes::Triangle(p) => Some(p.bounding_rect()),
            Shapes::Rect(p) => Some(p.bounding_rect()),
        };
        if let Some(rect) = rect_option {
            let rect_arc = Arc::new(rect);
            let initializer: PyClassInitializer<RustRect> = PyClassInitializer::from((
                RustRect {
                    rect: rect_arc.clone(),
                },
                RustShape {
                    inner: Shapes::Rect(rect_arc),
                },
            ));
            return Ok(Py::new(py, initializer)?.into_any());
        };
        Ok(py.None())
    }

    fn contains(&self, rhs: &RustShape) -> bool {
        match_shapes_method!(self, rhs, contains)
    }

    fn contains_properly(&self, rhs: &RustShape) -> bool {
        match_shapes_method!(self, rhs, contains_properly)
    }

    fn intersects(&self, rhs: &RustShape) -> bool {
        match_shapes_method!(self, rhs, intersects)
    }

    fn relate<'py>(
        &self,
        rhs: &RustShape,
        py: Python<'py>,
    ) -> PyResult<Py<RustIntersectionMatrix>> {
        let intersection_matrix = match_shapes_method!(self, rhs, relate);
        Py::new(
            py,
            RustIntersectionMatrix {
                intersection_matrix: intersection_matrix,
            },
        )
    }

    fn is_valid(&self) -> bool {
        match_shape!(self, is_valid)
    }

    fn to_wkt(&self) -> String {
        match_shape!(self, wkt_string)
    }

    fn buffer<'py>(&self, py: Python<'py>, radius: f64) -> PyResult<Py<PyAny>> {
        let polygons = match &self.inner {
            Shapes::Point(p) => p.buffer(radius),
            Shapes::MultiPoint(p) => p.buffer(radius),
            Shapes::LineString(p) => p.buffer(radius),
            Shapes::MultiLineString(p) => p.buffer(radius),
            Shapes::MultiPolygon(p) => p.buffer(radius),
            Shapes::Polygon(p) => p.buffer(radius),
            Shapes::Line(p) => p.buffer(radius),
            Shapes::Triangle(p) => p.buffer(radius),
            Shapes::Rect(p) => p.buffer(radius),
        };
        let multipolygon_arc = Arc::new(polygons);
        let initializer: PyClassInitializer<RustMultiPolygon> = PyClassInitializer::from((
            RustMultiPolygon {
                multipolygon: multipolygon_arc.clone(),
            },
            RustShape {
                inner: Shapes::MultiPolygon(multipolygon_arc),
            },
        ));
        Ok(Py::new(py, initializer)?.into_any())
    }

    fn intersection<'py>(&self, py: Python<'py>, rhs: &RustShape) -> PyResult<Py<PyAny>> {
        match (&self.inner, &rhs.inner) {
            (Shapes::Polygon(p), Shapes::Polygon(q)) => {
                mpg_to_pyany(py, p.as_ref().intersection(q.as_ref()))
            }
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

    fn union<'py>(&self, py: Python<'py>, rhs: &RustShape) -> PyResult<Py<PyAny>> {
        match (&self.inner, &rhs.inner) {
            (Shapes::Polygon(p), Shapes::Polygon(q)) => {
                mpg_to_pyany(py, p.as_ref().union(q.as_ref()))
            }
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

    fn difference<'py>(&self, py: Python<'py>, rhs: &RustShape) -> PyResult<Py<PyAny>> {
        match (&self.inner, &rhs.inner) {
            (Shapes::Polygon(p), Shapes::Polygon(q)) => {
                mpg_to_pyany(py, p.as_ref().difference(q.as_ref()))
            }
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
            Shapes::Line(p) => {
                let ps = p.points();
                let multipoint = MultiPoint::new(vec![ps.0, ps.1]);
                let multipoint_arc = Arc::new(multipoint);
                let initializer: PyClassInitializer<RustMultiPoint> = PyClassInitializer::from((
                    RustMultiPoint {
                        multipoint: multipoint_arc.clone(),
                    },
                    RustShape {
                        inner: Shapes::MultiPoint(multipoint_arc),
                    },
                ));
                Ok(Py::new(py, initializer)?.into_any())
            }
            Shapes::Triangle(p) => {
                let ls = p.to_polygon().exterior().clone();
                let linestring_arc = Arc::new(ls);
                let initializer: PyClassInitializer<RustLineString> = PyClassInitializer::from((
                    RustLineString {
                        linestring: linestring_arc.clone(),
                    },
                    RustShape {
                        inner: Shapes::LineString(linestring_arc),
                    },
                ));
                Ok(Py::new(py, initializer)?.into_any())
            }
            Shapes::Rect(p) => {
                let ls = p.to_polygon().exterior().clone();
                let linestring_arc = Arc::new(ls);
                let initializer: PyClassInitializer<RustLineString> = PyClassInitializer::from((
                    RustLineString {
                        linestring: linestring_arc.clone(),
                    },
                    RustShape {
                        inner: Shapes::LineString(linestring_arc),
                    },
                ));
                Ok(Py::new(py, initializer)?.into_any())
            }
            Shapes::LineString(p) => {
                let multipoint = p.points().collect::<MultiPoint>();
                let multipoint_arc = Arc::new(multipoint);
                let initializer: PyClassInitializer<RustMultiPoint> = PyClassInitializer::from((
                    RustMultiPoint {
                        multipoint: multipoint_arc.clone(),
                    },
                    RustShape {
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
                    RustShape {
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
                        RustShape {
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
                        RustShape {
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
        RustShape {
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
        RustShape {
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
        RustShape {
            inner: Shapes::MultiPolygon(multipolygon_arc),
        },
    ));
    Ok(Py::new(py, initializer)?.into_any())
}

#[pyfunction]
pub fn union_with_adapter<'py>(
    py: Python<'py>,
    rust_polygons: Vec<RustPolygon>,
    rust_rect: RustRect,
) -> PyResult<Py<PyAny>> {
    let polygons = rust_polygons
        .iter()
        .map(|x| x.polygon.as_ref())
        .collect::<Vec<&Polygon>>();
    let union = unary_union_with_adapter(polygons, rust_rect.rect.as_ref());
    let multipolygon_arc = Arc::new(union);
    let initializer: PyClassInitializer<RustMultiPolygon> = PyClassInitializer::from((
        RustMultiPolygon {
            multipolygon: multipolygon_arc.clone(),
        },
        RustShape {
            inner: Shapes::MultiPolygon(multipolygon_arc),
        },
    ));
    Ok(Py::new(py, initializer)?.into_any())
}

#[pyfunction(name = "intersect_tile")]
pub fn intersect_tile<'py>(
    py: Python<'py>,
    polygon: RustMultiPolygon,
    tile_polygon: RustPolygon,
    rust_rect: RustRect,
) -> PyResult<Py<PyAny>> {
    let mpg = intersect_tile_using_buffered_adapter_mpg(
        polygon.multipolygon.as_ref(),
        tile_polygon.polygon.as_ref(),
        rust_rect.rect.as_ref(),
    );
    mpg_to_pyany(py, mpg)
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
pub struct RustGeomVecCollection {
    geoms: Vec<Geometry>,
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
impl RustGeomVecCollection {
    #[new]
    fn new(shapes: Vec<RustShape>) -> Self {
        let geoms = shapes
            .iter()
            .map(|x| match_shape_geom!(x))
            .collect::<Vec<Geometry>>();
        RustGeomVecCollection { geoms: geoms }
    }

    fn distance<'py>(
        &self,
        py: Python<'py>,
        other: RustGeomVecCollection,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n_geoms = self.geoms.len();
        let n_geoms_other = other.geoms.len();
        let index = (0..n_geoms).flat_map(|i| (0..n_geoms_other).map(move |j| (i, j)));

        let shape = (n_geoms, n_geoms_other);
        let mut arr = Array2::zeros(shape);

        index.for_each(|(i, j)| {
            let a = self.geoms.get(i);
            let b = other.geoms.get(j);
            if let (Some(p), Some(q)) = (a, b) {
                let d = Euclidean.distance(p, q);
                arr[[i, j]] = d;
            }
        });

        Ok(arr.to_pyarray(py))
    }
}
