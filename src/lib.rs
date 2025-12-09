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
    use std::sync::Arc;
    use wkt::ToWkt;

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

    #[derive(Clone)]
    enum Shapes {
        Point(Arc<Point>),
        MultiPoint(Arc<MultiPoint>),
        LineString(Arc<LineString>),
        MultiLineString(Arc<MultiLineString>),
        Polygon(Arc<Polygon>),
        MultiPolygon(Arc<MultiPolygon>),
    }

    #[pyclass(subclass)]
    #[derive(Clone)]
    struct Shape {
        inner: Shapes,
    }

    #[pyclass(extends=Shape)]
    #[derive(Clone)]
    struct RustPoint {
        point: Arc<Point>,
    }
    #[pyclass(extends=Shape)]
    #[derive(Clone)]
    struct RustMultiPoint {
        multipoint: Arc<MultiPoint>,
    }
    #[pyclass(extends=Shape)]
    struct RustLineString {
        linestring: Arc<LineString>,
    }
    #[pyclass(extends=Shape)]
    #[derive(Clone)]
    struct RustPolygon {
        polygon: Arc<Polygon>,
    }
    #[pyclass(extends=Shape)]
    struct RustMultiLineString {
        multilinestring: Arc<MultiLineString>,
    }

    #[pyclass(extends=Shape)]
    struct RustMultiPolygon {
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
        fn new(
            pyarrays: Vec<(PyReadonlyArray2<f64>, Vec<PyReadonlyArray2<f64>>)>,
        ) -> (Self, Shape) {
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
                (Shapes::LineString(p), Shapes::Point(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
                (Shapes::Point(p), Shapes::LineString(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
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
                (Shapes::Polygon(p), Shapes::Point(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::LineString(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::MultiLineString(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::Polygon(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
                (Shapes::Point(p), Shapes::Polygon(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
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
                (Shapes::MultiPoint(p), Shapes::Point(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
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
                (Shapes::Point(p), Shapes::MultiPoint(q)) => {
                    Euclidean.distance(p.as_ref(), q.as_ref())
                }
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
                (Shapes::LineString(p), Shapes::Point(q)) => {
                    p.as_ref().hausdorff_distance(q.as_ref())
                }
                (Shapes::Point(p), Shapes::LineString(q)) => {
                    p.as_ref().hausdorff_distance(q.as_ref())
                }
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
                (Shapes::Polygon(p), Shapes::Polygon(q)) => {
                    p.as_ref().hausdorff_distance(q.as_ref())
                }
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
                (Shapes::MultiPoint(p), Shapes::Point(q)) => {
                    p.as_ref().hausdorff_distance(q.as_ref())
                }
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
                (Shapes::Point(p), Shapes::MultiPoint(q)) => {
                    p.as_ref().hausdorff_distance(q.as_ref())
                }
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
                (Shapes::MultiLineString(p), Shapes::LineString(q)) => {
                    p.as_ref().contains(q.as_ref())
                }
                (Shapes::MultiLineString(p), Shapes::MultiLineString(q)) => {
                    p.as_ref().contains(q.as_ref())
                }
                (Shapes::Point(p), Shapes::MultiLineString(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::LineString(p), Shapes::MultiLineString(q)) => {
                    p.as_ref().contains(q.as_ref())
                }
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
                (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => {
                    p.as_ref().contains(q.as_ref())
                }
                (Shapes::Point(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::LineString(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => {
                    p.as_ref().contains(q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::MultiPoint(p), Shapes::Point(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::MultiPoint(p), Shapes::LineString(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => {
                    p.as_ref().contains(q.as_ref())
                }
                (Shapes::MultiPoint(p), Shapes::Polygon(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::Point(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::LineString(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => {
                    p.as_ref().contains(q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
                (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => p.as_ref().contains(q.as_ref()),
            }
        }

        fn contains_properly(&self, rhs: &Shape) -> bool {
            match (&self.inner, &rhs.inner) {
                (Shapes::Point(p), Shapes::Point(q)) => p.as_ref().contains_properly(q.as_ref()),
                (Shapes::LineString(p), Shapes::Point(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::Point(p), Shapes::LineString(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
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
                (Shapes::Polygon(p), Shapes::LineString(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::MultiLineString(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::Polygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::Point(p), Shapes::Polygon(q)) => p.as_ref().contains_properly(q.as_ref()),
                (Shapes::LineString(p), Shapes::Polygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiLineString(p), Shapes::Polygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiPolygon(p), Shapes::Point(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
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
                (Shapes::Point(p), Shapes::MultiPolygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::LineString(p), Shapes::MultiPolygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiPoint(p), Shapes::Point(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiPoint(p), Shapes::LineString(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiPoint(p), Shapes::Polygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::Point(p), Shapes::MultiPoint(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::LineString(p), Shapes::MultiPoint(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::Polygon(p), Shapes::MultiPoint(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
                (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => {
                    p.as_ref().contains_properly(q.as_ref())
                }
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

        fn boundary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
            match &self.inner {
                Shapes::Point(_) => Ok(py.None()),
                Shapes::MultiPoint(_) => Ok(py.None()),
                Shapes::LineString(p) => {
                    let multipoint = p.points().collect::<MultiPoint>();
                    let multipoint_arc = Arc::new(multipoint);
                    let initializer: PyClassInitializer<RustMultiPoint> =
                        PyClassInitializer::from((
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
                    let initializer: PyClassInitializer<RustMultiPoint> =
                        PyClassInitializer::from((
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

    #[pyfunction]
    fn union<'py>(py: Python<'py>, rust_polygons: Vec<RustPolygon>) -> PyResult<Py<PyAny>> {
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
    fn count<'py>(rust_points: Vec<RustPoint>) -> PyResult<()> {
        println!("Some text {}", rust_points.len());
        Ok(())
    }

    #[pyfunction]
    fn point_in_polygon<'py>(rust_point: RustPoint, rust_polygon: RustPolygon) -> PyResult<bool> {
        let point = rust_point.point.as_ref();
        let polygon = rust_polygon.polygon;
        let is_in = polygon.as_ref().contains(point);
        Ok(is_in)
    }

    #[pyfunction(name = "intersection")]
    fn intersection<'py>(
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
}
