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
