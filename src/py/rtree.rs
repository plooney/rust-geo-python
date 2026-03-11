use pyo3::prelude::*;
use rstar::RTree;

use crate::py::enums::RustShape;

#[pyclass]
pub struct RustRTree {
    tree: RTree<RustShape>,
}

#[pymethods]
impl RustRTree {
    #[new]
    fn new() -> Self {
        RustRTree { tree: RTree::new() }
    }

    fn len(&self) -> usize {
        self.tree.size()
    }

    fn insert(&mut self, shape: RustShape) -> PyResult<()> {
        //let geom: GeoTypesGeometry<f64> = shape.to_geometry().into();
        self.tree.insert(shape);
        Ok(())
    }
}
