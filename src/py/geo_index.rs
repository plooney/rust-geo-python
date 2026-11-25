use geo_index::rtree::{
    DEFAULT_RTREE_NODE_SIZE, RTree, RTreeBuilder, RTreeIndex, sort::HilbertSort,
};
use pyo3::prelude::*;
use rstar::RTreeObject;

use crate::py::enums::RustShape;

#[pyclass]
pub struct RustGeoIndex {
    shapes: Vec<RustShape>,
    rtree: RTree<f64>,
}

#[pymethods]
impl RustGeoIndex {
    #[new]
    fn new(shapes: Vec<RustShape>, node_size: Option<u16>) -> Self {
        let mut builder = RTreeBuilder::<f64>::new_with_node_size(
            shapes.len() as u32,
            node_size.unwrap_or(DEFAULT_RTREE_NODE_SIZE),
        );
        for shape in &shapes {
            let env = shape.envelope();
            let lower = env.lower();
            let upper = env.upper();
            builder.add(lower[0], lower[1], upper[0], upper[1]);
        }
        let tree = builder.finish::<HilbertSort>();

        let rgi = RustGeoIndex {
            shapes: shapes,
            rtree: tree,
        };

        rgi
    }

    fn len(&self) -> usize {
        self.shapes.len()
    }

    fn search(&mut self, shape: RustShape) -> PyResult<Vec<RustShape>> {
        let env = shape.envelope();
        let lower = env.lower();
        let upper = env.upper();

        let matches = self.rtree.search(lower[0], lower[1], upper[0], upper[1]);

        Ok(matches
            .into_iter()
            .filter_map(|idx| self.shapes.get(idx as usize).cloned())
            .collect())
    }

    fn neighbors(
        &mut self,
        x: f64,
        y: f64,
        max_results: Option<usize>,
        max_distance: Option<f64>,
    ) -> PyResult<Vec<RustShape>> {
        let matches = self.rtree.neighbors(x, y, max_results, max_distance);

        Ok(matches
            .into_iter()
            .filter_map(|idx| self.shapes.get(idx as usize).cloned())
            .collect())
    }
}
