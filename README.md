# rust-geo-python

Fast 2D geometry for Python implemented in Rust. This project exposes `geo`-powered operations (distance, relations, boolean ops, buffering, etc.) to Python via PyO3 and NumPy arrays, with optional robust overlay helpers powered by `i_overlay`.

**Status:** early, API may change.

## Features
- Python classes for core shapes: point, line, line string, polygon, multipolygon, rect, triangle, and geometry collections.
- Geometry operations: distance, containment, intersection/union/difference, buffer, area, validity checks, and relations.
- Vectorized distance helpers that operate on NumPy arrays.
- Optional robust overlay utilities for tiling workflows.

## Install (from source)
This repo builds a Python extension with `maturin`.

Requirements:
- Rust toolchain (edition 2024)
- Python 3.7+ (uses `abi3-py37`)
- `maturin`

Build and install into the current Python environment:
```bash
pip install maturin
maturin develop --release
```

Build a wheel:
```bash
maturin build --release
```

## Quick start
```python
import numpy as np
import rust_geo_python as rg

# Points / lines
p = rg.RustPoint(1.0, 2.0)
line = rg.RustLine(0.0, 0.0, 3.0, 4.0)

# Polygons are defined by an exterior ring and optional interior rings
outer = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]])
hole = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]])
poly = rg.RustPolygon(outer, [hole])

print(poly.area())
print(poly.contains(p))
print(poly.to_wkt())
```

Vectorized distance helpers:
```python
points = np.array([[0.0, 0.0], [5.0, 5.0], [2.0, 3.0]])
polygon = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]])

dist = rg.points_polygon_distance(points, polygon)
print(dist)
```

## API overview
Python module: `rust_geo_python`

### Classes
- `RustShape` (base class)
- `RustPoint`
- `RustLine`
- `RustLineString`
- `RustMultiPoint`
- `RustMultiLineString`
- `RustPolygon`
- `RustMultiPolygon`
- `RustTriangle`
- `RustRect`
- `RustGeometryCollection`
- `RustGeomVecCollection`
- `RustIntersectionMatrix`

### Selected methods
`RustShape` provides:
- `distance(other)`
- `unsigned_area()`
- `hausdorff_distance(other)`
- `bounding_rect()`
- `contains(other)`
- `contains_properly(other)`
- `intersects(other)`
- `relate(other)` → `RustIntersectionMatrix`
- `is_valid()`
- `to_wkt()`
- `buffer(radius)`
- `intersection(other)`
- `union(other)`
- `difference(other)`
- `boundary()`

`RustPolygon` / `RustMultiPolygon`:
- `xy()` (NumPy arrays)
- `simplify(epsilon)`
- `area()`

`RustRect`:
- `xy()`
- `to_polygon()`

`RustGeomVecCollection`:
- `distance(other)` (pairwise distance matrix)

`RustGeometryCollection`:
- `len()`

### Functions
- `point_polygon_distance(point_xy, polygon_xy)`
- `points_polygon_distance(points_xy, polygon_xy)`
- `points_polygon_dist_mut(points_xy, polygon_xy)` (parallelized)
- `polygon_polygon_distance(polygon_xy, polygon_xy)`
- `from_wkt(wkt_string)`
- `union(polygons)`
- `union_with_adapter(polygons, rect)`
- `intersection(polygon, polygon)`
- `intersect_tile(multipolygon, tile_polygon, rect)`
- `point_in_polygon(point, polygon)`
- `union_set_shapes(...)`
- `intersection_shapes(...)`
- `difference_shapes(...)`

## Notes
- Coordinates are `float64`.
- Polygon rings are expected to be closed (first point == last point).
- Boolean ops on shapes currently support polygon and multipolygon combinations.
- `from_wkt` returns the most specific class (e.g., `RustPoint`, `RustPolygon`, `RustGeometryCollection`).
- `buffer()` is not supported for `RustGeometryCollection`.

## Development
Run tests:
```bash
cargo test
```

Build the Python extension in dev mode:
```bash
maturin develop
```
