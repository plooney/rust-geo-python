
import numpy as np
from rust_geo_python import RustLine, RustMultiPoint, RustPolygon, RustPoint, point_in_polygon

def test_contains():
    # Square [0,0] to [2,2]
    ext = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    poly = RustPolygon(ext, [])
    
    # Point inside
    p_in = RustPoint(1.0, 1.0)
    assert poly.contains(p_in)
    
    # Point outside
    p_out = RustPoint(3.0, 3.0)
    assert not poly.contains(p_out)

def test_contains_properly():
    # Square [0,0] to [2,2]
    ext = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    poly = RustPolygon(ext, [])
    
    # Point on boundary
    p_bound = RustPoint(0.0, 0.0)
    # Geo's contains excludes boundary points.
    assert not poly.contains(p_bound)
    assert not poly.contains_properly(p_bound)

def test_intersects():
    p1 = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    poly1 = RustPolygon(p1, [])
    
    # Overlapping
    p2 = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 1.0]], dtype=np.float64)
    poly2 = RustPolygon(p2, [])
    
    assert poly1.intersects(poly2)
    
    # Disjoint
    p3 = np.array([[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0], [3.0, 3.0]], dtype=np.float64)
    poly3 = RustPolygon(p3, [])
    assert not poly1.intersects(poly3)

def test_point_in_polygon_function():
    # Test the standalone helper if desired, but it takes RustPoint and RustPolygon
    ext = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    poly = RustPolygon(ext, [])
    p = RustPoint(1.0, 1.0)
    
    assert point_in_polygon(p, poly)


def test_closest_point_returns_multiple_points_for_ties():
    points = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    multipoint = RustMultiPoint(points)
    query = RustPoint(0.0, 0.0)

    closest = multipoint.closest_point(query)
    coords = closest.xy()

    assert coords.shape == (2, 2)
    assert {tuple(row) for row in coords.tolist()} == {(-1.0, 0.0), (1.0, 0.0)}


def test_closest_region_returns_overlapping_line_segment():
    poly_xy = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    polygon = RustPolygon(poly_xy, [])
    line = RustLine(-1.0, 0.5, 3.0, 0.5)

    region = line.closest_region(polygon)

    assert region.xy() == ((0.0, 0.5), (2.0, 0.5))


def test_closest_region_returns_parallel_closest_subsegment():
    poly_xy = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    polygon = RustPolygon(poly_xy, [])
    line = RustLine(-1.0, 2.0, 3.0, 2.0)

    region = line.closest_region(polygon)

    assert region.xy() == ((0.0, 2.0), (2.0, 2.0))
