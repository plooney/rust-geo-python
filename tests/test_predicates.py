
import pytest
import numpy as np
from rust_geo_python import RustPolygon, RustPoint, point_in_polygon

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
