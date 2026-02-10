
import pytest
import numpy as np
from rust_geo_python import RustPolygon, RustMultiPolygon, union, intersection_shapes, difference_shapes

def test_union_method():
    # Two overlapping squares
    p1 = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    poly1 = RustPolygon(p1, [])
    
    p2 = np.array([[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0], [1.0, 0.0]], dtype=np.float64)
    poly2 = RustPolygon(p2, [])
    
    # Union should be a rectangle [0,0] to [3,2] with area 6
    unioned = poly1.union(poly2)
    # Result is a Shape (wraps MultiPolygon likely)
    # The signature returns Py<PyAny> which is dynamically typed in Python.
    # It seems to return RustMultiPolygon from code reading.
    
    # Check area
    assert abs(unioned.area() - 6.0) < 1e-9

def test_intersection_method():
    p1 = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    poly1 = RustPolygon(p1, [])
    
    p2 = np.array([[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0], [1.0, 0.0]], dtype=np.float64)
    poly2 = RustPolygon(p2, [])
    
    # Intersection is [1,0] to [2,2], area 2
    inter = poly1.intersection(poly2)
    assert abs(inter.area() - 2.0) < 1e-9

def test_buffer():
    # Point buffer -> Circle (approx)
    from rust_geo_python import RustPoint
    p = RustPoint(0.0, 0.0)
    buf = p.buffer(1.0)
    # Area of unit circle is pi
    assert abs(buf.area() - np.pi) < 0.1 # default quad_segs might be low, loose tolerance

def test_union_function():
    # Helper to union list of polygons
    p1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    poly1 = RustPolygon(p1, [])
    
    p2 = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    poly2 = RustPolygon(p2, [])
    
    u = union([poly1, poly2])
    # Total area 2
    assert abs(u.area() - 2.0) < 1e-9
