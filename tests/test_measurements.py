
import pytest
import numpy as np
from rust_geo_python import RustPolygon, RustMultiPolygon, point_polygon_distance, polygon_polygon_distance

def test_polygon_area():
    # 2x2 square, area should be 4
    ext = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    poly = RustPolygon(ext, [])
    assert poly.area() == 4.0

def test_multipolygon_area():
    # Two 1x1 squares, total area 2
    sq1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    sq2 = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0], [2.0, 2.0]], dtype=np.float64)
    data = [(sq1, []), (sq2, [])]
    mpoly = RustMultiPolygon(data)
    assert mpoly.area() == 2.0

def test_shape_distance():
    # Distance between two disjoint polygons
    # Poly1: [0,0] to [1,1]
    # Poly2: [2,2] to [3,3]
    # Distance is sqrt((2-1)^2 + (2-1)^2) = sqrt(1+1) = sqrt(2) approx 1.414
    
    p1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    poly1 = RustPolygon(p1, [])
    
    p2 = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0], [2.0, 2.0]], dtype=np.float64)
    poly2 = RustPolygon(p2, [])
    
    # Using method on shape if available
    # The `distance` method is on `Shape` (base class). RustPolygon extends Shape.
    d = poly1.distance(poly2)
    assert abs(d - np.sqrt(2)) < 1e-9

def test_hausdorff_distance():
    # Simple test
    p1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    poly1 = RustPolygon(p1, [])
    
    p2 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    poly2 = RustPolygon(p2, [])
    
    # Hausdorff distance should be 1.0 (extra height of p2)
    hd = poly1.hausdorff_distance(poly2)
    assert abs(hd - 1.0) < 1e-9
