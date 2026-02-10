
import pytest
import numpy as np
import rust_geo_python
from rust_geo_python import point_polygon_distance, points_polygon_distance, polygon_polygon_distance, union_set_shapes

def test_point_polygon_distance_py():
    # x: 1D array point
    # y: 2D array polygon (exterior only)
    
    x = np.array([1.0, 1.0], dtype=np.float64)
    y = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    
    dist = point_polygon_distance(x, y)
    assert dist == 1.0

def test_points_polygon_distance_py():
    # x: 2D array of points
    xs = np.array([[1.0, 1.0], [3.0, 1.0]], dtype=np.float64)
    y = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    
    dists = points_polygon_distance(xs, y)
    assert np.allclose(dists, [1.0, 1.0])

def test_union_set_shapes():
    # This function takes raw numpy arrays representing polygons and unions them
    # Input: list of (ext, [ints])
    
    ext1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    poly1 = (ext1, [])
    
    ext2 = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    poly2 = (ext2, [])
    
    # Returns vector of (ext, [ints])
    result = union_set_shapes([poly1, poly2])
    
    # Should resolve to one polygon (rectangle 0,0 to 2,1)
    assert len(result) == 1
    res_ext, res_ints = result[0]
    
    # simple check: 0,0 and 2,1 should be in it, 1,0 is boundary now but not vertex necessarily if merged perfectly
    # or just check approximate bounds
    assert np.min(res_ext[:,0]) == 0.0
    assert np.max(res_ext[:,0]) == 2.0
