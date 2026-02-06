
import pytest
import numpy as np
from rust_geo_python import RustPoint, RustMultiPoint, RustLineString, RustPolygon, RustMultiLineString, RustMultiPolygon

def test_create_point():
    p = RustPoint(1.0, 2.0)
    assert p is not None

def test_point_xy():
    p = RustPoint(1.0, 2.0)
    # create dummy python to satisfy signature if needed, but creating from python side usually handles it.
    # The xy method might need to be called.
    # checking signature: fn xy<'py>(&self, py: Python<'py>)
    xy = p.xy()
    assert xy == (1.0, 2.0)

def test_create_multipoint():
    data = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    mp = RustMultiPoint(data)
    assert mp is not None
    assert np.allclose(mp.xy(), data)

def test_create_linestring():
    data = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    ls = RustLineString(data)
    assert ls is not None
    assert np.allclose(ls.xy(), data)

def test_create_polygon():
    # Exterior ring (square)
    ext = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    # Interiors (none)
    ints = []
    poly = RustPolygon(ext, ints)
    assert poly is not None
    
    # Check xy
    res_ext, res_ints = poly.xy()
    assert np.allclose(res_ext, ext)
    assert len(res_ints) == 0

def test_create_multipolygon():
    ext = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    ints = []
    # Vector of (ext, [ints]) tuples
    data = [(ext, ints)]
    mpoly = RustMultiPolygon(data)
    assert mpoly is not None
    
    res = mpoly.xy()
    assert len(res) == 1
    assert np.allclose(res[0][0], ext)
