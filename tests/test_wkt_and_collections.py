import pytest

import rust_geo_python as rg


def test_from_wkt_point():
    geom = rg.from_wkt("POINT (1 2)")
    assert isinstance(geom, rg.RustPoint)
    assert geom.xy() == (1.0, 2.0)


def test_from_wkt_polygon():
    geom = rg.from_wkt("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))")
    assert isinstance(geom, rg.RustPolygon)
    ext, holes = geom.xy()
    assert holes == []
    assert ext.shape == (5, 2)


def test_from_wkt_geometry_collection_iteration():
    geom = rg.from_wkt("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))")
    assert isinstance(geom, rg.RustGeometryCollection)
    assert len(geom) == 2

    items = list(geom)
    assert len(items) == 2
    assert isinstance(items[0], rg.RustPoint)
    assert isinstance(items[1], rg.RustLineString)


def test_geometry_collection_getitem():
    geom = rg.from_wkt("GEOMETRYCOLLECTION (POINT (1 2), POINT (3 4))")
    assert isinstance(geom[0], rg.RustPoint)
    assert isinstance(geom[-1], rg.RustPoint)

    with pytest.raises(IndexError):
        _ = geom[2]


def test_from_wkt_invalid():
    with pytest.raises(ValueError):
        rg.from_wkt("")

    with pytest.raises(ValueError):
        rg.from_wkt("NOT_A_WKT")
