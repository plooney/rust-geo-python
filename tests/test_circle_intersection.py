import numpy as np
import pytest

from rust_geo_python import update_circle_areas_using_boundaries


def boundary(outer, holes=()):
    return {
        "outer boundary": np.asarray(outer, dtype=np.float64),
        "interior boundary": [np.asarray(hole, dtype=np.float64) for hole in holes],
    }


def update_area(center, radius, regions, *, remain_within_connected_component=False):
    points = np.asarray([center], dtype=np.float64)
    areas = np.asarray([np.pi * radius**2], dtype=np.float64)
    distances = np.zeros((len(points), len(regions)), dtype=np.float64)
    result = update_circle_areas_using_boundaries(
        point_locations=points,
        areas=areas,
        distance_to_boundary=distances,
        radius=radius,
        remain_within_connected_component=remain_within_connected_component,
        boundaries_as_shapes=regions,
    )
    return areas, result


def test_tangent_hole_corner_has_analytic_intersection_area():
    outer = [(-10, -10), (-10, 10), (10, 10), (10, -10)]
    excluded_square = [(0, 0), (0, 1), (1, 1), (1, 0)]
    radius = np.sqrt(2.0)

    areas, result = update_area(
        (0.0, 2.0), radius, [boundary(outer, [excluded_square])]
    )

    expected = 7.0 * np.pi / 4.0 + 0.5
    assert result is areas
    assert areas[0] == pytest.approx(expected, abs=1e-12)


def test_circle_inside_region_returns_full_circle_area():
    region = boundary([(-3, -3), (3, -3), (3, 3), (-3, 3)])
    areas, _ = update_area((0.0, 0.0), 1.0, [region])
    assert areas[0] == pytest.approx(np.pi, abs=1e-12)


def test_circle_clipped_by_straight_boundary_returns_half_disk():
    region = boundary([(0, -2), (2, -2), (2, 2), (0, 2)])
    areas, _ = update_area((0.0, 0.0), 1.0, [region])
    assert areas[0] == pytest.approx(np.pi / 2.0, abs=1e-12)


def test_circle_inside_excluded_hole_has_zero_area():
    outer = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    hole = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    areas, _ = update_area((0.0, 0.0), 1.0, [boundary(outer, [hole])])
    assert areas[0] == pytest.approx(0.0, abs=1e-12)


def test_multiple_disjoint_regions_are_summed_when_component_limit_is_off():
    left = boundary([(-3, -3), (0, -3), (0, 3), (-3, 3)])
    right = boundary([(1, -3), (3, -3), (3, 3), (1, 3)])
    areas, _ = update_area((0.0, 0.0), 2.0, [left, right])
    right_cap = 4.0 * np.pi / 3.0 - np.sqrt(3.0)
    assert areas[0] == pytest.approx(2.0 * np.pi + right_cap, abs=1e-12)


def test_component_limit_uses_only_the_region_containing_the_center():
    left = boundary([(-3, -3), (0, -3), (0, 3), (-3, 3)])
    right = boundary([(1, -3), (3, -3), (3, 3), (1, 3)])
    areas, _ = update_area(
        (0.0, 0.0),
        2.0,
        [left, right],
        remain_within_connected_component=True,
    )
    assert areas[0] == pytest.approx(2.0 * np.pi, abs=1e-12)


def test_component_limit_uses_georust_polygon_boundary_semantics():
    outer = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
    hole = [(1, 1), (1, 3), (3, 3), (3, 1), (1, 1)]
    region = boundary(outer, [hole])

    exterior_boundary_area, _ = update_area(
        (0.0, 2.0), 1.0, [region], remain_within_connected_component=True
    )
    hole_boundary_area, _ = update_area(
        (1.0, 2.0), 1.0, [region], remain_within_connected_component=True
    )
    hole_interior_area, _ = update_area(
        (2.0, 2.0), 1.0, [region], remain_within_connected_component=True
    )

    assert exterior_boundary_area[0] == pytest.approx(np.pi / 2.0, abs=1e-12)
    assert hole_boundary_area[0] == pytest.approx(np.pi / 2.0, abs=1e-12)
    assert hole_interior_area[0] == 0.0


def test_multiple_points_are_updated_independently():
    points = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    areas = np.full(2, np.pi, dtype=np.float64)
    region = boundary([(-4, -4), (4, -4), (4, 4), (-4, 4)])
    result = update_circle_areas_using_boundaries(
        point_locations=points,
        areas=areas,
        distance_to_boundary=np.zeros((2, 1), dtype=np.float64),
        radius=1.0,
        remain_within_connected_component=False,
        boundaries_as_shapes=[region],
    )
    assert result is areas
    assert np.allclose(areas, [np.pi, np.pi], atol=1e-12)


def test_external_tangent_has_zero_intersection_area():
    region = boundary([(1, -2), (3, -2), (3, 2), (1, 2)])
    areas, _ = update_area((0.0, 0.0), 1.0, [region])
    assert areas[0] == pytest.approx(0.0, abs=1e-12)


def test_open_and_reversed_rings_are_supported():
    square = [(-3, -3), (-3, 3), (3, 3), (3, -3)]
    areas, _ = update_area((0.0, 0.0), 1.0, [boundary(square)])
    assert areas[0] == pytest.approx(np.pi, abs=1e-12)


def test_negative_radius_is_rejected():
    with pytest.raises(ValueError, match="radius"):
        update_area((0.0, 0.0), -1.0, [boundary([(-2, -2), (2, -2), (2, 2)])])
