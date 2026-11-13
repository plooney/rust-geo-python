use std::f64::consts::PI;
use std::fmt;

use geo::{Intersects, LineString, Point, Polygon};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircleIntersectionError {
    InvalidRadius,
    RadiusTooLarge,
    NonFiniteCoordinate,
}

impl fmt::Display for CircleIntersectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRadius => formatter.write_str("radius must be finite and non-negative"),
            Self::RadiusTooLarge => formatter.write_str("radius is too large for a finite area"),
            Self::NonFiniteCoordinate => {
                formatter.write_str("centers and polygon coordinates must be finite")
            }
        }
    }
}

impl std::error::Error for CircleIntersectionError {}

/// Area of the intersection between a circle and a closed linear ring.
///
/// Each edge is mapped into unit-circle coordinates and contributes either a
/// triangle area (for an edge portion inside the circle) or a circular-sector
/// area (for an edge portion outside it). The ring is supplied by GeoRust, so
/// its coordinates and edges use the same primitives as the rest of the core.
///
/// The line-circle discriminant is evaluated with ordinary `f64` arithmetic.
/// Near tangency, rounding can change its sign and therefore classify an edge
/// as a miss or a crossing, which can introduce a small area error. A future
/// improvement would use an adaptive robust predicate for the discriminant's
/// sign, while retaining floating-point arithmetic for the intersection roots.
fn ring_intersection_area(ring: &LineString<f64>, center: Point<f64>, radius: f64) -> f64 {
    if ring.0.len() < 3 || radius == 0.0 {
        return 0.0;
    }

    let mut signed_area = 0.0;
    for line in ring.lines() {
        let start = [
            (line.start.x - center.x()) / radius,
            (line.start.y - center.y()) / radius,
        ];
        let end = [
            (line.end.x - center.x()) / radius,
            (line.end.y - center.y()) / radius,
        ];
        let delta = [end[0] - start[0], end[1] - start[1]];
        let a = delta[0] * delta[0] + delta[1] * delta[1];
        let b = 2.0 * (start[0] * delta[0] + start[1] * delta[1]);
        let c = start[0] * start[0] + start[1] * start[1] - 1.0;
        let mut cuts = vec![0.0, 1.0];

        if a > 0.0 {
            let four_ac = 4.0 * a * c;
            let discriminant = b * b - four_ac;

            if discriminant >= 0.0 {
                let root = discriminant.sqrt();
                if root == 0.0 {
                    add_cut(&mut cuts, -b / (2.0 * a));
                } else {
                    // This form avoids cancellation when one of the roots is
                    // much smaller than the other.
                    let q = -0.5 * (b + b.signum() * root);
                    if q == 0.0 {
                        add_cut(&mut cuts, -b / (2.0 * a));
                    } else {
                        add_cut(&mut cuts, q / a);
                        add_cut(&mut cuts, c / q);
                    }
                }
            }
        }

        cuts.sort_by(f64::total_cmp);
        // Keep distinct nearby roots: a very short interval can still enclose
        // a real, non-zero sliver of circle/polygon intersection.
        cuts.dedup_by(|left, right| *left == *right);

        for interval in cuts.windows(2) {
            let t0 = interval[0];
            let t1 = interval[1];
            if t1 <= t0 {
                continue;
            }

            let p = [start[0] + t0 * delta[0], start[1] + t0 * delta[1]];
            let q = [start[0] + t1 * delta[0], start[1] + t1 * delta[1]];
            let midpoint = [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5];
            let cross = p[0] * q[1] - p[1] * q[0];

            if midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1] <= 1.0 {
                signed_area += 0.5 * cross;
            } else {
                let dot = p[0] * q[0] + p[1] * q[1];
                signed_area += 0.5 * cross.atan2(dot);
            }
        }
    }

    (signed_area.abs() * radius * radius).min(PI * radius * radius)
}

fn add_cut(cuts: &mut Vec<f64>, cut: f64) {
    if cut > 0.0 && cut < 1.0 {
        cuts.push(cut);
    }
}

fn circle_area_inside_region(center: Point<f64>, radius: f64, region: &Polygon<f64>) -> f64 {
    let exterior_area = ring_intersection_area(region.exterior(), center, radius);
    let holes_area = region
        .interiors()
        .iter()
        .map(|ring| ring_intersection_area(ring, center, radius))
        .sum::<f64>();
    (exterior_area - holes_area).clamp(0.0, PI * radius * radius)
}

/// Calculate circle intersection areas for GeoRust points and polygons.
///
/// If `remain_within_connected_component` is true, each circle is clipped to
/// the first polygon that GeoRust says intersects its center. Otherwise all
/// supplied polygon areas are accumulated and clamped to the circle area.
pub fn circle_areas_inside_regions(
    centers: &[Point<f64>],
    radius: f64,
    regions: &[Polygon<f64>],
    remain_within_connected_component: bool,
) -> Result<Vec<f64>, CircleIntersectionError> {
    if !radius.is_finite() || radius < 0.0 {
        return Err(CircleIntersectionError::InvalidRadius);
    }

    let circle_area = PI * radius * radius;
    if !circle_area.is_finite() {
        return Err(CircleIntersectionError::RadiusTooLarge);
    }

    if centers
        .iter()
        .any(|center| !center.x().is_finite() || !center.y().is_finite())
        || regions.iter().any(|region| {
            std::iter::once(region.exterior())
                .chain(region.interiors())
                .flat_map(|ring| ring.0.iter())
                .any(|coordinate| !coordinate.x.is_finite() || !coordinate.y.is_finite())
        })
    {
        return Err(CircleIntersectionError::NonFiniteCoordinate);
    }

    Ok(centers
        .iter()
        .map(|center| {
            if remain_within_connected_component {
                regions
                    .iter()
                    .find(|region| region.intersects(center))
                    .map_or(0.0, |region| {
                        circle_area_inside_region(*center, radius, region)
                    })
            } else {
                regions
                    .iter()
                    .map(|region| circle_area_inside_region(*center, radius, region))
                    .sum::<f64>()
                    .clamp(0.0, circle_area)
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::circle_areas_inside_regions;
    use geo::{Coord, LineString, Point, Polygon};

    fn ring(points: &[(f64, f64)]) -> LineString<f64> {
        LineString::new(points.iter().map(|(x, y)| Coord { x: *x, y: *y }).collect())
    }

    fn polygon(exterior: &[(f64, f64)], holes: &[&[(f64, f64)]]) -> Polygon<f64> {
        Polygon::new(
            ring(exterior),
            holes.iter().map(|hole| ring(hole)).collect(),
        )
    }

    #[test]
    fn tangent_hole_corner_has_analytic_intersection_area() {
        let region = polygon(
            &[
                (-10.0, -10.0),
                (-10.0, 10.0),
                (10.0, 10.0),
                (10.0, -10.0),
                (-10.0, -10.0),
            ],
            &[&[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]],
        );
        let actual =
            circle_areas_inside_regions(&[Point::new(0.0, 2.0)], 2.0_f64.sqrt(), &[region], false)
                .unwrap();

        assert!((actual[0] - (7.0 * std::f64::consts::PI / 4.0 + 0.5)).abs() < 1e-12);
    }

    #[test]
    fn component_selection_uses_georust_boundary_intersection_semantics() {
        let region = polygon(
            &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)],
            &[&[(1.0, 1.0), (1.0, 3.0), (3.0, 3.0), (3.0, 1.0), (1.0, 1.0)]],
        );
        let circle_area = std::f64::consts::PI;
        let actual = circle_areas_inside_regions(
            &[
                Point::new(0.0, 2.0), // exterior boundary
                Point::new(1.0, 2.0), // hole boundary is also polygon boundary
                Point::new(2.0, 2.0), // hole interior
            ],
            1.0,
            &[region],
            true,
        )
        .unwrap();

        assert!((actual[0] - circle_area / 2.0).abs() < 1e-12);
        assert!((actual[1] - circle_area / 2.0).abs() < 1e-12);
        assert_eq!(actual[2], 0.0);
    }

    #[test]
    fn circles_inside_a_polygon_return_full_area() {
        let region = polygon(
            &[
                (-3.0, -3.0),
                (3.0, -3.0),
                (3.0, 3.0),
                (-3.0, 3.0),
                (-3.0, -3.0),
            ],
            &[],
        );
        let actual =
            circle_areas_inside_regions(&[Point::new(0.0, 0.0)], 1.0, &[region], false).unwrap();
        assert!((actual[0] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn hole_area_is_subtracted() {
        let region = polygon(
            &[
                (-5.0, -5.0),
                (5.0, -5.0),
                (5.0, 5.0),
                (-5.0, 5.0),
                (-5.0, -5.0),
            ],
            &[&[
                (-2.0, -2.0),
                (-2.0, 2.0),
                (2.0, 2.0),
                (2.0, -2.0),
                (-2.0, -2.0),
            ]],
        );
        let actual =
            circle_areas_inside_regions(&[Point::new(0.0, 0.0)], 1.0, &[region], false).unwrap();
        assert_eq!(actual[0], 0.0);
    }

    #[test]
    fn rust_api_rejects_non_finite_geometry_coordinates() {
        let region = polygon(
            &[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)],
            &[],
        );
        let error =
            circle_areas_inside_regions(&[Point::new(f64::NAN, 0.0)], 1.0, &[region], false)
                .unwrap_err();
        assert_eq!(error, super::CircleIntersectionError::NonFiniteCoordinate);
    }
}
