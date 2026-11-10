use std::cmp::Ordering;
use std::f64::consts::PI;
use std::fmt;

use geo::{Intersects, LineString, Point, Polygon};
use num_bigint::BigInt;
use num_traits::ToPrimitive;

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

#[derive(Clone, Copy, Debug)]
struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    fn whole() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    fn add(self, other: Self) -> Self {
        Self::outward(self.lower + other.lower, self.upper + other.upper)
    }

    fn subtract(self, other: Self) -> Self {
        Self::outward(self.lower - other.upper, self.upper - other.lower)
    }

    fn multiply(self, other: Self) -> Self {
        let products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ];
        if products.iter().any(|value| !value.is_finite()) {
            return Self::whole();
        }
        let lower = products.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::outward(lower, upper)
    }

    fn outward(lower: f64, upper: f64) -> Self {
        if !lower.is_finite() || !upper.is_finite() {
            return Self::whole();
        }
        Self {
            lower: next_down(lower),
            upper: next_up(upper),
        }
    }

    fn sign(self) -> Option<Ordering> {
        if self.lower > 0.0 {
            Some(Ordering::Greater)
        } else if self.upper < 0.0 {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

/// An exact dyadic rational: `significand * 2^exponent`.
#[derive(Clone, Debug)]
struct Dyadic {
    significand: BigInt,
    exponent: i32,
}

impl Dyadic {
    fn from_f64(value: f64) -> Self {
        debug_assert!(value.is_finite());
        let bits = value.to_bits();
        let fraction = bits & ((1_u64 << 52) - 1);
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let (mantissa, exponent) = if exponent_bits == 0 {
            (fraction, -1074)
        } else {
            ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
        };
        let significand = BigInt::from(mantissa);
        Self {
            significand: if bits >> 63 == 0 {
                significand
            } else {
                -significand
            },
            exponent,
        }
    }

    fn add(&self, other: &Self) -> Self {
        let exponent = self.exponent.min(other.exponent);
        let left_shift = (self.exponent - exponent) as usize;
        let right_shift = (other.exponent - exponent) as usize;
        Self {
            significand: (&self.significand << left_shift) + (&other.significand << right_shift),
            exponent,
        }
    }

    fn subtract(&self, other: &Self) -> Self {
        let exponent = self.exponent.min(other.exponent);
        let left_shift = (self.exponent - exponent) as usize;
        let right_shift = (other.exponent - exponent) as usize;
        Self {
            significand: (&self.significand << left_shift) - (&other.significand << right_shift),
            exponent,
        }
    }

    fn multiply(&self, other: &Self) -> Self {
        Self {
            significand: &self.significand * &other.significand,
            exponent: self.exponent + other.exponent,
        }
    }

    fn sign(&self) -> Ordering {
        self.significand.cmp(&BigInt::from(0))
    }

    fn leading_mantissa_exponent(&self) -> Option<(f64, i32)> {
        if self.significand == BigInt::from(0) {
            return None;
        }
        let magnitude = if self.sign() == Ordering::Less {
            -self.significand.clone()
        } else {
            self.significand.clone()
        };
        let bit_count = magnitude.bits() as i32;
        let shift = (bit_count - 53).max(0);
        let leading = (magnitude >> shift as usize).to_u64()?;
        Some((leading as f64, self.exponent + shift))
    }

    fn normalized_discriminant(&self, radius: f64) -> f64 {
        let Some((discriminant_mantissa, discriminant_exponent)) = self.leading_mantissa_exponent()
        else {
            return 0.0;
        };
        let Some((radius_mantissa, radius_exponent)) =
            Dyadic::from_f64(radius).leading_mantissa_exponent()
        else {
            return 0.0;
        };
        let mantissa = 4.0 * discriminant_mantissa / radius_mantissa.powi(4);
        scale_by_power_of_two(mantissa, discriminant_exponent - 4 * radius_exponent)
    }
}

fn scale_by_power_of_two(mut value: f64, mut exponent: i32) -> f64 {
    while exponent > 1023 {
        value *= 2.0_f64.powi(1023);
        exponent -= 1023;
        if !value.is_finite() {
            return value;
        }
    }
    while exponent < -1022 {
        value *= 2.0_f64.powi(-1022);
        exponent += 1022;
        if value == 0.0 {
            return 0.0;
        }
    }
    value * 2.0_f64.powi(exponent)
}

fn exact_line_circle_discriminant(line: geo::Line<f64>, center: Point<f64>, radius: f64) -> Dyadic {
    let dx = Dyadic::from_f64(line.end.x).subtract(&Dyadic::from_f64(line.start.x));
    let dy = Dyadic::from_f64(line.end.y).subtract(&Dyadic::from_f64(line.start.y));
    let px = Dyadic::from_f64(line.start.x).subtract(&Dyadic::from_f64(center.x()));
    let py = Dyadic::from_f64(line.start.y).subtract(&Dyadic::from_f64(center.y()));
    let cross = dx.multiply(&py).subtract(&dy.multiply(&px));
    let direction_squared = dx.multiply(&dx).add(&dy.multiply(&dy));
    let radius_squared = Dyadic::from_f64(radius).multiply(&Dyadic::from_f64(radius));
    direction_squared
        .multiply(&radius_squared)
        .subtract(&cross.multiply(&cross))
}

/// Classify the exact line/circle discriminant. The interval filter is a fast
/// path; ambiguous cases use exact dyadic arithmetic on the input f64 values.
fn robust_line_circle_discriminant(
    line: geo::Line<f64>,
    center: Point<f64>,
    radius: f64,
    normalized_estimate: f64,
) -> (Ordering, Option<f64>) {
    let dx = Interval::point(line.end.x).subtract(Interval::point(line.start.x));
    let dy = Interval::point(line.end.y).subtract(Interval::point(line.start.y));
    let px = Interval::point(line.start.x).subtract(Interval::point(center.x()));
    let py = Interval::point(line.start.y).subtract(Interval::point(center.y()));
    let cross = dx.multiply(py).subtract(dy.multiply(px));
    let direction_squared = dx.multiply(dx).add(dy.multiply(dy));
    let radius_squared = Interval::point(radius).multiply(Interval::point(radius));
    let discriminant = direction_squared
        .multiply(radius_squared)
        .subtract(cross.multiply(cross));

    if let Some(sign) = discriminant.sign() {
        if sign != Ordering::Greater
            || (normalized_estimate.is_finite() && normalized_estimate > 0.0)
        {
            return (sign, None);
        }
    }

    let exact = exact_line_circle_discriminant(line, center, radius);
    let sign = exact.sign();
    let magnitude = (sign == Ordering::Greater).then(|| exact.normalized_discriminant(radius));
    (sign, magnitude)
}

/// Area of the intersection between a circle and a closed linear ring.
///
/// Each edge is mapped into unit-circle coordinates and contributes either a
/// triangle area (for an edge portion inside the circle) or a circular-sector
/// area (for an edge portion outside it). The ring is supplied by GeoRust, so
/// its coordinates and edges use the same primitives as the rest of the core.
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
            let approximate_discriminant = b * b - four_ac;
            let (sign, exact_discriminant) =
                robust_line_circle_discriminant(line, center, radius, approximate_discriminant);

            if sign != Ordering::Less {
                let discriminant = exact_discriminant.unwrap_or(approximate_discriminant);
                let root = discriminant.max(0.0).sqrt();
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
    use super::{circle_areas_inside_regions, next_down, next_up, robust_line_circle_discriminant};
    use geo::{Coord, LineString, Point, Polygon};
    use std::cmp::Ordering;

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
    fn robust_discriminant_classifies_tangency_and_adjacent_floats() {
        let center = Point::new(0.0, 0.0);
        let tangent_y = 1.0_f64;
        let just_outside_y = f64::from_bits(tangent_y.to_bits() + 1);
        let just_inside_y = f64::from_bits(tangent_y.to_bits() - 1);

        let (ordinary_sign, fallback_value) = robust_line_circle_discriminant(
            geo::Line::new(Coord { x: -2.0, y: 0.0 }, Coord { x: 2.0, y: 0.0 }),
            center,
            1.0,
            64.0,
        );
        assert_eq!(ordinary_sign, Ordering::Greater);
        assert_eq!(fallback_value, None);

        let classify = |y, estimate| {
            robust_line_circle_discriminant(
                geo::Line::new(Coord { x: -2.0, y }, Coord { x: 2.0, y }),
                center,
                1.0,
                estimate,
            )
        };

        assert_eq!(classify(tangent_y, f64::NAN).0, Ordering::Equal);
        assert_eq!(classify(just_outside_y, f64::NAN).0, Ordering::Less);
        let (inside_sign, recovered_discriminant) = classify(just_inside_y, f64::NAN);
        assert_eq!(inside_sign, Ordering::Greater);
        assert!(recovered_discriminant.unwrap() > 0.0);
    }

    #[test]
    fn robust_discriminant_handles_translation_and_scale() {
        let classify = |center: Point<f64>, radius: f64, y: f64| {
            robust_line_circle_discriminant(
                geo::Line::new(
                    Coord {
                        x: center.x() - 2.0 * radius,
                        y,
                    },
                    Coord {
                        x: center.x() + 2.0 * radius,
                        y,
                    },
                ),
                center,
                radius,
                f64::NAN,
            )
            .0
        };

        let translated_center = Point::new(1.0e12, 1.0e12);
        let translated_tangent = translated_center.y() + 1.0;
        assert_eq!(
            classify(translated_center, 1.0, translated_tangent),
            Ordering::Equal
        );

        for radius in [1.0e-100, 1.0e100] {
            let center = Point::new(0.0, 0.0);
            assert_eq!(classify(center, radius, radius), Ordering::Equal);
            assert_eq!(classify(center, radius, next_up(radius)), Ordering::Less);
            assert_eq!(
                classify(center, radius, next_down(radius)),
                Ordering::Greater
            );
        }
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
