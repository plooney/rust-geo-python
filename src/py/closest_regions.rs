use crate::py::enums::Shapes;
use geo::algorithm::bool_ops::BooleanOps;
use geo::line_intersection::{line_intersection, LineIntersection};
use geo::{
    Closest, ClosestPoint, Coord, Distance, Euclidean, Geometry, Line, LineString, MultiLineString,
    Point, Polygon,
};

const REGION_EPSILON: f64 = 1e-12;

#[derive(Clone, Copy)]
enum SegmentRegion {
    Point(Point<f64>),
    Line(Line<f64>),
}

#[derive(Clone, Copy)]
enum SourceKind {
    Line,
    LineString,
    MultiLineString,
}

fn push_unique_point(points: &mut Vec<Point<f64>>, candidate: Point<f64>) {
    if points.iter().any(|point| {
        (point.x() - candidate.x()).abs() <= REGION_EPSILON
            && (point.y() - candidate.y()).abs() <= REGION_EPSILON
    }) {
        return;
    }

    points.push(candidate);
}

fn push_unique_line(lines: &mut Vec<Line<f64>>, candidate: Line<f64>) {
    if lines.iter().any(|line| {
        ((line.start.x - candidate.start.x).abs() <= REGION_EPSILON
            && (line.start.y - candidate.start.y).abs() <= REGION_EPSILON
            && (line.end.x - candidate.end.x).abs() <= REGION_EPSILON
            && (line.end.y - candidate.end.y).abs() <= REGION_EPSILON)
            || ((line.start.x - candidate.end.x).abs() <= REGION_EPSILON
                && (line.start.y - candidate.end.y).abs() <= REGION_EPSILON
                && (line.end.x - candidate.start.x).abs() <= REGION_EPSILON
                && (line.end.y - candidate.start.y).abs() <= REGION_EPSILON)
    }) {
        return;
    }

    lines.push(candidate);
}

fn point_from_coord(coord: Coord<f64>) -> Point<f64> {
    Point::new(coord.x, coord.y)
}

fn line_to_linestring(line: Line<f64>) -> LineString<f64> {
    LineString::new(vec![line.start, line.end])
}

fn linestring_to_segments(linestring: &LineString<f64>) -> Vec<Line<f64>> {
    linestring.lines().collect()
}

fn multilinestring_to_segments(multilinestring: &MultiLineString<f64>) -> Vec<Line<f64>> {
    multilinestring
        .iter()
        .flat_map(|linestring| linestring.lines())
        .collect()
}

fn polygon_boundary_lines(polygon: &Polygon<f64>) -> Vec<LineString<f64>> {
    let mut lines = vec![polygon.exterior().clone()];
    lines.extend(polygon.interiors().iter().cloned());
    lines
}

fn polygon_boundary_segments(polygon: &Polygon<f64>) -> Vec<Line<f64>> {
    polygon_boundary_lines(polygon)
        .into_iter()
        .flat_map(|linestring| linestring.lines().collect::<Vec<_>>())
        .collect()
}

fn shape_to_linear_geometry(shape: &Shapes) -> Option<(Geometry<f64>, SourceKind)> {
    match shape {
        Shapes::Line(line) => Some((Geometry::Line(*line.as_ref()), SourceKind::Line)),
        Shapes::LineString(linestring) => Some((
            Geometry::LineString(linestring.as_ref().clone()),
            SourceKind::LineString,
        )),
        Shapes::MultiLineString(multilinestring) => Some((
            Geometry::MultiLineString(multilinestring.as_ref().clone()),
            SourceKind::MultiLineString,
        )),
        Shapes::Polygon(polygon) => Some((
            Geometry::MultiLineString(MultiLineString::new(polygon_boundary_lines(
                polygon.as_ref(),
            ))),
            SourceKind::MultiLineString,
        )),
        Shapes::MultiPolygon(multipolygon) => {
            let lines = multipolygon
                .iter()
                .flat_map(polygon_boundary_lines)
                .collect::<Vec<_>>();
            Some((
                Geometry::MultiLineString(MultiLineString::new(lines)),
                SourceKind::MultiLineString,
            ))
        }
        Shapes::Triangle(triangle) => Some((
            Geometry::LineString(triangle.to_polygon().exterior().clone()),
            SourceKind::LineString,
        )),
        Shapes::Rect(rect) => Some((
            Geometry::LineString(rect.to_polygon().exterior().clone()),
            SourceKind::LineString,
        )),
        _ => None,
    }
}

fn geometry_to_multilinestring(geometry: &Geometry<f64>) -> Option<MultiLineString<f64>> {
    match geometry {
        Geometry::Line(line) => Some(MultiLineString::new(vec![line_to_linestring(*line)])),
        Geometry::LineString(linestring) => Some(MultiLineString::new(vec![linestring.clone()])),
        Geometry::MultiLineString(multilinestring) => Some(multilinestring.clone()),
        _ => None,
    }
}

fn point_distance_to_segment(point: Point<f64>, segment: &Line<f64>) -> f64 {
    match segment.closest_point(&point) {
        Closest::Intersection(p) | Closest::SinglePoint(p) => Euclidean.distance(&point, &p),
        Closest::Indeterminate => Euclidean.distance(&point, &point_from_coord(segment.start)),
    }
}

fn segment_region_distance(region: SegmentRegion, target: &Line<f64>) -> f64 {
    match region {
        SegmentRegion::Point(point) => point_distance_to_segment(point, target),
        SegmentRegion::Line(line) => {
            point_distance_to_segment(point_from_coord(line.start), target)
        }
    }
}

fn closest_region_between_segments(source: &Line<f64>, target: &Line<f64>) -> SegmentRegion {
    if let Some(intersection) = line_intersection(*source, *target) {
        return match intersection {
            LineIntersection::SinglePoint { intersection, .. } => {
                SegmentRegion::Point(point_from_coord(intersection))
            }
            LineIntersection::Collinear { intersection } => SegmentRegion::Line(intersection),
        };
    }

    let source_dx = source.end.x - source.start.x;
    let source_dy = source.end.y - source.start.y;
    let target_dx = target.end.x - target.start.x;
    let target_dy = target.end.y - target.start.y;
    let source_length_squared = source_dx * source_dx + source_dy * source_dy;
    let cross = source_dx * target_dy - source_dy * target_dx;

    if source_length_squared > 0.0 && cross.abs() <= REGION_EPSILON * source_length_squared.sqrt() {
        let start_projection = ((target.start.x - source.start.x) * source_dx
            + (target.start.y - source.start.y) * source_dy)
            / source_length_squared;
        let end_projection = ((target.end.x - source.start.x) * source_dx
            + (target.end.y - source.start.y) * source_dy)
            / source_length_squared;
        let overlap_start = 0.0_f64.max(start_projection.min(end_projection));
        let overlap_end = 1.0_f64.min(start_projection.max(end_projection));

        if overlap_end - overlap_start > REGION_EPSILON {
            let point_at = |projection: f64| {
                Point::new(
                    source.start.x + projection * source_dx,
                    source.start.y + projection * source_dy,
                )
            };
            let start = point_at(overlap_start);
            let end = point_at(overlap_end);
            return SegmentRegion::Line(Line::new(
                Coord {
                    x: start.x(),
                    y: start.y(),
                },
                Coord {
                    x: end.x(),
                    y: end.y(),
                },
            ));
        }
    }

    let mut best_point = point_from_coord(source.start);
    let mut best_distance = f64::INFINITY;

    let endpoint_candidates = [point_from_coord(source.start), point_from_coord(source.end)];
    for candidate in endpoint_candidates {
        let distance = point_distance_to_segment(candidate, target);
        if distance + REGION_EPSILON < best_distance {
            best_distance = distance;
            best_point = candidate;
        }
    }

    let projected_candidates = [point_from_coord(target.start), point_from_coord(target.end)];
    for target_point in projected_candidates {
        let candidate = match source.closest_point(&target_point) {
            Closest::Intersection(point) | Closest::SinglePoint(point) => point,
            Closest::Indeterminate => point_from_coord(source.start),
        };
        let distance = point_distance_to_segment(candidate, target);
        if distance + REGION_EPSILON < best_distance {
            best_distance = distance;
            best_point = candidate;
        }
    }

    SegmentRegion::Point(best_point)
}

fn normalize_multilinestring(
    multilinestring: MultiLineString<f64>,
    source_kind: SourceKind,
) -> Option<Geometry<f64>> {
    let mut point_candidates = Vec::new();
    let mut line_candidates = Vec::new();
    let mut linestring_candidates = Vec::new();

    for linestring in multilinestring.0 {
        let segments = linestring_to_segments(&linestring);
        if segments.is_empty() {
            let points = linestring.points().collect::<Vec<_>>();
            if let Some(point) = points.first() {
                push_unique_point(&mut point_candidates, *point);
            }
            continue;
        }

        if segments.len() == 1 && linestring.0.len() == 2 {
            let line = segments[0];
            if (line.start.x - line.end.x).abs() <= REGION_EPSILON
                && (line.start.y - line.end.y).abs() <= REGION_EPSILON
            {
                push_unique_point(&mut point_candidates, point_from_coord(line.start));
            } else {
                push_unique_line(&mut line_candidates, line);
            }
        } else {
            linestring_candidates.push(linestring);
        }
    }

    if !linestring_candidates.is_empty() {
        if !line_candidates.is_empty() {
            linestring_candidates.extend(line_candidates.into_iter().map(line_to_linestring));
        }
        return Some(if linestring_candidates.len() == 1 {
            Geometry::LineString(linestring_candidates.remove(0))
        } else {
            Geometry::MultiLineString(MultiLineString::new(linestring_candidates))
        });
    }

    if !line_candidates.is_empty() {
        if line_candidates.len() == 1 {
            let line = line_candidates[0];
            return Some(match source_kind {
                SourceKind::Line => Geometry::Line(line),
                _ => Geometry::LineString(line_to_linestring(line)),
            });
        }

        return Some(Geometry::MultiLineString(MultiLineString::new(
            line_candidates
                .into_iter()
                .map(line_to_linestring)
                .collect(),
        )));
    }

    point_candidates
        .into_iter()
        .min_by(|left, right| {
            left.x()
                .total_cmp(&right.x())
                .then_with(|| left.y().total_cmp(&right.y()))
        })
        .map(Geometry::Point)
}

pub(crate) fn closest_region_to_polygon(
    shape: &Shapes,
    polygon: &Polygon<f64>,
) -> Option<Geometry<f64>> {
    match shape {
        Shapes::Point(point) => Some(Geometry::Point(*point.as_ref())),
        Shapes::Line(_)
        | Shapes::LineString(_)
        | Shapes::MultiLineString(_)
        | Shapes::Polygon(_)
        | Shapes::MultiPolygon(_)
        | Shapes::Triangle(_)
        | Shapes::Rect(_) => {
            let (source_geometry, source_kind) = shape_to_linear_geometry(shape)?;
            let source_multilinestring = geometry_to_multilinestring(&source_geometry)?;
            let zero_distance_region = polygon.clip(&source_multilinestring, false);
            if let Some(geometry) = normalize_multilinestring(zero_distance_region, source_kind) {
                return Some(geometry);
            }

            let source_segments = multilinestring_to_segments(&source_multilinestring);
            let polygon_segments = polygon_boundary_segments(polygon);
            let mut best_distance = f64::INFINITY;
            let mut best_points = Vec::new();
            let mut best_lines = Vec::new();

            for source_segment in &source_segments {
                for polygon_segment in &polygon_segments {
                    let region = closest_region_between_segments(source_segment, polygon_segment);
                    let distance = segment_region_distance(region, polygon_segment);

                    if distance + REGION_EPSILON < best_distance {
                        best_distance = distance;
                        best_points.clear();
                        best_lines.clear();
                    }

                    if (distance - best_distance).abs() <= REGION_EPSILON {
                        match region {
                            SegmentRegion::Point(point) => {
                                push_unique_point(&mut best_points, point)
                            }
                            SegmentRegion::Line(line) => push_unique_line(&mut best_lines, line),
                        }
                    }
                }
            }

            if !best_lines.is_empty() {
                return Some(if best_lines.len() == 1 {
                    match source_kind {
                        SourceKind::Line => Geometry::Line(best_lines[0]),
                        _ => Geometry::LineString(line_to_linestring(best_lines[0])),
                    }
                } else {
                    Geometry::MultiLineString(MultiLineString::new(
                        best_lines.into_iter().map(line_to_linestring).collect(),
                    ))
                });
            }

            best_points
                .into_iter()
                .min_by(|left, right| {
                    left.x()
                        .total_cmp(&right.x())
                        .then_with(|| left.y().total_cmp(&right.y()))
                })
                .map(Geometry::Point)
        }
        Shapes::MultiPoint(_) | Shapes::GeometryCollection(_) => None,
    }
}
