use crate::py::enums::Shapes;
use geo::{
    Closest, ClosestPoint, Distance, Euclidean, Geometry, Intersects, Line, LineString, Point,
};
use std::sync::Arc;

const CLOSEST_POINT_EPSILON: f64 = 1e-12;

fn push_unique_point(points: &mut Vec<Point<f64>>, candidate: Point<f64>) {
    if points.iter().any(|point| {
        (point.x() - candidate.x()).abs() <= CLOSEST_POINT_EPSILON
            && (point.y() - candidate.y()).abs() <= CLOSEST_POINT_EPSILON
    }) {
        return;
    }

    points.push(candidate);
}

fn collect_best_points(
    best_points: &mut Vec<Point<f64>>,
    best_distance: &mut Option<f64>,
    candidates: Vec<Point<f64>>,
    query: &Point<f64>,
) {
    for candidate in candidates {
        let distance = Euclidean.distance(&candidate, query);
        match *best_distance {
            None => {
                *best_distance = Some(distance);
                best_points.clear();
                push_unique_point(best_points, candidate);
            }
            Some(best) if distance + CLOSEST_POINT_EPSILON < best => {
                *best_distance = Some(distance);
                best_points.clear();
                push_unique_point(best_points, candidate);
            }
            Some(best) if (distance - best).abs() <= CLOSEST_POINT_EPSILON => {
                push_unique_point(best_points, candidate);
            }
            _ => {}
        }
    }
}

fn collect_closest_points_from_line(line: &Line<f64>, query: &Point<f64>) -> Vec<Point<f64>> {
    if line.start == line.end {
        return vec![Point::from(line.start)];
    }

    match line.closest_point(query) {
        Closest::Intersection(point) | Closest::SinglePoint(point) => vec![point],
        Closest::Indeterminate => vec![],
    }
}

fn collect_closest_points_from_linestring(
    line_string: &LineString<f64>,
    query: &Point<f64>,
) -> Vec<Point<f64>> {
    if line_string.0.is_empty() {
        return vec![];
    }

    if line_string.0.len() == 1 {
        return vec![Point::from(line_string.0[0])];
    }

    let mut best_points = Vec::new();
    let mut best_distance = None;

    for line in line_string.lines() {
        collect_best_points(
            &mut best_points,
            &mut best_distance,
            collect_closest_points_from_line(&line, query),
            query,
        );
    }

    best_points
}

pub(crate) fn collect_closest_points_from_shape(
    shape: &Shapes,
    query: &Point<f64>,
) -> Vec<Point<f64>> {
    match shape {
        Shapes::Point(point) => vec![point.as_ref().to_owned()],
        Shapes::Line(line) => collect_closest_points_from_line(line.as_ref(), query),
        Shapes::MultiPoint(multipoint) => {
            let mut best_points = Vec::new();
            let mut best_distance = None;

            for point in multipoint.iter() {
                collect_best_points(&mut best_points, &mut best_distance, vec![*point], query);
            }

            best_points
        }
        Shapes::LineString(line_string) => {
            collect_closest_points_from_linestring(line_string.as_ref(), query)
        }
        Shapes::MultiLineString(multiline_string) => {
            let mut best_points = Vec::new();
            let mut best_distance = None;

            for line_string in multiline_string.iter() {
                collect_best_points(
                    &mut best_points,
                    &mut best_distance,
                    collect_closest_points_from_linestring(&line_string, query),
                    query,
                );
            }

            best_points
        }
        Shapes::Polygon(polygon) => {
            if polygon.intersects(query) {
                return vec![*query];
            }

            let mut best_points = Vec::new();
            let mut best_distance = None;

            for ring in polygon.interiors().iter().chain(std::iter::once(polygon.exterior())) {
                collect_best_points(
                    &mut best_points,
                    &mut best_distance,
                    collect_closest_points_from_linestring(ring, query),
                    query,
                );
            }

            best_points
        }
        Shapes::MultiPolygon(multipolygon) => {
            let mut best_points = Vec::new();
            let mut best_distance = None;

            for polygon in multipolygon.iter() {
                collect_best_points(
                    &mut best_points,
                    &mut best_distance,
                    collect_closest_points_from_shape(&Shapes::Polygon(Arc::new(polygon.clone())), query),
                    query,
                );
            }

            best_points
        }
        Shapes::Triangle(triangle) => {
            if triangle.intersects(query) {
                return vec![*query];
            }

            let mut best_points = Vec::new();
            let mut best_distance = None;

            for line in triangle.to_lines() {
                collect_best_points(
                    &mut best_points,
                    &mut best_distance,
                    collect_closest_points_from_line(&line, query),
                    query,
                );
            }

            best_points
        }
        Shapes::Rect(rect) => {
            if rect.intersects(query) {
                return vec![*query];
            }

            let mut best_points = Vec::new();
            let mut best_distance = None;

            for line in rect.to_lines() {
                collect_best_points(
                    &mut best_points,
                    &mut best_distance,
                    collect_closest_points_from_line(&line, query),
                    query,
                );
            }

            best_points
        }
        Shapes::GeometryCollection(collection) => {
            let mut best_points = Vec::new();
            let mut best_distance = None;

            for geometry in collection.iter() {
                let shape = match geometry {
                    Geometry::Point(point) => Shapes::Point(Arc::new(point.clone())),
                    Geometry::Line(line) => Shapes::Line(Arc::new(line.clone())),
                    Geometry::LineString(line_string) => {
                        Shapes::LineString(Arc::new(line_string.clone()))
                    }
                    Geometry::MultiPoint(multipoint) => Shapes::MultiPoint(Arc::new(multipoint.clone())),
                    Geometry::MultiLineString(multiline_string) => {
                        Shapes::MultiLineString(Arc::new(multiline_string.clone()))
                    }
                    Geometry::Polygon(polygon) => Shapes::Polygon(Arc::new(polygon.clone())),
                    Geometry::MultiPolygon(multipolygon) => {
                        Shapes::MultiPolygon(Arc::new(multipolygon.clone()))
                    }
                    Geometry::Triangle(triangle) => Shapes::Triangle(Arc::new(triangle.clone())),
                    Geometry::Rect(rect) => Shapes::Rect(Arc::new(rect.clone())),
                    Geometry::GeometryCollection(inner) => {
                        Shapes::GeometryCollection(Arc::new(inner.clone()))
                    }
                };

                collect_best_points(
                    &mut best_points,
                    &mut best_distance,
                    collect_closest_points_from_shape(&shape, query),
                    query,
                );
            }

            best_points
        }
    }
}
