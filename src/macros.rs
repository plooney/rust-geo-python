macro_rules! match_shapes_method {
    ($self:ident, $rhs:ident, $method:ident) => {
        match (&$self.inner, &$rhs.inner) {
            (Shapes::Point(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Point(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiLineString(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiLineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiLineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::MultiLineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Point(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiLineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Point(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Point(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),

            // Line arms
            (Shapes::Line(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Line(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Line(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Line(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Line(p), Shapes::MultiLineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Line(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Line(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),

            (Shapes::Point(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),

            // Triangle arms
            (Shapes::Triangle(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Triangle(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Triangle(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Triangle(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Triangle(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Triangle(p), Shapes::MultiLineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Triangle(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Triangle(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),

            (Shapes::Line(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Point(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),

            // Rect arms
            (Shapes::Rect(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::MultiPoint(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::LineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::MultiLineString(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::MultiPolygon(q)) => p.as_ref().$method(q.as_ref()),

            (Shapes::Triangle(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Line(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Point(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Polygon(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),

            (Shapes::GeometryCollection(p), Shapes::Point(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::GeometryCollection(p), Shapes::Line(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::GeometryCollection(p), Shapes::LineString(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::MultiPoint(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::MultiLineString(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::Polygon(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::GeometryCollection(p), Shapes::MultiPolygon(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::Triangle(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::GeometryCollection(p), Shapes::Rect(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::GeometryCollection(p), Shapes::GeometryCollection(q)) => {
                p.as_ref().$method(q.as_ref())
            }

            (Shapes::Point(p), Shapes::GeometryCollection(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Line(p), Shapes::GeometryCollection(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::LineString(p), Shapes::GeometryCollection(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::GeometryCollection(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::GeometryCollection(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::GeometryCollection(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::GeometryCollection(q)) => {
                p.as_ref().$method(q.as_ref())
            }
            (Shapes::Triangle(p), Shapes::GeometryCollection(q)) => p.as_ref().$method(q.as_ref()),
            (Shapes::Rect(p), Shapes::GeometryCollection(q)) => p.as_ref().$method(q.as_ref()),
        }
    };
}

macro_rules! match_shapes_algo {
    ($self:ident, $rhs:ident, $algo:ident, $method:ident) => {
        match (&$self.inner, &$rhs.inner) {
            (Shapes::Point(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Point(p), Shapes::LineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::LineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::LineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiLineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiLineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiLineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Polygon(p), Shapes::LineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Polygon(p), Shapes::MultiLineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Polygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Point(p), Shapes::Polygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::Polygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Polygon(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::LineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::MultiLineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::Polygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPolygon(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Point(p), Shapes::MultiPolygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPolygon(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::MultiPolygon(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPolygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::LineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiLineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::Polygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::MultiPolygon(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::MultiPoint(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Point(p), Shapes::MultiPoint(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::MultiPoint(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::MultiPoint(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::MultiPoint(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::MultiPoint(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }

            // Line arms
            (Shapes::Line(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Line(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Line(p), Shapes::MultiPoint(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Line(p), Shapes::LineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Line(p), Shapes::MultiLineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Line(p), Shapes::Polygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Line(p), Shapes::MultiPolygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),

            (Shapes::Point(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Polygon(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),

            // Triangle arms
            (Shapes::Triangle(p), Shapes::Triangle(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Triangle(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Triangle(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Triangle(p), Shapes::MultiPoint(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Triangle(p), Shapes::LineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Triangle(p), Shapes::MultiLineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Triangle(p), Shapes::Polygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Triangle(p), Shapes::MultiPolygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),

            (Shapes::Line(p), Shapes::Triangle(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Point(p), Shapes::Triangle(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Triangle(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::Triangle(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Triangle(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::Triangle(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Triangle(q)) => $algo.$method(p.as_ref(), q.as_ref()),

            // Rect arms
            (Shapes::Rect(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Rect(p), Shapes::Triangle(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Rect(p), Shapes::Line(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Rect(p), Shapes::Point(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Rect(p), Shapes::MultiPoint(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Rect(p), Shapes::LineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Rect(p), Shapes::MultiLineString(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Rect(p), Shapes::Polygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Rect(p), Shapes::MultiPolygon(q)) => $algo.$method(p.as_ref(), q.as_ref()),

            (Shapes::Triangle(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Line(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Point(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPoint(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::LineString(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiLineString(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::Polygon(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),
            (Shapes::MultiPolygon(p), Shapes::Rect(q)) => $algo.$method(p.as_ref(), q.as_ref()),

            (Shapes::GeometryCollection(p), Shapes::Point(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::Line(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::LineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::MultiPoint(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::MultiLineString(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::Polygon(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::MultiPolygon(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::Triangle(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::Rect(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::GeometryCollection(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }

            (Shapes::Point(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Line(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::LineString(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPoint(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiLineString(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Polygon(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::MultiPolygon(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Triangle(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
            (Shapes::Rect(p), Shapes::GeometryCollection(q)) => {
                $algo.$method(p.as_ref(), q.as_ref())
            }
        }
    };
}

macro_rules! match_shape {
    ($self:ident, $method:ident) => {
        match &$self.inner {
            Shapes::Point(p) => p.$method(),
            Shapes::MultiPoint(p) => p.$method(),
            Shapes::LineString(p) => p.$method(),
            Shapes::MultiLineString(p) => p.$method(),
            Shapes::MultiPolygon(p) => p.$method(),
            Shapes::Polygon(p) => p.$method(),
            Shapes::Line(p) => p.$method(),
            Shapes::Triangle(p) => p.$method(),
            Shapes::Rect(p) => p.$method(),
            Shapes::GeometryCollection(p) => p.$method(),
        }
    };
}

macro_rules! match_shape_arg {
    ($self:ident, $method:ident, $arg: ident) => {
        match &$self.inner {
            Shapes::Point(p) => p.$method($arg),
            Shapes::MultiPoint(p) => p.$method($arg),
            Shapes::LineString(p) => p.$method($arg),
            Shapes::MultiLineString(p) => p.$method($arg),
            Shapes::MultiPolygon(p) => p.$method($arg),
            Shapes::Polygon(p) => p.$method($arg),
            Shapes::Line(p) => p.$method($arg),
            Shapes::Triangle(p) => p.$method($arg),
            Shapes::Rect(p) => p.$method($arg),
            Shapes::GeometryCollection(p) => p.$method($arg),
        }
    };
}
