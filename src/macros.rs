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
        }
    };
}

macro_rules! match_shape {
    ($self:ident, $method:ident) => {
        match &$self.inner {
            Shapes::Point(p) => p.as_ref().$method(),
            Shapes::MultiPoint(p) => p.as_ref().$method(),
            Shapes::LineString(p) => p.$method(),
            Shapes::MultiLineString(p) => p.$method(),
            Shapes::MultiPolygon(p) => p.$method(),
            Shapes::Polygon(p) => p.$method(),
        }
    };
}
