use convert::*;
use geo::BooleanOps;
use geo::GeoFloat;
use geo::MultiPolygon;
use geo::Polygon;
use geo::Rect;
use geo::geometry::Coord;
use geo::winding_order::{Winding, WindingOrder};
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay::ShapeType;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::overlay::FloatOverlay;
use i_overlay::i_float::adapter::FloatPointAdapter;
use i_overlay::i_float::float::compatible::FloatPointCompatible;
use i_overlay::i_float::float::number::FloatNumber;
use i_overlay::i_float::float::rect::FloatRect;
use num_traits::FromPrimitive;

/// A geometry coordinate scalar suitable for performing geometric boolean operations.
pub trait BoolOpsNum: GeoFloat + FloatNumber + FromPrimitive {}
impl<T: GeoFloat + FloatNumber + FromPrimitive> BoolOpsNum for T {}

/// New type for `Coord` that implements `FloatPointCompatible` for `BoolOpsNum` to
/// circumvent orphan rule, since Coord is defined in geo_types.
#[derive(Copy, Clone, Debug)]
pub struct BoolOpsCoord<T: BoolOpsNum>(pub(crate) Coord<T>);

impl<T: BoolOpsNum> FloatPointCompatible<T> for BoolOpsCoord<T> {
    fn from_xy(x: T, y: T) -> Self {
        Self(Coord { x, y })
    }

    fn x(&self) -> T {
        self.0.x
    }

    fn y(&self) -> T {
        self.0.y
    }
}

pub(crate) mod convert {
    use super::BoolOpsCoord;
    use super::BoolOpsNum;
    use geo::geometry::{Coord, LineString, MultiLineString, MultiPolygon, Polygon};

    pub fn line_string_from_path<T: BoolOpsNum>(path: Vec<BoolOpsCoord<T>>) -> LineString<T> {
        let coords = path.into_iter().map(|bops_coord| bops_coord.0).collect();
        LineString(coords)
    }

    #[allow(dead_code)]
    pub fn multi_line_string_from_paths<T: BoolOpsNum>(
        paths: Vec<Vec<BoolOpsCoord<T>>>,
    ) -> MultiLineString<T> {
        let line_strings = paths.into_iter().map(|p| line_string_from_path(p));
        MultiLineString(line_strings.collect())
    }

    pub fn polygon_from_shape<T: BoolOpsNum>(shape: Vec<Vec<BoolOpsCoord<T>>>) -> Polygon<T> {
        let mut rings = shape.into_iter().map(|path| {
            let mut line_string = line_string_from_path(path);
            line_string.close();
            line_string
        });
        let exterior = rings.next().unwrap_or(LineString::empty());

        Polygon::new(exterior, rings.collect())
    }

    pub fn multi_polygon_from_shapes<T: BoolOpsNum>(
        shapes: Vec<Vec<Vec<BoolOpsCoord<T>>>>,
    ) -> MultiPolygon<T> {
        let polygons = shapes.into_iter().map(|s| polygon_from_shape(s));
        MultiPolygon(polygons.collect())
    }

    pub fn ring_to_shape_path<T: BoolOpsNum>(line_string: &LineString<T>) -> Vec<BoolOpsCoord<T>> {
        if line_string.0.is_empty() {
            return vec![];
        }
        // In geo, Polygon rings are explicitly closed LineStrings — their final coordinate is the same as their first coordinate,
        // however in i_overlay, shape paths are implicitly closed, so we skip the last coordinate.
        let coords = &line_string.0[..line_string.0.len() - 1];
        coords.iter().copied().map(BoolOpsCoord).collect()
    }

    #[allow(dead_code)]
    pub fn line_string_to_shape_path<T: BoolOpsNum>(
        line_string: &LineString<T>,
    ) -> Vec<BoolOpsCoord<T>> {
        line_string.coords().copied().map(BoolOpsCoord).collect()
    }

    impl<T: BoolOpsNum> From<Coord<T>> for BoolOpsCoord<T> {
        fn from(value: Coord<T>) -> Self {
            BoolOpsCoord(value)
        }
    }
}

pub fn intersect_tile_using_buffered_adapter(
    polygon: &Polygon,
    tile_polygon: &Polygon,
    adapter: FloatPointAdapter<BoolOpsCoord<f64>, f64>,
) -> MultiPolygon {
    let subject = polygon.rings().map(ring_to_shape_path).collect::<Vec<_>>();
    let clip = tile_polygon
        .rings()
        .map(ring_to_shape_path)
        .collect::<Vec<_>>();

    let shapes = FloatOverlay::with_adapter(adapter, subject.len() + clip.len())
        .unsafe_add_source(&subject, ShapeType::Subject)
        .unsafe_add_source(&clip, ShapeType::Clip)
        .overlay(OverlayRule::Intersect, FillRule::EvenOdd);
    return multi_polygon_from_shapes(shapes);
}

pub fn intersect_tile_using_buffered_adapter_mpg(
    multipolygon: &MultiPolygon,
    tile_polygon: &Polygon,
    rect: &Rect,
) -> MultiPolygon {
    let buffer_rect = FloatRect::new(rect.min().x, rect.max().x, rect.min().y, rect.max().y);
    let adapter = FloatPointAdapter::new(buffer_rect);
    let mpgs = multipolygon
        .iter()
        .flat_map(|x| intersect_tile_using_buffered_adapter(x, tile_polygon, adapter.clone()))
        .collect::<MultiPolygon>();
    return mpgs;
}

pub fn unary_union_with_adapter<'a, B: BooleanOps + 'a>(
    boppables: impl IntoIterator<Item = &'a B>,
    rect: &Rect,
) -> MultiPolygon<B::Scalar>
where
    BoolOpsCoord<<B as BooleanOps>::Scalar>: FloatPointCompatible<f64>,
{
    let mut winding_order: Option<WindingOrder> = None;
    let subject = boppables
        .into_iter()
        .flat_map(|boppable| {
            let rings = boppable.rings();
            rings
                .map(|ring| {
                    if winding_order.is_none() {
                        winding_order = ring.winding_order();
                    }
                    ring_to_shape_path(ring)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let fill_rule = if winding_order == Some(WindingOrder::CounterClockwise) {
        FillRule::Positive
    } else {
        FillRule::Negative
    };

    let buffer_rect = FloatRect::new(rect.min().x, rect.max().x, rect.min().y, rect.max().y);
    let adapter = FloatPointAdapter::new(buffer_rect);
    let shapes = FloatOverlay::with_adapter(adapter, subject.len())
        .unsafe_add_source(&subject, ShapeType::Subject)
        .overlay(OverlayRule::Subject, fill_rule);
    multi_polygon_from_shapes(shapes)
}
