package org.bytedeco.pytorch.dataframe.geo;

import org.bytedeco.pytorch.dataframe.dtype.AbstractDataValue;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;
import org.locationtech.jts.geom.PrecisionModel;
import org.locationtech.jts.io.WKBReader;
import org.locationtech.jts.io.WKBWriter;
import org.locationtech.jts.io.WKTReader;
import org.locationtech.jts.io.WKTWriter;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Geometry cell value: WKT / WKB / GeoJSON backed by JTS {@link Geometry}.
 *
 * <p>Uses {@code org.locationtech.jts} already on the project classpath (via parquet).
 * GeoJSON support is a minimal pure-Java subset (Point / simple Polygon) — full
 * GeoJSON module ({@code jts-io-common}) is intentionally not required.
 * Complex topology beyond client-side predicates should be pushed to DuckDB (plan non-goal).
 */
public final class GeoData extends AbstractDataValue {

    private static final long serialVersionUID = 1L;
    private static final GeometryFactory GF = new GeometryFactory(new PrecisionModel(), 4326);

    private final Geometry geometry;
    private final CRS crs;

    public GeoData(Geometry geometry, CRS crs) {
        this.geometry = Objects.requireNonNull(geometry, "geometry").copy();
        this.crs = crs == null ? CRS.WGS84 : crs;
        if (this.geometry.getSRID() == 0) {
            this.crs.epsg().ifPresent(this.geometry::setSRID);
        }
    }

    public static GeoData fromWkt(String wkt) {
        return fromWkt(wkt, CRS.WGS84);
    }

    public static GeoData fromWkt(String wkt, CRS crs) {
        Objects.requireNonNull(wkt, "wkt");
        try {
            Geometry g = new WKTReader(GF).read(wkt);
            return new GeoData(g, crs);
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid WKT: " + wkt, e);
        }
    }

    public static GeoData fromWkb(byte[] wkb) {
        return fromWkb(wkb, CRS.WGS84);
    }

    public static GeoData fromWkb(byte[] wkb, CRS crs) {
        Objects.requireNonNull(wkb, "wkb");
        try {
            Geometry g = new WKBReader(GF).read(wkb);
            return new GeoData(g, crs);
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid WKB (" + wkb.length + " bytes)", e);
        }
    }

    public static GeoData fromGeoJson(String geoJson) {
        return fromGeoJson(geoJson, CRS.WGS84);
    }

    /**
     * Minimal GeoJSON parser (no jts-io dependency):
     * <ul>
     *   <li>{@code {"type":"Point","coordinates":[lon,lat]}}</li>
     *   <li>{@code {"type":"Polygon","coordinates":[[[lon,lat],...] ]}}</li>
     * </ul>
     */
    public static GeoData fromGeoJson(String geoJson, CRS crs) {
        Objects.requireNonNull(geoJson, "geoJson");
        String s = geoJson.trim();
        String lower = s.toLowerCase(Locale.ROOT);
        try {
            if (lower.contains("\"point\"")) {
                double[] xy = firstCoordPair(s);
                return point(xy[0], xy[1], crs);
            }
            if (lower.contains("\"polygon\"")) {
                java.util.List<Coordinate> coords = allCoordPairs(s);
                if (coords.size() < 4) {
                    throw new IllegalArgumentException("Polygon needs >= 4 positions (closed ring)");
                }
                // ensure closed
                Coordinate first = coords.get(0);
                Coordinate last = coords.get(coords.size() - 1);
                if (!first.equals2D(last)) coords.add(new Coordinate(first));
                Coordinate[] arr = coords.toArray(new Coordinate[0]);
                Geometry g = GF.createPolygon(arr);
                return new GeoData(g, crs);
            }
            if (lower.contains("\"linestring\"")) {
                java.util.List<Coordinate> coords = allCoordPairs(s);
                if (coords.size() < 2) throw new IllegalArgumentException("LineString needs >= 2 positions");
                Geometry g = GF.createLineString(coords.toArray(new Coordinate[0]));
                return new GeoData(g, crs);
            }
            // fallback: first coordinate pair as point
            double[] xy = firstCoordPair(s);
            return point(xy[0], xy[1], crs);
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalArgumentException("invalid GeoJSON", e);
        }
    }

    private static double[] firstCoordPair(String s) {
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\\[\\s*([-0-9.eE+]+)\\s*,\\s*([-0-9.eE+]+)")
                .matcher(s);
        if (!m.find()) throw new IllegalArgumentException("no coordinates in GeoJSON");
        return new double[]{Double.parseDouble(m.group(1)), Double.parseDouble(m.group(2))};
    }

    private static java.util.List<Coordinate> allCoordPairs(String s) {
        java.util.List<Coordinate> out = new java.util.ArrayList<>();
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\\[\\s*([-0-9.eE+]+)\\s*,\\s*([-0-9.eE+]+)\\s*(?:,\\s*[-0-9.eE+]+)?\\s*\\]")
                .matcher(s);
        while (m.find()) {
            out.add(new Coordinate(Double.parseDouble(m.group(1)), Double.parseDouble(m.group(2))));
        }
        return out;
    }

    /** POINT (lon lat) factory. */
    public static GeoData point(double lon, double lat) {
        return point(lon, lat, CRS.WGS84);
    }

    public static GeoData point(double lon, double lat, CRS crs) {
        Point p = GF.createPoint(new Coordinate(lon, lat));
        return new GeoData(p, crs);
    }

    public static GeoData parse(Object raw) {
        return parse(raw, CRS.WGS84);
    }

    public static GeoData parse(Object raw, CRS crs) {
        if (raw == null) return null;
        if (raw instanceof GeoData gd) return gd;
        if (raw instanceof Geometry g) return new GeoData(g, crs);
        if (raw instanceof byte[] wkb) return fromWkb(wkb, crs);
        String s = raw.toString().trim();
        if (s.isEmpty()) return null;
        if (s.startsWith("{")) return fromGeoJson(s, crs);
        String u = s.toUpperCase(Locale.ROOT);
        if (u.startsWith("POINT") || u.startsWith("LINE") || u.startsWith("POLYGON")
                || u.startsWith("MULTI") || u.startsWith("GEOMETRY") || u.startsWith("SRID=")) {
            // EWKT: SRID=4326;POINT(...)
            if (u.startsWith("SRID=")) {
                int semi = s.indexOf(';');
                if (semi > 0) {
                    String sridPart = s.substring(5, semi).trim();
                    try {
                        CRS c = CRS.ofEpsg(Integer.parseInt(sridPart));
                        return fromWkt(s.substring(semi + 1).trim(), c);
                    } catch (NumberFormatException ignored) {}
                }
            }
            return fromWkt(s, crs);
        }
        // "lon,lat" or "lon lat"
        String[] parts = s.split("[,\\s]+");
        if (parts.length >= 2) {
            try {
                return point(Double.parseDouble(parts[0]), Double.parseDouble(parts[1]), crs);
            } catch (NumberFormatException ignored) {}
        }
        throw new IllegalArgumentException("cannot parse geometry: " + s);
    }

    public Geometry geometry() {
        return geometry;
    }

    public CRS crs() {
        return crs;
    }

    public String toWkt() {
        return new WKTWriter().write(geometry);
    }

    public byte[] toWkb() {
        return new WKBWriter().write(geometry);
    }

    /** Minimal GeoJSON writer (Point / Polygon / LineString / fallback with WKT). */
    public String toGeoJson() {
        if (geometry instanceof Point p) {
            Coordinate c = p.getCoordinate();
            if (c == null) return "{\"type\":\"Point\",\"coordinates\":[]}";
            return String.format(Locale.ROOT,
                    "{\"type\":\"Point\",\"coordinates\":[%.10f,%.10f]}", c.x, c.y);
        }
        String type = geometry.getGeometryType();
        Coordinate[] cs = geometry.getCoordinates();
        StringBuilder sb = new StringBuilder();
        sb.append("{\"type\":\"").append(type).append("\",\"coordinates\":");
        if ("Polygon".equalsIgnoreCase(type)) {
            sb.append("[[");
            for (int i = 0; i < cs.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(String.format(Locale.ROOT, "[%.10f,%.10f]", cs[i].x, cs[i].y));
            }
            sb.append("]]");
        } else if ("LineString".equalsIgnoreCase(type)) {
            sb.append('[');
            for (int i = 0; i < cs.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(String.format(Locale.ROOT, "[%.10f,%.10f]", cs[i].x, cs[i].y));
            }
            sb.append(']');
        } else {
            // generic fallback
            return "{\"type\":\"GeometryCollection\",\"geometries\":[],\"wkt\":\""
                    + toWkt().replace("\\", "\\\\").replace("\"", "\\\"") + "\"}";
        }
        sb.append('}');
        return sb.toString();
    }

    public boolean isEmpty() {
        return geometry.isEmpty();
    }

    public String geometryType() {
        return geometry.getGeometryType();
    }

    /** Centroid lon/lat (x=lon, y=lat for geographic). */
    public double[] centroidXy() {
        Point c = geometry.getCentroid();
        if (c == null || c.isEmpty()) return new double[]{Double.NaN, Double.NaN};
        return new double[]{c.getX(), c.getY()};
    }

    public boolean evaluate(SpatialPredicate pred, GeoData other) {
        return evaluate(pred, other, 0.0);
    }

    public boolean evaluate(SpatialPredicate pred, GeoData other, double tolerance) {
        Objects.requireNonNull(pred, "pred");
        if (other == null) return false;
        Geometry a = this.geometry;
        Geometry b = other.geometry;
        return switch (pred) {
            case WITHIN -> a.within(b);
            case CONTAINS -> a.contains(b);
            case INTERSECTS -> a.intersects(b);
            case DISJOINT -> a.disjoint(b);
            case TOUCHES -> a.touches(b);
            case CROSSES -> a.crosses(b);
            case EQUALS -> a.equalsTopo(b);
            case DWITHIN -> distance(other) <= tolerance;
        };
    }

    /**
     * Distance: for two points in geographic CRS uses haversine meters;
     * otherwise JTS cartesian distance in CRS units.
     */
    public double distance(GeoData other) {
        Objects.requireNonNull(other, "other");
        if (geometry instanceof Point && other.geometry instanceof Point
                && crs.isGeographic() && other.crs.isGeographic()) {
            Coordinate a = geometry.getCoordinate();
            Coordinate b = other.geometry.getCoordinate();
            if (a == null || b == null) return Double.NaN;
            return haversineMeters(a.y, a.x, b.y, b.x);
        }
        return geometry.distance(other.geometry);
    }

    /** Earth-mean-radius haversine; lat/lon in degrees. */
    public static double haversineMeters(double lat1, double lon1, double lat2, double lon2) {
        final double R = 6_371_000.0;
        double p1 = Math.toRadians(lat1);
        double p2 = Math.toRadians(lat2);
        double dp = Math.toRadians(lat2 - lat1);
        double dl = Math.toRadians(lon2 - lon1);
        double h = Math.sin(dp / 2) * Math.sin(dp / 2)
                + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
        return 2 * R * Math.asin(Math.min(1.0, Math.sqrt(h)));
    }

    @Override
    public String getDataType() {
        return "GEOMETRY";
    }

    @Override
    public Object toArrowCompatible() {
        Map<String, Object> m = new java.util.LinkedHashMap<>();
        m.put("wkb", toWkb());
        m.put("wkt", toWkt());
        m.put("crs", crs.toString());
        m.put("type", geometryType());
        return m;
    }

    @Override
    public String getShortDesc() {
        String wkt = toWkt();
        if (wkt.length() > 64) wkt = wkt.substring(0, 61) + "...";
        return geometryType() + "(" + wkt + "), crs=" + crs;
    }

    @Override
    public Number getNumericValue() {
        double[] c = centroidXy();
        if (Double.isNaN(c[0])) return null;
        return c[0];
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof GeoData that)) return false;
        return geometry.equalsExact(that.geometry) && Objects.equals(crs, that.crs);
    }

    @Override
    public int hashCode() {
        return Objects.hash(toWkt(), crs);
    }

    @Override
    public String toString() {
        return "GeoData[" + getShortDesc() + "]";
    }
}
