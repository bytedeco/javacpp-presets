package org.bytedeco.pytorch.dataframe.geo;

import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

/**
 * Coordinate Reference System descriptor.
 *
 * <p>Supports EPSG codes and WKT CRS strings. Full PROJ pipeline transforms are
 * <b>not</b> embedded; {@link #transformLonLatApprox} provides a documented
 * identity/approx path for common geographic CRS (EPSG:4326 / CRS84).
 * Complex projected transforms should be pushed to DuckDB / PROJ externally.
 */
public final class CRS {

    public static final CRS WGS84 = ofEpsg(4326);
    public static final CRS CRS84 = ofAuthority("OGC", "CRS84");

    private final String authority;
    private final String code;
    private final String wkt;
    private final String name;

    private CRS(String authority, String code, String wkt, String name) {
        this.authority = authority;
        this.code = code;
        this.wkt = wkt;
        this.name = name;
    }

    public static CRS ofEpsg(int epsg) {
        String wkt = switch (epsg) {
            case 4326 -> "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],"
                    + "PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433],AUTHORITY[\"EPSG\",\"4326\"]]";
            case 3857 -> "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\"],AUTHORITY[\"EPSG\",\"3857\"]]";
            default -> "EPSG:" + epsg;
        };
        String name = switch (epsg) {
            case 4326 -> "WGS 84";
            case 3857 -> "WGS 84 / Pseudo-Mercator";
            default -> "EPSG:" + epsg;
        };
        return new CRS("EPSG", String.valueOf(epsg), wkt, name);
    }

    public static CRS ofAuthority(String authority, String code) {
        Objects.requireNonNull(authority, "authority");
        Objects.requireNonNull(code, "code");
        return new CRS(authority, code, authority + ":" + code, authority + ":" + code);
    }

    public static CRS ofWkt(String wkt) {
        Objects.requireNonNull(wkt, "wkt");
        String auth = null, code = null;
        // best-effort AUTHORITY["EPSG","4326"] extract
        int idx = wkt.toUpperCase(Locale.ROOT).lastIndexOf("AUTHORITY");
        if (idx >= 0) {
            String tail = wkt.substring(idx);
            java.util.regex.Matcher m = java.util.regex.Pattern
                    .compile("AUTHORITY\\s*\\[\\s*\"([^\"]+)\"\\s*,\\s*\"([^\"]+)\"\\s*\\]",
                            java.util.regex.Pattern.CASE_INSENSITIVE)
                    .matcher(tail);
            if (m.find()) {
                auth = m.group(1);
                code = m.group(2);
            }
        }
        return new CRS(auth, code, wkt, auth != null ? auth + ":" + code : "WKT");
    }

    public static CRS parse(String s) {
        if (s == null || s.isBlank()) return WGS84;
        String t = s.trim();
        if (t.toUpperCase(Locale.ROOT).startsWith("EPSG:")) {
            return ofEpsg(Integer.parseInt(t.substring(5).trim()));
        }
        if (t.toUpperCase(Locale.ROOT).startsWith("GEOGCS") || t.toUpperCase(Locale.ROOT).startsWith("PROJCS")) {
            return ofWkt(t);
        }
        try {
            return ofEpsg(Integer.parseInt(t));
        } catch (NumberFormatException e) {
            return ofAuthority("CUSTOM", t);
        }
    }

    public String authority() { return authority; }
    public String code() { return code; }
    public String wkt() { return wkt; }
    public String name() { return name; }

    public Optional<Integer> epsg() {
        if ("EPSG".equalsIgnoreCase(authority) && code != null) {
            try { return Optional.of(Integer.parseInt(code)); }
            catch (NumberFormatException e) { return Optional.empty(); }
        }
        return Optional.empty();
    }

    public boolean isGeographic() {
        Optional<Integer> e = epsg();
        return e.isPresent() && (e.get() == 4326 || e.get() == 4269);
    }

    /**
     * Approx lon/lat transform between common geographic CRS only.
     * EPSG:4326 and OGC:CRS84 differ by axis order (lat/lon vs lon/lat) in some
     * encodings; this method treats both as lon/lat numeric pairs and is identity.
     * Projected CRS (e.g. 3857) throws — push those to DuckDB/PROJ.
     */
    public static double[] transformLonLatApprox(double lon, double lat, CRS from, CRS to) {
        Objects.requireNonNull(from, "from");
        Objects.requireNonNull(to, "to");
        if (from.equals(to)) return new double[]{lon, lat};
        if (from.isGeographic() && to.isGeographic()) return new double[]{lon, lat};
        // WebMercator ↔ WGS84 (well-known formulas) — only 3857 ↔ 4326
        Integer fe = from.epsg().orElse(null);
        Integer te = to.epsg().orElse(null);
        if (fe != null && te != null) {
            if (fe == 4326 && te == 3857) return wgs84ToWebMercator(lon, lat);
            if (fe == 3857 && te == 4326) return webMercatorToWgs84(lon, lat);
        }
        throw new UnsupportedOperationException(
                "CRS transform not embedded for " + from.name() + " → " + to.name()
                        + "; use DuckDB/PROJ for projected transforms (plan non-goal).");
    }

    private static double[] wgs84ToWebMercator(double lon, double lat) {
        double x = lon * 20037508.342789244 / 180.0;
        double y = Math.log(Math.tan((90.0 + lat) * Math.PI / 360.0)) / (Math.PI / 180.0);
        y = y * 20037508.342789244 / 180.0;
        return new double[]{x, y};
    }

    private static double[] webMercatorToWgs84(double x, double y) {
        double lon = x * 180.0 / 20037508.342789244;
        double lat = y * 180.0 / 20037508.342789244;
        lat = 180.0 / Math.PI * (2.0 * Math.atan(Math.exp(lat * Math.PI / 180.0)) - Math.PI / 2.0);
        return new double[]{lon, lat};
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CRS crs)) return false;
        if (authority != null && code != null && crs.authority != null && crs.code != null) {
            return authority.equalsIgnoreCase(crs.authority) && code.equals(crs.code);
        }
        return Objects.equals(wkt, crs.wkt);
    }

    @Override
    public int hashCode() {
        if (authority != null && code != null) {
            return Objects.hash(authority.toUpperCase(Locale.ROOT), code);
        }
        return Objects.hash(wkt);
    }

    @Override
    public String toString() {
        return name != null ? name : String.valueOf(wkt);
    }
}
