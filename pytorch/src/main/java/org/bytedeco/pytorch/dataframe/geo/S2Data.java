package org.bytedeco.pytorch.dataframe.geo;

import org.bytedeco.pytorch.dataframe.dtype.AbstractDataValue;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * S2 cell id index (Google S2) — pure-Java subset.
 *
 * <p><b>Scope (honest):</b> encodes lon/lat → 64-bit S2-like cell id using face
 * + Hilbert-order quantization at a given level (0–30). Sufficient as a spatial
 * partition / join key. <b>Not</b> bit-identical to {@code s2-geometry-library-java}
 * (exact cubic projection and Hilbert lookup tables differ). Documented so
 * cross-language tests use the same encoder or swap the real S2 library later
 * without changing this API.
 *
 * @see <a href="https://s2geometry.io/devguide/s2cell_hierarchy">S2 cell hierarchy</a>
 */
public final class S2Data extends AbstractDataValue {

    private static final long serialVersionUID = 1L;

    private final long cellId;
    private final int level;

    public S2Data(long cellId) {
        this.cellId = cellId;
        this.level = extractLevel(cellId);
    }

    public static S2Data of(long cellId) {
        return new S2Data(cellId);
    }

    public static S2Data fromLonLat(double lon, double lat, int level) {
        return new S2Data(indexLonLat(lon, lat, level));
    }

    public static S2Data fromGeo(GeoData geo, int level) {
        Objects.requireNonNull(geo, "geo");
        double[] c = geo.centroidXy();
        return fromLonLat(c[0], c[1], level);
    }

    public long cellId() {
        return cellId;
    }

    public int level() {
        return level;
    }

    /** Token: unsigned hex (S2 python token style, trimmed trailing zeros not applied). */
    public String toToken() {
        return Long.toUnsignedString(cellId, 16);
    }

    public static S2Data fromToken(String token) {
        Objects.requireNonNull(token, "token");
        return new S2Data(Long.parseUnsignedLong(token.trim(), 16));
    }

    /**
     * Face (0..5) from lon/lat via cube-face mapping (approx equi-rectangular → cube).
     * Level bits packed: {@code face(3) | level(5) | hilbert(56)} modified layout
     * that keeps level recoverable.
     */
    public static long indexLonLat(double lon, double lat, int level) {
        if (level < 0 || level > 30) {
            throw new IllegalArgumentException("S2 level must be 0..30, got " + level);
        }
        lat = Math.max(-90, Math.min(90, lat));
        lon = Math.max(-180, Math.min(180, lon));

        // Convert to 3D unit vector on sphere
        double lonRad = Math.toRadians(lon);
        double latRad = Math.toRadians(lat);
        double cosLat = Math.cos(latRad);
        double x = Math.cos(lonRad) * cosLat;
        double y = Math.sin(lonRad) * cosLat;
        double z = Math.sin(latRad);

        // Dominant axis → face 0..5
        double ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
        int face;
        double u, v;
        if (ax >= ay && ax >= az) {
            face = x >= 0 ? 0 : 1;
            u = y / ax;
            v = z / ax;
            if (face == 1) u = -u;
        } else if (ay >= ax && ay >= az) {
            face = y >= 0 ? 2 : 3;
            u = x / ay;
            v = z / ay;
            if (face == 2) u = -u;
            if (face == 3) { u = -u; /* keep */ }
        } else {
            face = z >= 0 ? 4 : 5;
            u = x / az;
            v = y / az;
            if (face == 5) v = -v;
        }
        // u,v in [-1,1] → [0,1]
        double su = 0.5 * (u + 1.0);
        double sv = 0.5 * (v + 1.0);
        int n = 1 << Math.min(level, 30);
        long ix = Math.min(n - 1L, Math.max(0, (long) Math.floor(su * n)));
        long iy = Math.min(n - 1L, Math.max(0, (long) Math.floor(sv * n)));
        long hilbert = xyToHilbert(ix, iy, Math.min(level, 30));

        // Pack: [face:3][level:5][hilbert:56]
        return ((long) (face & 7) << 61) | ((long) (level & 31) << 56) | (hilbert & ((1L << 56) - 1));
    }

    public double[] centerLonLat() {
        int face = (int) ((cellId >>> 61) & 7);
        int lvl = level;
        int n = 1 << Math.min(lvl, 30);
        long hilbert = cellId & ((1L << 56) - 1);
        long[] xy = hilbertToXy(hilbert, Math.min(lvl, 30));
        double su = (xy[0] + 0.5) / n;
        double sv = (xy[1] + 0.5) / n;
        double u = 2.0 * su - 1.0;
        double v = 2.0 * sv - 1.0;
        double x, y, z;
        switch (face) {
            case 0 -> { x = 1; y = u; z = v; }
            case 1 -> { x = -1; y = -u; z = v; }
            case 2 -> { x = -u; y = 1; z = v; }
            case 3 -> { x = u; y = -1; z = v; }
            case 4 -> { x = u; y = v; z = 1; }
            default -> { x = u; y = -v; z = -1; } // face 5
        }
        double norm = Math.sqrt(x * x + y * y + z * z);
        x /= norm; y /= norm; z /= norm;
        double lat = Math.toDegrees(Math.asin(Math.max(-1, Math.min(1, z))));
        double lon = Math.toDegrees(Math.atan2(y, x));
        return new double[]{lon, lat};
    }

    public S2Data toParent(int parentLevel) {
        if (parentLevel < 0 || parentLevel > level) {
            throw new IllegalArgumentException("parentLevel must be in 0..level");
        }
        double[] c = centerLonLat();
        return fromLonLat(c[0], c[1], parentLevel);
    }

    public static int extractLevel(long cellId) {
        return (int) ((cellId >>> 56) & 31);
    }

    public static int extractFace(long cellId) {
        return (int) ((cellId >>> 61) & 7);
    }

    /** Hilbert curve encode (standard non-recursive bit algorithm). */
    static long xyToHilbert(long x, long y, int order) {
        long n = 1L << order;
        long rx, ry, d = 0;
        for (long s = n / 2; s > 0; s /= 2) {
            rx = (x & s) > 0 ? 1 : 0;
            ry = (y & s) > 0 ? 1 : 0;
            d += s * s * ((3 * rx) ^ ry);
            // rotate
            if (ry == 0) {
                if (rx == 1) {
                    x = n - 1 - x;
                    y = n - 1 - y;
                }
                long t = x; x = y; y = t;
            }
        }
        return d;
    }

    static long[] hilbertToXy(long d, int order) {
        long n = 1L << order;
        long x = 0, y = 0;
        long rx, ry;
        long t = d;
        for (long s = 1; s < n; s *= 2) {
            rx = 1 & (t / 2);
            ry = 1 & (t ^ rx);
            // rotate
            if (ry == 0) {
                if (rx == 1) {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                long tmp = x; x = y; y = tmp;
            }
            x += s * rx;
            y += s * ry;
            t /= 4;
        }
        return new long[]{x, y};
    }

    public static List<Long> indexAll(double[] lons, double[] lats, int level) {
        Objects.requireNonNull(lons); Objects.requireNonNull(lats);
        if (lons.length != lats.length) throw new IllegalArgumentException("length mismatch");
        List<Long> out = new ArrayList<>(lons.length);
        for (int i = 0; i < lons.length; i++) out.add(indexLonLat(lons[i], lats[i], level));
        return out;
    }

    @Override
    public String getDataType() {
        return "S2";
    }

    @Override
    public Object toArrowCompatible() {
        return cellId;
    }

    @Override
    public String getShortDesc() {
        return "s2=" + toToken() + " level=" + level + " face=" + extractFace(cellId);
    }

    @Override
    public Number getNumericValue() {
        return cellId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof S2Data s2Data)) return false;
        return cellId == s2Data.cellId;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(cellId);
    }

    @Override
    public String toString() {
        return "S2Data[" + getShortDesc() + "]";
    }
}
