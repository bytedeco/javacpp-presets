package org.bytedeco.pytorch.dataframe.geo;

import org.bytedeco.pytorch.dataframe.dtype.AbstractDataValue;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * H3 hexagonal hierarchical spatial index (Uber H3) — pure-Java subset.
 *
 * <p><b>Scope (honest):</b> encodes lon/lat → 64-bit H3-like cell id using the
 * public H3 digit layout (resolution in bits 52..55, base-cell + cubical digits).
 * This is a <em>deterministic indexing key</em> suitable for group-by / join /
 * partition, <b>not</b> a bit-identical reimplementation of {@code libh3}
 * (boundary geometry, gridDisk, and exact base-cell numbering require the
 * full H3 library). Cross-language equality tests should compare <em>our</em>
 * encoder on both sides, or swap in {@code com.uber:h3} later without changing
 * the {@link H3Data} API.
 *
 * <p>Resolution range: 0–15 (H3 standard).
 *
 * @see <a href="https://h3geo.org/docs/core-library/h3Indexing">H3 Indexing</a>
 */
public final class H3Data extends AbstractDataValue {

    private static final long serialVersionUID = 1L;

    /** H3 mode bit for hexagon cell (binary 0001b in mode field). */
    private static final long MODE_HEXAGON = 1L << 59;

    private final long h3Index;
    private final int resolution;

    public H3Data(long h3Index) {
        this.h3Index = h3Index;
        this.resolution = extractResolution(h3Index);
    }

    public static H3Data of(long h3Index) {
        return new H3Data(h3Index);
    }

    public static H3Data fromLonLat(double lon, double lat, int resolution) {
        return new H3Data(indexLonLat(lon, lat, resolution));
    }

    public static H3Data fromGeo(GeoData geo, int resolution) {
        Objects.requireNonNull(geo, "geo");
        double[] c = geo.centroidXy();
        return fromLonLat(c[0], c[1], resolution);
    }

    public long h3Index() {
        return h3Index;
    }

    public int resolution() {
        return resolution;
    }

    /** Hex string form (no 0x prefix), lowercase 16 hex digits. */
    public String toHex() {
        return String.format("%016x", h3Index);
    }

    public static H3Data fromHex(String hex) {
        Objects.requireNonNull(hex, "hex");
        String s = hex.trim();
        if (s.startsWith("0x") || s.startsWith("0X")) s = s.substring(2);
        return new H3Data(Long.parseUnsignedLong(s, 16));
    }

    /**
     * Deterministic lon/lat → cell id.
     * Layout (compatible with H3 bit positions, not base-cell table):
     * <pre>
     *   bits 63..59 : mode=1 (hexagon)
     *   bits 58..56 : reserved 0
     *   bits 55..52 : resolution 0..15
     *   bits 51..0  : quantized Morton-like key of (lon,lat) at res
     * </pre>
     */
    public static long indexLonLat(double lon, double lat, int resolution) {
        if (resolution < 0 || resolution > 15) {
            throw new IllegalArgumentException("H3 resolution must be 0..15, got " + resolution);
        }
        if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
            // clamp rather than fail — matches common GIS tolerance
            lat = Math.max(-90, Math.min(90, lat));
            lon = Math.max(-180, Math.min(180, lon));
        }
        // cells per axis roughly 7^res (H3 aperture 7); use 8^res power-of-two for bit packing
        int n = 1 << Math.min(resolution + 3, 20); // 8 .. ~1M bins per axis
        double x = (lon + 180.0) / 360.0; // 0..1
        double y = (lat + 90.0) / 180.0;  // 0..1
        long ix = Math.min(n - 1, Math.max(0, (long) Math.floor(x * n)));
        long iy = Math.min(n - 1, Math.max(0, (long) Math.floor(y * n)));
        long morton = interleaveBits(ix, iy) & ((1L << 52) - 1);
        long resBits = ((long) resolution & 0xFL) << 52;
        return MODE_HEXAGON | resBits | morton;
    }

    /** Approximate cell center lon/lat (inverse of {@link #indexLonLat}). */
    public double[] centerLonLat() {
        int res = resolution;
        int n = 1 << Math.min(res + 3, 20);
        long morton = h3Index & ((1L << 52) - 1);
        long[] xy = deinterleaveBits(morton);
        double lon = xy[0] * 360.0 / n - 180.0 + 180.0 / n;
        double lat = xy[1] * 180.0 / n - 90.0 + 90.0 / n;
        return new double[]{lon, lat};
    }

    /** Parent cell at coarser resolution (res must be &lt; this.resolution). */
    public H3Data toParent(int parentRes) {
        if (parentRes < 0 || parentRes > resolution) {
            throw new IllegalArgumentException("parentRes must be in 0..resolution");
        }
        double[] c = centerLonLat();
        return fromLonLat(c[0], c[1], parentRes);
    }

    /** k-ring approximation: cells whose morton neighborhood is within chebyshev k. */
    public List<H3Data> gridDisk(int k) {
        if (k < 0) throw new IllegalArgumentException("k >= 0");
        double[] c = centerLonLat();
        int res = resolution;
        int n = 1 << Math.min(res + 3, 20);
        double x = (c[0] + 180.0) / 360.0;
        double y = (c[1] + 90.0) / 180.0;
        long ix = Math.min(n - 1, Math.max(0, (long) Math.floor(x * n)));
        long iy = Math.min(n - 1, Math.max(0, (long) Math.floor(y * n)));
        List<H3Data> out = new ArrayList<>();
        for (long dx = -k; dx <= k; dx++) {
            for (long dy = -k; dy <= k; dy++) {
                long nx = ix + dx;
                long ny = iy + dy;
                if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
                long morton = interleaveBits(nx, ny) & ((1L << 52) - 1);
                long resBits = ((long) res & 0xFL) << 52;
                out.add(new H3Data(MODE_HEXAGON | resBits | morton));
            }
        }
        return out;
    }

    public static int extractResolution(long h3) {
        return (int) ((h3 >>> 52) & 0xFL);
    }

    /** Morton interleave of two non-negative integers (low 26 bits each). */
    static long interleaveBits(long x, long y) {
        return spread(x & 0x3FFFFFFL) | (spread(y & 0x3FFFFFFL) << 1);
    }

    static long[] deinterleaveBits(long z) {
        return new long[]{compact(z), compact(z >>> 1)};
    }

    private static long spread(long x) {
        x &= 0x3FFFFFFL;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFFL;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FFL;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FL;
        x = (x | (x << 2)) & 0x3333333333333333L;
        x = (x | (x << 1)) & 0x5555555555555555L;
        return x;
    }

    private static long compact(long x) {
        x &= 0x5555555555555555L;
        x = (x | (x >>> 1)) & 0x3333333333333333L;
        x = (x | (x >>> 2)) & 0x0F0F0F0F0F0F0F0FL;
        x = (x | (x >>> 4)) & 0x00FF00FF00FF00FFL;
        x = (x | (x >>> 8)) & 0x0000FFFF0000FFFFL;
        x = (x | (x >>> 16)) & 0x00000000FFFFFFFFL;
        return x & 0x3FFFFFFL;
    }

    /** Batch encode. */
    public static List<Long> indexAll(double[] lons, double[] lats, int resolution) {
        Objects.requireNonNull(lons, "lons");
        Objects.requireNonNull(lats, "lats");
        if (lons.length != lats.length) throw new IllegalArgumentException("lons/lats length mismatch");
        List<Long> out = new ArrayList<>(lons.length);
        for (int i = 0; i < lons.length; i++) out.add(indexLonLat(lons[i], lats[i], resolution));
        return out;
    }

    @Override
    public String getDataType() {
        return "H3";
    }

    @Override
    public Object toArrowCompatible() {
        return h3Index; // uint64
    }

    @Override
    public String getShortDesc() {
        return "h3=" + toHex() + " res=" + resolution;
    }

    @Override
    public Number getNumericValue() {
        return h3Index;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof H3Data h3Data)) return false;
        return h3Index == h3Data.h3Index;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(h3Index);
    }

    @Override
    public String toString() {
        return "H3Data[" + getShortDesc() + "]";
    }
}
