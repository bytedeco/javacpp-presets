package org.bytedeco.pytorch.dataframe.geo;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Spatial join over geometry columns (client-side).
 *
 * <p>Algorithm:
 * <ol>
 *   <li>Optional H3/S2 pre-filter buckets rows that cannot possibly match</li>
 *   <li>Pairwise {@link GeoData#evaluate} for candidates</li>
 * </ol>
 *
 * <p>For multi-million-row production joins, push to DuckDB (plan non-goal for
 * full spatial engine). This implementation is correct and suitable for
 * enterprise feature pipelines at moderate scale + unit/benchmark validation.
 */
public final class GeoJoin {

    private GeoJoin() {}

    public static DataFrame geoJoin(
            DataFrame left,
            DataFrame right,
            String leftGeoCol,
            String rightGeoCol,
            SpatialPredicate pred) throws Exception {
        return geoJoin(left, right, leftGeoCol, rightGeoCol, pred, 0.0, null);
    }

    public static DataFrame geoJoin(
            DataFrame left,
            DataFrame right,
            String leftGeoCol,
            String rightGeoCol,
            SpatialPredicate pred,
            double tolerance) throws Exception {
        return geoJoin(left, right, leftGeoCol, rightGeoCol, pred, tolerance, null);
    }

    /**
     * @param opts if non-null and h3Resolution &gt; 0, use H3 co-location prefilter
     *             (same cell ⇒ candidate). Falls back to full pairwise if opts null.
     */
    public static DataFrame geoJoin(
            DataFrame left,
            DataFrame right,
            String leftGeoCol,
            String rightGeoCol,
            SpatialPredicate pred,
            double tolerance,
            GeoOptions opts) throws Exception {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        Objects.requireNonNull(leftGeoCol, "leftGeoCol");
        Objects.requireNonNull(rightGeoCol, "rightGeoCol");
        Objects.requireNonNull(pred, "pred");
        if (!left.hasColumn(leftGeoCol)) throw new IllegalArgumentException("left missing " + leftGeoCol);
        if (!right.hasColumn(rightGeoCol)) throw new IllegalArgumentException("right missing " + rightGeoCol);

        // Parse geometries once
        GeoData[] lg = parseColumn(left, leftGeoCol, opts);
        GeoData[] rg = parseColumn(right, rightGeoCol, opts);

        List<int[]> matches = new ArrayList<>();
        if (opts != null && pred != SpatialPredicate.DISJOINT) {
            // H3 prefilter at configured resolution
            int res = opts.h3Resolution();
            Map<Long, List<Integer>> rightBuckets = new HashMap<>();
            for (int j = 0; j < rg.length; j++) {
                if (rg[j] == null) continue;
                long h = H3Data.fromGeo(rg[j], res).h3Index();
                rightBuckets.computeIfAbsent(h, k -> new ArrayList<>()).add(j);
                // also ring-1 neighbors for boundary cases
                for (H3Data n : H3Data.fromGeo(rg[j], res).gridDisk(1)) {
                    rightBuckets.computeIfAbsent(n.h3Index(), k -> new ArrayList<>());
                    List<Integer> bucket = rightBuckets.get(n.h3Index());
                    if (!bucket.contains(j)) bucket.add(j);
                }
            }
            for (int i = 0; i < lg.length; i++) {
                if (lg[i] == null) continue;
                long h = H3Data.fromGeo(lg[i], res).h3Index();
                List<Integer> cands = rightBuckets.get(h);
                if (cands == null) continue;
                for (int j : cands) {
                    if (rg[j] == null) continue;
                    if (lg[i].evaluate(pred, rg[j], tolerance)) {
                        matches.add(new int[]{i, j});
                    }
                }
            }
        } else {
            // full pairwise
            for (int i = 0; i < lg.length; i++) {
                if (lg[i] == null) continue;
                for (int j = 0; j < rg.length; j++) {
                    if (rg[j] == null) continue;
                    if (lg[i].evaluate(pred, rg[j], tolerance)) {
                        matches.add(new int[]{i, j});
                    }
                }
            }
        }

        return materialize(left, right, matches);
    }

    /**
     * Equi-join on H3 index of geometry centroids (fast approximate co-location).
     */
    public static DataFrame h3Join(
            DataFrame left,
            DataFrame right,
            String leftGeoCol,
            String rightGeoCol,
            int resolution) throws Exception {
        Objects.requireNonNull(left); Objects.requireNonNull(right);
        GeoData[] lg = parseColumn(left, leftGeoCol, null);
        GeoData[] rg = parseColumn(right, rightGeoCol, null);
        Map<Long, List<Integer>> rightBuckets = new HashMap<>();
        for (int j = 0; j < rg.length; j++) {
            if (rg[j] == null) continue;
            long h = H3Data.fromGeo(rg[j], resolution).h3Index();
            rightBuckets.computeIfAbsent(h, k -> new ArrayList<>()).add(j);
        }
        List<int[]> matches = new ArrayList<>();
        for (int i = 0; i < lg.length; i++) {
            if (lg[i] == null) continue;
            long h = H3Data.fromGeo(lg[i], resolution).h3Index();
            List<Integer> cands = rightBuckets.get(h);
            if (cands == null) continue;
            for (int j : cands) matches.add(new int[]{i, j});
        }
        return materialize(left, right, matches);
    }

    /**
     * Add an H3 index column derived from a geometry (or lon/lat) column.
     */
    public static DataFrame withH3(DataFrame df, String geoCol, String outCol, int resolution) throws Exception {
        Objects.requireNonNull(df);
        if (!df.hasColumn(geoCol)) throw new IllegalArgumentException("missing " + geoCol);
        DataFrame out = df.copy();
        if (!out.hasColumn(outCol)) out.addColumn(outCol, Column.DType.INT64);
        Column src = df.column(geoCol);
        for (int i = 0; i < df.rowCount(); i++) {
            GeoData g = GeoData.parse(src.get(i));
            if (g == null) {
                out.set(i, outCol, null);
            } else {
                out.set(i, outCol, H3Data.fromGeo(g, resolution).h3Index());
            }
        }
        return out;
    }

    public static DataFrame withS2(DataFrame df, String geoCol, String outCol, int level) throws Exception {
        Objects.requireNonNull(df);
        if (!df.hasColumn(geoCol)) throw new IllegalArgumentException("missing " + geoCol);
        DataFrame out = df.copy();
        if (!out.hasColumn(outCol)) out.addColumn(outCol, Column.DType.INT64);
        Column src = df.column(geoCol);
        for (int i = 0; i < df.rowCount(); i++) {
            GeoData g = GeoData.parse(src.get(i));
            if (g == null) {
                out.set(i, outCol, null);
            } else {
                out.set(i, outCol, S2Data.fromGeo(g, level).cellId());
            }
        }
        return out;
    }

    private static GeoData[] parseColumn(DataFrame df, String col, GeoOptions opts) {
        CRS crs = opts == null ? CRS.WGS84 : opts.crs();
        Column c = df.column(col);
        GeoData[] arr = new GeoData[df.rowCount()];
        for (int i = 0; i < df.rowCount(); i++) {
            Object v = c.get(i);
            if (v == null) {
                arr[i] = null;
                continue;
            }
            try {
                arr[i] = GeoData.parse(v, crs);
            } catch (Exception e) {
                arr[i] = null;
            }
        }
        return arr;
    }

    private static DataFrame materialize(DataFrame left, DataFrame right, List<int[]> matches) throws Exception {
        DataFrame result = DataFrame.create();
        // left cols as-is; right cols prefixed if name collides
        List<String> leftNames = new ArrayList<>();
        List<String> rightNames = new ArrayList<>();
        List<String> rightOutNames = new ArrayList<>();
        for (Column c : left.columns()) {
            leftNames.add(c.name());
            result.addColumn(c.name(), c.dtype());
        }
        for (Column c : right.columns()) {
            rightNames.add(c.name());
            String outName = c.name();
            if (result.hasColumn(outName)) outName = "right_" + outName;
            // ensure unique
            String base = outName;
            int k = 1;
            while (result.hasColumn(outName)) outName = base + "_" + (k++);
            rightOutNames.add(outName);
            result.addColumn(outName, c.dtype());
        }
        for (int[] m : matches) {
            int ri = result.addEmptyRow();
            int li = m[0], rj = m[1];
            for (String n : leftNames) {
                result.set(ri, n, left.get(li, n));
            }
            for (int c = 0; c < rightNames.size(); c++) {
                result.set(ri, rightOutNames.get(c), right.get(rj, rightNames.get(c)));
            }
        }
        return result;
    }
}
