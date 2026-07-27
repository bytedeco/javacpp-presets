package org.bytedeco.pytorch.data.dataframe;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Advanced / cold-path DataFrame ops matching industrial Pandas/Polars APIs
 * that most developers under-use: asof joins, semi/anti, set algebra,
 * reindex, interpolate, qcut, pipe, take, mask/where, combine, memory stats.
 *
 * <p>Kept out of {@link DataFrame} to avoid further bloating the façade —
 * {@link DataFrame} re-exports the most common entry points.
 */
public final class AdvancedOps {
    private AdvancedOps() {}

    // ================================================================
    // Pipe / take / mask / where / combine
    // ================================================================

    /** Pandas {@code DataFrame.pipe(func, *args)} — chain external function. */
    public static <R> R pipe(DataFrame df, Function<DataFrame, R> func) {
        return func.apply(df);
    }

    /**
     * Pandas {@code DataFrame.take(indices)} — positional row selection.
     * Negative indices count from the end.
     */
    public static DataFrame take(DataFrame df, int... indices) {
        if (indices == null || indices.length == 0) return DataFrame.create();
        int n = df.rowCount();
        int[] resolved = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (idx < 0) idx = n + idx;
            if (idx < 0 || idx >= n) {
                throw new IndexOutOfBoundsException("take index " + indices[i] + " out of range [0," + n + ")");
            }
            resolved[i] = idx;
        }
        return df.loc(resolved);
    }

    /**
     * Pandas {@code Series.mask(cond, other)} — where cond is true, replace with other.
     * {@code cond} is a boolean column name or a row predicate.
     */
    public static DataFrame mask(DataFrame df, String boolCol, Object other) {
        Column cond = df.column(boolCol);
        DataFrame result = df.copy();
        for (Column c : result.columns()) {
            if (c.name().equals(boolCol)) continue;
            for (int i = 0; i < result.rowCount(); i++) {
                if (isTrue(cond.get(i))) c.set(i, other);
            }
        }
        return result;
    }

    public static DataFrame mask(DataFrame df, Predicate<Map<String, Object>> cond, Object other) {
        DataFrame result = df.copy();
        for (int i = 0; i < result.rowCount(); i++) {
            if (cond.test(df.toDict(i))) {
                for (Column c : result.columns()) c.set(i, other);
            }
        }
        return result;
    }

    /**
     * Pandas {@code Series.where(cond, other)} — keep value where cond true, else other.
     * Inverse of {@link #mask}.
     */
    public static DataFrame where(DataFrame df, String boolCol, Object other) {
        Column cond = df.column(boolCol);
        DataFrame result = df.copy();
        for (Column c : result.columns()) {
            if (c.name().equals(boolCol)) continue;
            for (int i = 0; i < result.rowCount(); i++) {
                if (!isTrue(cond.get(i))) c.set(i, other);
            }
        }
        return result;
    }

    /**
     * Pandas {@code DataFrame.combine_first(other)} — fill nulls in this with other
     * (aligned by row position; columns are the union).
     */
    public static DataFrame combineFirst(DataFrame left, DataFrame right) throws Exception {
        Set<String> names = new LinkedHashSet<>();
        for (Column c : left.columns()) names.add(c.name());
        for (Column c : right.columns()) names.add(c.name());
        DataFrame result = DataFrame.create();
        int n = Math.max(left.rowCount(), right.rowCount());
        for (String name : names) {
            Column.DType dt = left.hasColumn(name) ? left.column(name).dtype()
                : right.column(name).dtype();
            result.addColumn(name, dt);
        }
        for (int i = 0; i < n; i++) {
            int ri = result.addEmptyRow();
            for (String name : names) {
                Object lv = (i < left.rowCount() && left.hasColumn(name)) ? left.get(i, name) : null;
                Object rv = (i < right.rowCount() && right.hasColumn(name)) ? right.get(i, name) : null;
                result.set(ri, name, lv != null ? lv : rv);
            }
        }
        return result;
    }

    /**
     * Pandas {@code DataFrame.combine(other, func)} — element-wise binary combine
     * for shared columns (row-position aligned).
     */
    public static DataFrame combine(DataFrame left, DataFrame right,
                                    BiFunction<Object, Object, Object> func) throws Exception {
        DataFrame result = DataFrame.create();
        int n = Math.max(left.rowCount(), right.rowCount());
        for (Column c : left.columns()) {
            if (!right.hasColumn(c.name())) {
                result.addColumn(c.copy());
                while (result.column(c.name()).size() < n) result.column(c.name()).add(null);
                continue;
            }
            result.addColumn(c.name(), c.dtype());
            Column out = result.column(c.name());
            Column rc = right.column(c.name());
            for (int i = 0; i < n; i++) {
                Object a = i < left.rowCount() ? c.get(i) : null;
                Object b = i < right.rowCount() ? rc.get(i) : null;
                out.add(func.apply(a, b));
            }
        }
        // columns only in right
        for (Column c : right.columns()) {
            if (result.hasColumn(c.name())) continue;
            result.addColumn(c.copy());
            while (result.column(c.name()).size() < n) result.column(c.name()).add(null);
        }
        return result;
    }

    // ================================================================
    // Join variants: asof / semi / anti / cross
    // ================================================================

    /**
     * Pandas {@code merge_asof} / Polars {@code join_asof}.
     * Both frames must be sorted ascending by the on-key (within each {@code by} group).
     *
     * @param direction {@code "backward"} (default), {@code "forward"}, {@code "nearest"}
     * @param tolerance max absolute key distance; null = unlimited (numeric/comparable keys)
     * @param by optional group columns — asof is performed independently within each group
     * @param allowExactMatches when false, exact key matches are skipped (search strictly before/after)
     */
    public static DataFrame joinAsof(DataFrame left, DataFrame right,
                                     String leftOn, String rightOn,
                                     String direction, Double tolerance,
                                     String[] by, boolean allowExactMatches) throws Exception {
        String dir = direction == null ? "backward" : direction.toLowerCase(Locale.ROOT);
        String[] byCols = by == null ? new String[0] : by;

        DataFrame result = DataFrame.create();
        List<String> leftCols = new ArrayList<>();
        for (Column c : left.columns()) leftCols.add(c.name());
        List<String> rightCols = new ArrayList<>();
        Set<String> bySet = new HashSet<>(Arrays.asList(byCols));
        for (Column c : right.columns()) {
            String n = c.name();
            if (n.equals(rightOn)) continue;
            if (bySet.contains(n)) continue; // by keys already on left
            if (left.hasColumn(n) && !bySet.contains(n)) {
                // collision — suffix
                rightCols.add(n);
            } else {
                rightCols.add(n);
            }
        }
        // de-dup right col names against left
        List<String> rightOutNames = new ArrayList<>();
        for (String n : rightCols) {
            String out = left.hasColumn(n) ? n + "_right" : n;
            rightOutNames.add(out);
        }
        for (String n : leftCols) result.addColumn(n, left.column(n).dtype());
        for (int i = 0; i < rightCols.size(); i++) {
            result.addColumn(rightOutNames.get(i), right.column(rightCols.get(i)).dtype());
        }

        // Build right index: groupKey -> list of (rowIndex) sorted by rightOn
        Map<String, List<Integer>> rightGroups = new HashMap<>();
        for (int i = 0; i < right.rowCount(); i++) {
            String gk = groupKey(right, i, byCols);
            rightGroups.computeIfAbsent(gk, k -> new ArrayList<>()).add(i);
        }
        // ensure each group is sorted by rightOn
        Column rKeyCol = right.column(rightOn);
        for (List<Integer> idxs : rightGroups.values()) {
            idxs.sort((a, b) -> Expression.compareVals(rKeyCol.get(a), rKeyCol.get(b)));
        }

        Column lKey = left.column(leftOn);
        for (int li = 0; li < left.rowCount(); li++) {
            Object lk = lKey.get(li);
            String gk = groupKey(left, li, byCols);
            List<Integer> candidates = rightGroups.getOrDefault(gk, List.of());
            int match = -1;
            if (lk != null && !candidates.isEmpty()) {
                match = switch (dir) {
                    case "forward" -> asofForwardGrouped(rKeyCol, candidates, lk, tolerance, allowExactMatches);
                    case "nearest" -> asofNearestGrouped(rKeyCol, candidates, lk, tolerance, allowExactMatches);
                    default -> asofBackwardGrouped(rKeyCol, candidates, lk, tolerance, allowExactMatches);
                };
            }
            int ri = result.addEmptyRow();
            for (String n : leftCols) result.set(ri, n, left.get(li, n));
            if (match >= 0) {
                for (int i = 0; i < rightCols.size(); i++) {
                    result.set(ri, rightOutNames.get(i), right.get(match, rightCols.get(i)));
                }
            }
        }
        return result;
    }

    public static DataFrame joinAsof(DataFrame left, DataFrame right,
                                     String leftOn, String rightOn,
                                     String direction, Double tolerance) throws Exception {
        return joinAsof(left, right, leftOn, rightOn, direction, tolerance, null, true);
    }

    public static DataFrame joinAsof(DataFrame left, DataFrame right, String on) throws Exception {
        return joinAsof(left, right, on, on, "backward", null, null, true);
    }

    public static DataFrame mergeAsof(DataFrame left, DataFrame right,
                                      String leftOn, String rightOn,
                                      String direction, Double tolerance) throws Exception {
        return joinAsof(left, right, leftOn, rightOn, direction, tolerance, null, true);
    }

    /**
     * Full Pandas {@code merge_asof} with {@code by} and {@code allow_exact_matches}.
     */
    public static DataFrame mergeAsof(DataFrame left, DataFrame right,
                                      String leftOn, String rightOn,
                                      String direction, Double tolerance,
                                      String[] by, boolean allowExactMatches) throws Exception {
        return joinAsof(left, right, leftOn, rightOn, direction, tolerance, by, allowExactMatches);
    }

    public static DataFrame mergeAsof(DataFrame left, DataFrame right, String on,
                                      String[] by, boolean allowExactMatches) throws Exception {
        return joinAsof(left, right, on, on, "backward", null, by, allowExactMatches);
    }

    /**
     * Polars semi-join — keep left rows whose key exists in right (no right columns).
     */
    public static DataFrame joinSemi(DataFrame left, DataFrame right, String on) {
        return joinSemi(left, right, on, on);
    }

    public static DataFrame joinSemi(DataFrame left, DataFrame right, String leftOn, String rightOn) {
        Set<Object> keys = new HashSet<>();
        Column rc = right.column(rightOn);
        for (int i = 0; i < right.rowCount(); i++) keys.add(rc.get(i));
        Column lc = left.column(leftOn);
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < left.rowCount(); i++) {
            if (keys.contains(lc.get(i))) keep.add(i);
        }
        return left.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Polars anti-join — keep left rows whose key does NOT exist in right.
     */
    public static DataFrame joinAnti(DataFrame left, DataFrame right, String on) {
        return joinAnti(left, right, on, on);
    }

    public static DataFrame joinAnti(DataFrame left, DataFrame right, String leftOn, String rightOn) {
        Set<Object> keys = new HashSet<>();
        Column rc = right.column(rightOn);
        for (int i = 0; i < right.rowCount(); i++) keys.add(rc.get(i));
        Column lc = left.column(leftOn);
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < left.rowCount(); i++) {
            if (!keys.contains(lc.get(i))) keep.add(i);
        }
        return left.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Cartesian / cross join.
     * Warning: O(n*m) — use only for small frames.
     */
    public static DataFrame joinCross(DataFrame left, DataFrame right) throws Exception {
        DataFrame result = DataFrame.create();
        for (Column c : left.columns()) result.addColumn(c.name(), c.dtype());
        for (Column c : right.columns()) {
            String name = c.name();
            if (result.hasColumn(name)) name = name + "_right";
            result.addColumn(name, c.dtype());
        }
        List<String> rightNames = new ArrayList<>();
        for (Column c : right.columns()) {
            String name = c.name();
            rightNames.add(result.hasColumn(name) && left.hasColumn(name) ? name + "_right" : name);
        }
        // fix: recompute actual result right col names
        rightNames.clear();
        int li = 0;
        for (Column c : left.columns()) li++;
        for (int i = li; i < result.columnCount(); i++) rightNames.add(result.column(i).name());

        for (int i = 0; i < left.rowCount(); i++) {
            for (int j = 0; j < right.rowCount(); j++) {
                int ri = result.addEmptyRow();
                int col = 0;
                for (Column c : left.columns()) {
                    result.set(ri, c.name(), left.get(i, c.name()));
                    col++;
                }
                for (int k = 0; k < right.columnCount(); k++) {
                    result.set(ri, rightNames.get(k), right.get(j, right.column(k).name()));
                }
            }
        }
        return result;
    }

    // ================================================================
    // Set algebra (Polars set_*)
    // ================================================================

    /** Row-wise set difference (left minus right) by full-row key. */
    public static DataFrame setDifference(DataFrame left, DataFrame right) {
        Set<String> rightKeys = rowKeySet(right);
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < left.rowCount(); i++) {
            if (!rightKeys.contains(rowKey(left, i))) keep.add(i);
        }
        return left.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /** Row-wise set intersection. */
    public static DataFrame setIntersection(DataFrame left, DataFrame right) {
        Set<String> rightKeys = rowKeySet(right);
        List<Integer> keep = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < left.rowCount(); i++) {
            String k = rowKey(left, i);
            if (rightKeys.contains(k) && seen.add(k)) keep.add(i);
        }
        return left.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /** Row-wise set union (left then right rows not already in left). */
    public static DataFrame setUnion(DataFrame left, DataFrame right) throws Exception {
        Set<String> seen = rowKeySet(left);
        List<Integer> extra = new ArrayList<>();
        for (int i = 0; i < right.rowCount(); i++) {
            String k = rowKey(right, i);
            if (seen.add(k)) extra.add(i);
        }
        if (extra.isEmpty()) return left.copy();
        DataFrame tail = right.loc(extra.stream().mapToInt(Integer::intValue).toArray());
        return DataFrame.vstack(left, tail);
    }

    // ================================================================
    // Reindex / asof-lookup / truncate / searchsorted / interpolate / qcut
    // ================================================================

    /**
     * Pandas {@code reindex(labels)} on a key column — align rows to {@code labels}
     * order; missing keys produce null rows. Optional fill method:
     * {@code null}, {@code "ffill"}, {@code "bfill"}, {@code "nearest"}.
     */
    public static DataFrame reindex(DataFrame df, String keyCol, List<?> labels,
                                    String method) throws Exception {
        Column key = df.column(keyCol);
        Map<Object, Integer> pos = new HashMap<>();
        for (int i = 0; i < df.rowCount(); i++) pos.putIfAbsent(key.get(i), i);

        DataFrame result = DataFrame.create();
        for (Column c : df.columns()) result.addColumn(c.name(), c.dtype());

        List<Integer> mapped = new ArrayList<>(labels.size());
        for (Object lab : labels) {
            Integer p = pos.get(lab);
            mapped.add(p); // may be null
            int ri = result.addEmptyRow();
            if (p != null) {
                for (Column c : df.columns()) result.set(ri, c.name(), df.get(p, c.name()));
            } else {
                result.set(ri, keyCol, lab);
            }
        }

        if (method != null && !method.isEmpty()) {
            fillAlong(result, method.toLowerCase(Locale.ROOT));
        }
        return result;
    }

    /**
     * Pandas {@code Series.searchsorted(v, side)} — binary search insertion positions
     * into a sorted column. side = {@code "left"} or {@code "right"}.
     */
    public static int[] searchsorted(DataFrame df, String col, Object[] values, String side) {
        Column c = df.column(col);
        boolean right = side != null && side.equalsIgnoreCase("right");
        int[] out = new int[values.length];
        for (int k = 0; k < values.length; k++) {
            Object v = values[k];
            int lo = 0, hi = df.rowCount();
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                int cmp = Expression.compareVals(c.get(mid), v);
                if (right ? cmp <= 0 : cmp < 0) lo = mid + 1;
                else hi = mid;
            }
            out[k] = lo;
        }
        return out;
    }

    /**
     * Linear / ffill / bfill / nearest interpolation of nulls in a numeric column.
     * Methods: {@code "linear"}, {@code "ffill"}, {@code "bfill"}, {@code "nearest"}.
     */
    public static DataFrame interpolate(DataFrame df, String col, String method) {
        String m = method == null ? "linear" : method.toLowerCase(Locale.ROOT);
        DataFrame result = df.copy();
        Column c = result.column(col);
        int n = result.rowCount();
        switch (m) {
            case "ffill" -> {
                Object last = null;
                for (int i = 0; i < n; i++) {
                    Object v = c.get(i);
                    if (v == null) c.set(i, last);
                    else last = v;
                }
            }
            case "bfill" -> {
                Object next = null;
                for (int i = n - 1; i >= 0; i--) {
                    Object v = c.get(i);
                    if (v == null) c.set(i, next);
                    else next = v;
                }
            }
            case "nearest" -> {
                // nearest non-null neighbor
                for (int i = 0; i < n; i++) {
                    if (c.get(i) != null) continue;
                    int L = i - 1, R = i + 1;
                    while (L >= 0 && c.get(L) == null) L--;
                    while (R < n && c.get(R) == null) R++;
                    if (L < 0 && R >= n) continue;
                    if (L < 0) c.set(i, c.get(R));
                    else if (R >= n) c.set(i, c.get(L));
                    else c.set(i, (i - L) <= (R - i) ? c.get(L) : c.get(R));
                }
            }
            default -> { // linear
                int i = 0;
                while (i < n) {
                    if (c.get(i) != null) { i++; continue; }
                    int start = i - 1; // last known
                    int j = i;
                    while (j < n && c.get(j) == null) j++;
                    // [i, j) are null; start may be -1, j may be n
                    if (start >= 0 && j < n) {
                        double a = DataValues.asDouble(c.get(start));
                        double b = DataValues.asDouble(c.get(j));
                        int span = j - start;
                        for (int k = i; k < j; k++) {
                            double t = (k - start) / (double) span;
                            c.set(k, a + t * (b - a));
                        }
                    }
                    i = j;
                }
            }
        }
        return result;
    }

    /**
     * Pandas {@code qcut} — equal-frequency binning of a numeric column.
     * @param q number of quantiles (e.g. 4 = quartiles)
     * @param labels optional bin labels (length = q); null → "0".."q-1"
     * @param duplicates {@code "drop"} drops duplicate edges; {@code "raise"} throws
     */
    public static DataFrame qcut(DataFrame df, String col, int q,
                                 String[] labels, String duplicates) {
        if (q < 2) throw new IllegalArgumentException("q must be >= 2");
        Column src = df.column(col);
        List<Double> vals = new ArrayList<>();
        for (int i = 0; i < df.rowCount(); i++) {
            double d = DataValues.asDouble(src.get(i));
            if (!Double.isNaN(d)) vals.add(d);
        }
        Collections.sort(vals);
        if (vals.isEmpty()) {
            DataFrame empty = df.copy();
            empty.addColumn(col + "_qcut", Column.DType.STRING);
            Column out = empty.column(col + "_qcut");
            while (out.size() < empty.rowCount()) out.add(null);
            return empty;
        }
        double[] edges = new double[q + 1];
        edges[0] = vals.get(0);
        edges[q] = vals.get(vals.size() - 1);
        for (int i = 1; i < q; i++) {
            double pos = (vals.size() - 1) * (i / (double) q);
            int lo = (int) Math.floor(pos);
            int hi = (int) Math.ceil(pos);
            if (lo == hi) edges[i] = vals.get(lo);
            else {
                double t = pos - lo;
                edges[i] = vals.get(lo) * (1 - t) + vals.get(hi) * t;
            }
        }
        // handle duplicate edges
        List<Double> uniq = new ArrayList<>();
        uniq.add(edges[0]);
        for (int i = 1; i < edges.length; i++) {
            if (edges[i] > uniq.get(uniq.size() - 1)) uniq.add(edges[i]);
            else if ("raise".equalsIgnoreCase(duplicates)) {
                throw new IllegalArgumentException("Duplicate bin edges at qcut; try duplicates=\"drop\"");
            }
        }
        edges = uniq.stream().mapToDouble(Double::doubleValue).toArray();
        int bins = edges.length - 1;
        String[] labs = labels;
        if (labs == null || labs.length != bins) {
            labs = new String[bins];
            for (int i = 0; i < bins; i++) labs[i] = String.valueOf(i);
        }

        DataFrame result = df.copy();
        String outName = col + "_qcut";
        if (result.hasColumn(outName)) result.removeColumn(outName);
        result.addColumn(outName, Column.DType.STRING);
        Column out = result.column(outName);
        while (out.size() < result.rowCount()) out.add(null);
        for (int i = 0; i < result.rowCount(); i++) {
            double d = DataValues.asDouble(src.get(i));
            if (Double.isNaN(d)) { out.set(i, null); continue; }
            int b = bins - 1;
            for (int k = 0; k < bins; k++) {
                boolean last = k == bins - 1;
                if (last ? d <= edges[k + 1] : d < edges[k + 1] || (k == 0 && d == edges[0])) {
                    // include left edge of first bin, right edge of last
                    if (d >= edges[k] || (k == 0)) { b = k; break; }
                }
                if (d >= edges[k] && (last ? d <= edges[k + 1] : d < edges[k + 1])) {
                    b = k; break;
                }
            }
            // simpler assignment
            b = bins - 1;
            for (int k = 0; k < bins; k++) {
                if (k == bins - 1) {
                    if (d >= edges[k] && d <= edges[k + 1]) { b = k; break; }
                } else if (d >= edges[k] && d < edges[k + 1]) {
                    b = k; break;
                }
            }
            out.set(i, labs[Math.max(0, Math.min(b, labs.length - 1))]);
        }
        return result;
    }

    public static DataFrame qcut(DataFrame df, String col, int q) {
        return qcut(df, col, q, null, "drop");
    }

    // ================================================================
    // Monotonic / unique / memory / clip helpers
    // ================================================================

    public static boolean isMonotonicIncreasing(DataFrame df, String col) {
        Column c = df.column(col);
        for (int i = 1; i < df.rowCount(); i++) {
            if (Expression.compareVals(c.get(i - 1), c.get(i)) > 0) return false;
        }
        return true;
    }

    public static boolean isMonotonicDecreasing(DataFrame df, String col) {
        Column c = df.column(col);
        for (int i = 1; i < df.rowCount(); i++) {
            if (Expression.compareVals(c.get(i - 1), c.get(i)) < 0) return false;
        }
        return true;
    }

    public static boolean isUnique(DataFrame df, String col) {
        Set<Object> seen = new HashSet<>();
        Column c = df.column(col);
        for (int i = 0; i < df.rowCount(); i++) {
            if (!seen.add(c.get(i))) return false;
        }
        return true;
    }

    /**
     * Approximate deep memory usage per column (bytes). Object/string columns
     * use UTF-16 length estimate; numeric use fixed width.
     */
    public static Map<String, Long> memoryUsage(DataFrame df, boolean deep) {
        Map<String, Long> out = new LinkedHashMap<>();
        for (Column c : df.columns()) {
            long bytes = 0;
            switch (c.dtype()) {
                case INT32, FLOAT32, BOOLEAN -> bytes = (long) df.rowCount() * 4;
                case INT64, FLOAT64 -> bytes = (long) df.rowCount() * 8;
                default -> {
                    if (deep) {
                        for (int i = 0; i < df.rowCount(); i++) {
                            Object v = c.get(i);
                            if (v == null) bytes += 8;
                            else if (v instanceof String s) bytes += 24L + (long) s.length() * 2;
                            else if (v instanceof List<?> list) bytes += 24L + list.size() * 16L;
                            else if (v instanceof Map<?, ?> map) bytes += 32L + map.size() * 32L;
                            else bytes += 32;
                        }
                    } else {
                        bytes = (long) df.rowCount() * 24;
                    }
                }
            }
            out.put(c.name(), bytes);
        }
        return out;
    }

    /** Polars {@code estimate_size()} — total estimated bytes. */
    public static long estimateSize(DataFrame df) {
        return memoryUsage(df, true).values().stream().mapToLong(Long::longValue).sum();
    }

    /**
     * Weighted sample without/with replacement.
     * @param weights per-row weights (same length as df); null = uniform
     */
    public static DataFrame sampleWeighted(DataFrame df, int n, double[] weights,
                                           boolean replace, Long seed) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        int N = df.rowCount();
        if (N == 0) return DataFrame.create();
        Random rng = seed == null ? new Random() : new Random(seed);
        if (weights != null && weights.length != N) {
            throw new IllegalArgumentException("weights length must equal rowCount");
        }
        double[] w = weights == null ? uniform(N) : weights.clone();
        // normalize non-negative
        double sum = 0;
        for (int i = 0; i < N; i++) {
            if (w[i] < 0) w[i] = 0;
            sum += w[i];
        }
        if (sum <= 0) throw new IllegalArgumentException("weights sum must be > 0");
        for (int i = 0; i < N; i++) w[i] /= sum;

        int[] pick = new int[n];
        if (replace) {
            for (int i = 0; i < n; i++) pick[i] = weightedDraw(w, rng);
        } else {
            if (n > N) n = N;
            // sequential sampling without replacement (Efraimidis-Spirakis)
            double[] keys = new double[N];
            for (int i = 0; i < N; i++) {
                double u = Math.max(rng.nextDouble(), 1e-12);
                keys[i] = w[i] <= 0 ? Double.NEGATIVE_INFINITY : Math.log(u) / w[i];
            }
            Integer[] order = new Integer[N];
            for (int i = 0; i < N; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(keys[b], keys[a]));
            pick = new int[n];
            for (int i = 0; i < n; i++) pick[i] = order[i];
        }
        return df.loc(pick);
    }

    // ================================================================
    // helpers
    // ================================================================

    private static boolean isTrue(Object v) {
        if (v == null) return false;
        if (v instanceof Boolean b) return b;
        if (v instanceof Number n) return n.doubleValue() != 0;
        return Boolean.parseBoolean(v.toString());
    }

    private static String rowKey(DataFrame df, int row) {
        StringBuilder sb = new StringBuilder();
        for (Column c : df.columns()) {
            Object v = c.get(row);
            sb.append(v == null ? "\0" : v.toString()).append('');
        }
        return sb.toString();
    }

    private static Set<String> rowKeySet(DataFrame df) {
        Set<String> s = new HashSet<>();
        for (int i = 0; i < df.rowCount(); i++) s.add(rowKey(df, i));
        return s;
    }

    private static void fillAlong(DataFrame df, String method) {
        for (Column c : df.columns()) {
            if ("ffill".equals(method)) {
                Object last = null;
                for (int i = 0; i < df.rowCount(); i++) {
                    Object v = c.get(i);
                    if (v == null) c.set(i, last);
                    else last = v;
                }
            } else if ("bfill".equals(method)) {
                Object next = null;
                for (int i = df.rowCount() - 1; i >= 0; i--) {
                    Object v = c.get(i);
                    if (v == null) c.set(i, next);
                    else next = v;
                }
            }
            // nearest left unimplemented for multi-col reindex fill
        }
    }

    private static String groupKey(DataFrame df, int row, String[] byCols) {
        if (byCols == null || byCols.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < byCols.length; i++) {
            if (i > 0) sb.append('|');
            Object v = df.get(row, byCols[i]);
            sb.append(v == null ? "\0" : v.toString());
        }
        return sb.toString();
    }

    private static int asofBackward(Column rKey, Object lk, Double tol) {
        return asofBackwardGrouped(rKey, null, lk, tol, true);
    }

    private static int asofForward(Column rKey, Object lk, Double tol) {
        return asofForwardGrouped(rKey, null, lk, tol, true);
    }

    private static int asofNearest(Column rKey, Object lk, Double tol) {
        return asofNearestGrouped(rKey, null, lk, tol, true);
    }

    /** @param candidates null = scan full column 0..size-1 */
    private static int asofBackwardGrouped(Column rKey, List<Integer> candidates,
                                           Object lk, Double tol, boolean allowExact) {
        int best = -1;
        int n = candidates == null ? rKey.size() : candidates.size();
        for (int i = 0; i < n; i++) {
            int idx = candidates == null ? i : candidates.get(i);
            Object rk = rKey.get(idx);
            if (rk == null) continue;
            int cmp = Expression.compareVals(rk, lk);
            if (cmp < 0 || (cmp == 0 && allowExact)) {
                if (tol != null && !withinTol(rk, lk, tol)) continue;
                best = idx;
            } else if (cmp > 0) {
                break; // sorted ascending
            } else {
                // cmp==0 && !allowExact → skip exact, continue looking for strictly less
            }
        }
        return best;
    }

    private static int asofForwardGrouped(Column rKey, List<Integer> candidates,
                                          Object lk, Double tol, boolean allowExact) {
        int n = candidates == null ? rKey.size() : candidates.size();
        for (int i = 0; i < n; i++) {
            int idx = candidates == null ? i : candidates.get(i);
            Object rk = rKey.get(idx);
            if (rk == null) continue;
            int cmp = Expression.compareVals(rk, lk);
            if (cmp > 0 || (cmp == 0 && allowExact)) {
                if (tol != null && !withinTol(rk, lk, tol)) return -1;
                return idx;
            }
            // cmp==0 && !allowExact → keep searching for strictly greater
        }
        return -1;
    }

    private static int asofNearestGrouped(Column rKey, List<Integer> candidates,
                                          Object lk, Double tol, boolean allowExact) {
        int best = -1;
        double bestDist = Double.POSITIVE_INFINITY;
        int n = candidates == null ? rKey.size() : candidates.size();
        for (int i = 0; i < n; i++) {
            int idx = candidates == null ? i : candidates.get(i);
            Object rk = rKey.get(idx);
            if (rk == null) continue;
            int cmp = Expression.compareVals(rk, lk);
            if (cmp == 0 && !allowExact) continue;
            double d = absDist(rk, lk);
            if (d < bestDist) {
                bestDist = d;
                best = idx;
            }
        }
        if (best >= 0 && tol != null && bestDist > tol) return -1;
        return best;
    }

    private static boolean withinTol(Object a, Object b, double tol) {
        return absDist(a, b) <= tol;
    }

    private static double absDist(Object a, Object b) {
        if (a instanceof Number && b instanceof Number) {
            return Math.abs(((Number) a).doubleValue() - ((Number) b).doubleValue());
        }
        // non-numeric: 0 if equal else +inf
        return Objects.equals(a, b) ? 0 : Double.POSITIVE_INFINITY;
    }

    private static double[] uniform(int n) {
        double[] w = new double[n];
        Arrays.fill(w, 1.0);
        return w;
    }

    private static int weightedDraw(double[] w, Random rng) {
        double r = rng.nextDouble();
        double acc = 0;
        for (int i = 0; i < w.length; i++) {
            acc += w[i];
            if (r <= acc) return i;
        }
        return w.length - 1;
    }
}
