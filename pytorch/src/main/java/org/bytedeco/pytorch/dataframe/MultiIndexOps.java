package org.bytedeco.pytorch.dataframe;

import java.util.*;

/**
 * MultiIndex-style level ops for DataFrames that store index levels as
 * leading columns (or any named level columns).
 *
 * <p>Convention: levels are ordinary columns listed in {@code levels} order.
 * This mirrors a "column MultiIndex" view without a separate Index type:
 * <pre>
 *   df.swaplevel(0, 1, "g1", "g2")
 *   df.stack(List.of("g1","g2"), "value")
 *   df.unstack("g2", "value")
 * </pre>
 */
public final class MultiIndexOps {
    private MultiIndexOps() {}

    /**
     * Swap two level columns by position in {@code levels} (Pandas {@code swaplevel}).
     * Returns a frame with those two columns swapped in column order; data unchanged.
     */
    public static DataFrame swaplevel(DataFrame df, int i, int j, String... levels) {
        String[] lv = requireLevels(df, levels);
        int a = normalize(i, lv.length);
        int b = normalize(j, lv.length);
        if (a == b) return df.copy();
        String[] order = df.getColumnNames().toArray(new String[0]);
        // swap positions of lv[a] and lv[b] in column order
        int pa = indexOf(order, lv[a]);
        int pb = indexOf(order, lv[b]);
        if (pa < 0 || pb < 0) throw new IllegalArgumentException("level columns missing");
        String tmp = order[pa];
        order[pa] = order[pb];
        order[pb] = tmp;
        return df.reorderColumns(order);
    }

    /** Swap by level name. */
    public static DataFrame swaplevel(DataFrame df, String levelA, String levelB, String... levels) {
        String[] lv = requireLevels(df, levels);
        return swaplevel(df, indexOf(lv, levelA), indexOf(lv, levelB), levels);
    }

    /**
     * Reorder level columns to {@code order} (subset of levels), keeping other columns after.
     * Pandas {@code reorder_levels}.
     */
    public static DataFrame reorderLevels(DataFrame df, String[] order, String... levels) {
        String[] lv = requireLevels(df, levels);
        Set<String> levelSet = new LinkedHashSet<>(Arrays.asList(lv));
        for (String o : order) {
            if (!levelSet.contains(o)) throw new IllegalArgumentException("unknown level: " + o);
        }
        List<String> cols = new ArrayList<>();
        for (String o : order) cols.add(o);
        // remaining levels not in order keep original relative order
        for (String l : lv) if (!cols.contains(l)) cols.add(l);
        for (String c : df.getColumnNames()) {
            if (!levelSet.contains(c)) cols.add(c);
        }
        return df.reorderColumns(cols.toArray(new String[0]));
    }

    /**
     * Drop one or more level columns (Pandas {@code droplevel}).
     * @param level names or integer positions into {@code levels}
     */
    public static DataFrame droplevel(DataFrame df, Object level, String... levels) {
        String[] lv = requireLevels(df, levels);
        List<String> drop = new ArrayList<>();
        if (level instanceof Object[] arr) {
            for (Object o : arr) drop.add(resolveLevel(lv, o));
        } else if (level instanceof List<?> list) {
            for (Object o : list) drop.add(resolveLevel(lv, o));
        } else {
            drop.add(resolveLevel(lv, level));
        }
        return df.drop(drop.toArray(new String[0]));
    }

    /**
     * Wide → long: melt non-id columns into (variable, value) while keeping level cols.
     * Pandas {@code stack} analogue when levels are columns.
     *
     * @param idLevels columns that stay as identifiers (index levels)
     * @param valueVars columns to stack; null = all non-id columns
     * @param varName name of stacked column names
     * @param valueName name of stacked values
     */
    public static DataFrame stack(DataFrame df, List<String> idLevels,
                                  List<String> valueVars,
                                  String varName, String valueName) {
        List<String> ids = idLevels == null ? List.of() : idLevels;
        List<String> vals = valueVars;
        if (vals == null || vals.isEmpty()) {
            vals = new ArrayList<>();
            for (String c : df.getColumnNames()) {
                if (!ids.contains(c)) vals.add(c);
            }
        }
        String vn = varName == null ? "variable" : varName;
        String valn = valueName == null ? "value" : valueName;
        return df.melt(ids, vals, vn, valn);
    }

    /**
     * Long → wide: pivot {@code valueCol} using {@code levelCol} as new columns.
     * Pandas {@code unstack} analogue.
     *
     * @param indexCols remaining index columns (may be empty)
     * @param levelCol column whose distinct values become new columns
     * @param valueCol values to place in the wide table
     * @param fillValue optional fill for missing combos (null = leave null)
     */
    public static DataFrame unstack(DataFrame df, List<String> indexCols,
                                    String levelCol, String valueCol,
                                    Object fillValue) throws Exception {
        Objects.requireNonNull(levelCol, "levelCol");
        Objects.requireNonNull(valueCol, "valueCol");
        List<String> idx = indexCols == null ? List.of() : indexCols;

        // collect unique level values in first-seen order
        LinkedHashSet<Object> levelVals = new LinkedHashSet<>();
        Column lc = df.column(levelCol);
        for (int i = 0; i < df.rowCount(); i++) levelVals.add(lc.get(i));

        // key = index composite → map levelVal → value
        Map<String, Map<Object, Object>> grid = new LinkedHashMap<>();
        Map<String, Object[]> indexValues = new LinkedHashMap<>();
        for (int r = 0; r < df.rowCount(); r++) {
            StringBuilder kb = new StringBuilder();
            Object[] iv = new Object[idx.size()];
            for (int i = 0; i < idx.size(); i++) {
                Object v = df.get(r, idx.get(i));
                iv[i] = v;
                kb.append(v == null ? "\0" : v).append('');
            }
            String key = kb.toString();
            indexValues.putIfAbsent(key, iv);
            grid.computeIfAbsent(key, k -> new LinkedHashMap<>())
                .put(lc.get(r), df.get(r, valueCol));
        }

        DataFrame result = DataFrame.create();
        for (String c : idx) result.addColumn(c, df.column(c).dtype());
        for (Object lv : levelVals) {
            String name = lv == null ? "null" : String.valueOf(lv);
            // avoid collision with index cols
            if (result.hasColumn(name)) name = valueCol + "_" + name;
            result.addColumn(name, df.column(valueCol).dtype());
        }

        for (Map.Entry<String, Map<Object, Object>> e : grid.entrySet()) {
            int ri = result.addEmptyRow();
            Object[] iv = indexValues.get(e.getKey());
            for (int i = 0; i < idx.size(); i++) result.set(ri, idx.get(i), iv[i]);
            for (Object lv : levelVals) {
                String name = lv == null ? "null" : String.valueOf(lv);
                if (!result.hasColumn(name)) name = valueCol + "_" + name;
                Object v = e.getValue().get(lv);
                result.set(ri, name, v != null ? v : fillValue);
            }
        }
        return result;
    }

    public static DataFrame unstack(DataFrame df, String levelCol, String valueCol) throws Exception {
        // all other columns except level+value are index
        List<String> idx = new ArrayList<>();
        for (String c : df.getColumnNames()) {
            if (!c.equals(levelCol) && !c.equals(valueCol)) idx.add(c);
        }
        return unstack(df, idx, levelCol, valueCol, null);
    }

    // ---- helpers ----

    private static String[] requireLevels(DataFrame df, String... levels) {
        if (levels == null || levels.length == 0) {
            throw new IllegalArgumentException("levels required (column names acting as MultiIndex)");
        }
        for (String l : levels) {
            if (!df.hasColumn(l)) throw new IllegalArgumentException("no such level column: " + l);
        }
        return levels;
    }

    private static int normalize(int i, int n) {
        int x = i < 0 ? n + i : i;
        if (x < 0 || x >= n) throw new IndexOutOfBoundsException("level index " + i);
        return x;
    }

    private static String resolveLevel(String[] levels, Object level) {
        if (level instanceof Number n) return levels[normalize(n.intValue(), levels.length)];
        String name = String.valueOf(level);
        for (String l : levels) if (l.equals(name)) return l;
        throw new IllegalArgumentException("unknown level: " + level);
    }

    private static int indexOf(String[] arr, String name) {
        for (int i = 0; i < arr.length; i++) if (arr[i].equals(name)) return i;
        return -1;
    }
}
