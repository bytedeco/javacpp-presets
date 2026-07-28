package org.bytedeco.pytorch.dataframe;

import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataset;
import org.bytedeco.pytorch.dataframe.feature.pipeline.DataFramePipeline;
import org.bytedeco.pytorch.dataframe.feature.pipeline.Pipeline;

import java.util.*;

/**
 * Structural reshape operators for {@link DataFrame}:
 * row/column {@code split}, {@code cat}/{@code stack}, {@code expand}, {@code compress},
 * and {@code trainTestSplit}.
 *
 * <p>Bound as instance methods on {@link DataFrame} for fluent use:
 * <pre>
 *   DataFrame[] parts = df.splitRows(4);           // 4 roughly equal row chunks
 *   DataFrame[] cols  = df.splitCols(2);           // 2 column chunks
 *   DataFrame cat     = DataFrame.cat(a, b);       // axis=0 concat
 *   DataFrame stacked = DataFrame.stack(a, b);     // axis=1 concat
 *   DataFrame wide    = df.expand("item_seq");      // list → item_seq_0..N
 *   DataFrame packed  = wide.compress("item_seq", "item_seq_0", "item_seq_1", ...);
 *   TrainTestSplit tt = df.trainTestSplit(0.2, 42L);
 * </pre>
 */
public final class DataFrameOps {
    private DataFrameOps() {}

    // ---- split (rows / columns) ----

    /**
     * Split into {@code n} roughly equal row chunks (last chunk absorbs remainder).
     * Empty frame → empty array when n&lt;=0; otherwise n frames (some may be empty).
     */
    public static DataFrame[] splitRows(DataFrame df, int n) {
        Objects.requireNonNull(df, "df");
        if (n <= 0) throw new IllegalArgumentException("n must be > 0");
        int rows = df.rowCount();
        DataFrame[] out = new DataFrame[n];
        if (rows == 0) {
            for (int i = 0; i < n; i++) out[i] = emptyLike(df);
            return out;
        }
        int base = rows / n;
        int rem = rows % n;
        int start = 0;
        for (int i = 0; i < n; i++) {
            int size = base + (i < rem ? 1 : 0);
            out[i] = df.iloc(start, start + size);
            start += size;
        }
        return out;
    }

    /**
     * Split rows at explicit indices (like numpy.array_split with indices).
     * {@code indices} are exclusive end positions of each chunk except the last
     * which runs to the end. Example: indices={3,7} on 10 rows → [0:3), [3:7), [7:10).
     */
    public static DataFrame[] splitRowsAt(DataFrame df, int... indices) {
        Objects.requireNonNull(df, "df");
        int rows = df.rowCount();
        if (indices == null || indices.length == 0) return new DataFrame[]{df.copy()};
        int[] cuts = Arrays.copyOf(indices, indices.length);
        Arrays.sort(cuts);
        List<DataFrame> parts = new ArrayList<>(cuts.length + 1);
        int start = 0;
        for (int cut : cuts) {
            int end = Math.max(0, Math.min(rows, cut));
            if (end < start) end = start;
            parts.add(df.iloc(start, end));
            start = end;
        }
        parts.add(df.iloc(start, rows));
        return parts.toArray(new DataFrame[0]);
    }

    /** Split into {@code n} roughly equal column chunks. */
    public static DataFrame[] splitCols(DataFrame df, int n) {
        Objects.requireNonNull(df, "df");
        if (n <= 0) throw new IllegalArgumentException("n must be > 0");
        int cols = df.columnCount();
        DataFrame[] out = new DataFrame[n];
        if (cols == 0) {
            for (int i = 0; i < n; i++) out[i] = DataFrame.create();
            return out;
        }
        int base = cols / n;
        int rem = cols % n;
        int start = 0;
        for (int i = 0; i < n; i++) {
            int size = base + (i < rem ? 1 : 0);
            out[i] = df.iloc(0, df.rowCount(), start, start + size);
            start += size;
        }
        return out;
    }

    /** Split columns by name groups: each String[] is one output frame's columns. */
    public static DataFrame[] splitColsByName(DataFrame df, String[]... groups) {
        Objects.requireNonNull(df, "df");
        if (groups == null || groups.length == 0) return new DataFrame[]{df.copy()};
        DataFrame[] out = new DataFrame[groups.length];
        for (int i = 0; i < groups.length; i++) {
            out[i] = groups[i] == null || groups[i].length == 0
                ? emptyLike(df).iloc(0, 0) // empty cols, 0 rows? keep rows with no cols
                : df.select(groups[i]);
            // select drops rows? no - select keeps all rows
            if (groups[i] == null || groups[i].length == 0) {
                DataFrame empty = DataFrame.create();
                // preserve row count with no columns via sync: just create empty
                out[i] = empty;
            }
        }
        return out;
    }

    // ---- cat / stack (aliases of concat) ----

    /** Vertical concat (axis=0). Alias of {@link DataFrame#vstack(DataFrame...)}. */
    public static DataFrame cat(DataFrame... frames) throws Exception {
        return DataFrame.vstack(frames);
    }

    public static DataFrame cat(List<DataFrame> frames) throws Exception {
        return DataFrame.vstack(frames);
    }

    /**
     * Horizontal stack (axis=1). Alias of {@link DataFrame#hstack(DataFrame...)}.
     * Column name collisions are disambiguated with {@code _1}, {@code _2}, …
     */
    public static DataFrame stack(DataFrame... frames) throws Exception {
        return stack(Arrays.asList(frames), true);
    }

    public static DataFrame stack(List<DataFrame> frames) throws Exception {
        return stack(frames, true);
    }

    /**
     * Horizontal stack with optional rename on collision.
     * @param renameOnCollision when true, duplicate names become {@code name_1}, …
     */
    public static DataFrame stack(List<DataFrame> frames, boolean renameOnCollision) throws Exception {
        if (frames == null || frames.isEmpty()) return DataFrame.create();
        if (!renameOnCollision) return DataFrame.hstack(frames);

        DataFrame result = DataFrame.create();
        Set<String> used = new LinkedHashSet<>();
        int maxRows = 0;
        for (DataFrame f : frames) if (f != null) maxRows = Math.max(maxRows, f.rowCount());

        for (DataFrame f : frames) {
            if (f == null) continue;
            for (Column c : f.columns()) {
                String name = uniqueName(c.name(), used);
                used.add(name);
                Column copy = new Column(name, c.dtype());
                for (int i = 0; i < maxRows; i++) {
                    copy.add(i < c.size() ? c.get(i) : null);
                }
                result.addColumn(copy);
            }
        }
        // sync row count
        if (result.columnCount() > 0) {
            // rowCount is driven by columns; force via reflection-safe path: add empty rows if needed
            // Column sizes already == maxRows; DataFrame.rowCount may still be 0 until sync
            result = forceRowCount(result, maxRows);
        }
        return result;
    }

    private static DataFrame forceRowCount(DataFrame df, int rows) {
        // DataFrame tracks rowCount separately; re-materialize via loc of all rows
        if (df.rowCount() == rows) return df;
        if (df.columnCount() == 0) return df;
        // When columns were added with full data but rowCount not updated, use first col size
        int n = df.column(0).size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        return df.loc(idx);
    }

    private static String uniqueName(String base, Set<String> used) {
        if (!used.contains(base)) return base;
        int i = 1;
        while (used.contains(base + "_" + i)) i++;
        return base + "_" + i;
    }

    // ---- expand / compress (list ↔ wide) ----

    /**
     * Expand a LIST / VECTOR / array / List cell column into {@code col_0 .. col_{width-1}}.
     * Width is the max length across rows (or {@code maxWidth} if &gt;0).
     * Original column is dropped by default.
     */
    public static DataFrame expand(DataFrame df, String listCol) {
        return expand(df, listCol, -1, true, listCol + "_");
    }

    public static DataFrame expand(DataFrame df, String listCol, int maxWidth,
                                   boolean dropOriginal, String prefix) {
        Objects.requireNonNull(df, "df");
        Column col = df.column(listCol);
        int width = 0;
        Object[] cells = new Object[df.rowCount()];
        for (int i = 0; i < df.rowCount(); i++) {
            Object v = col.get(i);
            cells[i] = v;
            width = Math.max(width, listLen(v));
        }
        if (maxWidth > 0) width = Math.min(width, maxWidth);
        if (width == 0) {
            DataFrame out = df.copy();
            if (dropOriginal && out.hasColumn(listCol)) out.removeColumn(listCol);
            return out;
        }

        // Infer element dtype from first non-null element
        Column.DType elemDt = Column.DType.FLOAT64;
        outer:
        for (Object cell : cells) {
            Object e0 = listElem(cell, 0);
            if (e0 == null) continue;
            if (e0 instanceof Integer) { elemDt = Column.DType.INT32; break; }
            if (e0 instanceof Long) { elemDt = Column.DType.INT64; break; }
            if (e0 instanceof Float) { elemDt = Column.DType.FLOAT32; break; }
            if (e0 instanceof Double) { elemDt = Column.DType.FLOAT64; break; }
            if (e0 instanceof Boolean) { elemDt = Column.DType.BOOLEAN; break; }
            if (e0 instanceof String) { elemDt = Column.DType.STRING; break; }
            elemDt = Column.DType.STRING;
            break outer;
        }

        DataFrame out = DataFrame.create();
        for (Column c : df.columns()) {
            if (dropOriginal && c.name().equals(listCol)) continue;
            out.addColumn(c.copy());
        }
        String pfx = prefix == null ? listCol + "_" : prefix;
        for (int j = 0; j < width; j++) {
            String name = pfx + j;
            // avoid collision
            String n = name;
            int k = 1;
            while (out.hasColumn(n)) n = name + "_" + (k++);
            Column nc = new Column(n, elemDt);
            for (int i = 0; i < df.rowCount(); i++) {
                nc.add(listElem(cells[i], j));
            }
            out.addColumn(nc);
        }
        return forceRowCount(out, df.rowCount());
    }

    /**
     * Compress wide columns {@code prefix0, prefix1, ...} (or explicit names) into one LIST column.
     * Numeric homogeneous → primitive array (long[]/int[]/float[]/double[]); else List.
     */
    public static DataFrame compress(DataFrame df, String outCol, String... sourceCols) {
        Objects.requireNonNull(df, "df");
        if (sourceCols == null || sourceCols.length == 0)
            throw new IllegalArgumentException("sourceCols required");
        DataFrame out = DataFrame.create();
        Set<String> drop = new HashSet<>(Arrays.asList(sourceCols));
        for (Column c : df.columns()) {
            if (!drop.contains(c.name())) out.addColumn(c.copy());
        }
        Column listCol = new Column(outCol, Column.DType.LIST);
        for (int i = 0; i < df.rowCount(); i++) {
            List<Object> elems = new ArrayList<>(sourceCols.length);
            for (String sc : sourceCols) {
                elems.add(df.get(i, sc));
            }
            listCol.add(densify(elems));
        }
        if (out.hasColumn(outCol)) out.removeColumn(outCol);
        out.addColumn(listCol);
        return forceRowCount(out, df.rowCount());
    }

    /**
     * Compress columns matching {@code prefix + index} for index in {@code [0, width)}.
     */
    public static DataFrame compressPrefix(DataFrame df, String outCol, String prefix, int width) {
        String[] cols = new String[width];
        for (int i = 0; i < width; i++) cols[i] = prefix + i;
        return compress(df, outCol, cols);
    }

    // ---- train / test split ----

    /**
     * Random train/test split by rows.
     * @param testSize fraction in (0,1) or absolute count if &gt;=1
     */
    public static TrainTestSplit trainTestSplit(DataFrame df, double testSize) {
        return trainTestSplit(df, testSize, true, null, null);
    }

    public static TrainTestSplit trainTestSplit(DataFrame df, double testSize, long seed) {
        return trainTestSplit(df, testSize, true, seed, null);
    }

    /**
     * @param stratifyCol optional column name for stratified split (class labels)
     */
    public static TrainTestSplit trainTestSplit(DataFrame df, double testSize,
                                                boolean shuffle, Long seed,
                                                String stratifyCol) {
        Objects.requireNonNull(df, "df");
        int n = df.rowCount();
        if (n == 0) return new TrainTestSplit(emptyLike(df), emptyLike(df), new int[0], new int[0]);

        int nTest;
        if (testSize >= 1.0) nTest = (int) Math.min(n, Math.round(testSize));
        else if (testSize <= 0.0) nTest = 0;
        else nTest = (int) Math.round(n * testSize);
        nTest = Math.max(0, Math.min(n, nTest));
        int nTrain = n - nTest;

        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;

        if (stratifyCol != null && !stratifyCol.isEmpty()) {
            idx = stratifiedIndices(df, stratifyCol, nTest, seed);
        } else if (shuffle) {
            Random rng = seed == null ? new Random() : new Random(seed);
            for (int i = n - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }

        int[] testIdx = Arrays.copyOfRange(idx, 0, nTest);
        int[] trainIdx = Arrays.copyOfRange(idx, nTest, n);
        // keep train first for stable "train then test" ordering of remainder
        // actually after shuffle first nTest are test — swap semantics to match sklearn:
        // sklearn takes last portion as test after shuffle. Reorder:
        trainIdx = Arrays.copyOfRange(idx, 0, nTrain);
        testIdx = Arrays.copyOfRange(idx, nTrain, n);

        return new TrainTestSplit(df.loc(trainIdx), df.loc(testIdx), trainIdx, testIdx);
    }

    private static int[] stratifiedIndices(DataFrame df, String col, int nTest, Long seed) {
        int n = df.rowCount();
        Map<Object, List<Integer>> byClass = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) {
            Object k = df.get(i, col);
            byClass.computeIfAbsent(k == null ? "__null__" : k, x -> new ArrayList<>()).add(i);
        }
        Random rng = seed == null ? new Random() : new Random(seed);
        List<Integer> test = new ArrayList<>(nTest);
        List<Integer> train = new ArrayList<>(n - nTest);
        double frac = n == 0 ? 0 : (double) nTest / n;
        for (List<Integer> members : byClass.values()) {
            Collections.shuffle(members, rng);
            int take = (int) Math.round(members.size() * frac);
            // ensure at least 1 test if class large enough and nTest>0
            if (nTest > 0 && take == 0 && members.size() > 1) take = 1;
            take = Math.min(take, members.size());
            test.addAll(members.subList(0, take));
            train.addAll(members.subList(take, members.size()));
        }
        // adjust sizes if rounding drifted
        Collections.shuffle(test, rng);
        Collections.shuffle(train, rng);
        while (test.size() > nTest && !test.isEmpty()) {
            train.add(test.remove(test.size() - 1));
        }
        while (test.size() < nTest && !train.isEmpty()) {
            test.add(train.remove(train.size() - 1));
        }
        int[] out = new int[n];
        int p = 0;
        for (int i : train) out[p++] = i;
        for (int i : test) out[p++] = i;
        return out;
    }

    // ---- feature / label selection ----

    /**
     * Select feature columns + label columns into a structured view.
     * Does not copy data beyond column projection.
     */
    public static FeatureLabelSplit featureLabel(DataFrame df,
                                                 String[] featureCols,
                                                 String... labelCols) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(featureCols, "featureCols");
        String[] labels = labelCols == null ? new String[0] : labelCols;
        DataFrame X = featureCols.length == 0 ? DataFrame.create() : df.select(featureCols);
        DataFrame y = labels.length == 0 ? DataFrame.create() : df.select(labels);
        // align rows
        if (X.rowCount() != df.rowCount() && featureCols.length > 0) {
            // select keeps rows
        }
        return new FeatureLabelSplit(X, y, featureCols, labels);
    }

    /** Features = all columns except labels. */
    public static FeatureLabelSplit featureLabelExclude(DataFrame df, String... labelCols) {
        Set<String> labs = new HashSet<>(Arrays.asList(labelCols == null ? new String[0] : labelCols));
        List<String> feats = new ArrayList<>();
        for (Column c : df.columns()) {
            if (!labs.contains(c.name())) feats.add(c.name());
        }
        return featureLabel(df, feats.toArray(new String[0]), labelCols);
    }

    // ---- helpers ----

    private static DataFrame emptyLike(DataFrame df) {
        DataFrame out = DataFrame.create();
        for (Column c : df.columns()) out.addColumn(c.name(), c.dtype());
        return out;
    }

    private static int listLen(Object v) {
        if (v == null) return 0;
        if (v instanceof long[]) return ((long[]) v).length;
        if (v instanceof int[]) return ((int[]) v).length;
        if (v instanceof float[]) return ((float[]) v).length;
        if (v instanceof double[]) return ((double[]) v).length;
        if (v instanceof short[]) return ((short[]) v).length;
        if (v instanceof byte[]) return ((byte[]) v).length;
        if (v instanceof boolean[]) return ((boolean[]) v).length;
        if (v instanceof Object[]) return ((Object[]) v).length;
        if (v instanceof List) return ((List<?>) v).size();
        return 1; // scalar treated as length-1
    }

    private static Object listElem(Object v, int j) {
        if (v == null) return null;
        if (v instanceof long[]) {
            long[] a = (long[]) v;
            return j < a.length ? a[j] : null;
        }
        if (v instanceof int[]) {
            int[] a = (int[]) v;
            return j < a.length ? a[j] : null;
        }
        if (v instanceof float[]) {
            float[] a = (float[]) v;
            return j < a.length ? a[j] : null;
        }
        if (v instanceof double[]) {
            double[] a = (double[]) v;
            return j < a.length ? a[j] : null;
        }
        if (v instanceof short[]) {
            short[] a = (short[]) v;
            return j < a.length ? (int) a[j] : null;
        }
        if (v instanceof byte[]) {
            byte[] a = (byte[]) v;
            return j < a.length ? (int) a[j] : null;
        }
        if (v instanceof boolean[]) {
            boolean[] a = (boolean[]) v;
            return j < a.length ? a[j] : null;
        }
        if (v instanceof Object[]) {
            Object[] a = (Object[]) v;
            return j < a.length ? a[j] : null;
        }
        if (v instanceof List) {
            List<?> a = (List<?>) v;
            return j < a.size() ? a.get(j) : null;
        }
        return j == 0 ? v : null;
    }

    private static Object densify(List<Object> elems) {
        if (elems == null || elems.isEmpty()) return new long[0];
        boolean allLong = true, allInt = true, allFloat = true, allDouble = true;
        for (Object o : elems) {
            if (o == null) { allLong = allInt = allFloat = allDouble = false; break; }
            if (!(o instanceof Number)) { allLong = allInt = allFloat = allDouble = false; break; }
            if (!(o instanceof Long) && !(o instanceof Integer) && !(o instanceof Short) && !(o instanceof Byte))
                allLong = false;
            if (!(o instanceof Integer) && !(o instanceof Short) && !(o instanceof Byte))
                allInt = false;
            if (!(o instanceof Float)) allFloat = false;
            if (!(o instanceof Double) && !(o instanceof Float)) allDouble = false;
        }
        if (allInt) {
            int[] a = new int[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).intValue();
            return a;
        }
        if (allLong) {
            long[] a = new long[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).longValue();
            return a;
        }
        if (allFloat) {
            float[] a = new float[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).floatValue();
            return a;
        }
        if (allDouble) {
            double[] a = new double[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).doubleValue();
            return a;
        }
        return elems;
    }

    // ---- result types ----

    /** Result of {@link #trainTestSplit}. */
    public static final class TrainTestSplit {
        public final DataFrame train;
        public final DataFrame test;
        public final int[] trainIndices;
        public final int[] testIndices;

        public TrainTestSplit(DataFrame train, DataFrame test, int[] trainIndices, int[] testIndices) {
            this.train = train;
            this.test = test;
            this.trainIndices = trainIndices;
            this.testIndices = testIndices;
        }

        public DataFrame train() { return train; }
        public DataFrame test() { return test; }
        public int[] trainIndices() { return trainIndices; }
        public int[] testIndices() { return testIndices; }

        /** Further split train into features/labels. */
        public FeatureLabelSplit featuresLabels(String[] featureCols, String... labelCols) {
            return featureLabel(train, featureCols, labelCols);
        }
    }

    /** Feature matrix X + label frame y. */
    public static final class FeatureLabelSplit {
        public final DataFrame X;
        public final DataFrame y;
        public final String[] featureCols;
        public final String[] labelCols;

        public FeatureLabelSplit(DataFrame X, DataFrame y, String[] featureCols, String[] labelCols) {
            this.X = X;
            this.y = y;
            this.featureCols = featureCols == null ? new String[0] : featureCols.clone();
            this.labelCols = labelCols == null ? new String[0] : labelCols.clone();
        }

        public DataFrame X() { return X; }
        public DataFrame y() { return y; }
        public String[] featureCols() { return featureCols.clone(); }
        public String[] labelCols() { return labelCols.clone(); }

        /** Apply an optional feature pipeline to X (fitTransform). */
        public FeatureLabelSplit transformFeatures(
                DataFramePipeline pipe)
                throws Exception {
            if (pipe == null) return this;
            DataFrame Xt = pipe.fitTransform(X);
            return new FeatureLabelSplit(Xt, y, featureCols, labelCols);
        }

        public FeatureLabelSplit transformFeatures(
                Pipeline pipe)
                throws Exception {
            if (pipe == null) return this;
            DataFrame Xt = pipe.fitTransform(X);
            return new FeatureLabelSplit(Xt, y, featureCols, labelCols);
        }

        /** Build a {@link DataFrameDataset} from X/y. */
        public DataFrameDataset toDataset() {
            return DataFrameDataset.of(X, y);
        }

        public DataFrameDataset toDataset(
                DataFrameDataset.Options opts) {
            return DataFrameDataset.of(X, y, opts);
        }
    }
}
