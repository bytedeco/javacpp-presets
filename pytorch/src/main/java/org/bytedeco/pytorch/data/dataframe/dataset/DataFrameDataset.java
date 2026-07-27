package org.bytedeco.pytorch.data.dataframe.dataset;

import java.util.*;
import java.util.function.Function;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.SizeTOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.datasets.TensorDataset;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.DataFrameOps;
import org.bytedeco.pytorch.data.dataframe.feature.pipeline.DataFramePipeline;
import org.bytedeco.pytorch.data.dataframe.feature.pipeline.Pipeline;
import org.bytedeco.pytorch.global.torch;

/**
 * DataFrame-backed dataset that yields (features, labels) per index.
 *
 * <p>Supports:
 * <ul>
 *   <li>Scalar numeric feature columns → stacked float tensor {@code [n_feat]}</li>
 *   <li>LIST / VECTOR / long[] / float[] columns → dense tensors (e.g. item_seq → long[64])</li>
 *   <li>Optional label columns (classification / regression)</li>
 *   <li>Optional feature-engineering {@link Pipeline} / {@link DataFramePipeline}</li>
 *   <li>Pure-Java {@link DataFrameDataLoader} with named multi-feature batches</li>
 *   <li>Interop with native Dataset / DataLoader via {@link #asDataset()} /
 *       {@link #asJavaTensorDataset()} / {@link #nativeDataLoader()}</li>
 *   <li>Materialized conversion to {@link TensorDataset} and reverse
 *       {@link #fromFeatureLabelTensors} / {@link #fromTensorDataset}</li>
 * </ul>
 *
 * <pre>
 *   DataFrame df = DataFrame.readParquet("train.parquet");
 *   DataFrameDataset ds = DataFrameDataset.builder(df)
 *       .features("user_id", "item_id", "likes_level", "views_level")
 *       .sequenceFeature("item_seq")   // long[64] LIST
 *       .labels("label")
 *       .pipeline(optionalPipe)       // or null
 *       .build();
 *
 *   // Pure-Java multi-feature loader (named sequences + scalars)
 *   DataFrameDataLoader loader = ds.dataloader()
 *       .batchSize(256)
 *       .shuffle(true)
 *       .dropLast(false)
 *       .build();
 *   for (DataFrameDataLoader.Batch b : loader) {
 *       Tensor x = b.features();  // or b.feature("item_seq")
 *       Tensor y = b.labels();
 *   }
 *
 *   // Native Dataset / DataLoader interop (MNIST-style Example loops)
 *   DataFrameNativeDataset nativeDs = ds.asDataset(); // extends Dataset
 *   SequentialDataLoader nloader = ds.nativeDataLoader()
 *       .batchSize(256).shuffle(false).buildSequential();
 * </pre>
 *
 * <p><b>When to use which path</b>
 * <ul>
 *   <li>{@link #dataloader()} — named multi-feature batches (scalars + sequences)</li>
 *   <li>{@link #asDataset()} + {@link #nativeDataLoader()} — plug into native
 *       {@code RandomDataLoader}/{@code SequentialDataLoader} training loops</li>
 *   <li>{@link #asJavaTensorDataset()} — features-only {@code JavaTensorDataset}</li>
 *   <li>{@link #toTensorDataset()} — materialize a single tensor store</li>
 * </ul>
 */
public final class DataFrameDataset implements Iterable<DataFrameDataset.Sample> {

    /** Per-row sample: named feature tensors + label tensor (may be null). */
    public static final class Sample {
        private final long index;
        private final Map<String, Tensor> features;
        private final Tensor labels;
        private final Tensor stackedFeatures;

        Sample(long index, Map<String, Tensor> features, Tensor stackedFeatures, Tensor labels) {
            this.index = index;
            this.features = features;
            this.stackedFeatures = stackedFeatures;
            this.labels = labels;
        }

        public long index() { return index; }
        public Map<String, Tensor> features() { return features; }
        public Tensor feature(String name) {
            Tensor t = features.get(name);
            if (t == null) throw new IllegalArgumentException("No feature: " + name);
            return t;
        }
        /** Stacked scalar numeric features, or first sequence feature if no scalars. */
        public Tensor data() {
            if (stackedFeatures != null) return stackedFeatures;
            if (!features.isEmpty()) return features.values().iterator().next();
            return torch.empty(new long[]{0});
        }
        public Tensor labels() { return labels; }
        public Tensor target() { return labels; }

        /** Native {@link Example} (data=stacked or first feature, target=labels or empty). */
        public Example toExample() {
            Tensor d = data();
            Tensor t = labels != null ? labels : torch.empty(new long[]{0});
            return new Example(d, t);
        }
    }

    /** Builder / runtime options. */
    public static final class Options {
        public String[] featureCols = new String[0];
        public String[] sequenceCols = new String[0];
        public String[] labelCols = new String[0];
        public boolean stackScalarFeatures = true;
        public boolean labelsAsLong = true; // classification default
        public Pipeline pipeline;
        public DataFramePipeline dfPipeline;
        public Function<DataFrame, DataFrame> customTransform;

        public Options features(String... cols) { this.featureCols = cols == null ? new String[0] : cols; return this; }
        public Options sequenceFeatures(String... cols) { this.sequenceCols = cols == null ? new String[0] : cols; return this; }
        public Options labels(String... cols) { this.labelCols = cols == null ? new String[0] : cols; return this; }
        public Options stackScalarFeatures(boolean v) { this.stackScalarFeatures = v; return this; }
        public Options labelsAsLong(boolean v) { this.labelsAsLong = v; return this; }
        public Options pipeline(Pipeline p) { this.pipeline = p; return this; }
        public Options pipeline(DataFramePipeline p) { this.dfPipeline = p; return this; }
        public Options transform(Function<DataFrame, DataFrame> fn) { this.customTransform = fn; return this; }
    }

    private final DataFrame featuresDf;
    private final DataFrame labelsDf;
    private final Options options;
    private final String[] scalarFeatureNames;
    private final String[] sequenceFeatureNames;
    private final String[] labelNames;

    // caches for packed sequence columns: name → {data, dim, isLong}
    private final Map<String, PackedColumn> packedSequences = new LinkedHashMap<>();
    private float[] packedScalars; // [n * nFeat] row-major
    private int nScalarFeat;
    private long[] packedLabelsLong;
    private float[] packedLabelsFloat;
    private int nLabel;

    private static final class PackedColumn {
        final long[] longData;   // [n * dim] or null
        final float[] floatData; // [n * dim] or null
        final int dim;
        final boolean isLong;
        PackedColumn(long[] d, int dim) { this.longData = d; this.floatData = null; this.dim = dim; this.isLong = true; }
        PackedColumn(float[] d, int dim) { this.longData = null; this.floatData = d; this.dim = dim; this.isLong = false; }
    }

    private DataFrameDataset(DataFrame featuresDf, DataFrame labelsDf, Options options) {
        this.featuresDf = featuresDf;
        this.labelsDf = labelsDf == null ? DataFrame.create() : labelsDf;
        this.options = options == null ? new Options() : options;
        this.scalarFeatureNames = resolveScalarFeatures();
        this.sequenceFeatureNames = resolveSequenceFeatures();
        this.labelNames = resolveLabels();
        materialize();
    }

    public static DataFrameDataset of(DataFrame X, DataFrame y) {
        return of(X, y, new Options());
    }

    public static DataFrameDataset of(DataFrame X, DataFrame y, Options opts) {
        Options o = opts == null ? new Options() : opts;
        if ((o.featureCols == null || o.featureCols.length == 0) && X != null) {
            List<String> names = new ArrayList<>();
            for (Column c : X.columns()) names.add(c.name());
            o.featureCols = names.toArray(new String[0]);
        }
        if ((o.labelCols == null || o.labelCols.length == 0) && y != null) {
            List<String> names = new ArrayList<>();
            for (Column c : y.columns()) names.add(c.name());
            o.labelCols = names.toArray(new String[0]);
        }
        return new DataFrameDataset(X, y, o);
    }

    public static Builder builder(DataFrame df) {
        return new Builder(df);
    }

    public static final class Builder {
        private final DataFrame source;
        private final Options opts = new Options();
        private String[] featureCols;
        private String[] sequenceCols;
        private String[] labelCols;

        Builder(DataFrame source) {
            this.source = Objects.requireNonNull(source, "df");
        }

        public Builder features(String... cols) { this.featureCols = cols; return this; }
        public Builder sequenceFeature(String... cols) { this.sequenceCols = cols; return this; }
        public Builder sequenceFeatures(String... cols) { return sequenceFeature(cols); }
        public Builder labels(String... cols) { this.labelCols = cols; return this; }
        public Builder pipeline(Pipeline p) { opts.pipeline(p); return this; }
        public Builder pipeline(DataFramePipeline p) { opts.pipeline(p); return this; }
        public Builder transform(Function<DataFrame, DataFrame> fn) { opts.transform(fn); return this; }
        public Builder stackScalarFeatures(boolean v) { opts.stackScalarFeatures(v); return this; }
        public Builder labelsAsLong(boolean v) { opts.labelsAsLong(v); return this; }

        public DataFrameDataset build() throws Exception {
            DataFrame df = source;
            // optional full-frame custom transform first
            if (opts.customTransform != null) df = opts.customTransform.apply(df);

            String[] feats = featureCols != null ? featureCols : new String[0];
            String[] seqs = sequenceCols != null ? sequenceCols : new String[0];
            String[] labs = labelCols != null ? labelCols : new String[0];

            // default: all non-label columns are features; list/vector → sequence
            if (feats.length == 0 && seqs.length == 0) {
                List<String> f = new ArrayList<>();
                List<String> s = new ArrayList<>();
                Set<String> labSet = new HashSet<>(Arrays.asList(labs));
                for (Column c : df.columns()) {
                    if (labSet.contains(c.name())) continue;
                    if (isSequenceDtype(c)) s.add(c.name());
                    else f.add(c.name());
                }
                feats = f.toArray(new String[0]);
                seqs = s.toArray(new String[0]);
            }

            opts.features(feats).sequenceFeatures(seqs).labels(labs);

            // Build X from feature + sequence cols
            List<String> xcols = new ArrayList<>();
            xcols.addAll(Arrays.asList(feats));
            xcols.addAll(Arrays.asList(seqs));
            DataFrame X = xcols.isEmpty() ? DataFrame.create() : df.select(xcols.toArray(new String[0]));
            DataFrame y = labs.length == 0 ? DataFrame.create() : df.select(labs);

            // optional pipeline on features only
            if (opts.dfPipeline != null) {
                X = opts.dfPipeline.fitTransform(X);
            } else if (opts.pipeline != null) {
                X = opts.pipeline.fitTransform(X);
            }

            return new DataFrameDataset(X, y, opts);
        }
    }

    private static boolean isSequenceDtype(Column c) {
        Column.DType dt = c.dtype();
        if (dt == Column.DType.LIST || dt == Column.DType.VECTOR
            || dt == Column.DType.EMBEDDING || dt == Column.DType.TENSOR) return true;
        // peek cell
        for (int i = 0; i < Math.min(c.size(), 8); i++) {
            Object v = c.get(i);
            if (v == null) continue;
            return v instanceof long[] || v instanceof int[] || v instanceof float[]
                || v instanceof double[] || v instanceof List || v instanceof Object[];
        }
        return false;
    }

    private String[] resolveScalarFeatures() {
        if (options.featureCols != null && options.featureCols.length > 0) {
            List<String> out = new ArrayList<>();
            for (String n : options.featureCols) {
                if (featuresDf.hasColumn(n) && !isSequenceDtype(featuresDf.column(n))) out.add(n);
            }
            return out.toArray(new String[0]);
        }
        List<String> out = new ArrayList<>();
        for (Column c : featuresDf.columns()) {
            if (!isSequenceDtype(c)) out.add(c.name());
        }
        return out.toArray(new String[0]);
    }

    private String[] resolveSequenceFeatures() {
        if (options.sequenceCols != null && options.sequenceCols.length > 0) {
            List<String> out = new ArrayList<>();
            for (String n : options.sequenceCols) if (featuresDf.hasColumn(n)) out.add(n);
            return out.toArray(new String[0]);
        }
        List<String> out = new ArrayList<>();
        for (Column c : featuresDf.columns()) {
            if (isSequenceDtype(c)) out.add(c.name());
        }
        return out.toArray(new String[0]);
    }

    private String[] resolveLabels() {
        if (options.labelCols != null && options.labelCols.length > 0) {
            List<String> out = new ArrayList<>();
            for (String n : options.labelCols) if (labelsDf.hasColumn(n)) out.add(n);
            return out.toArray(new String[0]);
        }
        List<String> out = new ArrayList<>();
        for (Column c : labelsDf.columns()) out.add(c.name());
        return out.toArray(new String[0]);
    }

    private void materialize() {
        int n = size();
        nScalarFeat = scalarFeatureNames.length;
        if (nScalarFeat > 0) {
            packedScalars = new float[n * nScalarFeat];
            for (int j = 0; j < nScalarFeat; j++) {
                Column c = featuresDf.column(scalarFeatureNames[j]);
                for (int i = 0; i < n; i++) {
                    Object v = c.get(i);
                    packedScalars[i * nScalarFeat + j] = v instanceof Number ? ((Number) v).floatValue() : 0f;
                }
            }
        }
        for (String name : sequenceFeatureNames) {
            packedSequences.put(name, packSequence(featuresDf.column(name)));
        }
        nLabel = labelNames.length;
        if (nLabel > 0) {
            if (options.labelsAsLong) {
                packedLabelsLong = new long[n * nLabel];
                for (int j = 0; j < nLabel; j++) {
                    Column c = labelsDf.column(labelNames[j]);
                    for (int i = 0; i < n; i++) {
                        Object v = c.get(i);
                        packedLabelsLong[i * nLabel + j] = v instanceof Number ? ((Number) v).longValue() : 0L;
                    }
                }
            } else {
                packedLabelsFloat = new float[n * nLabel];
                for (int j = 0; j < nLabel; j++) {
                    Column c = labelsDf.column(labelNames[j]);
                    for (int i = 0; i < n; i++) {
                        Object v = c.get(i);
                        packedLabelsFloat[i * nLabel + j] = v instanceof Number ? ((Number) v).floatValue() : 0f;
                    }
                }
            }
        }
    }

    private static PackedColumn packSequence(Column col) {
        int n = col.size();
        int dim = 0;
        boolean preferLong = true;
        Object[] cells = new Object[n];
        for (int i = 0; i < n; i++) {
            Object v = col.get(i);
            cells[i] = v;
            int len = seqLen(v);
            if (len > dim) dim = len;
            if (v instanceof float[] || v instanceof double[]) preferLong = false;
            if (v instanceof List && !((List<?>) v).isEmpty()) {
                Object e = ((List<?>) v).get(0);
                if (e instanceof Float || e instanceof Double) preferLong = false;
            }
        }
        if (dim == 0) dim = 1;
        if (preferLong) {
            long[] data = new long[n * dim];
            for (int i = 0; i < n; i++) fillLong(data, i * dim, dim, cells[i]);
            return new PackedColumn(data, dim);
        } else {
            float[] data = new float[n * dim];
            for (int i = 0; i < n; i++) fillFloat(data, i * dim, dim, cells[i]);
            return new PackedColumn(data, dim);
        }
    }

    private static int seqLen(Object v) {
        if (v == null) return 0;
        if (v instanceof long[]) return ((long[]) v).length;
        if (v instanceof int[]) return ((int[]) v).length;
        if (v instanceof float[]) return ((float[]) v).length;
        if (v instanceof double[]) return ((double[]) v).length;
        if (v instanceof List) return ((List<?>) v).size();
        if (v instanceof Object[]) return ((Object[]) v).length;
        if (v instanceof Number) return 1;
        return 0;
    }

    private static void fillLong(long[] dest, int off, int dim, Object v) {
        if (v == null) return;
        if (v instanceof long[]) {
            long[] a = (long[]) v;
            System.arraycopy(a, 0, dest, off, Math.min(dim, a.length));
            return;
        }
        if (v instanceof int[]) {
            int[] a = (int[]) v;
            for (int i = 0; i < Math.min(dim, a.length); i++) dest[off + i] = a[i];
            return;
        }
        if (v instanceof float[]) {
            float[] a = (float[]) v;
            for (int i = 0; i < Math.min(dim, a.length); i++) dest[off + i] = (long) a[i];
            return;
        }
        if (v instanceof double[]) {
            double[] a = (double[]) v;
            for (int i = 0; i < Math.min(dim, a.length); i++) dest[off + i] = (long) a[i];
            return;
        }
        if (v instanceof List) {
            List<?> a = (List<?>) v;
            for (int i = 0; i < Math.min(dim, a.size()); i++) {
                Object e = a.get(i);
                dest[off + i] = e instanceof Number ? ((Number) e).longValue() : 0L;
            }
            return;
        }
        if (v instanceof Number) dest[off] = ((Number) v).longValue();
    }

    private static void fillFloat(float[] dest, int off, int dim, Object v) {
        if (v == null) return;
        if (v instanceof float[]) {
            float[] a = (float[]) v;
            System.arraycopy(a, 0, dest, off, Math.min(dim, a.length));
            return;
        }
        if (v instanceof double[]) {
            double[] a = (double[]) v;
            for (int i = 0; i < Math.min(dim, a.length); i++) dest[off + i] = (float) a[i];
            return;
        }
        if (v instanceof long[]) {
            long[] a = (long[]) v;
            for (int i = 0; i < Math.min(dim, a.length); i++) dest[off + i] = a[i];
            return;
        }
        if (v instanceof int[]) {
            int[] a = (int[]) v;
            for (int i = 0; i < Math.min(dim, a.length); i++) dest[off + i] = a[i];
            return;
        }
        if (v instanceof List) {
            List<?> a = (List<?>) v;
            for (int i = 0; i < Math.min(dim, a.size()); i++) {
                Object e = a.get(i);
                dest[off + i] = e instanceof Number ? ((Number) e).floatValue() : 0f;
            }
            return;
        }
        if (v instanceof Number) dest[off] = ((Number) v).floatValue();
    }

    public int size() {
        return Math.max(featuresDf.rowCount(), labelsDf.rowCount());
    }

    public long sizeLong() { return size(); }

    public String[] scalarFeatureNames() { return scalarFeatureNames.clone(); }
    public String[] sequenceFeatureNames() { return sequenceFeatureNames.clone(); }
    public String[] labelNames() { return labelNames.clone(); }
    public DataFrame featuresFrame() { return featuresDf; }
    public DataFrame labelsFrame() { return labelsDf; }
    public Options options() { return options; }

    public Sample get(long index) {
        int i = (int) index;
        if (i < 0 || i >= size()) throw new IndexOutOfBoundsException("index " + index + " size " + size());

        Map<String, Tensor> feats = new LinkedHashMap<>();
        Tensor stacked = null;

        if (nScalarFeat > 0) {
            float[] row = new float[nScalarFeat];
            System.arraycopy(packedScalars, i * nScalarFeat, row, 0, nScalarFeat);
            Tensor t = torch.tensor(row);
            if (options.stackScalarFeatures) {
                stacked = t;
                feats.put("__stacked__", t);
                for (int j = 0; j < nScalarFeat; j++) {
                    feats.put(scalarFeatureNames[j], torch.tensor(new float[]{row[j]}));
                }
            } else {
                for (int j = 0; j < nScalarFeat; j++) {
                    feats.put(scalarFeatureNames[j], torch.tensor(new float[]{row[j]}));
                }
                stacked = t;
            }
        }

        for (Map.Entry<String, PackedColumn> e : packedSequences.entrySet()) {
            PackedColumn pc = e.getValue();
            if (pc.isLong) {
                long[] row = new long[pc.dim];
                System.arraycopy(pc.longData, i * pc.dim, row, 0, pc.dim);
                feats.put(e.getKey(), torch.tensor(row));
            } else {
                float[] row = new float[pc.dim];
                System.arraycopy(pc.floatData, i * pc.dim, row, 0, pc.dim);
                feats.put(e.getKey(), torch.tensor(row));
            }
        }

        Tensor labels = null;
        if (nLabel > 0) {
            if (options.labelsAsLong) {
                if (nLabel == 1) {
                    labels = torch.tensor(new long[]{packedLabelsLong[i]});
                } else {
                    long[] row = new long[nLabel];
                    System.arraycopy(packedLabelsLong, i * nLabel, row, 0, nLabel);
                    labels = torch.tensor(row);
                }
            } else {
                if (nLabel == 1) {
                    labels = torch.tensor(new float[]{packedLabelsFloat[i]});
                } else {
                    float[] row = new float[nLabel];
                    System.arraycopy(packedLabelsFloat, i * nLabel, row, 0, nLabel);
                    labels = torch.tensor(row);
                }
            }
        }
        return new Sample(index, feats, stacked, labels);
    }

    @Override
    public Iterator<Sample> iterator() {
        return new Iterator<>() {
            int i = 0;
            @Override public boolean hasNext() { return i < size(); }
            @Override public Sample next() {
                if (!hasNext()) throw new NoSuchElementException();
                return get(i++);
            }
        };
    }

    /** All scalar features as {@code [n, n_feat]} float tensor. */
    public Tensor featuresTensor() {
        int n = size();
        if (nScalarFeat == 0) return torch.empty(new long[]{n, 0});
        Tensor t = torch.tensor(packedScalars);
        return t.reshape(new long[]{n, nScalarFeat});
    }

    /** Named sequence feature as {@code [n, dim]} tensor. */
    public Tensor sequenceTensor(String name) {
        PackedColumn pc = packedSequences.get(name);
        if (pc == null) throw new IllegalArgumentException("No sequence feature: " + name);
        int n = size();
        if (pc.isLong) {
            Tensor t = torch.tensor(pc.longData);
            return t.reshape(new long[]{n, pc.dim});
        }
        Tensor t = torch.tensor(pc.floatData);
        return t.reshape(new long[]{n, pc.dim});
    }

    /** Labels as {@code [n]} or {@code [n, n_label]} tensor. */
    public Tensor labelsTensor() {
        int n = size();
        if (nLabel == 0) return torch.empty(new long[]{n});
        if (options.labelsAsLong) {
            Tensor t = torch.tensor(packedLabelsLong);
            return nLabel == 1 ? t.reshape(new long[]{n}) : t.reshape(new long[]{n, nLabel});
        }
        Tensor t = torch.tensor(packedLabelsFloat);
        return nLabel == 1 ? t.reshape(new long[]{n}) : t.reshape(new long[]{n, nLabel});
    }

    /**
     * Convert to native {@link TensorDataset}.
     * Uses stacked scalar features when present; otherwise first sequence feature.
     * Labels are not stored in TensorDataset (single-tensor dataset) — use
     * {@link #toTensorDatasetWithLabels()} for a TensorVector of [features, labels].
     */
    public TensorDataset toTensorDataset() {
        Tensor data = nScalarFeat > 0 ? featuresTensor()
            : (!packedSequences.isEmpty()
                ? sequenceTensor(packedSequences.keySet().iterator().next())
                : torch.empty(new long[]{size(), 0}));
        return new TensorDataset(data);
    }

    /**
     * Native {@link TensorDataset} of features with labels concatenated on the
     * last dimension as {@code [N, F+L]} (float). The JavaCPP
     * {@code TensorDataset(TensorVector)} binding stacks equal-shaped tensors
     * and cannot hold mismatched {@code [N,F]} + {@code [N]} pairs, so we
     * {@code cat} instead. Prefer {@link #dataloader()} when you need separate
     * feature / label / sequence batches.
     */
    public TensorDataset toTensorDatasetWithLabels() {
        Tensor x = nScalarFeat > 0 ? featuresTensor()
            : (!packedSequences.isEmpty()
                ? sequenceTensor(packedSequences.keySet().iterator().next())
                    .to(torch.ScalarType.Float)
                : torch.empty(new long[]{size(), 0}));
        if (nLabel == 0) return new TensorDataset(x);
        Tensor y = labelsTensor();
        // ensure 2-D [N, L] float for cat with x [N, F]
        if (y.dim() == 1) y = y.unsqueeze(1);
        y = y.to(torch.ScalarType.Float);
        x = x.to(torch.ScalarType.Float);
        Tensor combined = torch.cat(new TensorVector(x, y), 1);
        return new TensorDataset(combined);
    }

    /**
     * Features tensor and labels tensor as a pair (not stacked).
     * Use this when you need separate X/y for native training loops.
     */
    public Tensor[] toFeatureLabelTensors() {
        Tensor x = nScalarFeat > 0 ? featuresTensor()
            : (!packedSequences.isEmpty()
                ? sequenceTensor(packedSequences.keySet().iterator().next())
                : torch.empty(new long[]{size(), 0}));
        Tensor y = nLabel > 0 ? labelsTensor() : torch.empty(new long[]{size()});
        return new Tensor[]{x, y};
    }

    /** Fluent dataloader builder. */
    public DataFrameDataLoader.Builder dataloader() {
        return DataFrameDataLoader.builder(this);
    }

    public DataFrameDataLoader dataloader(int batchSize) {
        return DataFrameDataLoader.builder(this).batchSize(batchSize).build();
    }

    /** Convenience: feature/label split then dataset. */
    public static DataFrameDataset fromFeatureLabel(DataFrameOps.FeatureLabelSplit fl) {
        return of(fl.X, fl.y);
    }

    // ---- efficient batch gather (used by DataFrameDataLoader) ----

    /** Gather scalar features for row indices → float[B * nFeat]. */
    float[] gatherScalars(int[] idx) {
        if (nScalarFeat == 0 || packedScalars == null) return new float[0];
        float[] out = new float[idx.length * nScalarFeat];
        for (int b = 0; b < idx.length; b++) {
            System.arraycopy(packedScalars, idx[b] * nScalarFeat, out, b * nScalarFeat, nScalarFeat);
        }
        return out;
    }

    /** Number of packed scalar feature columns. */
    public int scalarFeatureCount() { return nScalarFeat; }

    /** Gather sequence feature → primitive array (long[] or float[]). */
    Object gatherSequence(String name, int[] idx) {
        PackedColumn pc = packedSequences.get(name);
        if (pc == null) throw new IllegalArgumentException("No sequence feature: " + name);
        if (pc.isLong) {
            long[] out = new long[idx.length * pc.dim];
            for (int b = 0; b < idx.length; b++) {
                System.arraycopy(pc.longData, idx[b] * pc.dim, out, b * pc.dim, pc.dim);
            }
            return out;
        }
        float[] out = new float[idx.length * pc.dim];
        for (int b = 0; b < idx.length; b++) {
            System.arraycopy(pc.floatData, idx[b] * pc.dim, out, b * pc.dim, pc.dim);
        }
        return out;
    }

    int sequenceDim(String name) {
        PackedColumn pc = packedSequences.get(name);
        if (pc == null) throw new IllegalArgumentException("No sequence feature: " + name);
        return pc.dim;
    }

    boolean sequenceIsLong(String name) {
        PackedColumn pc = packedSequences.get(name);
        if (pc == null) throw new IllegalArgumentException("No sequence feature: " + name);
        return pc.isLong;
    }

    Object gatherLabels(int[] idx) {
        if (nLabel == 0) return null;
        if (options.labelsAsLong) {
            long[] out = new long[idx.length * nLabel];
            for (int b = 0; b < idx.length; b++) {
                System.arraycopy(packedLabelsLong, idx[b] * nLabel, out, b * nLabel, nLabel);
            }
            return out;
        }
        float[] out = new float[idx.length * nLabel];
        for (int b = 0; b < idx.length; b++) {
            System.arraycopy(packedLabelsFloat, idx[b] * nLabel, out, b * nLabel, nLabel);
        }
        return out;
    }

    public int labelCount() { return nLabel; }
    public boolean labelsAsLong() { return options.labelsAsLong; }

    // ---- native Dataset / DataLoader interop --------------------------------

    /**
     * Adapter that <em>extends</em> virtualized native {@link org.bytedeco.pytorch.data.Dataset}.
     * Use with {@link org.bytedeco.pytorch.data.dataloader.RandomDataLoader} /
     * {@link org.bytedeco.pytorch.data.dataloader.SequentialDataLoader}.
     */
    public DataFrameNativeDataset asDataset() {
        return asDataset(NativeViewOptions.defaults());
    }

    public DataFrameNativeDataset asDataset(NativeViewOptions viewOpts) {
        return new DataFrameNativeDataset(this, viewOpts);
    }

    /** Alias of {@link #asDataset()}. */
    public DataFrameNativeDataset toNativeDataset() {
        return asDataset();
    }

    public DataFrameNativeDataset toNativeDataset(NativeViewOptions viewOpts) {
        return asDataset(viewOpts);
    }

    /**
     * Features-only adapter that <em>extends</em>
     * {@link org.bytedeco.pytorch.data.datasets.JavaTensorDataset}.
     */
    public DataFrameJavaTensorDataset asJavaTensorDataset() {
        return asJavaTensorDataset(NativeViewOptions.defaults());
    }

    public DataFrameJavaTensorDataset asJavaTensorDataset(NativeViewOptions viewOpts) {
        return new DataFrameJavaTensorDataset(this, viewOpts);
    }

    /** Fluent builder for native Random/Sequential DataLoader over {@link #asDataset()}. */
    public NativeDataLoaderBuilder nativeDataLoader() {
        return asDataset().nativeDataLoader();
    }

    public NativeDataLoaderBuilder nativeDataLoader(NativeViewOptions viewOpts) {
        return asDataset(viewOpts).nativeDataLoader();
    }

    /** Fluent builder for native tensor (NoTarget) loaders. */
    public NativeTensorDataLoaderBuilder nativeTensorDataLoader() {
        return asJavaTensorDataset().nativeTensorDataLoader();
    }

    public NativeTensorDataLoaderBuilder nativeTensorDataLoader(NativeViewOptions viewOpts) {
        return asJavaTensorDataset(viewOpts).nativeTensorDataLoader();
    }

    // ---- reverse conversion: tensors / TensorDataset → DataFrameDataset -----

    /**
     * Build a dataset from feature matrix {@code X [N,F]} and label tensor
     * {@code y [N]} or {@code [N,L]}.
     *
     * @param x       feature tensor (rank 1 or 2)
     * @param y       label tensor (may be null)
     * @param xNames  optional feature column names (length F); defaults to {@code f0..}
     * @param yNames  optional label column names; defaults to {@code label} / {@code y0..}
     */
    public static DataFrameDataset fromFeatureLabelTensors(
            Tensor x, Tensor y, String[] xNames, String[] yNames) {
        Objects.requireNonNull(x, "x");
        DataFrame X = tensorToFeatureFrame(x, xNames);
        DataFrame Y;
        if (y == null || y.numel() == 0) {
            Y = DataFrame.create();
        } else {
            Y = tensorToFeatureFrame(y, yNames != null && yNames.length > 0
                ? yNames
                : defaultLabelNames(y));
        }
        Options opts = new Options();
        if (y != null && y.numel() > 0) {
            // preserve integer labels when source is long/int
            opts.labelsAsLong(isIntegralScalarType(y.scalar_type()));
        }
        return of(X, Y, opts);
    }

    public static DataFrameDataset fromFeatureLabelTensors(Tensor x, Tensor y) {
        return fromFeatureLabelTensors(x, y, null, null);
    }

    /**
     * Features-only dataset from a native {@link TensorDataset}'s stored tensor.
     *
     * @param td      source TensorDataset
     * @param dataCol optional base name; for 2-D tensors columns are {@code dataCol_i}
     *                or {@code f0..} when null
     */
    public static DataFrameDataset fromTensorDataset(TensorDataset td, String dataCol) {
        Objects.requireNonNull(td, "td");
        Tensor t = td.tensor();
        String[] names = null;
        if (dataCol != null && !dataCol.isEmpty() && t.dim() == 2) {
            int f = (int) t.size(1);
            names = new String[f];
            for (int i = 0; i < f; i++) names[i] = dataCol + "_" + i;
        } else if (dataCol != null && !dataCol.isEmpty() && t.dim() <= 1) {
            names = new String[]{dataCol};
        }
        return fromFeatureLabelTensors(t, null, names, null);
    }

    public static DataFrameDataset fromTensorDataset(TensorDataset td) {
        return fromTensorDataset(td, null);
    }

    /**
     * Materialize any native {@link org.bytedeco.pytorch.data.Dataset} by calling
     * {@code get(i)} for all indices, stacking into X/y, then
     * {@link #fromFeatureLabelTensors}.
     */
    public static DataFrameDataset fromNativeDataset(org.bytedeco.pytorch.data.Dataset nativeDs,
                                                     String[] xNames, String[] yNames) {
        Objects.requireNonNull(nativeDs, "nativeDs");
        SizeTOptional sizeOpt = nativeDs.size();
        if (sizeOpt == null || !sizeOpt.has_value()) {
            throw new IllegalArgumentException("native dataset has no size()");
        }
        long n = sizeOpt.get();
        if (n <= 0) {
            return of(DataFrame.create(), DataFrame.create());
        }
        TensorVector dataTv = new TensorVector();
        TensorVector targetTv = new TensorVector();
        boolean hasTarget = false;
        for (long i = 0; i < n; i++) {
            Example ex = nativeDs.get(i);
            dataTv.push_back(ex.data());
            Tensor tgt = ex.target();
            if (tgt != null && tgt.numel() > 0) {
                hasTarget = true;
                targetTv.push_back(tgt);
            } else {
                targetTv.push_back(torch.empty(new long[]{0}));
            }
        }
        Tensor x = torch.stack(dataTv, 0);
        Tensor y = hasTarget ? torch.stack(targetTv, 0) : null;
        // squeeze trailing singleton label dim: [N,1] → keep 2-D for col names
        return fromFeatureLabelTensors(x, y, xNames, yNames);
    }

    public static DataFrameDataset fromNativeDataset(org.bytedeco.pytorch.data.Dataset nativeDs) {
        return fromNativeDataset(nativeDs, null, null);
    }

    private static String[] defaultLabelNames(Tensor y) {
        if (y.dim() <= 1) return new String[]{"label"};
        int L = (int) y.size(y.dim() - 1);
        if (L <= 1) return new String[]{"label"};
        String[] names = new String[L];
        for (int i = 0; i < L; i++) names[i] = "y" + i;
        return names;
    }

    /** Convert rank-1/2 numeric tensor into a DataFrame of scalar columns. */
    private static DataFrame tensorToFeatureFrame(Tensor t, String[] names) {
        Tensor cpu = t.contiguous().cpu();
        long[] shape = cpu.shape();
        int rank = shape.length;
        if (rank == 0) {
            DataFrame df = DataFrame.create();
            String name = names != null && names.length > 0 ? names[0] : "f0";
            df.addColumn(name, Column.DType.FLOAT32);
            int ri = df.addRow();
            df.set(ri, name, cpu.item_float());
            return df;
        }
        if (rank == 1) {
            int n = (int) shape[0];
            String name = names != null && names.length > 0 ? names[0] : "f0";
            boolean asLong = isIntegralScalarType(cpu.scalar_type());
            DataFrame df = DataFrame.create();
            df.addColumn(name, asLong ? Column.DType.INT64 : Column.DType.FLOAT32);
            if (asLong) {
                Tensor l = cpu.to(torch.ScalarType.Long);
                LongPointer ptr = l.data_ptr_long();
                for (int i = 0; i < n; i++) {
                    int ri = df.addRow();
                    df.set(ri, name, ptr.get(i));
                }
            } else {
                Tensor f = cpu.to(torch.ScalarType.Float);
                FloatPointer ptr = f.data_ptr_float();
                for (int i = 0; i < n; i++) {
                    int ri = df.addRow();
                    df.set(ri, name, ptr.get(i));
                }
            }
            return df;
        }
        // rank >= 2: use leading dim as rows, flatten trailing into columns if rank>2
        int rows = (int) shape[0];
        int cols;
        if (rank == 2) {
            cols = (int) shape[1];
        } else {
            long c = 1;
            for (int i = 1; i < rank; i++) c *= shape[i];
            cols = (int) c;
            cpu = cpu.reshape(new long[]{rows, cols});
        }
        String[] colNames = names;
        if (colNames == null || colNames.length < cols) {
            colNames = new String[cols];
            for (int i = 0; i < cols; i++) {
                colNames[i] = (names != null && i < names.length && names[i] != null)
                    ? names[i] : ("f" + i);
            }
        }
        boolean asLong = isIntegralScalarType(cpu.scalar_type());
        DataFrame df = DataFrame.create();
        for (int c = 0; c < cols; c++) {
            df.addColumn(colNames[c], asLong ? Column.DType.INT64 : Column.DType.FLOAT32);
        }
        if (asLong) {
            Tensor l = cpu.to(torch.ScalarType.Long).reshape(new long[]{-1});
            LongPointer ptr = l.data_ptr_long();
            for (int r = 0; r < rows; r++) {
                int ri = df.addRow();
                for (int c = 0; c < cols; c++) {
                    df.set(ri, colNames[c], ptr.get((long) r * cols + c));
                }
            }
        } else {
            Tensor f = cpu.to(torch.ScalarType.Float).reshape(new long[]{-1});
            FloatPointer ptr = f.data_ptr_float();
            for (int r = 0; r < rows; r++) {
                int ri = df.addRow();
                for (int c = 0; c < cols; c++) {
                    df.set(ri, colNames[c], ptr.get((long) r * cols + c));
                }
            }
        }
        return df;
    }

    /** Always {@code intern()} ScalarType before equality — see project memory. */
    private static boolean isIntegralScalarType(torch.ScalarType st) {
        if (st == null) return false;
        torch.ScalarType s = st.intern();
        return s == torch.ScalarType.Long
            || s == torch.ScalarType.Int
            || s == torch.ScalarType.Short
            || s == torch.ScalarType.Byte
            || s == torch.ScalarType.Char;
    }
}
