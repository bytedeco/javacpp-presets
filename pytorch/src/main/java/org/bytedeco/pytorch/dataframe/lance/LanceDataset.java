package org.bytedeco.pytorch.dataframe.lance;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.ann.AnnSearchResult;
import org.bytedeco.pytorch.dataframe.ann.Distance;
import org.bytedeco.pytorch.dataframe.ann.HnswIndex;
import org.bytedeco.pytorch.dataframe.ann.VectorColumn;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.ai.EmbeddingMath;

/**
 * Pure-Java <b>training</b> vector dataset layout for multimodal DataFrames with
 * first-class embedding columns + optional local HNSW.
 *
 * <p><b>Not</b> byte-compatible with lance-rs / official {@code org.lance:lance-core}.
 * Prefer {@link DataFrame#writeLance(String)} for
 * official Lance, or {@code writeLanceTraining}/{@code readLanceTraining} for this layout.
 *
 * <p>Original: Lance-inspired vector dataset: versioned directory layout for multimodal
 * DataFrames with first-class embedding columns + optional HNSW index.
 *
 * <p>Layout:
 * <pre>
 *   my.lance/
 *     _manifest.json          # schema, versions, vector column meta
 *     data/
 *       fragment-0.bin        # row-oriented cell payload (Java serialization of row maps)
 *     vectors/
 *       &lt;col&gt;.f32             # raw little-endian float32 [n * dim]
 *       &lt;col&gt;.meta.json       # dim, n, metric, model
 *     indices/
 *       &lt;col&gt;.hnsw            # serialized HnswIndex (optional)
 * </pre>
 *
 * <p>Not a byte-compatible port of lance-rs; API mirrors Daft
 * {@code read_lance}/{@code write_lance} for training pipelines.
 *
 * <pre>
 *   df.writeLance("clips.lance", "image_emb");
 *   DataFrame back = DataFrame.readLance("clips.lance");
 *   LanceDataset ds = LanceDataset.open("clips.lance");
 *   AnnSearchResult top = ds.search("image_emb", query, 10);
 * </pre>
 */
public final class LanceDataset implements Closeable {
    public static final String MANIFEST = "_manifest.json";
    public static final String DATA_DIR = "data";
    public static final String VECTORS_DIR = "vectors";
    public static final String INDICES_DIR = "indices";

    private final Path root;
    private final Manifest manifest;
    private final Map<String, float[]> vectorCache = new HashMap<>();
    private final Map<String, HnswIndex> indexCache = new HashMap<>();

    private LanceDataset(Path root, Manifest manifest) {
        this.root = root;
        this.manifest = manifest;
    }

    public Path root() { return root; }
    public Manifest manifest() { return manifest; }
    public int rowCount() { return manifest.rowCount; }
    public List<String> vectorColumns() { return Collections.unmodifiableList(manifest.vectorColumns); }

    // ── open / create ──────────────────────────────────────────────────

    public static LanceDataset open(String path) throws IOException {
        return open(Path.of(path));
    }

    public static LanceDataset open(Path path) throws IOException {
        Path manifestPath = path.resolve(MANIFEST);
        if (!Files.isRegularFile(manifestPath)) {
            throw new FileNotFoundException("Not a lance dataset (missing _manifest.json): " + path);
        }
        Manifest m = Manifest.read(manifestPath);
        return new LanceDataset(path, m);
    }

    public static boolean isLanceDataset(String path) {
        return Files.isRegularFile(Path.of(path).resolve(MANIFEST));
    }

    // ── write ──────────────────────────────────────────────────────────

    /**
     * Write a DataFrame as a lance dataset. Embedding / VECTOR / float[] columns
     * listed in {@code vectorCols} (or auto-detected) are stored as dense f32
     * matrices; remaining columns go to the fragment payload.
     */
    public static LanceDataset write(DataFrame df, String path, String... vectorCols) throws Exception {
        return write(df, Path.of(path), WriteOptions.defaults(), vectorCols);
    }

    public static LanceDataset write(DataFrame df, Path path, WriteOptions opts, String... vectorCols) throws Exception {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(path, "path");
        if (opts == null) opts = WriteOptions.defaults();

        Files.createDirectories(path);
        Files.createDirectories(path.resolve(DATA_DIR));
        Files.createDirectories(path.resolve(VECTORS_DIR));
        Files.createDirectories(path.resolve(INDICES_DIR));

        // resolve vector columns
        List<String> vcols = new ArrayList<>();
        if (vectorCols != null && vectorCols.length > 0) {
            for (String c : vectorCols) if (df.hasColumn(c)) vcols.add(c);
        } else {
            for (Column c : df.columns()) {
                if (c.dtype() == Column.DType.EMBEDDING
                    || c.dtype() == Column.DType.VECTOR
                    || c.dtype() == Column.DType.TENSOR) {
                    vcols.add(c.name());
                } else if (looksLikeVector(c)) {
                    vcols.add(c.name());
                }
            }
        }

        Manifest manifest = new Manifest();
        manifest.version = 1;
        manifest.rowCount = df.rowCount();
        manifest.createdAt = System.currentTimeMillis();
        manifest.vectorColumns = new ArrayList<>(vcols);
        manifest.columns = new ArrayList<>();
        for (Column c : df.columns()) {
            ColMeta cm = new ColMeta();
            cm.name = c.name();
            cm.dtype = c.dtype().name();
            cm.vector = vcols.contains(c.name());
            manifest.columns.add(cm);
        }

        // write vector matrices
        for (String vc : vcols) {
            VectorMatrix vm = packVectors(df.column(vc));
            Path f32 = path.resolve(VECTORS_DIR).resolve(vc + ".f32");
            writeF32Matrix(f32, vm.data, vm.n, vm.dim);

            VectorMeta meta = new VectorMeta();
            meta.name = vc;
            meta.n = vm.n;
            meta.dim = vm.dim;
            meta.metric = opts.metric == null ? "cosine" : opts.metric;
            meta.model = vm.modelName;
            writeJson(path.resolve(VECTORS_DIR).resolve(vc + ".meta.json"), meta.toJson());

            ColMeta cm = manifest.find(vc);
            if (cm != null) {
                cm.dim = vm.dim;
                cm.model = vm.modelName;
                cm.metric = meta.metric;
            }

            if (opts.buildIndex && vm.n > 0 && vm.dim > 0) {
                Distance space = "cosine".equalsIgnoreCase(meta.metric) || "ip".equalsIgnoreCase(meta.metric)
                    ? Distance.IP : Distance.L2;
                float[] matrix = vm.data;
                if ("cosine".equalsIgnoreCase(meta.metric)) {
                    matrix = Arrays.copyOf(vm.data, vm.data.length);
                    VectorColumn.l2Normalize(matrix, vm.n, vm.dim);
                }
                HnswIndex idx = HnswIndex.builder(vm.dim)
                    .M(opts.hnswM)
                    .efConstruction(opts.hnswEfConstruction)
                    .space(space)
                    .normalize("cosine".equalsIgnoreCase(meta.metric))
                    .vectors(matrix, vm.n)
                    .build();
                idx.save(path.resolve(INDICES_DIR).resolve(vc + ".hnsw"));
                if (cm != null) cm.hasIndex = true;
            }
        }

        // write row payload (non-vector + metadata); vectors stored by row index reference
        writeFragment(path.resolve(DATA_DIR).resolve("fragment-0.bin"), df, vcols);

        manifest.fragmentCount = 1;
        writeJson(path.resolve(MANIFEST), manifest.toJson());
        return new LanceDataset(path, manifest);
    }

    // ── read ───────────────────────────────────────────────────────────

    /** Load full DataFrame (payload + vector columns reconstituted as EMBEDDING). */
    public DataFrame toDataFrame() throws Exception {
        DataFrame df = readFragment(root.resolve(DATA_DIR).resolve("fragment-0.bin"));
        for (String vc : manifest.vectorColumns) {
            VectorMatrix vm = loadVectors(vc);
            if (vm == null) continue;
            List<Object> cells = new ArrayList<>(vm.n);
            String model = vm.modelName == null ? "lance" : vm.modelName;
            for (int i = 0; i < vm.n; i++) {
                float[] row = new float[vm.dim];
                System.arraycopy(vm.data, i * vm.dim, row, 0, vm.dim);
                cells.add(new EmbeddingData(row, model));
            }
            // pad / trim to df rows
            while (cells.size() < df.rowCount()) cells.add(null);
            if (df.hasColumn(vc)) {
                // replace
                df = df.drop(vc).withColumn(vc, cells.subList(0, df.rowCount()));
            } else {
                df = df.withColumn(vc, cells.subList(0, Math.min(cells.size(), df.rowCount())));
            }
        }
        return df;
    }

    public static DataFrame read(String path) throws Exception {
        return open(path).toDataFrame();
    }

    public static DataFrame read(Path path) throws Exception {
        return open(path).toDataFrame();
    }

    // ── vector access / search ─────────────────────────────────────────

    public float[][] getVectors(String column) throws IOException {
        VectorMatrix vm = loadVectors(column);
        if (vm == null) return new float[0][];
        float[][] out = new float[vm.n][vm.dim];
        for (int i = 0; i < vm.n; i++) {
            System.arraycopy(vm.data, i * vm.dim, out[i], 0, vm.dim);
        }
        return out;
    }

    public int vectorDim(String column) throws IOException {
        VectorMatrix vm = loadVectors(column);
        return vm == null ? 0 : vm.dim;
    }

    /**
     * ANN search over a vector column. Uses HNSW index when present, else brute-force cosine/L2.
     */
    public AnnSearchResult search(String vectorCol, float[] query, int k) throws Exception {
        return search(vectorCol, query, k, 64);
    }

    public AnnSearchResult search(String vectorCol, float[] query, int k, int efSearch) throws Exception {
        Objects.requireNonNull(query, "query");
        HnswIndex idx = loadIndex(vectorCol);
        if (idx != null) {
            float[] q = query;
            ColMeta cm = manifest.find(vectorCol);
            if (cm != null && "cosine".equalsIgnoreCase(cm.metric)) {
                q = EmbeddingMath.l2Normalize(Arrays.copyOf(query, query.length));
            }
            return idx.search(q, k, efSearch);
        }
        // brute force
        VectorMatrix vm = loadVectors(vectorCol);
        if (vm == null || vm.n == 0) {
            return new AnnSearchResult(new int[0], new float[0], new long[0]);
        }
        String metric = Optional.ofNullable(manifest.find(vectorCol)).map(c -> c.metric).orElse("cosine");
        float[] q = "cosine".equalsIgnoreCase(metric)
            ? EmbeddingMath.l2Normalize(Arrays.copyOf(query, query.length)) : query;

        double[] scores = new double[vm.n];
        for (int i = 0; i < vm.n; i++) {
            float[] row = new float[vm.dim];
            System.arraycopy(vm.data, i * vm.dim, row, 0, vm.dim);
            if ("cosine".equalsIgnoreCase(metric) || "ip".equalsIgnoreCase(metric)) {
                scores[i] = EmbeddingMath.cosine(
                    "cosine".equalsIgnoreCase(metric) ? EmbeddingMath.l2Normalize(row) : row, q);
            } else {
                double s = 0;
                int d = Math.min(vm.dim, q.length);
                for (int j = 0; j < d; j++) {
                    double diff = row[j] - q[j];
                    s += diff * diff;
                }
                scores[i] = -s; // higher better
            }
        }
        Integer[] order = new Integer[vm.n];
        for (int i = 0; i < vm.n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(scores[b], scores[a]));
        int take = Math.min(k, vm.n);
        int[] indices = new int[take];
        long[] ids = new long[take];
        float[] dists = new float[take];
        for (int i = 0; i < take; i++) {
            indices[i] = order[i];
            ids[i] = order[i];
            dists[i] = (float) scores[order[i]];
        }
        return new AnnSearchResult(indices, dists, ids);
    }

    /**
     * Return DataFrame rows corresponding to ANN hits (id = row index).
     */
    public DataFrame searchAsDataFrame(String vectorCol, float[] query, int k) throws Exception {
        AnnSearchResult r = search(vectorCol, query, k);
        DataFrame all = toDataFrame();
        int[] rows = r.indices();
        if (rows == null || rows.length == 0) return DataFrame.create();
        DataFrame hits = all.loc(rows);
        // attach score
        List<Object> scores = new ArrayList<>(rows.length);
        float[] d = r.distances();
        for (int i = 0; i < rows.length; i++) scores.add(i < d.length ? d[i] : null);
        return hits.withColumn("_score", scores);
    }

    // ── internals ──────────────────────────────────────────────────────

    private VectorMatrix loadVectors(String column) throws IOException {
        float[] cached = vectorCache.get(column);
        Path f32 = root.resolve(VECTORS_DIR).resolve(column + ".f32");
        Path metaP = root.resolve(VECTORS_DIR).resolve(column + ".meta.json");
        if (!Files.isRegularFile(f32) || !Files.isRegularFile(metaP)) return null;
        VectorMeta meta = VectorMeta.fromJson(Files.readString(metaP));
        if (cached != null) {
            VectorMatrix vm = new VectorMatrix();
            vm.data = cached; vm.n = meta.n; vm.dim = meta.dim; vm.modelName = meta.model;
            return vm;
        }
        float[] data = readF32Matrix(f32, meta.n, meta.dim);
        vectorCache.put(column, data);
        VectorMatrix vm = new VectorMatrix();
        vm.data = data; vm.n = meta.n; vm.dim = meta.dim; vm.modelName = meta.model;
        return vm;
    }

    private HnswIndex loadIndex(String column) {
        HnswIndex cached = indexCache.get(column);
        if (cached != null) return cached;
        Path p = root.resolve(INDICES_DIR).resolve(column + ".hnsw");
        if (!Files.isRegularFile(p)) return null;
        try {
            HnswIndex idx = HnswIndex.load(p);
            indexCache.put(column, idx);
            return idx;
        } catch (Exception e) {
            return null;
        }
    }

    private static boolean looksLikeVector(Column c) {
        for (int i = 0; i < Math.min(c.size(), 8); i++) {
            Object v = c.get(i);
            if (v == null) continue;
            if (v instanceof float[] || v instanceof double[] || v instanceof EmbeddingData) return true;
            return false;
        }
        return false;
    }

    private static VectorMatrix packVectors(Column col) {
        int n = col.size();
        int dim = 0;
        String model = null;
        List<float[]> rows = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            float[] v = toFloatVec(col.get(i));
            if (v != null) {
                dim = Math.max(dim, v.length);
                if (model == null && col.get(i) instanceof EmbeddingData ed) model = ed.getModelName();
            }
            rows.add(v);
        }
        if (dim == 0) dim = 1;
        float[] data = new float[n * dim];
        for (int i = 0; i < n; i++) {
            float[] v = rows.get(i);
            if (v == null) continue;
            System.arraycopy(v, 0, data, i * dim, Math.min(dim, v.length));
        }
        VectorMatrix vm = new VectorMatrix();
        vm.data = data; vm.n = n; vm.dim = dim; vm.modelName = model;
        return vm;
    }

    private static float[] toFloatVec(Object v) {
        if (v == null) return null;
        if (v instanceof EmbeddingData ed) return ed.getVector();
        if (v instanceof float[] f) return f;
        if (v instanceof double[] d) {
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return f;
        }
        if (v instanceof List<?> list) {
            float[] f = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object o = list.get(i);
                f[i] = o instanceof Number ? ((Number) o).floatValue() : 0f;
            }
            return f;
        }
        return VectorColumn.asFloatArray(v);
    }

    private static void writeF32Matrix(Path path, float[] data, int n, int dim) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(4 + 4 + data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(n);
        buf.putInt(dim);
        for (float v : data) buf.putFloat(v);
        buf.flip();
        Files.write(path, buf.array());
    }

    private static float[] readF32Matrix(Path path, int n, int dim) throws IOException {
        byte[] raw = Files.readAllBytes(path);
        ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int rn = buf.getInt();
        int rd = buf.getInt();
        if (n <= 0) n = rn;
        if (dim <= 0) dim = rd;
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length && buf.remaining() >= 4; i++) data[i] = buf.getFloat();
        return data;
    }

    private static void writeFragment(Path path, DataFrame df, List<String> vectorCols) throws Exception {
        Set<String> skip = new HashSet<>(vectorCols);
        List<Map<String, Object>> rows = new ArrayList<>(df.rowCount());
        for (int i = 0; i < df.rowCount(); i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("_rowid", (long) i);
            for (Column c : df.columns()) {
                if (skip.contains(c.name())) continue;
                Object v = c.get(i);
                // only store serializable-friendly values
                row.put(c.name(), sanitize(v));
            }
            rows.add(row);
        }
        try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            oos.writeObject(rows);
            // also store column dtypes
            Map<String, String> dtypes = new LinkedHashMap<>();
            for (Column c : df.columns()) {
                if (!skip.contains(c.name())) dtypes.put(c.name(), c.dtype().name());
            }
            oos.writeObject(dtypes);
        }
    }

    @SuppressWarnings("unchecked")
    private static DataFrame readFragment(Path path) throws Exception {
        if (!Files.isRegularFile(path)) return DataFrame.create();
        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            List<Map<String, Object>> rows = (List<Map<String, Object>>) ois.readObject();
            Map<String, String> dtypes = (Map<String, String>) ois.readObject();
            DataFrame df = DataFrame.create();
            if (rows == null || rows.isEmpty()) {
                if (dtypes != null) {
                    for (Map.Entry<String, String> e : dtypes.entrySet()) {
                        df.addColumn(e.getKey(), parseDType(e.getValue()));
                    }
                }
                return df;
            }
            // column order from dtypes or first row
            List<String> names = new ArrayList<>();
            if (dtypes != null && !dtypes.isEmpty()) names.addAll(dtypes.keySet());
            else names.addAll(rows.get(0).keySet());
            for (String name : names) {
                Column.DType dt = dtypes != null && dtypes.containsKey(name)
                    ? parseDType(dtypes.get(name)) : Column.DType.STRING;
                // skip internal
                if ("_rowid".equals(name) && !dtypes.containsKey(name)) {
                    // still add as INT64
                    dt = Column.DType.INT64;
                }
                if (!df.hasColumn(name)) df.addColumn(name, dt);
            }
            for (Map<String, Object> row : rows) {
                int ri = df.addEmptyRow();
                for (String name : names) {
                    if (df.hasColumn(name)) df.set(ri, name, row.get(name));
                }
            }
            return df;
        }
    }

    private static Object sanitize(Object v) {
        if (v == null) return null;
        if (v instanceof String || v instanceof Number || v instanceof Boolean) return v;
        if (v instanceof EmbeddingData || v instanceof float[] || v instanceof double[]) return null; // stored elsewhere
        if (v instanceof Map || v instanceof List) return v;
        return String.valueOf(v);
    }

    private static Column.DType parseDType(String s) {
        try { return Column.DType.valueOf(s); }
        catch (Exception e) { return Column.DType.STRING; }
    }

    private static void writeJson(Path path, String json) throws IOException {
        Files.writeString(path, json, StandardCharsets.UTF_8);
    }

    @Override
    public void close() {
        vectorCache.clear();
        indexCache.clear();
    }

    // ── options / meta types ───────────────────────────────────────────

    public static final class WriteOptions {
        public String metric = "cosine"; // cosine | l2 | ip
        public boolean buildIndex = true;
        public int hnswM = 16;
        public int hnswEfConstruction = 200;

        public static WriteOptions defaults() { return new WriteOptions(); }
        public WriteOptions metric(String m) { this.metric = m; return this; }
        public WriteOptions buildIndex(boolean v) { this.buildIndex = v; return this; }
        public WriteOptions hnswM(int m) { this.hnswM = m; return this; }
        public WriteOptions hnswEfConstruction(int ef) { this.hnswEfConstruction = ef; return this; }
    }

    public static final class Manifest {
        public int version;
        public int rowCount;
        public long createdAt;
        public int fragmentCount;
        public List<String> vectorColumns = new ArrayList<>();
        public List<ColMeta> columns = new ArrayList<>();

        ColMeta find(String name) {
            for (ColMeta c : columns) if (name.equals(c.name)) return c;
            return null;
        }

        String toJson() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"version\": ").append(version).append(",\n");
            sb.append("  \"row_count\": ").append(rowCount).append(",\n");
            sb.append("  \"created_at\": ").append(createdAt).append(",\n");
            sb.append("  \"fragment_count\": ").append(fragmentCount).append(",\n");
            sb.append("  \"vector_columns\": [");
            for (int i = 0; i < vectorColumns.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append('"').append(esc(vectorColumns.get(i))).append('"');
            }
            sb.append("],\n  \"columns\": [\n");
            for (int i = 0; i < columns.size(); i++) {
                ColMeta c = columns.get(i);
                sb.append("    {\"name\":\"").append(esc(c.name)).append("\",\"dtype\":\"")
                    .append(esc(c.dtype)).append("\",\"vector\":").append(c.vector);
                if (c.dim > 0) sb.append(",\"dim\":").append(c.dim);
                if (c.model != null) sb.append(",\"model\":\"").append(esc(c.model)).append('"');
                if (c.metric != null) sb.append(",\"metric\":\"").append(esc(c.metric)).append('"');
                if (c.hasIndex) sb.append(",\"has_index\":true");
                sb.append('}');
                if (i + 1 < columns.size()) sb.append(',');
                sb.append('\n');
            }
            sb.append("  ]\n}\n");
            return sb.toString();
        }

        static Manifest read(Path path) throws IOException {
            String json = Files.readString(path);
            Manifest m = new Manifest();
            m.version = (int) extractLong(json, "version", 1);
            m.rowCount = (int) extractLong(json, "row_count", 0);
            m.createdAt = extractLong(json, "created_at", 0);
            m.fragmentCount = (int) extractLong(json, "fragment_count", 1);
            m.vectorColumns = extractStringArray(json, "vector_columns");
            m.columns = extractColumns(json);
            return m;
        }
    }

    public static final class ColMeta {
        public String name;
        public String dtype;
        public boolean vector;
        public int dim;
        public String model;
        public String metric;
        public boolean hasIndex;
    }

    public static final class VectorMeta {
        public String name;
        public int n;
        public int dim;
        public String metric;
        public String model;

        String toJson() {
            return "{\n  \"name\":\"" + esc(name) + "\",\n  \"n\":" + n
                + ",\n  \"dim\":" + dim + ",\n  \"metric\":\"" + esc(metric)
                + "\",\n  \"model\":\"" + esc(model == null ? "" : model) + "\"\n}\n";
        }

        static VectorMeta fromJson(String json) {
            VectorMeta m = new VectorMeta();
            m.name = extractString(json, "name", "");
            m.n = (int) extractLong(json, "n", 0);
            m.dim = (int) extractLong(json, "dim", 0);
            m.metric = extractString(json, "metric", "cosine");
            m.model = extractString(json, "model", "");
            return m;
        }
    }

    private static final class VectorMatrix {
        float[] data;
        int n, dim;
        String modelName;
    }

    // minimal JSON helpers (avoid extra deps)
    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static long extractLong(String json, String key, long def) {
        String pat = "\"" + key + "\"";
        int i = json.indexOf(pat);
        if (i < 0) return def;
        int c = json.indexOf(':', i + pat.length());
        if (c < 0) return def;
        int s = c + 1;
        while (s < json.length() && Character.isWhitespace(json.charAt(s))) s++;
        int e = s;
        while (e < json.length() && (Character.isDigit(json.charAt(e)) || json.charAt(e) == '-')) e++;
        try { return Long.parseLong(json.substring(s, e)); } catch (Exception ex) { return def; }
    }

    private static String extractString(String json, String key, String def) {
        String pat = "\"" + key + "\"";
        int i = json.indexOf(pat);
        if (i < 0) return def;
        int c = json.indexOf(':', i + pat.length());
        if (c < 0) return def;
        int q1 = json.indexOf('"', c + 1);
        if (q1 < 0) return def;
        int q2 = json.indexOf('"', q1 + 1);
        if (q2 < 0) return def;
        return json.substring(q1 + 1, q2);
    }

    private static List<String> extractStringArray(String json, String key) {
        List<String> out = new ArrayList<>();
        String pat = "\"" + key + "\"";
        int i = json.indexOf(pat);
        if (i < 0) return out;
        int b = json.indexOf('[', i);
        int e = json.indexOf(']', b);
        if (b < 0 || e < 0) return out;
        String body = json.substring(b + 1, e);
        int q = 0;
        while (q < body.length()) {
            int q1 = body.indexOf('"', q);
            if (q1 < 0) break;
            int q2 = body.indexOf('"', q1 + 1);
            if (q2 < 0) break;
            out.add(body.substring(q1 + 1, q2));
            q = q2 + 1;
        }
        return out;
    }

    private static List<ColMeta> extractColumns(String json) {
        List<ColMeta> out = new ArrayList<>();
        int idx = json.indexOf("\"columns\"");
        if (idx < 0) return out;
        int b = json.indexOf('[', idx);
        int e = json.lastIndexOf(']');
        if (b < 0 || e < b) return out;
        String body = json.substring(b + 1, e);
        int pos = 0;
        while (pos < body.length()) {
            int o1 = body.indexOf('{', pos);
            if (o1 < 0) break;
            int o2 = body.indexOf('}', o1);
            if (o2 < 0) break;
            String obj = body.substring(o1, o2 + 1);
            ColMeta cm = new ColMeta();
            cm.name = extractString(obj, "name", "");
            cm.dtype = extractString(obj, "dtype", "STRING");
            cm.vector = obj.contains("\"vector\":true") || obj.contains("\"vector\": true");
            cm.dim = (int) extractLong(obj, "dim", 0);
            cm.model = extractString(obj, "model", null);
            cm.metric = extractString(obj, "metric", null);
            cm.hasIndex = obj.contains("\"has_index\":true") || obj.contains("\"has_index\": true");
            if (!cm.name.isEmpty()) out.add(cm);
            pos = o2 + 1;
        }
        return out;
    }
}
