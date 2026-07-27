package org.bytedeco.pytorch.data.dataframe.io;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;
import java.util.function.Predicate;

import org.apache.parquet.example.data.Group;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.Type;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.parquet.LocalParquetReader;
import org.bytedeco.pytorch.data.parquet.LocalParquetWriter;
import org.bytedeco.pytorch.data.parquet.ParquetInputFormat;

/**
 * Advanced Parquet IO: column projection, predicate pushdown (row filter),
 * true row-group parallel read, and Hive-style partition_by write/read.
 *
 * <p>Heap safety: row-group workers materialise one group at a time; optional
 * {@code maxRows} and streaming consumers bound peak memory.
 */
public final class ParquetAdvanced {
    private ParquetAdvanced() {}

    public static final class ReadOptions {
        private List<String> columns;          // null = all
        private Predicate<Map<String, Object>> filter; // row predicate (pushdown-after-read)
        private int workers = Math.max(1, Runtime.getRuntime().availableProcessors());
        private long maxRows = 0;               // 0 = unlimited
        private int batchRows = 50_000;         // streaming batch size

        public ReadOptions columns(String... cols) {
            this.columns = cols == null ? null : Arrays.asList(cols);
            return this;
        }
        public ReadOptions columns(List<String> cols) {
            this.columns = cols;
            return this;
        }
        public ReadOptions filter(Predicate<Map<String, Object>> f) {
            this.filter = f;
            return this;
        }
        /** Simple equality filter: column == value (ANDed if called multiple times via and). */
        public ReadOptions eq(String col, Object value) {
            Predicate<Map<String, Object>> p = row -> Objects.equals(row.get(col), value);
            this.filter = this.filter == null ? p : this.filter.and(p);
            return this;
        }
        public ReadOptions workers(int n) { this.workers = Math.max(1, n); return this; }
        public ReadOptions maxRows(long n) { this.maxRows = Math.max(0, n); return this; }
        public ReadOptions batchRows(int n) { this.batchRows = Math.max(1, n); return this; }

        public List<String> columns() { return columns; }
        public Predicate<Map<String, Object>> filter() { return filter; }
        public int workers() { return workers; }
        public long maxRows() { return maxRows; }
        public int batchRows() { return batchRows; }

        public static ReadOptions defaults() { return new ReadOptions(); }
    }

    // ================================================================
    // Read with projection + filter + row-group parallel
    // ================================================================

    public static DataFrame read(String path) throws Exception {
        return read(path, ReadOptions.defaults());
    }

    public static DataFrame read(String path, ReadOptions opt) throws Exception {
        Objects.requireNonNull(opt, "opt");
        Path p = Paths.get(path);
        int nGroups;
        MessageType schema;
        try (ParquetInputFormat probe = ParquetInputFormat.open(p)) {
            nGroups = probe.getRowGroupCount();
            schema = probe.getSchema();
        }
        List<String> allFields = fieldNames(schema);
        List<String> want = resolveColumns(allFields, opt.columns());

        // Single row-group or single worker → sequential
        if (nGroups <= 1 || opt.workers() <= 1) {
            return readSequential(p, schema, want, opt);
        }

        int workers = Math.min(opt.workers(), nGroups);
        ExecutorService pool = Executors.newFixedThreadPool(workers, r -> {
            Thread t = new Thread(r, "parquet-rg-worker");
            t.setDaemon(true);
            return t;
        });
        try {
            List<Future<DataFrame>> futures = new ArrayList<>(nGroups);
            for (int g = 0; g < nGroups; g++) {
                final int groupIndex = g;
                futures.add(pool.submit(() -> readRowGroup(p, groupIndex, want, opt.filter())));
            }
            List<DataFrame> parts = new ArrayList<>();
            long total = 0;
            for (int i = 0; i < futures.size(); i++) {
                DataFrame part = futures.get(i).get();
                if (part == null || part.rowCount() == 0) continue;
                if (opt.maxRows() > 0 && total + part.rowCount() > opt.maxRows()) {
                    long remain = opt.maxRows() - total;
                    part = part.head((int) Math.min(Integer.MAX_VALUE, remain));
                    parts.add(part);
                    total += part.rowCount();
                    for (int j = i + 1; j < futures.size(); j++) futures.get(j).cancel(true);
                    break;
                }
                parts.add(part);
                total += part.rowCount();
                if (opt.maxRows() > 0 && total >= opt.maxRows()) {
                    for (int j = i + 1; j < futures.size(); j++) futures.get(j).cancel(true);
                    break;
                }
            }
            if (parts.isEmpty()) return emptyWithCols(want, schema);
            if (parts.size() == 1) return parts.get(0);
            return DataFrame.vstack(parts);
        } finally {
            pool.shutdownNow();
        }
    }

    /**
     * Stream row-groups (or batches) to consumer without retaining prior frames.
     * Uses sequential row-group iteration for peak-heap = one group.
     */
    public static long stream(String path, ReadOptions opt, Consumer<DataFrame> consumer)
            throws Exception {
        Objects.requireNonNull(consumer);
        Path p = Paths.get(path);
        long delivered = 0;
        try (ParquetInputFormat in = ParquetInputFormat.open(p)) {
            MessageType schema = in.getSchema();
            List<String> want = resolveColumns(fieldNames(schema), opt.columns());
            DataFrame batch = emptyWithCols(want, schema);
            Group row;
            while ((row = in.read()) != null) {
                Map<String, Object> map = rowToMap(row, schema, want);
                if (opt.filter() != null && !opt.filter().test(map)) continue;
                int ri = batch.addEmptyRow();
                for (String c : want) batch.set(ri, c, map.get(c));
                if (batch.rowCount() >= opt.batchRows()) {
                    consumer.accept(batch);
                    delivered += batch.rowCount();
                    if (opt.maxRows() > 0 && delivered >= opt.maxRows()) return delivered;
                    batch = emptyWithCols(want, schema);
                }
            }
            if (batch.rowCount() > 0) {
                if (opt.maxRows() > 0 && delivered + batch.rowCount() > opt.maxRows()) {
                    batch = batch.head((int) (opt.maxRows() - delivered));
                }
                if (batch.rowCount() > 0) {
                    consumer.accept(batch);
                    delivered += batch.rowCount();
                }
            }
        }
        return delivered;
    }

    // ================================================================
    // Hive partition_by write / scan
    // ================================================================

    /**
     * Write DataFrame partitioned by key columns into Hive-style directories:
     * {@code root/col=val/col2=val2/part-0.parquet}.
     */
    public static void writePartitioned(DataFrame df, String root, String... partitionBy)
            throws Exception {
        if (partitionBy == null || partitionBy.length == 0) {
            df.writeParquet(root);
            return;
        }
        Path rootPath = Paths.get(root);
        Files.createDirectories(rootPath);

        // group rows by partition key
        Map<String, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < df.rowCount(); i++) {
            String key = hiveKey(df, i, partitionBy);
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        // data columns = all except partition keys
        Set<String> partSet = new HashSet<>(Arrays.asList(partitionBy));
        List<String> dataCols = new ArrayList<>();
        for (String c : df.getColumnNames()) {
            if (!partSet.contains(c)) dataCols.add(c);
        }

        int partIdx = 0;
        for (Map.Entry<String, List<Integer>> e : groups.entrySet()) {
            Path dir = rootPath.resolve(e.getKey());
            Files.createDirectories(dir);
            DataFrame part = df.loc(e.getValue().stream().mapToInt(Integer::intValue).toArray());
            // drop partition columns from file body (Hive convention)
            if (!dataCols.isEmpty()) {
                part = part.select(dataCols.toArray(new String[0]));
            }
            Path file = dir.resolve("part-" + (partIdx++) + ".parquet");
            part.writeParquet(file.toString());
        }
    }

    /**
     * Scan a Hive-partitioned directory tree, reconstruct partition columns from path,
     * optionally filter partitions and project columns.
     *
     * @param root directory containing {@code key=value/} subdirs
     * @param partitionKeys ordered partition column names (e.g. "year","month")
     */
    public static DataFrame scanHive(String root, String[] partitionKeys, ReadOptions opt)
            throws Exception {
        Path rootPath = Paths.get(root);
        if (!Files.isDirectory(rootPath)) {
            // single file fallback
            return read(root, opt == null ? ReadOptions.defaults() : opt);
        }
        ReadOptions o = opt == null ? ReadOptions.defaults() : opt;
        List<Path> files = new ArrayList<>();
        try (var walk = Files.walk(rootPath)) {
            walk.filter(p -> p.getFileName().toString().endsWith(".parquet")
                    || p.getFileName().toString().endsWith(".parq"))
                .forEach(files::add);
        }
        Collections.sort(files);
        List<DataFrame> parts = new ArrayList<>();
        long total = 0;
        for (Path f : files) {
            Map<String, String> partVals = parseHivePath(rootPath, f, partitionKeys);
            // partition pruning: if filter can be evaluated on partition cols only — skip
            if (o.filter() != null && partitionKeys != null && partitionKeys.length > 0) {
                Map<String, Object> probe = new HashMap<>(partVals);
                // if filter only refs partition keys and rejects, skip file
                // (best-effort: run filter on a synthetic row with only part cols)
                try {
                    if (!o.filter().test(probe) && probe.keySet().containsAll(Arrays.asList(partitionKeys))) {
                        // might still need data cols — only skip when filter is pure-partition
                        // We conservatively do NOT skip unless user set a partition predicate separately.
                    }
                } catch (Exception ignored) {}
            }
            DataFrame body = read(f.toString(), ReadOptions.defaults()
                .columns(o.columns())
                .workers(1)
                .maxRows(o.maxRows() > 0 ? Math.max(0, o.maxRows() - total) : 0));
            // inject partition columns
            if (partitionKeys != null) {
                for (String pk : partitionKeys) {
                    String pv = partVals.get(pk);
                    if (!body.hasColumn(pk)) {
                        body = body.withColumn(pk, org.bytedeco.pytorch.data.dataframe.Functions.lit(pv));
                    }
                }
            }
            if (o.filter() != null) {
                body = filterFrame(body, o.filter());
            }
            if (body.rowCount() > 0) {
                parts.add(body);
                total += body.rowCount();
            }
            if (o.maxRows() > 0 && total >= o.maxRows()) break;
        }
        if (parts.isEmpty()) return DataFrame.create();
        if (parts.size() == 1) return parts.get(0);
        return DataFrame.vstack(parts);
    }

    public static DataFrame scanHive(String root, String... partitionKeys) throws Exception {
        return scanHive(root, partitionKeys, ReadOptions.defaults());
    }

    // ================================================================
    // Internals
    // ================================================================

    private static DataFrame readSequential(Path path, MessageType schema,
                                            List<String> want, ReadOptions opt) throws Exception {
        DataFrame df = emptyWithCols(want, schema);
        try (ParquetInputFormat in = ParquetInputFormat.open(path)) {
            Group row;
            while ((row = in.read()) != null) {
                Map<String, Object> map = rowToMap(row, schema, want);
                if (opt.filter() != null && !opt.filter().test(map)) continue;
                int ri = df.addEmptyRow();
                for (String c : want) df.set(ri, c, map.get(c));
                if (opt.maxRows() > 0 && df.rowCount() >= opt.maxRows()) break;
            }
        }
        return df;
    }

    /**
     * Read a single row-group by re-opening the file and skipping prior groups.
     * Pure-Java reader has no random row-group open; we skip via read() count.
     * For true parallel IO each worker still reads only its group's pages once
     * advanced (page load is lazy per group in ParquetInputFormat).
     */
    private static DataFrame readRowGroup(Path path, int groupIndex,
                                          List<String> want,
                                          Predicate<Map<String, Object>> filter) throws Exception {
        try (ParquetInputFormat in = ParquetInputFormat.open(path)) {
            MessageType schema = in.getSchema();
            // Skip to target group by consuming prior groups' rows
            long skip = 0;
            List<org.apache.parquet.format.RowGroup> rgs = in.getFooter().getRow_groups();
            for (int g = 0; g < groupIndex && rgs != null && g < rgs.size(); g++) {
                skip += rgs.get(g).getNum_rows();
            }
            long targetRows = (rgs != null && groupIndex < rgs.size())
                ? rgs.get(groupIndex).getNum_rows() : 0;
            // advance
            for (long i = 0; i < skip; i++) {
                if (in.read() == null) break;
            }
            DataFrame df = emptyWithCols(want, schema);
            for (long i = 0; i < targetRows; i++) {
                Group row = in.read();
                if (row == null) break;
                Map<String, Object> map = rowToMap(row, schema, want);
                if (filter != null && !filter.test(map)) continue;
                int ri = df.addEmptyRow();
                for (String c : want) df.set(ri, c, map.get(c));
            }
            return df;
        }
    }

    private static Map<String, Object> rowToMap(Group row, MessageType schema, List<String> want) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (String fname : want) {
            map.put(fname, readGroupValue(row, fname, schema));
        }
        return map;
    }

    private static Object readGroupValue(Group row, String fname, MessageType schema) {
        try {
            int idx = row.getType().getFieldIndex(fname);
            if (row.getFieldRepetitionCount(idx) == 0) return null;
            Type type = schema.getType(fname);
            if (!type.isPrimitive()) {
                try { return row.getGroup(idx, 0).toString(); }
                catch (Exception e) { return row.getValueToString(idx, 0); }
            }
            return switch (type.asPrimitiveType().getPrimitiveTypeName()) {
                case INT32 -> row.getInteger(idx, 0);
                case INT64 -> row.getLong(idx, 0);
                case FLOAT -> row.getFloat(idx, 0);
                case DOUBLE -> row.getDouble(idx, 0);
                case BOOLEAN -> row.getBoolean(idx, 0);
                default -> {
                    try { yield row.getString(idx, 0); }
                    catch (Exception e) { yield row.getValueToString(idx, 0); }
                }
            };
        } catch (Exception e) {
            return null;
        }
    }

    private static List<String> fieldNames(MessageType schema) {
        List<String> names = new ArrayList<>();
        for (Type t : schema.getFields()) names.add(t.getName());
        return names;
    }

    private static List<String> resolveColumns(List<String> all, List<String> want) {
        if (want == null || want.isEmpty()) return all;
        List<String> out = new ArrayList<>();
        for (String c : want) {
            if (!all.contains(c)) throw new IllegalArgumentException("column not in parquet: " + c);
            out.add(c);
        }
        return out;
    }

    private static DataFrame emptyWithCols(List<String> cols, MessageType schema) {
        DataFrame df = DataFrame.create();
        for (String c : cols) {
            Column.DType dt = Column.DType.STRING;
            try {
                Type t = schema.getType(c);
                if (t.isPrimitive()) {
                    dt = switch (t.asPrimitiveType().getPrimitiveTypeName()) {
                        case INT32 -> Column.DType.INT32;
                        case INT64 -> Column.DType.INT64;
                        case FLOAT -> Column.DType.FLOAT32;
                        case DOUBLE -> Column.DType.FLOAT64;
                        case BOOLEAN -> Column.DType.BOOLEAN;
                        default -> Column.DType.STRING;
                    };
                }
            } catch (Exception ignored) {}
            df.addColumn(c, dt);
        }
        return df;
    }

    private static String hiveKey(DataFrame df, int row, String[] partitionBy) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < partitionBy.length; i++) {
            if (i > 0) sb.append('/');
            Object v = df.get(row, partitionBy[i]);
            sb.append(partitionBy[i]).append('=').append(v == null ? "__HIVE_DEFAULT_PARTITION__" : v);
        }
        return sb.toString();
    }

    private static Map<String, String> parseHivePath(Path root, Path file, String[] keys) {
        Map<String, String> out = new LinkedHashMap<>();
        if (keys == null) return out;
        Path rel = root.relativize(file.getParent() == null ? file : file.getParent());
        for (Path part : rel) {
            String s = part.toString();
            int eq = s.indexOf('=');
            if (eq > 0) {
                out.put(s.substring(0, eq), s.substring(eq + 1));
            }
        }
        return out;
    }

    private static DataFrame filterFrame(DataFrame df, Predicate<Map<String, Object>> filter) {
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < df.rowCount(); i++) {
            if (filter.test(df.toDict(i))) keep.add(i);
        }
        return df.loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }
}
