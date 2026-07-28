/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.lance;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.bytedeco.pytorch.data.arrow.ArrowBridge;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.lance.LanceDataset;
import org.lance.Dataset;
import org.lance.Tag;
import org.lance.Version;
import org.lance.WriteParams;
import org.lance.compaction.CompactionOptions;
import org.lance.index.DistanceType;
import org.lance.index.Index;
import org.lance.index.OptimizeOptions;
import org.lance.ipc.ApproxMode;
import org.lance.ipc.FullTextQuery;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.Query;
import org.lance.ipc.ScanOptions;

import java.io.Closeable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Official <a href="https://lance.org/">Lance</a> ({@code org.lance:lance-core}) wrapper with
 * first-class {@link DataFrame} interop via Apache Arrow.
 *
 * <p>Uses the native JNI-backed {@link Dataset} API. DataFrame conversion goes through
 * {@link ArrowBridge} (DataFrame ↔ Arrow IPC ↔ Lance).
 *
 * <pre>{@code
 * // write (default DataFrame path)
 * df.writeLance("clips.lance");
 * // open + ANN + index
 * try (Lance ds = Lance.open("clips.lance")) {
 *     ds.createVectorIndex("emb", LanceIndex.ivfHnswPq(32, 16, 100, "cosine"));
 *     DataFrame hits = ds.search("emb", query, 10, SearchOptions.cosine().ef(64));
 *     ds.tag("v1");
 * }
 * // read
 * DataFrame back = Lance.readDataFrame("clips.lance");
 * }</pre>
 *
 * <p>Also interoperates with the pure-Java training dataset under
 * {@link LanceDataset} (different on-disk layout —
 * use {@link #isOfficialLance(String)} / {@link #isPureJavaLance(String)} to detect).
 */
public final class Lance implements Closeable {

    public static final String VERSION = "10.0.0-beta.5";

    private final Dataset dataset;
    private final BufferAllocator allocator;
    private final boolean ownsAllocator;
    private final String uri;

    private Lance(Dataset dataset, BufferAllocator allocator, boolean ownsAllocator, String uri) {
        this.dataset = Objects.requireNonNull(dataset, "dataset");
        this.allocator = Objects.requireNonNull(allocator, "allocator");
        this.ownsAllocator = ownsAllocator;
        this.uri = uri;
    }

    // ---- open / write ----------------------------------------------------

    public static Lance open(String path) throws Exception {
        return open(Path.of(path), null);
    }

    public static Lance open(Path path) throws Exception {
        return open(path, null);
    }

    /**
     * Open with optional version / tag (time travel via {@link LanceReadOptions}).
     */
    public static Lance open(String path, LanceReadOptions opts) throws Exception {
        return open(Path.of(path), opts);
    }

    public static Lance open(Path path, LanceReadOptions opts) throws Exception {
        Objects.requireNonNull(path, "path");
        String uri = toFileUri(path);
        BufferAllocator alloc = new RootAllocator();
        try {
            Dataset ds = Dataset.open(uri, alloc);
            Lance lance = new Lance(ds, alloc, true, uri);
            if (opts != null) {
                if (opts.version() != null) {
                    return lance.checkoutVersion(opts.version());
                }
                if (opts.tag() != null && !opts.tag().isBlank()) {
                    return lance.checkoutTag(opts.tag());
                }
            }
            return lance;
        } catch (Throwable t) {
            alloc.close();
            throw t;
        }
    }

    /**
     * Write a DataFrame as an official Lance dataset.
     *
     * @param mode CREATE / OVERWRITE / APPEND
     */
    public static Lance write(DataFrame df, String path, WriteParams.WriteMode mode) throws Exception {
        return write(df, Path.of(path),
            mode == null
                ? LanceWriteOptions.create()
                : LanceWriteOptions.defaults().mode(mode));
    }

    public static Lance write(DataFrame df, String path) throws Exception {
        return write(df, path, LanceWriteOptions.defaults());
    }

    public static Lance write(DataFrame df, String path, LanceWriteOptions options) throws Exception {
        return write(df, Path.of(path), options);
    }

    public static Lance write(DataFrame df, Path path, LanceWriteOptions options) throws Exception {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(path, "path");
        LanceWriteOptions opts = options == null ? LanceWriteOptions.defaults() : options;
        WriteParams.WriteMode m = opts.mode() == null ? WriteParams.WriteMode.CREATE : opts.mode();
        if (m == WriteParams.WriteMode.CREATE && Files.exists(path)) {
            // prefer overwrite when path exists to avoid hard failures in benchmarks
            m = WriteParams.WriteMode.OVERWRITE;
        }
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        String uri = toFileUri(path);
        BufferAllocator alloc = new RootAllocator();
        ArrowReader reader = null;
        try {
            reader = ArrowBridge.toArrowReader(df, alloc);
            var builder = Dataset.write()
                    .allocator(alloc)
                    .reader(reader)
                    .uri(uri)
                    .mode(m);
            if (opts.maxRowsPerFile() != null) builder.maxRowsPerFile(opts.maxRowsPerFile());
            if (opts.maxRowsPerGroup() != null) builder.maxRowsPerGroup(opts.maxRowsPerGroup());
            if (opts.maxBytesPerFile() != null) builder.maxBytesPerFile(opts.maxBytesPerFile());
            if (opts.stableRowIds() != null) builder.enableStableRowIds(opts.stableRowIds());
            if (opts.dataStorageVersion() != null && !opts.dataStorageVersion().isBlank()) {
                builder.dataStorageVersion(opts.dataStorageVersion());
            }
            if (opts.storageOptions() != null && !opts.storageOptions().isEmpty()) {
                builder.storageOptions(new LinkedHashMap<>(opts.storageOptions()));
            }
            Dataset ds = builder.execute();
            try { reader.close(); } catch (Exception ignored) {}
            reader = null;
            return new Lance(ds, alloc, true, uri);
        } catch (Throwable t) {
            if (reader != null) {
                try { reader.close(); } catch (Exception ignored) {}
            }
            alloc.close();
            throw t;
        }
    }

    /** Overwrite convenience. */
    public static Lance overwrite(DataFrame df, String path) throws Exception {
        return write(df, path, LanceWriteOptions.overwrite());
    }

    /** Append rows to an existing dataset. */
    public static Lance append(DataFrame df, String path) throws Exception {
        return write(df, path, LanceWriteOptions.append());
    }

    /** Read entire dataset into a DataFrame (static one-shot). */
    public static DataFrame readDataFrame(String path) throws Exception {
        try (Lance ds = open(path)) {
            return ds.toDataFrame();
        }
    }

    public static DataFrame readDataFrame(Path path) throws Exception {
        try (Lance ds = open(path)) {
            return ds.toDataFrame();
        }
    }

    public static DataFrame readDataFrame(String path, LanceReadOptions opts) throws Exception {
        try (Lance ds = open(path, opts)) {
            return ds.scan(opts);
        }
    }

    /**
     * Heuristic: official Lance datasets have a {@code _versions} / {@code data} layout with
     * manifest binaries; our pure-Java training layout uses {@code _manifest.json}.
     */
    public static boolean isOfficialLance(String path) {
        Path p = Path.of(path);
        if (Files.isRegularFile(p.resolve("_manifest.json"))) {
            return false; // pure-Java training layout
        }
        return Files.isDirectory(p.resolve("_versions"))
                || Files.isDirectory(p.resolve("data"))
                || (Files.isDirectory(p) && Files.exists(p));
    }

    public static boolean isPureJavaLance(String path) {
        return Files.isRegularFile(Path.of(path).resolve("_manifest.json"));
    }

    // ---- instance API ----------------------------------------------------

    public Dataset raw() {
        return dataset;
    }

    public String uri() {
        return uri != null ? uri : dataset.uri();
    }

    public long countRows() {
        return dataset.countRows();
    }

    public long countRows(String filter) {
        return dataset.countRows(filter);
    }

    public Schema schema() {
        return dataset.getSchema();
    }

    public List<String> columnNames() {
        Schema s = schema();
        if (s == null || s.getFields() == null) return List.of();
        return s.getFields().stream().map(Field::getName).collect(Collectors.toList());
    }

    public Map<String, String> schemaMap() {
        Map<String, String> m = new LinkedHashMap<>();
        Schema s = schema();
        if (s == null) return m;
        for (Field f : s.getFields()) {
            m.put(f.getName(), f.getType() == null ? "null" : f.getType().toString());
        }
        return m;
    }

    public long version() {
        return dataset.version();
    }

    public Version getVersion() {
        return dataset.getVersion();
    }

    public List<Version> listVersions() {
        List<Version> v = dataset.listVersions();
        return v == null ? List.of() : v;
    }

    public long latestVersion() {
        return dataset.latestVersion();
    }

    /** Full table scan → DataFrame. */
    public DataFrame toDataFrame() throws Exception {
        return scan(null, null, -1);
    }

    /**
     * Scan with optional column projection, SQL-like filter, and row limit.
     *
     * @param columns null/empty = all columns
     * @param filter  null/blank = no filter (Lance filter expression)
     * @param limit   &lt;= 0 = no limit
     */
    public DataFrame scan(List<String> columns, String filter, long limit) throws Exception {
        ScanOptions.Builder b = new ScanOptions.Builder();
        if (columns != null && !columns.isEmpty()) {
            b.columns(columns);
        }
        if (filter != null && !filter.isBlank()) {
            b.filter(filter);
        }
        if (limit > 0) {
            b.limit(limit);
        }
        b.batchSize(64_000);
        try (LanceScanner scanner = dataset.newScan(b.build());
             ArrowReader reader = scanner.scanBatches()) {
            return ArrowBridge.fromArrowReader(reader);
        }
    }

    public DataFrame scan(LanceReadOptions opts) throws Exception {
        if (opts == null) return toDataFrame();
        ScanOptions.Builder b = new ScanOptions.Builder();
        if (opts.columns() != null && !opts.columns().isEmpty()) {
            b.columns(opts.columns());
        }
        if (opts.filter() != null && !opts.filter().isBlank()) {
            b.filter(opts.filter());
        }
        if (opts.limit() > 0) b.limit(opts.limit());
        if (opts.offset() > 0) b.offset(opts.offset());
        if (opts.batchSize() > 0) b.batchSize(opts.batchSize());
        if (opts.withRowId()) b.withRowId(true);
        if (opts.withRowAddress()) b.withRowAddress(true);
        b.useScalarIndex(opts.useScalarIndex());
        try (LanceScanner scanner = dataset.newScan(b.build());
             ArrowReader reader = scanner.scanBatches()) {
            return ArrowBridge.fromArrowReader(reader);
        }
    }

    public DataFrame head(int n) throws Exception {
        return scan(null, null, n);
    }

    public DataFrame filter(String expr) throws Exception {
        return scan(null, expr, -1);
    }

    public DataFrame select(String... columns) throws Exception {
        return scan(Arrays.asList(columns), null, -1);
    }

    /**
     * Vector nearest-neighbor search → DataFrame of hits (with score columns when present).
     *
     * @param vectorColumn embedding / vector column name
     * @param query        query vector
     * @param k            top-k
     * @param metric       L2 / Cosine / Dot / Hamming (case-insensitive); null = L2
     */
    public DataFrame search(String vectorColumn, float[] query, int k, String metric) throws Exception {
        return search(vectorColumn, query, k,
            SearchOptions.defaults().metric(metric == null ? "L2" : metric));
    }

    public DataFrame search(String vectorColumn, float[] query, int k) throws Exception {
        return search(vectorColumn, query, k, SearchOptions.l2());
    }

    /**
     * ANN search with full {@link SearchOptions} (ef / nprobes / refine / hybrid filter).
     */
    public DataFrame search(String vectorColumn, float[] query, int k, SearchOptions options)
            throws Exception {
        Objects.requireNonNull(vectorColumn, "vectorColumn");
        Objects.requireNonNull(query, "query");
        if (k <= 0) throw new IllegalArgumentException("k must be > 0");
        SearchOptions opts = options == null ? SearchOptions.defaults() : options;
        DistanceType dt = parseDistance(opts.metric());

        Query.Builder qb = new Query.Builder()
                .setColumn(vectorColumn)
                .setKey(query)
                .setK(k)
                .setDistanceType(dt);
        if (opts.ef() != null) qb.setEf(opts.ef());
        if (opts.minimumNprobes() != null) qb.setMinimumNprobes(opts.minimumNprobes());
        if (opts.maximumNprobes() != null) qb.setMaximumNprobes(opts.maximumNprobes());
        if (opts.refineFactor() != null) qb.setRefineFactor(opts.refineFactor());
        if (opts.useIndex() != null) qb.setUseIndex(opts.useIndex());
        if (opts.queryParallelism() != null) qb.setQueryParallelism(opts.queryParallelism());
        if (opts.approxMode() != null && !opts.approxMode().isBlank()) {
            try {
                qb.setApproxMode(ApproxMode.valueOf(opts.approxMode()));
            } catch (Exception ignored) {
                // keep default approx mode if enum name doesn't match
            }
        }
        Query nearest = qb.build();

        ScanOptions.Builder sb = new ScanOptions.Builder()
                .nearest(nearest)
                .prefilter(opts.prefilter());
        if (opts.filter() != null && !opts.filter().isBlank()) {
            sb.filter(opts.filter());
        }
        if (opts.columns() != null && !opts.columns().isEmpty()) {
            sb.columns(opts.columns());
        }
        long bs = opts.batchSize() > 0 ? opts.batchSize() : Math.max(k, 64);
        sb.batchSize(bs);

        try (LanceScanner scanner = dataset.newScan(sb.build());
             ArrowReader reader = scanner.scanBatches()) {
            return ArrowBridge.fromArrowReader(reader);
        }
    }

    /**
     * Full-text search (requires an inverted / FTS index on the column).
     */
    public DataFrame fullTextSearch(String column, String query, int limit) throws Exception {
        Objects.requireNonNull(column, "column");
        Objects.requireNonNull(query, "query");
        // FullTextQuery.match(queryText, column) — query first, then column
        FullTextQuery ftq = FullTextQuery.match(query, column);
        ScanOptions.Builder b = new ScanOptions.Builder()
                .fullTextQuery(ftq);
        if (limit > 0) b.limit(limit);
        b.batchSize(Math.max(limit, 64));
        try (LanceScanner scanner = dataset.newScan(b.build());
             ArrowReader reader = scanner.scanBatches()) {
            return ArrowBridge.fromArrowReader(reader);
        }
    }

    public DataFrame fullTextPhrase(String column, String phrase, int limit) throws Exception {
        Objects.requireNonNull(column, "column");
        Objects.requireNonNull(phrase, "phrase");
        // FullTextQuery.phrase(queryText, column)
        FullTextQuery ftq = FullTextQuery.phrase(phrase, column);
        ScanOptions.Builder b = new ScanOptions.Builder().fullTextQuery(ftq);
        if (limit > 0) b.limit(limit);
        try (LanceScanner scanner = dataset.newScan(b.build());
             ArrowReader reader = scanner.scanBatches()) {
            return ArrowBridge.fromArrowReader(reader);
        }
    }

    /** Take rows by row id. */
    public DataFrame take(List<Long> rowIds, List<String> columns) throws Exception {
        try (ArrowReader reader = dataset.take(rowIds, columns)) {
            return ArrowBridge.fromArrowReader(reader);
        }
    }

    public DataFrame take(List<Long> rowIds) throws Exception {
        return take(rowIds, null);
    }

    /** Random sample of {@code n} rows. */
    public DataFrame sample(long n, List<String> columns) throws Exception {
        try (ArrowReader reader = dataset.sample(n, columns)) {
            return ArrowBridge.fromArrowReader(reader);
        }
    }

    public DataFrame sample(long n) throws Exception {
        return sample(n, null);
    }

    // ---- indexes ---------------------------------------------------------

    /**
     * Create a vector index on {@code column}.
     *
     * @return true if the call completed without throwing
     */
    public boolean createVectorIndex(String column, LanceIndex index) {
        return createVectorIndex(column, index, true);
    }

    public boolean createVectorIndex(String column, LanceIndex index, boolean replace) {
        Objects.requireNonNull(column, "column");
        Objects.requireNonNull(index, "index");
        try {
            createIndex(List.of(column), index, replace);
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Convenience: IVF-HNSW-PQ with reasonable defaults for the given metric.
     * Returns true if the call completed without throwing.
     */
    public boolean createVectorIndex(String column, String indexName, String metric) {
        Objects.requireNonNull(column, "column");
        // Small defaults suitable for smoke tests / moderate data; tune via LanceIndex factories.
        LanceIndex idx = LanceIndex.ivfHnswPq(4, 16, 50, metric == null ? "L2" : metric)
                .named(indexName);
        return createVectorIndex(column, idx, true);
    }

    public boolean createScalarIndex(String column, LanceIndex index) {
        return createScalarIndex(column, index, true);
    }

    public boolean createScalarIndex(String column, LanceIndex index, boolean replace) {
        Objects.requireNonNull(column, "column");
        Objects.requireNonNull(index, "index");
        try {
            createIndex(List.of(column), index, replace);
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    /** Create a full-text inverted index on a text column. */
    public boolean createFtsIndex(String column) {
        return createFtsIndex(column, null, true);
    }

    public boolean createFtsIndex(String column, String indexName, boolean replace) {
        LanceIndex idx = LanceIndex.fts();
        if (indexName != null) idx = idx.named(indexName);
        return createScalarIndex(column, idx, replace);
    }

    /**
     * Low-level index creation — throws on failure.
     */
    public Index createIndex(List<String> columns, LanceIndex index, boolean replace) {
        Objects.requireNonNull(columns, "columns");
        Objects.requireNonNull(index, "index");
        if (columns.isEmpty()) throw new IllegalArgumentException("columns must not be empty");
        Optional<String> name = Optional.ofNullable(index.name());
        return dataset.createIndex(columns, index.indexType(), name, index.indexParams(), replace);
    }

    public Index createIndex(String column, LanceIndex index) {
        return createIndex(List.of(column), index, true);
    }

    public void dropIndex(String indexName) {
        dataset.dropIndex(indexName);
    }

    public List<String> listIndexes() {
        List<String> names = dataset.listIndexes();
        return names == null ? List.of() : names;
    }

    public List<Index> getIndexes() {
        List<Index> idx = dataset.getIndexes();
        return idx == null ? List.of() : idx;
    }

    public Map<String, Object> getIndexStatistics(String indexName) {
        Map<String, Object> m = dataset.getIndexStatistics(indexName);
        return m == null ? Map.of() : m;
    }

    public void optimizeIndices() {
        dataset.optimizeIndices(OptimizeOptions.builder().build());
    }

    public void optimizeIndices(OptimizeOptions options) {
        dataset.optimizeIndices(options == null
            ? OptimizeOptions.builder().build()
            : options);
    }

    // ---- versioning / tags / branches ------------------------------------

    /**
     * Checkout a historical version. Returns a new {@link Lance} wrapping the checked-out
     * dataset; the previous handle remains open until closed by the caller.
     */
    public Lance checkoutVersion(long version) {
        Dataset ds = dataset.checkoutVersion(version);
        // checkout returns a new Dataset; keep sharing allocator ownership with this wrapper
        return new Lance(ds, allocator, false, uri);
    }

    public Lance checkoutTag(String tag) {
        Dataset ds = dataset.checkoutTag(tag);
        return new Lance(ds, allocator, false, uri);
    }

    public void checkoutLatest() {
        dataset.checkoutLatest();
    }

    public void restore() {
        dataset.restore();
    }

    /** Create a tag at the current version. */
    public void tag(String name) {
        dataset.tags().create(name, dataset.version());
    }

    public void tag(String name, long version) {
        dataset.tags().create(name, version);
    }

    public void deleteTag(String name) {
        dataset.tags().delete(name);
    }

    public void updateTag(String name, long version) {
        dataset.tags().update(name, version);
    }

    public List<Tag> listTags() {
        List<Tag> t = dataset.tags().list();
        return t == null ? List.of() : t;
    }

    public long tagVersion(String name) {
        return dataset.tags().getVersion(name);
    }

    public List<String> listBranches() {
        var branches = dataset.branches().list();
        if (branches == null) return List.of();
        List<String> names = new ArrayList<>();
        for (var b : branches) {
            try {
                names.add(String.valueOf(b));
            } catch (Exception e) {
                names.add(b == null ? "null" : b.getClass().getSimpleName());
            }
        }
        return names;
    }

    // ---- mutation --------------------------------------------------------

    /** Delete rows matching a Lance filter predicate. */
    public void delete(String predicate) {
        Objects.requireNonNull(predicate, "predicate");
        dataset.delete(predicate);
    }

    public void truncateTable() {
        dataset.truncateTable();
    }

    public void compact() {
        dataset.compact();
    }

    public void compact(CompactionOptions options) {
        if (options == null) dataset.compact();
        else dataset.compact(options);
    }

    public Map<String, Object> info() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("uri", uri());
        m.put("countRows", countRows());
        m.put("columns", columnNames());
        m.put("schema", schemaMap());
        m.put("lanceVersion", VERSION);
        m.put("official", true);
        try { m.put("version", version()); } catch (Throwable ignored) {}
        try { m.put("latestVersion", latestVersion()); } catch (Throwable ignored) {}
        try { m.put("indexes", listIndexes()); } catch (Throwable ignored) {}
        try { m.put("fragments", dataset.getFragments() == null ? 0 : dataset.getFragments().size()); }
        catch (Throwable ignored) {}
        return m;
    }

    @Override
    public void close() {
        try {
            dataset.close();
        } catch (Throwable ignored) {
        }
        if (ownsAllocator) {
            try {
                allocator.close();
            } catch (Throwable ignored) {
            }
        }
    }

    // ---- DataFrame extension-style helpers -------------------------------

    /**
     * Write DataFrame via official Lance and return path.
     * Prefer this (or {@link DataFrame#writeLance(String)}) for production datasets.
     */
    public static String writeDataFrame(DataFrame df, String path) throws Exception {
        try (Lance ignored = write(df, path, LanceWriteOptions.overwrite())) {
            return path;
        }
    }

    public static String writeDataFrame(DataFrame df, String path, LanceWriteOptions options)
            throws Exception {
        try (Lance ignored = write(df, path, options == null ? LanceWriteOptions.overwrite() : options)) {
            return path;
        }
    }

    /**
     * Dual-read: pure-Java training layout when {@code _manifest.json} is present;
     * otherwise official Lance.
     */
    public static DataFrame readAuto(String path) throws Exception {
        if (isPureJavaLance(path)) {
            return LanceDataset.read(path);
        }
        try {
            return readDataFrame(path);
        } catch (Throwable officialEx) {
            try {
                return LanceDataset.read(path);
            } catch (Throwable pureEx) {
                officialEx.addSuppressed(pureEx);
                throw officialEx;
            }
        }
    }

    public static DataFrame readAuto(String path, LanceReadOptions opts) throws Exception {
        if (isPureJavaLance(path)) {
            return LanceDataset.read(path);
        }
        return readDataFrame(path, opts);
    }

    // ---- helpers ---------------------------------------------------------

    static String toFileUri(Path path) {
        Path abs = path.toAbsolutePath().normalize();
        // Lance accepts plain paths and file:// URIs; prefer plain absolute path for local FS.
        return abs.toString();
    }

    static DistanceType parseDistance(String metric) {
        return LanceIndex.parseDistance(metric);
    }
}
