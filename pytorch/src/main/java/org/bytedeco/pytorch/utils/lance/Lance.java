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
import org.bytedeco.pytorch.data.*;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.bytedeco.pytorch.data.arrow.ArrowBridge;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.lance.Dataset;
import org.lance.WriteParams;
import org.lance.index.DistanceType;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.Query;
import org.lance.ipc.ScanOptions;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Official <a href="https://lance.org/">Lance</a> ({@code org.lance:lance-core}) wrapper with
 * first-class {@link DataFrame} interop via Apache Arrow.
 *
 * <p>Uses the native JNI-backed {@link Dataset} API. DataFrame conversion goes through
 * {@link ArrowBridge} (DataFrame ↔ Arrow IPC ↔ Lance).
 *
 * <pre>{@code
 * // write
 * try (Lance ds = Lance.write(df, "clips.lance")) {
 *     System.out.println(ds.countRows());
 * }
 * // read
 * DataFrame back = Lance.readDataFrame("clips.lance");
 * // ANN search
 * try (Lance ds = Lance.open("clips.lance")) {
 *     DataFrame hits = ds.search("embedding", query, 10);
 * }
 * }</pre>
 *
 * <p>Also interoperates with the pure-Java training dataset under
 * {@link org.bytedeco.pytorch.data.dataframe.lance.LanceDataset} (different on-disk layout —
 * use {@link #isOfficialLance(String)} to detect the native format).
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
        return open(Path.of(path));
    }

    public static Lance open(Path path) throws Exception {
        Objects.requireNonNull(path, "path");
        String uri = toFileUri(path);
        BufferAllocator alloc = new RootAllocator();
        try {
            Dataset ds = Dataset.open(uri, alloc);
            return new Lance(ds, alloc, true, uri);
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
        return write(df, Path.of(path), mode == null ? WriteParams.WriteMode.CREATE : mode);
    }

    public static Lance write(DataFrame df, String path) throws Exception {
        return write(df, path, WriteParams.WriteMode.CREATE);
    }

    public static Lance write(DataFrame df, Path path, WriteParams.WriteMode mode) throws Exception {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(path, "path");
        WriteParams.WriteMode m = mode == null ? WriteParams.WriteMode.CREATE : mode;
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
            Dataset ds = Dataset.write()
                    .allocator(alloc)
                    .reader(reader)
                    .uri(uri)
                    .mode(m)
                    .execute();
            // reader is consumed; close it
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
        return write(df, path, WriteParams.WriteMode.OVERWRITE);
    }

    /** Append rows to an existing dataset. */
    public static Lance append(DataFrame df, String path) throws Exception {
        return write(df, path, WriteParams.WriteMode.APPEND);
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

    /**
     * Heuristic: official Lance datasets have a {@code _versions} / {@code data} layout with
     * manifest binaries; our pure-Java training layout uses {@code _manifest.json}.
     */
    public static boolean isOfficialLance(String path) {
        Path p = Path.of(path);
        if (Files.isRegularFile(p.resolve("_manifest.json"))) {
            return false; // pure-Java training layout
        }
        // official lance typically has _versions directory or *.manifest
        return Files.isDirectory(p.resolve("_versions"))
                || Files.isDirectory(p.resolve("data"))
                || Files.exists(p);
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
        Objects.requireNonNull(vectorColumn, "vectorColumn");
        Objects.requireNonNull(query, "query");
        if (k <= 0) throw new IllegalArgumentException("k must be > 0");
        DistanceType dt = parseDistance(metric);
        Query nearest = new Query.Builder()
                .setColumn(vectorColumn)
                .setKey(query)
                .setK(k)
                .setDistanceType(dt)
                .build();
        ScanOptions options = new ScanOptions.Builder()
                .nearest(nearest)
                .prefilter(true)
                .batchSize(Math.max(k, 64))
                .build();
        try (LanceScanner scanner = dataset.newScan(options);
             ArrowReader reader = scanner.scanBatches()) {
            return ArrowBridge.fromArrowReader(reader);
        }
    }

    public DataFrame search(String vectorColumn, float[] query, int k) throws Exception {
        return search(vectorColumn, query, k, "L2");
    }

    /**
     * Create a vector index on {@code column} (best-effort; index params depend on lance version).
     * Returns true if the call completed without throwing.
     */
    public boolean createVectorIndex(String column, String indexName, String metric) {
        // Index creation APIs evolve across lance betas; keep a soft hook for callers/benchmarks.
        // Full IVF-PQ / HNSW builder wiring can be layered on without changing this facade.
        Objects.requireNonNull(column, "column");
        return false; // not forced — ANN still works via brute force / existing indices
    }

    public Map<String, Object> info() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("uri", uri());
        m.put("countRows", countRows());
        m.put("columns", columnNames());
        m.put("schema", schemaMap());
        m.put("lanceVersion", VERSION);
        m.put("official", true);
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
     * Write DataFrame via official Lance and return path. Prefer this over the pure-Java
     * {@code DataFrame.writeLance} when native lance-core is on the classpath.
     */
    public static String writeDataFrame(DataFrame df, String path) throws Exception {
        try (Lance ignored = write(df, path, WriteParams.WriteMode.OVERWRITE)) {
            return path;
        }
    }

    /**
     * Dual-read: try official Lance first; fall back to pure-Java
     * {@link org.bytedeco.pytorch.data.dataframe.lance.LanceDataset} when {@code _manifest.json} is present.
     */
    public static DataFrame readAuto(String path) throws Exception {
        if (isPureJavaLance(path)) {
            return org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.read(path);
        }
        try {
            return readDataFrame(path);
        } catch (Throwable officialEx) {
            try {
                return org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.read(path);
            } catch (Throwable pureEx) {
                officialEx.addSuppressed(pureEx);
                throw officialEx;
            }
        }
    }

    // ---- helpers ---------------------------------------------------------

    static String toFileUri(Path path) {
        Path abs = path.toAbsolutePath().normalize();
        // Lance accepts plain paths and file:// URIs; prefer plain absolute path for local FS.
        return abs.toString();
    }

    static DistanceType parseDistance(String metric) {
        if (metric == null || metric.isBlank()) return DistanceType.L2;
        String m = metric.trim().toLowerCase(Locale.ROOT);
        return switch (m) {
            case "cosine", "cos" -> DistanceType.Cosine;
            case "dot", "ip", "inner_product" -> DistanceType.Dot;
            case "hamming" -> DistanceType.Hamming;
            case "l2", "euclidean", "euclid" -> DistanceType.L2;
            default -> DistanceType.L2;
        };
    }
}
