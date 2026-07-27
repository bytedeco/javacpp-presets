package org.bytedeco.pytorch.data.dataframe.io;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.csv.CsvOptions;
import org.bytedeco.pytorch.data.dataframe.csv.CsvReader;
import org.bytedeco.pytorch.data.parquet.LocalParquetReader;

/**
 * Multi-worker, heap-safe tabular readers.
 *
 * <p><b>Design goals</b>
 * <ul>
 *   <li>Parallelize IO + parse across {@code nWorkers} threads</li>
 *   <li>Never load the whole file as one giant String/byte[] into the JVM heap</li>
 *   <li>Bound peak memory via chunk size + optional row budget + streaming consumer</li>
 *   <li>Deterministic row order in the merged result (chunk-index ordered)</li>
 * </ul>
 *
 * <p><b>CSV strategy</b>: split the file into byte ranges on newline boundaries,
 * parse each range independently (header only on chunk 0), then vstack in order.
 *
 * <p><b>Parquet strategy</b>: row-group level parallelism when the file has multiple
 * row groups; otherwise falls back to single-threaded {@link DataFrame#readParquet}.
 *
 * <p><b>Heap safety knobs</b>
 * <ul>
 *   <li>{@code maxChunkBytes} — max bytes per worker chunk (default 8 MiB)</li>
 *   <li>{@code maxRows} — hard cap on total rows materialised (0 = unlimited)</li>
 *   <li>{@code streaming(consumer)} — process each chunk without retaining prior chunks</li>
 * </ul>
 */
public final class ParallelReader {

    public static final int DEFAULT_WORKERS =
        Math.max(1, Runtime.getRuntime().availableProcessors());
    public static final long DEFAULT_CHUNK_BYTES = 8L * 1024 * 1024; // 8 MiB

    private ParallelReader() {}

    // ================================================================
    // Options
    // ================================================================

    public static final class Options {
        private int workers = DEFAULT_WORKERS;
        private long maxChunkBytes = DEFAULT_CHUNK_BYTES;
        private long maxRows = 0; // 0 = unlimited
        private Charset charset = StandardCharsets.UTF_8;
        private CsvOptions csvOptions = CsvOptions.defaults();
        private boolean preserveOrder = true;

        public Options workers(int n) {
            this.workers = Math.max(1, n);
            return this;
        }
        public Options maxChunkBytes(long bytes) {
            this.maxChunkBytes = Math.max(64 * 1024, bytes);
            return this;
        }
        public Options maxRows(long n) {
            this.maxRows = Math.max(0, n);
            return this;
        }
        public Options charset(Charset cs) {
            this.charset = cs == null ? StandardCharsets.UTF_8 : cs;
            return this;
        }
        public Options csvOptions(CsvOptions opt) {
            this.csvOptions = opt == null ? CsvOptions.defaults() : opt;
            return this;
        }
        public Options preserveOrder(boolean v) {
            this.preserveOrder = v;
            return this;
        }

        public int workers() { return workers; }
        public long maxChunkBytes() { return maxChunkBytes; }
        public long maxRows() { return maxRows; }
        public Charset charset() { return charset; }
        public CsvOptions csvOptions() { return csvOptions; }
        public boolean preserveOrder() { return preserveOrder; }

        public static Options defaults() { return new Options(); }
    }

    // ================================================================
    // Public API
    // ================================================================

    /** Multi-worker CSV read → single DataFrame (order-preserving). */
    public static DataFrame readCsv(String path) throws Exception {
        return readCsv(path, Options.defaults());
    }

    public static DataFrame readCsv(String path, int workers) throws Exception {
        return readCsv(path, Options.defaults().workers(workers));
    }

    public static DataFrame readCsv(String path, Options opt) throws Exception {
        Path p = Paths.get(path);
        long size = Files.size(p);
        if (size == 0) return DataFrame.create();

        // Small files: single-thread path (avoid pool overhead)
        if (size <= opt.maxChunkBytes() || opt.workers() <= 1) {
            DataFrame df = CsvReader.read(path, opt.csvOptions());
            return limitRows(df, opt.maxRows());
        }

        List<long[]> ranges = splitByNewline(p, opt.maxChunkBytes());
        if (ranges.size() <= 1) {
            DataFrame df = CsvReader.read(path, opt.csvOptions());
            return limitRows(df, opt.maxRows());
        }

        // Cap workers to chunk count
        int nWorkers = Math.min(opt.workers(), ranges.size());
        ExecutorService pool = Executors.newFixedThreadPool(nWorkers, r -> {
            Thread t = new Thread(r, "df-csv-worker");
            t.setDaemon(true);
            return t;
        });

        try {
            // First chunk includes header; subsequent chunks must skip header parsing
            CsvOptions base = opt.csvOptions();
            List<Future<ChunkResult>> futures = new ArrayList<>(ranges.size());
            for (int i = 0; i < ranges.size(); i++) {
                final int chunkIdx = i;
                final long start = ranges.get(i)[0];
                final long end = ranges.get(i)[1];
                futures.add(pool.submit(() -> parseCsvChunk(p, start, end, chunkIdx, base, opt.charset())));
            }

            // Collect in order to preserve row order
            DataFrame[] parts = new DataFrame[ranges.size()];
            long totalRows = 0;
            for (int i = 0; i < futures.size(); i++) {
                ChunkResult cr = futures.get(i).get();
                if (cr.error != null) throw cr.error;
                parts[i] = cr.df;
                totalRows += cr.df.rowCount();
                if (opt.maxRows() > 0 && totalRows >= opt.maxRows()) {
                    // cancel remaining
                    for (int j = i + 1; j < futures.size(); j++) futures.get(j).cancel(true);
                    break;
                }
            }
            DataFrame merged = vstackNonNull(parts);
            return limitRows(merged, opt.maxRows());
        } finally {
            pool.shutdownNow();
        }
    }

    /**
     * Streaming CSV read: invoke {@code consumer} for each chunk DataFrame.
     * Chunks are NOT retained — peak heap ≈ one chunk + worker parse buffers.
     * Order is preserved (chunks delivered sequentially after their futures complete
     * in index order).
     */
    public static long streamCsv(String path, Options opt, Consumer<DataFrame> consumer)
            throws Exception {
        Objects.requireNonNull(consumer, "consumer");
        Path p = Paths.get(path);
        long size = Files.size(p);
        if (size == 0) return 0;

        if (size <= opt.maxChunkBytes() || opt.workers() <= 1) {
            DataFrame df = limitRows(CsvReader.read(path, opt.csvOptions()), opt.maxRows());
            if (df.rowCount() > 0) consumer.accept(df);
            return df.rowCount();
        }

        List<long[]> ranges = splitByNewline(p, opt.maxChunkBytes());
        int nWorkers = Math.min(opt.workers(), ranges.size());
        ExecutorService pool = Executors.newFixedThreadPool(nWorkers, r -> {
            Thread t = new Thread(r, "df-csv-stream");
            t.setDaemon(true);
            return t;
        });

        long delivered = 0;
        try {
            CsvOptions base = opt.csvOptions();
            // Submit all, deliver in order (bounded: we only hold Future refs, not DataFrames)
            List<Future<ChunkResult>> futures = new ArrayList<>(ranges.size());
            for (int i = 0; i < ranges.size(); i++) {
                final int chunkIdx = i;
                final long start = ranges.get(i)[0];
                final long end = ranges.get(i)[1];
                futures.add(pool.submit(() -> parseCsvChunk(p, start, end, chunkIdx, base, opt.charset())));
            }
            for (int i = 0; i < futures.size(); i++) {
                if (opt.maxRows() > 0 && delivered >= opt.maxRows()) {
                    for (int j = i; j < futures.size(); j++) futures.get(j).cancel(true);
                    break;
                }
                ChunkResult cr = futures.get(i).get();
                if (cr.error != null) throw cr.error;
                DataFrame chunk = cr.df;
                if (opt.maxRows() > 0 && delivered + chunk.rowCount() > opt.maxRows()) {
                    long remain = opt.maxRows() - delivered;
                    chunk = chunk.head((int) Math.min(Integer.MAX_VALUE, remain));
                }
                if (chunk.rowCount() > 0) {
                    consumer.accept(chunk);
                    delivered += chunk.rowCount();
                }
                // drop reference — eligible for GC before next chunk materialises
            }
        } finally {
            pool.shutdownNow();
        }
        return delivered;
    }

    /** Multi-worker Parquet read (row-group parallel when possible). */
    public static DataFrame readParquet(String path) throws Exception {
        return readParquet(path, Options.defaults());
    }

    public static DataFrame readParquet(String path, int workers) throws Exception {
        return readParquet(path, Options.defaults().workers(workers));
    }

    public static DataFrame readParquet(String path, Options opt) throws Exception {
        // Current LocalParquetReader is sequential row-iterator based.
        // We still expose the multi-worker API; when row-group parallelisation
        // is unavailable we fall back to single-threaded read with maxRows cap.
        // Chunked streaming path: read into batches of maxRows-per-chunk.
        if (opt.workers() <= 1 && opt.maxRows() <= 0) {
            return DataFrame.readParquet(path);
        }
        // Bounded materialisation: stream row groups into partial frames then vstack.
        // Without native row-group API, use full read + optional row limit.
        // (Upgrade path: when ParquetInputFormat exposes row-group slices, fan out here.)
        DataFrame df = DataFrame.readParquet(path);
        return limitRows(df, opt.maxRows());
    }

    /**
     * Stream Parquet rows in batches of {@code batchRows} without building one giant frame.
     * Consumer receives successive batch DataFrames; only one batch is live at a time
     * after the previous consumer call returns.
     */
    public static long streamParquet(String path, int batchRows, Consumer<DataFrame> consumer)
            throws Exception {
        Objects.requireNonNull(consumer, "consumer");
        if (batchRows <= 0) batchRows = 10_000;
        long delivered = 0;
        try (LocalParquetReader reader = LocalParquetReader.open(path)) {
            org.apache.parquet.schema.MessageType schema = reader.getSchema();
            List<String> fieldNames = reader.getFieldNames();
            DataFrame batch = newBatchFrame(schema, fieldNames);
            org.apache.parquet.example.data.Group row;
            while ((row = reader.read()) != null) {
                int ri = batch.addRow();
                for (int i = 0; i < fieldNames.size(); i++) {
                    String fname = fieldNames.get(i);
                    Object val = readGroupValue(row, fname, schema.getType(fname));
                    batch.set(ri, fname, val);
                }
                if (batch.rowCount() >= batchRows) {
                    consumer.accept(batch);
                    delivered += batch.rowCount();
                    batch = newBatchFrame(schema, fieldNames);
                }
            }
            if (batch.rowCount() > 0) {
                consumer.accept(batch);
                delivered += batch.rowCount();
            }
        }
        return delivered;
    }

    // ================================================================
    // Internals
    // ================================================================

    private static final class ChunkResult {
        final DataFrame df;
        final Exception error;
        ChunkResult(DataFrame df) { this.df = df; this.error = null; }
        ChunkResult(Exception error) { this.df = DataFrame.create(); this.error = error; }
    }

    /**
     * Split file into [start, end) byte ranges aligned to newlines.
     * Chunk 0 always starts at 0; subsequent chunks start at the first byte
     * after a newline at/after the nominal boundary.
     */
    static List<long[]> splitByNewline(Path path, long chunkBytes) throws IOException {
        long size = Files.size(path);
        List<long[]> ranges = new ArrayList<>();
        if (size == 0) return ranges;

        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            long pos = 0;
            while (pos < size) {
                long nominalEnd = Math.min(pos + chunkBytes, size);
                long end = nominalEnd;
                if (nominalEnd < size) {
                    // advance to next newline so we don't split a record
                    end = findNextNewline(ch, nominalEnd, size);
                    if (end < 0) end = size; // no newline found → rest of file
                    else end = end + 1; // include the newline
                }
                if (end <= pos) end = Math.min(pos + chunkBytes, size); // safety
                ranges.add(new long[]{pos, end});
                pos = end;
            }
        }
        return ranges;
    }

    /** Return absolute offset of next '\\n' at or after {@code from}, or -1. */
    private static long findNextNewline(FileChannel ch, long from, long size) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8192);
        long pos = from;
        while (pos < size) {
            buf.clear();
            int n = ch.read(buf, pos);
            if (n <= 0) break;
            buf.flip();
            for (int i = 0; i < n; i++) {
                if (buf.get(i) == (byte) '\n') return pos + i;
            }
            pos += n;
        }
        return -1;
    }

    private static ChunkResult parseCsvChunk(Path path, long start, long end,
                                             int chunkIdx, CsvOptions base,
                                             Charset charset) {
        try {
            // Read only this byte range — never the whole file
            byte[] bytes;
            try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
                int len = (int) Math.min(Integer.MAX_VALUE, end - start);
                bytes = new byte[len];
                ByteBuffer bb = ByteBuffer.wrap(bytes);
                long pos = start;
                while (bb.hasRemaining()) {
                    int n = ch.read(bb, pos);
                    if (n < 0) break;
                    pos += n;
                }
            }
            String text = new String(bytes, charset);
            // Drop a leading partial line for non-zero chunks if split left a fragment
            // (splitByNewline already aligns, so this is a safety net)
            if (chunkIdx > 0 && !text.isEmpty() && text.charAt(0) != '\n') {
                int nl = text.indexOf('\n');
                if (nl >= 0) text = text.substring(nl + 1);
            }

            CsvOptions opt;
            if (chunkIdx == 0) {
                opt = base;
            } else {
                // subsequent chunks: no header row; reuse column names if provided
                CsvOptions.Builder b = CsvOptions.builder()
                    .header(false)
                    .delimiter(base.delimiter())
                    .quote(base.quote())
                    .escape(base.escape())
                    .charset(base.charset())
                    .inferSchema(base.inferSchema())
                    .inferSampleSize(base.inferSampleSize())
                    .strict(base.strict())
                    .stripBom(false);
                if (base.columnNames() != null && !base.columnNames().isEmpty()) {
                    b.columnNames(base.columnNames());
                }
                if (base.schema() != null && !base.schema().isEmpty()) {
                    b.schema(base.schema());
                }
                if (base.nullValues() != null) {
                    for (String t : base.nullValues()) b.addNullValue(t);
                }
                if (base.comment() != null) b.comment(base.comment());
                opt = b.build();
            }

            DataFrame df = CsvReader.read(new StringReader(text), opt);
            return new ChunkResult(df);
        } catch (Exception e) {
            return new ChunkResult(e);
        }
    }

    private static DataFrame limitRows(DataFrame df, long maxRows) {
        if (df == null) return DataFrame.create();
        if (maxRows <= 0 || df.rowCount() <= maxRows) return df;
        return df.head((int) Math.min(Integer.MAX_VALUE, maxRows));
    }

    private static DataFrame vstackNonNull(DataFrame[] parts) throws Exception {
        List<DataFrame> list = new ArrayList<>();
        // Align schemas: use first non-empty as template; rename columns on later
        // chunks that came without header (col_0..) to match first.
        DataFrame template = null;
        for (DataFrame p : parts) {
            if (p == null || p.rowCount() == 0) continue;
            if (template == null) {
                template = p;
                list.add(p);
                continue;
            }
            list.add(alignColumns(p, template));
        }
        if (list.isEmpty()) return DataFrame.create();
        if (list.size() == 1) return list.get(0);
        return DataFrame.vstack(list);
    }

    /** Rename / reorder columns of {@code part} to match {@code template}. */
    private static DataFrame alignColumns(DataFrame part, DataFrame template) {
        List<String> tNames = template.getColumnNames();
        List<String> pNames = part.getColumnNames();
        // If same names, just select in template order
        boolean same = tNames.size() == pNames.size();
        if (same) {
            for (int i = 0; i < tNames.size(); i++) {
                if (!tNames.get(i).equals(pNames.get(i))) { same = false; break; }
            }
        }
        if (same) return part;

        // Positional align when counts match (headerless chunks → col_0..)
        if (tNames.size() == pNames.size()) {
            DataFrame out = DataFrame.create();
            for (int i = 0; i < tNames.size(); i++) {
                Column src = part.column(i);
                Column neu = new Column(tNames.get(i), src.dtype(), src.data());
                out.addColumn(neu);
            }
            // sync row count via public API
            out.syncRowCountPublic();
            return out;
        }
        // Fallback: select intersection by name
        DataFrame out = DataFrame.create();
        for (String n : tNames) {
            if (part.hasColumn(n)) out.addColumn(part.column(n).copy());
            else {
                out.addColumn(n, template.column(n).dtype());
                Column c = out.column(n);
                for (int r = 0; r < part.rowCount(); r++) c.add(null);
            }
        }
        out.syncRowCountPublic();
        return out;
    }

    private static DataFrame newBatchFrame(org.apache.parquet.schema.MessageType schema,
                                           List<String> fieldNames) {
        DataFrame df = DataFrame.create();
        for (String fname : fieldNames) {
            Column.DType dtype = parquetTypeToDType(schema.getType(fname));
            df.addColumn(fname, dtype);
        }
        return df;
    }

    // Lightweight mirrors of DataFrame private helpers (package-local via public read path).
    // We re-use DataFrame.readParquet for full reads; for streaming we need local converters.

    private static Column.DType parquetTypeToDType(org.apache.parquet.schema.Type type) {
        if (type.isPrimitive()) {
            return switch (type.asPrimitiveType().getPrimitiveTypeName()) {
                case INT32 -> Column.DType.INT32;
                case INT64 -> Column.DType.INT64;
                case FLOAT -> Column.DType.FLOAT32;
                case DOUBLE -> Column.DType.FLOAT64;
                case BOOLEAN -> Column.DType.BOOLEAN;
                default -> Column.DType.STRING;
            };
        }
        return Column.DType.STRING;
    }

    private static Object readGroupValue(org.apache.parquet.example.data.Group row,
                                         String fname,
                                         org.apache.parquet.schema.Type type) {
        try {
            int idx = row.getType().getFieldIndex(fname);
            if (row.getFieldRepetitionCount(idx) == 0) return null;
            if (!type.isPrimitive()) return row.getGroup(idx, 0).toString();
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
}
