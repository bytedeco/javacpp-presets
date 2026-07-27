package org.bytedeco.pytorch.data.dataframe;
import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.function.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.time.*;
import java.time.temporal.ChronoUnit;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.data.numpy.*;
import org.bytedeco.pytorch.data.pickle.Pickle;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.gguf.GGUFReader;
import org.bytedeco.pytorch.data.parquet.LocalParquetReader;
import org.bytedeco.pytorch.data.parquet.LocalParquetWriter;
import org.bytedeco.pytorch.data.dataframe.csv.CsvOptions;
import org.bytedeco.pytorch.data.dataframe.csv.CsvReader;
import org.bytedeco.pytorch.data.dataframe.csv.CsvWriter;
import org.bytedeco.pytorch.data.dataframe.json.JsonOptions;
import org.bytedeco.pytorch.data.dataframe.json.JsonReader;
import org.bytedeco.pytorch.data.dataframe.json.JsonWriter;
import org.bytedeco.pytorch.data.dataframe.io.FormatDetect;

/**
 * Pandas-like DataFrame for PyTorch with Polars-style lazy expressions.
 *
 * <p>Supports reading and writing: parquet, numpy/npz, pickle, safetensors,
 * GGUF, CSV/TSV, JSON/JSONL, Arrow IPC/Feather, Excel, SQL (SQLite/JDBC),
 * HDF5, Avro, ORC, and torch {@link Tensor} columns.
 *
 * <p>Example:
 * <pre>
 *   DataFrame df = DataFrame.readParquet("data.parquet");
 *   LazyDataFrame ldf = df.lazy();
 *   ldf.withColumn("x2", col("x").plus(lit(1)))
 *       .filter(col("x").gt(lit(0)))
 *       .sort(asc("x"), null)
 *       .collect();
 *   df.writeParquet("output.parquet");
 * </pre>
 */
public final class DataFrame implements AutoCloseable, Serializable {
    private static final long serialVersionUID = 1L;

    private final List<Column> columns;
    private final Map<String, Column> columnMap;
    private int rowCount;
    private final List<AutoCloseable> resources = new ArrayList<>();

    // ---- constructors ----

    public DataFrame() {
        this.columns = new ArrayList<>();
        this.columnMap = new LinkedHashMap<>();
        this.rowCount = 0;
    }

    public static DataFrame create() { return new DataFrame(); }

    // ---- column management ----

    public List<Column> columns() { return new ArrayList<>(columns); }
    public int columnCount() { return columns.size(); }
    public int rowCount() { return rowCount; }

    /** Spark-style row count alias of {@link #rowCount()}. */
    public long countRows() { return rowCount; }
    public int[] shape() { return new int[]{rowCount, columns.size()}; }

    public Column column(String name) {
        Column c = columnMap.get(name);
        if (c == null) throw new IllegalArgumentException("No such column: " + name);
        return c;
    }

    public Column column(int index) { return columns.get(index); }
    public boolean hasColumn(String name) { return columnMap.containsKey(name); }

    public void addColumn(String name, Column.DType dtype) {
        if (columnMap.containsKey(name)) throw new IllegalArgumentException("Column exists: " + name);
        Column col = new Column(name, dtype);
        // Pad to current rowCount so set(row, col, val) works on existing frames.
        for (int i = 0; i < rowCount; i++) col.add(null);
        columns.add(col);
        columnMap.put(name, col);
    }

    public void addColumn(Column col) {
        if (columnMap.containsKey(col.name()))
            throw new IllegalArgumentException("Column exists: " + col.name());
        // Align shorter columns to current rowCount (or grow rowCount via sync).
        while (col.size() < rowCount) col.add(null);
        columns.add(col);
        columnMap.put(col.name(), col);
        syncRowCount();
    }

    public Column removeColumn(String name) {
        Column col = columnMap.remove(name);
        if (col == null) throw new IllegalArgumentException("No such column: " + name);
        columns.remove(col);
        return col;
    }

    public void renameColumn(String oldName, String newName) {
        Column col = columnMap.remove(oldName);
        if (col == null) throw new IllegalArgumentException("No such column: " + oldName);
        Column renamed = new Column(newName, col.dtype(), col.data());
        columns.set(columns.indexOf(col), renamed);
        columnMap.put(newName, renamed);
    }

    // ---- Lazy API ----

    /** Returns a lazy view of this DataFrame. */
    public LazyDataFrame lazy() {
        return new LazyDataFrame(this);
    }

    // ---- row access ----

    public DataFrame loc(int[] rowIndices, String... colNames) {
        DataFrame result = DataFrame.create();
        List<Column> srcCols;
        if (colNames == null || colNames.length == 0) srcCols = columns;
        else {
            srcCols = new ArrayList<>();
            for (String n : colNames) srcCols.add(columnMap.get(n));
        }
        for (Column c : srcCols) {
            if (c == null) continue;
            result.addColumn(c.name(), c.dtype());
        }
        for (int ri : rowIndices) {
            if (ri < 0) ri = rowCount + ri;
            int resultRow = result.addRow();
            for (Column c : srcCols) {
                if (c == null) continue;
                result.set(resultRow, c.name(), c.get(ri));
            }
        }
        return result;
    }

    public DataFrame iloc(int startRow, int endRow, int startCol, int endCol) {
        DataFrame result = DataFrame.create();
        int actualStart = startRow < 0 ? rowCount + startRow : startRow;
        int actualEnd = endRow < 0 ? rowCount + endRow : endRow;
        int actualStartC = startCol < 0 ? columns.size() + startCol : startCol;
        int actualEndC = endCol < 0 ? columns.size() + endCol : endCol;
        actualStart = Math.max(0, actualStart);
        actualEnd = Math.min(rowCount, actualEnd);
        actualStartC = Math.max(0, actualStartC);
        actualEndC = Math.min(columns.size(), actualEndC);
        for (int ci = actualStartC; ci < actualEndC; ci++) {
            Column src = columns.get(ci);
            Column newCol = new Column(src.name(), src.dtype());
            for (int ri = actualStart; ri < actualEnd; ri++) newCol.add(src.get(ri));
            result.addColumn(newCol);
        }
        result.syncRowCount();
        return result;
    }

    public DataFrame iloc(int startRow, int endRow) { return iloc(startRow, endRow, 0, columns.size()); }
    public DataFrame iloc(int rowIndex) { return iloc(rowIndex, rowIndex + 1, 0, columns.size()); }
    public DataFrame head(int n) { return iloc(0, Math.min(n, rowCount)); }
    /** Daft {@code limit(n)} alias for {@link #head(int)}. */
    public DataFrame limit(int n) { return head(n); }
    public DataFrame tail(int n) { return iloc(Math.max(0, rowCount - n), rowCount); }
    /** Daft {@code where(condition)} alias for {@link #filter(Expression)}. */
    public DataFrame where(Expression condition) { return filter(condition); }

    // ---- cell access ----

    public Object get(int rowIndex, String colName) { return column(colName).get(rowIndex); }
    public void set(int rowIndex, String colName, Object value) { column(colName).set(rowIndex, value); }

    public int addCell(int rowIndex, String colName, Object value) {
        if (rowIndex < 0) {
            if (rowCount == 0) {
                for (Column c : columns) c.add(value);
                rowCount = 1;
                return 0;
            }
            for (Column c : columns) {
                if (c.name().equals(colName)) c.add(value);
                else c.add(null);
            }
            return rowCount++;
        }
        column(colName).set(rowIndex, value);
        return rowIndex;
    }

    public int addRow(Object... values) {
        // no-arg / empty varargs: append a row of nulls
        if (values == null || values.length == 0) {
            for (Column c : columns) c.add(null);
            return rowCount++;
        }
        if (values.length != columns.size())
            throw new IllegalArgumentException("Row size mismatch");
        for (int i = 0; i < columns.size(); i++) columns.get(i).add(values[i]);
        return rowCount++;
    }

    // ---- I/O: Parquet ----

    public static DataFrame readParquet(String path) throws Exception {
        DataFrame df = DataFrame.create();
        try (LocalParquetReader reader = LocalParquetReader.open(path)) {
            org.apache.parquet.schema.MessageType schema = reader.getSchema();
            List<String> fieldNames = reader.getFieldNames();
            for (String fname : fieldNames) {
                Column.DType dtype = parquetTypeToDType(schema.getType(fname));
                df.addColumn(fname, dtype);
            }
            for (org.apache.parquet.example.data.Group row = reader.read(); row != null; row = reader.read()) {
                int ri = df.addRow();
                for (int i = 0; i < df.columnCount(); i++) {
                    String fname = fieldNames.get(i);
                    Object val = readGroupValue(row, fname, schema.getType(fname));
                    df.set(ri, fname, val);
                }
            }
        }
        return df;
    }

    /**
     * Multi-worker Parquet read with optional row budget (heap-safe when combined with
     * {@link #streamParquet(String, int, java.util.function.Consumer)}).
     */
    public static DataFrame readParquet(String path, int workers) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParallelReader.readParquet(path, workers);
    }

    /**
     * Stream Parquet in fixed-size batches without materialising the whole file.
     * Peak heap ≈ one batch. Returns total rows delivered.
     */
    public static long streamParquet(String path, int batchRows,
                                     java.util.function.Consumer<DataFrame> consumer) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParallelReader.streamParquet(path, batchRows, consumer);
    }

    public void writeParquet(String path) throws Exception {
        org.apache.parquet.schema.MessageType schema = buildParquetSchema();
        // Dictionary off: nested LIST/VECTOR columns can trip parquet-column
        // "writing empty page" on flush when dictionary pages are empty.
        // Larger page size keeps fixed-size list elements in fewer pages.
        try (LocalParquetWriter w = LocalParquetWriter.builder(path, schema)
            .withCompression(org.apache.parquet.hadoop.metadata.CompressionCodecName.ZSTD)
            .withDictionary(false)
            .withPageSize(8 * 1024 * 1024)
            .build()) {
            for (int i = 0; i < rowCount; i++) {
                org.apache.parquet.example.data.simple.SimpleGroup g = w.makeGroup();
                for (int ci = 0; ci < columnCount(); ci++) {
                    Column col = columns.get(ci);
                    Object val = col.get(i);
                    writeGroupField(g, col.name(), col.dtype(), val);
                }
                w.write(g);
            }
        }
    }

    // ---- I/O: NumPy ----

    public static DataFrame readNpy(String path) throws Exception {
        NDArray arr = NP.load(path);
        Column.DType dtype = numpyDType(arr.dtype);
        Column col = new Column("data", dtype);
        long total = 1;
        for (long d : arr.shape) total *= d;
        if (NDArray.isFloatFamily(arr.dtype)) {
            for (int i = 0; i < total; i++) col.add(arr.getDouble(i));
        } else {
            for (int i = 0; i < total; i++) col.add(arr.getLong(i));
        }
        DataFrame df = DataFrame.create();
        df.addColumn(col);
        return df;
    }

    public void toNumpy(String path) throws Exception {
        Column col = findFirstNumeric();
        NDArray arr = columnToNDArray(col);
        NP.save(arr, path);
    }

    // ---- I/O: Pickle (pandas-compatible layouts) ----

    public static DataFrame readPickle(String path) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.pickle.PandasDataFramePickle.load(path);
    }

    public static DataFrame readPickle(String path,
            org.bytedeco.pytorch.data.dataframe.pickle.PickleOptions options) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.pickle.PandasDataFramePickle.load(path, options);
    }

    public void toPickle(String path) throws Exception {
        org.bytedeco.pytorch.data.dataframe.pickle.PandasDataFramePickle.dump(this, path);
    }

    public void toPickle(String path,
            org.bytedeco.pytorch.data.dataframe.pickle.PickleOptions options) throws Exception {
        org.bytedeco.pytorch.data.dataframe.pickle.PandasDataFramePickle.dump(this, path, options);
    }

    /** Legacy list-of-dicts pickle (explicit). */
    public void toPickleRecords(String path) throws Exception {
        toPickle(path, org.bytedeco.pytorch.data.dataframe.pickle.PickleOptions.records());
    }

    // ---- I/O: SafeTensors ----

    public static DataFrame readSafetensors(String path) throws Exception {
        DataFrame df = DataFrame.create();
        Map<String, Tensor> tensors = SafeTensors.loadAsTensors(new File(path), false);
        for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
            Column col = tensorToColumn(e.getKey(), e.getValue());
            df.addColumn(col);
        }
        return df;
    }

    public void toSafetensors(String path) throws Exception {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        for (Column col : columns) {
            if (!isNumeric(col.dtype())) continue;
            Tensor t = columnToTensor(col);
            tensors.put(col.name(), t);
        }
        SafeTensors.save(tensors, new File(path));
    }

    /** Daft {@code write_safetensors} alias for {@link #toSafetensors(String)}. */
    public void writeSafetensors(String path) throws Exception { toSafetensors(path); }

    /** Daft {@code write_parquet} alias for {@link #writeParquet(String)}. */
    public void write_parquet(String path) throws Exception { writeParquet(path); }

    // ---- I/O: multimodal (Daft-style + OpenCV/FFmpeg backends) ----

    /**
     * Batch-load images from a directory or glob-like path prefix
     * (Daft {@code read_images}).
     * <p>Uses {@link org.bytedeco.pytorch.data.dataframe.media.MediaBridge} —
     * OpenCV when available, pure-Java ImageIO otherwise.
     * Result columns: {@code path} (STRING), {@code image} (IMAGE),
     * plus {@code width}/{@code height}/{@code channels} metadata.
     */
    public static DataFrame readImages(String pathOrGlob) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readImages(pathOrGlob);
    }

    /** Batch-load images with explicit {@link org.bytedeco.pytorch.data.dataframe.media.MediaBridge.ImageOptions}. */
    public static DataFrame readImages(String pathOrGlob,
            org.bytedeco.pytorch.data.dataframe.media.MediaBridge.ImageOptions opts) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readImages(pathOrGlob, opts);
    }

    /**
     * Load audio files (Daft {@code read_audio}).
     * Columns: {@code path}, {@code audio}, plus sample_rate/channels/duration/num_samples.
     * FFmpeg preferred for compressed formats; pure-Java for WAV.
     */
    public static DataFrame readAudio(String pathOrGlob) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readAudio(pathOrGlob);
    }

    public static DataFrame readAudio(String pathOrGlob, int sampleRate, boolean mono) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readAudio(pathOrGlob, sampleRate, mono);
    }

    public static DataFrame readAudio(String pathOrGlob,
            org.bytedeco.pytorch.data.dataframe.media.MediaBridge.AudioOptions opts) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readAudio(pathOrGlob, opts, true);
    }

    /**
     * Load video files as frame sequences (Daft {@code read_video}).
     * Real FFmpeg decode via {@link org.bytedeco.pytorch.data.dataframe.media.MediaBridge} when available.
     * Columns: {@code path}, {@code video}, plus width/height/fps/duration/frame_count.
     */
    public static DataFrame readVideo(String pathOrGlob) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readVideo(pathOrGlob);
    }

    public static DataFrame readVideo(String pathOrGlob,
            org.bytedeco.pytorch.data.dataframe.media.MediaBridge.VideoOptions opts) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readVideo(pathOrGlob, opts);
    }

    /** torchvision ImageFolder layout: {@code root/class_x/*.{jpg,png,…}}. */
    public static DataFrame readImageFolder(String root) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readImageFolder(root);
    }

    /** torchaudio-style audio folder: {@code root/class_x/*.{wav,mp3,…}}. */
    public static DataFrame readAudioFolder(String root) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readAudioFolder(root);
    }

    /** torchtext-style text folder: {@code root/class_x/*.{txt,md,…}}. */
    public static DataFrame readTextFolder(String root) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readTextFolder(root);
    }

    /** Mixed image/audio/video/text directory → multimodal DataFrame. */
    public static DataFrame readMultimodal(String pathOrDir) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.readMultimodalDir(pathOrDir);
    }

    /**
     * Build an image DataFrame from in-memory {@link org.bytedeco.pytorch.data.dataframe.dtype.ImageData} cells.
     */
    public static DataFrame fromImages(String imageCol, List<org.bytedeco.pytorch.data.dataframe.dtype.ImageData> images) {
        DataFrame df = DataFrame.create();
        df.addColumn(imageCol == null ? "image" : imageCol, Column.DType.IMAGE);
        if (images != null) {
            for (org.bytedeco.pytorch.data.dataframe.dtype.ImageData img : images) {
                int ri = df.addEmptyRow();
                df.set(ri, imageCol == null ? "image" : imageCol, img);
            }
        }
        return df;
    }

    /** Build image DataFrame from OpenCV/torchvision CHW tensors. */
    public static DataFrame fromOpenCV(String imageCol, List<org.bytedeco.pytorch.Tensor> tensors) {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.fromOpenCVTensors(imageCol, tensors);
    }

    /** Build frame DataFrame from FFmpeg-decoded frame tensors. */
    public static DataFrame fromFFmpegFrames(List<org.bytedeco.pytorch.Tensor> frames, double fps) {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.fromFFmpegFrames("frame", frames, fps);
    }

    /** Build audio DataFrame from torchaudio-style [C,T] waveforms. */
    public static DataFrame fromAudioTensors(List<org.bytedeco.pytorch.Tensor> waves, int sampleRate) {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.fromAudioTensors("audio", waves, sampleRate);
    }

    /** Explode video column into per-frame image rows. */
    public DataFrame extractVideoFrames(String videoCol, double fps) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.extractVideoFrames(this, videoCol, fps);
    }

    /** Batch-embed image column → embedding column. */
    public DataFrame embedImages(String imageCol, String outCol, int dim) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.embedImages(this, imageCol, outCol, dim);
    }

    /** Batch-embed audio column → embedding column. */
    public DataFrame embedAudioCol(String audioCol, String outCol, int dim) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.embedAudio(this, audioCol, outCol, dim);
    }

    /** Batch-embed video column → embedding column. */
    public DataFrame embedVideos(String videoCol, String outCol, int dim) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MultimodalIO.embedVideo(this, videoCol, outCol, dim);
    }

    /** Stack image column as NCHW float tensor in [0,1] (torchvision batch). */
    public org.bytedeco.pytorch.Tensor toVisionBatch(String imageCol) {
        return org.bytedeco.pytorch.data.dataframe.media.MediaInterop.toVisionBatch(this, imageCol);
    }

    /** Apply torchvision-style transform to image column. */
    public DataFrame transformImages(String imageCol, Object transform) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.media.MediaInterop.applyVisionTransform(this, imageCol, transform);
    }

    /**
     * Build a DataFrame from embedding vectors (float[] per row).
     * Columns: {@code embedding} (EMBEDDING) plus optional id.
     */
    public static DataFrame fromEmbeddings(String embCol, float[][] vectors, String modelName) {
        DataFrame df = DataFrame.create();
        String col = embCol == null ? "embedding" : embCol;
        df.addColumn(col, Column.DType.EMBEDDING);
        if (vectors != null) {
            for (float[] v : vectors) {
                int ri = df.addEmptyRow();
                df.set(ri, col, new org.bytedeco.pytorch.data.dataframe.dtype.EmbeddingData(
                    v, modelName == null ? "unknown" : modelName));
            }
        }
        return df;
    }

    // ---- I/O: Lance (official org.lance:lance-core by default) ----

    /**
     * Write this DataFrame as an <b>official</b> Lance dataset
     * ({@code org.lance:lance-core} via Arrow IPC). Overwrites if the path exists.
     *
     * <p>For the pure-Java training layout ({@code _manifest.json} + local HNSW),
     * use {@link #writeLanceTraining(String, String...)}.
     *
     * @param path dataset directory (created if missing)
     */
    public void writeLance(String path) throws Exception {
        org.bytedeco.pytorch.utils.lance.Lance.writeDataFrame(this, path);
    }

    /**
     * Write as official Lance with {@link org.bytedeco.pytorch.utils.lance.LanceWriteOptions}
     * (mode, maxRowsPerFile, stableRowIds, …).
     */
    public void writeLance(String path,
                           org.bytedeco.pytorch.utils.lance.LanceWriteOptions options) throws Exception {
        org.bytedeco.pytorch.utils.lance.Lance.writeDataFrame(this, path, options);
    }

    /**
     * Pure-Java training layout (Daft-style {@code write_lance}): dense f32 vector matrices
     * + optional local HNSW. <b>Not</b> byte-compatible with lance-rs / official Lance.
     *
     * @param path       dataset directory (created if missing)
     * @param vectorCols explicit vector column names; empty → auto-detect EMBEDDING/VECTOR/TENSOR
     */
    public void writeLanceTraining(String path, String... vectorCols) throws Exception {
        org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.write(this, path, vectorCols);
    }

    public void writeLanceTraining(String path,
                                   org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.WriteOptions opts,
                                   String... vectorCols) throws Exception {
        org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.write(
            this, java.nio.file.Path.of(path), opts, vectorCols);
    }

    /**
     * @deprecated Use {@link #writeLance(String)} (now official) or
     * {@link #writeLanceTraining(String, String...)} for the pure-Java layout.
     * Kept for binary compatibility: when {@code vectorCols} is empty this delegates to
     * official write; otherwise to training write.
     */
    @Deprecated
    public void writeLance(String path, String... vectorCols) throws Exception {
        if (vectorCols == null || vectorCols.length == 0) {
            writeLance(path);
        } else {
            writeLanceTraining(path, vectorCols);
        }
    }

    /**
     * @deprecated Use {@link #writeLanceTraining(String, org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.WriteOptions, String...)}.
     */
    @Deprecated
    public void writeLance(String path,
                           org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.WriteOptions opts,
                           String... vectorCols) throws Exception {
        writeLanceTraining(path, opts, vectorCols);
    }

    /** Alias of {@link #writeLance(String)} — official Lance. */
    public void writeLanceOfficial(String path) throws Exception {
        writeLance(path);
    }

    /**
     * Read a Lance dataset. Auto-detects pure-Java training layout
     * ({@code _manifest.json}); otherwise uses official {@code org.lance:lance-core}.
     *
     * @see org.bytedeco.pytorch.utils.lance.Lance#readAuto(String)
     */
    public static DataFrame readLance(String path) throws Exception {
        return org.bytedeco.pytorch.utils.lance.Lance.readAuto(path);
    }

    /**
     * Read official Lance with column projection / filter / limit / version.
     * Falls back to full pure-Java read when {@code _manifest.json} is present
     * (options other than path are ignored for the training layout).
     */
    public static DataFrame readLance(String path,
                                      org.bytedeco.pytorch.utils.lance.LanceReadOptions options)
            throws Exception {
        return org.bytedeco.pytorch.utils.lance.Lance.readAuto(path, options);
    }

    /** Force pure-Java training layout read. */
    public static DataFrame readLanceTraining(String path) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.read(path);
    }

    /** @deprecated Prefer {@link #readLance(String)} which already auto-detects. */
    @Deprecated
    public static DataFrame readLanceAuto(String path) throws Exception {
        return readLance(path);
    }

    /** Alias of {@link #readLance(String)} for official-only callers; still auto-falls-back. */
    public static DataFrame readLanceOfficial(String path) throws Exception {
        return org.bytedeco.pytorch.utils.lance.Lance.readDataFrame(path);
    }

    /**
     * Open an official native Lance dataset for ANN search, indexing, versioning, etc.
     *
     * @see org.bytedeco.pytorch.utils.lance.Lance
     */
    public static org.bytedeco.pytorch.utils.lance.Lance openLance(String path) throws Exception {
        return org.bytedeco.pytorch.utils.lance.Lance.open(path);
    }

    public static org.bytedeco.pytorch.utils.lance.Lance openLance(
            String path, org.bytedeco.pytorch.utils.lance.LanceReadOptions options) throws Exception {
        return org.bytedeco.pytorch.utils.lance.Lance.open(path, options);
    }

    /** Open pure-Java training dataset (local HNSW, not official format). */
    public static org.bytedeco.pytorch.data.dataframe.lance.LanceDataset openLanceTraining(String path)
            throws Exception {
        return org.bytedeco.pytorch.data.dataframe.lance.LanceDataset.open(path);
    }

    /** @deprecated Use {@link #openLance(String)} (official). */
    @Deprecated
    public static org.bytedeco.pytorch.utils.lance.Lance openLanceOfficial(String path) throws Exception {
        return openLance(path);
    }

    // ---- I/O: DuckDB (official org.duckdb:duckdb_jdbc) ----

    /**
     * Scan a file with embedded DuckDB ({@code read_parquet} / {@code read_csv_auto} /
     * {@code read_json_auto} / {@code read_orc} auto-detected by extension).
     *
     * @see #readDuckDBParquet(String)
     * @see #readDuckDBCsv(String)
     * @see #readDuckDBJson(String)
     */
    public static DataFrame readDuckDB(String path) throws Exception {
        try (org.bytedeco.pytorch.utils.duckdb.DuckDB db =
                     org.bytedeco.pytorch.utils.duckdb.DuckDB.inMemory()) {
            var fmt = org.bytedeco.pytorch.utils.duckdb.DuckDB.detectFormat(path);
            return db.query("SELECT * FROM " + fmt.tableFunction(path));
        }
    }

    /** DuckDB {@code read_parquet(path)} → DataFrame. */
    public static DataFrame readDuckDBParquet(String path) throws Exception {
        return org.bytedeco.pytorch.utils.duckdb.DuckDB.scanParquet(path);
    }

    /** DuckDB {@code read_csv_auto(path)} → DataFrame. */
    public static DataFrame readDuckDBCsv(String path) throws Exception {
        return org.bytedeco.pytorch.utils.duckdb.DuckDB.scanCsv(path);
    }

    /** DuckDB {@code read_json_auto(path)} → DataFrame. */
    public static DataFrame readDuckDBJson(String path) throws Exception {
        return org.bytedeco.pytorch.utils.duckdb.DuckDB.scanJson(path);
    }

    /** DuckDB {@code read_orc(path)} → DataFrame. */
    public static DataFrame readDuckDBOrc(String path) throws Exception {
        return org.bytedeco.pytorch.utils.duckdb.DuckDB.scanOrc(path);
    }

    /** Run arbitrary DuckDB SQL in an in-memory session → DataFrame. */
    public static DataFrame readDuckDBSql(String sql) throws Exception {
        return org.bytedeco.pytorch.utils.duckdb.DuckDB.scanSql(sql);
    }

    /**
     * Open a persistent DuckDB database file and read a table (or run SQL).
     * If {@code tableOrSql} looks like SQL ({@code SELECT}/{@code WITH}/…), it is executed
     * as a query; otherwise it is treated as a table name.
     */
    public static DataFrame readDuckDB(java.nio.file.Path dbFile, String tableOrSql) throws Exception {
        try (org.bytedeco.pytorch.utils.duckdb.DuckDB db =
                     org.bytedeco.pytorch.utils.duckdb.DuckDB.open(dbFile)) {
            String s = tableOrSql == null ? "" : tableOrSql.trim();
            String lower = s.toLowerCase(java.util.Locale.ROOT);
            if (lower.startsWith("select") || lower.startsWith("with") || lower.startsWith("from")
                    || lower.startsWith("describe") || lower.startsWith("show")
                    || lower.startsWith("summarize") || s.contains(" ")) {
                return db.query(s);
            }
            return db.tableToDataFrame(s);
        }
    }

    /** Read a named table from a persistent DuckDB database file. */
    public static DataFrame readDuckDBTable(String dbPath, String table) throws Exception {
        return readDuckDB(java.nio.file.Path.of(dbPath), table);
    }

    /** Export this DataFrame to Parquet via DuckDB {@code COPY ... TO}. */
    public void writeDuckDBParquet(String path) throws Exception {
        try (org.bytedeco.pytorch.utils.duckdb.DuckDB db =
                     org.bytedeco.pytorch.utils.duckdb.DuckDB.inMemory()) {
            db.exportParquet(this, path);
        }
    }

    /** Export this DataFrame to CSV via DuckDB {@code COPY ... TO (FORMAT CSV)}. */
    public void writeDuckDBCsv(String path) throws Exception {
        try (org.bytedeco.pytorch.utils.duckdb.DuckDB db =
                     org.bytedeco.pytorch.utils.duckdb.DuckDB.inMemory()) {
            db.exportCsv(this, path);
        }
    }

    /** Export this DataFrame to JSON via DuckDB {@code COPY ... TO (FORMAT JSON)}. */
    public void writeDuckDBJson(String path) throws Exception {
        try (org.bytedeco.pytorch.utils.duckdb.DuckDB db =
                     org.bytedeco.pytorch.utils.duckdb.DuckDB.inMemory()) {
            db.exportJson(this, path);
        }
    }

    /**
     * Persist this DataFrame as a table inside a DuckDB database file
     * (creates the file / parent dirs if missing). Default {@code if_exists=REPLACE}.
     */
    public void toDuckDB(String dbPath, String table) throws Exception {
        toDuckDB(java.nio.file.Path.of(dbPath), table,
                org.bytedeco.pytorch.data.dataframe.sql.SqlOptions.builder()
                        .ifExists(org.bytedeco.pytorch.data.dataframe.sql.SqlOptions.IfExists.REPLACE)
                        .build());
    }

    /**
     * Persist this DataFrame as a table inside a DuckDB database file with write options.
     */
    public void toDuckDB(java.nio.file.Path dbFile, String table,
                         org.bytedeco.pytorch.data.dataframe.sql.SqlOptions options) throws Exception {
        try (org.bytedeco.pytorch.utils.duckdb.DuckDB db =
                     org.bytedeco.pytorch.utils.duckdb.DuckDB.open(dbFile)) {
            if (options == null) {
                db.register(table, this);
            } else {
                toSql(db.connection(), table, options);
            }
        }
    }

    /** Alias of {@link #toDuckDB(String, String)}. */
    public void writeDuckDB(String dbPath, String table) throws Exception {
        toDuckDB(dbPath, table);
    }

    /** Register this DataFrame as a DuckDB table and run {@code sql} (may reference table name). */
    public DataFrame duckDBQuery(String tableName, String sql) throws Exception {
        try (org.bytedeco.pytorch.utils.duckdb.DuckDB db =
                     org.bytedeco.pytorch.utils.duckdb.DuckDB.inMemory()) {
            db.register(tableName, this);
            return db.query(sql);
        }
    }

    /**
     * Open a short-lived DuckDB handle. Caller owns the resource and must {@code close()}.
     *
     * @see org.bytedeco.pytorch.utils.duckdb.DuckDB#inMemory()
     * @see org.bytedeco.pytorch.utils.duckdb.DuckDB#open(java.nio.file.Path)
     */
    public static org.bytedeco.pytorch.utils.duckdb.DuckDB openDuckDB() throws Exception {
        return org.bytedeco.pytorch.utils.duckdb.DuckDB.inMemory();
    }

    public static org.bytedeco.pytorch.utils.duckdb.DuckDB openDuckDB(String dbPath) throws Exception {
        return org.bytedeco.pytorch.utils.duckdb.DuckDB.open(java.nio.file.Path.of(dbPath));
    }

    public static org.bytedeco.pytorch.utils.duckdb.DuckDB openDuckDB(java.nio.file.Path dbFile)
            throws Exception {
        return org.bytedeco.pytorch.utils.duckdb.DuckDB.open(dbFile);
    }

    // ---- AI batch embedding façade ----

    /**
     * Batch-embed multimodal columns (Daft {@code functions.ai.embed_*}).
     * <pre>
     *   df.embed().model("clip-vit-base-patch32")
     *     .imageColumn("image", "image_emb")
     *     .textColumn("caption", "text_emb")
     *     .transform();
     * </pre>
     */
    public org.bytedeco.pytorch.data.dataframe.ai.BatchEmbedder embed() {
        return org.bytedeco.pytorch.data.dataframe.ai.BatchEmbedder.create();
    }

    /** Embed all detectable modality columns with one model. */
    public DataFrame embedAll(String modelId) {
        return org.bytedeco.pytorch.data.dataframe.ai.BatchEmbedder.embedAll(this, modelId);
    }

    /** Download URL/path column values as binary (Daft {@code col.url.download()} table helper). */
    public DataFrame download(String urlCol, String outCol) throws Exception {
        Column src = column(urlCol);
        List<Object> data = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) {
            Object v = src.get(i);
            if (v == null) { data.add(null); continue; }
            String pathOrUrl = v.toString();
            try {
                byte[] bytes;
                if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
                    try (InputStream in = new java.net.URI(pathOrUrl).toURL().openStream()) {
                        bytes = in.readAllBytes();
                    }
                } else {
                    bytes = Files.readAllBytes(Path.of(pathOrUrl));
                }
                data.add(new org.bytedeco.pytorch.data.dataframe.dtype.BinaryData("download", bytes));
            } catch (Exception e) {
                data.add(null);
            }
        }
        return withColumn(outCol, data);
    }

    private static List<Path> expandMediaPaths(String pathOrGlob, String... extensions) throws Exception {
        List<Path> out = new ArrayList<>();
        if (pathOrGlob == null || pathOrGlob.isBlank()) return out;
        Set<String> exts = new HashSet<>();
        for (String e : extensions) exts.add(e.toLowerCase(Locale.ROOT));

        // comma-separated list
        String[] parts = pathOrGlob.split(",");
        for (String part : parts) {
            String p = part.trim();
            if (p.isEmpty()) continue;
            Path path = Path.of(p);
            if (Files.isRegularFile(path)) {
                out.add(path);
            } else if (Files.isDirectory(path)) {
                try (var stream = Files.list(path)) {
                    stream.filter(Files::isRegularFile)
                        .filter(f -> {
                            String name = f.getFileName().toString().toLowerCase(Locale.ROOT);
                            for (String e : exts) if (name.endsWith(e)) return true;
                            return false;
                        })
                        .sorted()
                        .forEach(out::add);
                }
            } else {
                // glob relative to parent or cwd
                Path parent = path.getParent() != null ? path.getParent() : Path.of(".");
                String pattern = path.getFileName() != null ? path.getFileName().toString() : "*";
                if (Files.isDirectory(parent)) {
                    try (var stream = Files.newDirectoryStream(parent, pattern)) {
                        for (Path f : stream) {
                            if (Files.isRegularFile(f)) out.add(f);
                        }
                    } catch (Exception ignored) {}
                }
            }
        }
        return out;
    }

    // ---- I/O: GGUF ----

    public static DataFrame readGguf(String path) throws Exception {
        DataFrame df = DataFrame.create();
        try (GGUFReader reader = new GGUFReader(new File(path))) {
            for (Map.Entry<String, Object> me : reader.metadata().entrySet()) {
                df.addColumn(me.getKey(), inferDType(me.getValue()));
                int ri = df.addRow();
                df.set(ri, me.getKey(), me.getValue());
            }
            for (Map.Entry<String, GGUFReader.TensorInfo> te : reader.tensorInfos().entrySet()) {
                Tensor t = reader.loadTensor(te.getKey());
                Column col = tensorToColumn(te.getKey(), t);
                df.addColumn(col);
            }
        }
        return df;
    }

    // ---- I/O: CSV ----

    public static DataFrame readCsv(String path) throws Exception {
        return CsvReader.read(path, CsvOptions.defaults());
    }

    public static DataFrame readCsv(String path, boolean hasHeader, char delimiter) throws Exception {
        return CsvReader.read(path, CsvOptions.builder().header(hasHeader).delimiter(delimiter).build());
    }

    public static DataFrame readCsv(String path, CsvOptions options) throws Exception {
        return CsvReader.read(path, options);
    }

    public static DataFrame readCsv(java.nio.file.Path path, CsvOptions options) throws Exception {
        return CsvReader.read(path, options);
    }

    public static DataFrame readCsv(java.io.Reader reader, CsvOptions options) throws Exception {
        return CsvReader.read(reader, options);
    }

    public static DataFrame readCsv(java.io.InputStream in, CsvOptions options) throws Exception {
        return CsvReader.read(in, options);
    }

    /**
     * Multi-worker CSV read. Splits the file on newline boundaries across
     * {@code workers} threads; never loads the whole file as one String.
     * Peak heap ≈ {@code workers × chunkBytes} parse buffers + merged frame.
     * <pre>
     *   DataFrame df = DataFrame.readCsvParallel("big.csv", 8);
     * </pre>
     */
    public static DataFrame readCsvParallel(String path, int workers) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParallelReader.readCsv(path, workers);
    }

    public static DataFrame readCsvParallel(String path) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParallelReader.readCsv(path);
    }

    public static DataFrame readCsvParallel(String path,
                                            org.bytedeco.pytorch.data.dataframe.io.ParallelReader.Options opt)
            throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParallelReader.readCsv(path, opt);
    }

    /**
     * Stream CSV in chunks with multi-worker parse. Consumer receives each chunk;
     * prior chunks are not retained — use this for multi-GB files that must not
     * blow the JVM heap.
     * @return total rows delivered
     */
    public static long streamCsv(String path, int workers, long maxChunkBytes,
                                 java.util.function.Consumer<DataFrame> consumer) throws Exception {
        var opt = org.bytedeco.pytorch.data.dataframe.io.ParallelReader.Options.defaults()
            .workers(workers)
            .maxChunkBytes(maxChunkBytes);
        return org.bytedeco.pytorch.data.dataframe.io.ParallelReader.streamCsv(path, opt, consumer);
    }

    public static long streamCsv(String path, java.util.function.Consumer<DataFrame> consumer)
            throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParallelReader.streamCsv(
            path, org.bytedeco.pytorch.data.dataframe.io.ParallelReader.Options.defaults(), consumer);
    }

    public void toCsv(String path) throws Exception {
        CsvWriter.write(this, path, CsvOptions.defaults());
    }

    public void toCsv(String path, boolean includeHeader, char delimiter) throws Exception {
        CsvWriter.write(this, path, CsvOptions.builder().header(includeHeader).delimiter(delimiter).build());
    }

    public void toCsv(String path, CsvOptions options) throws Exception {
        CsvWriter.write(this, path, options);
    }

    public void toCsv(java.nio.file.Path path, CsvOptions options) throws Exception {
        CsvWriter.write(this, path, options);
    }

    public void toCsv(java.io.Writer writer, CsvOptions options) throws Exception {
        CsvWriter.write(this, writer, options);
    }

    // ---- I/O: JSON / JSONL ----

    /**
     * Read a JSON file into a DataFrame (records orient by default; auto-detects JSONL by extension).
     * <pre>
     *   DataFrame df = DataFrame.readJson("data.json");
     *   DataFrame df2 = DataFrame.readJson("rows.jsonl", JsonOptions.lines());
     * </pre>
     */
    public static DataFrame readJson(String path) throws Exception {
        return JsonReader.read(path, JsonOptions.defaults());
    }

    public static DataFrame readJson(String path, JsonOptions options) throws Exception {
        return JsonReader.read(path, options);
    }

    public static DataFrame readJson(java.nio.file.Path path, JsonOptions options) throws Exception {
        return JsonReader.read(path, options);
    }

    public static DataFrame readJson(java.io.Reader reader, JsonOptions options) throws Exception {
        return JsonReader.read(reader, options);
    }

    public static DataFrame readJson(java.io.InputStream in, JsonOptions options) throws Exception {
        return JsonReader.read(in, options);
    }

    /** Parse an in-memory JSON string into a DataFrame. */
    public static DataFrame readJsonString(String json) {
        return JsonReader.readString(json, JsonOptions.defaults());
    }

    public static DataFrame readJsonString(String json, JsonOptions options) {
        return JsonReader.readString(json, options);
    }

    /** Read JSON Lines / NDJSON (one JSON object per line). */
    public static DataFrame readJsonl(String path) throws Exception {
        return JsonReader.readJsonl(path, JsonOptions.lines());
    }

    public static DataFrame readJsonl(String path, JsonOptions options) throws Exception {
        return JsonReader.readJsonl(path, options == null ? JsonOptions.lines() : options);
    }

    public static DataFrame readJsonl(java.nio.file.Path path, JsonOptions options) throws Exception {
        return JsonReader.readJsonl(path, options == null ? JsonOptions.lines() : options);
    }

    public static DataFrame readJsonl(java.io.Reader reader, JsonOptions options) throws Exception {
        return JsonReader.readJsonl(reader, options == null ? JsonOptions.lines() : options);
    }

    /** Write this DataFrame as JSON (records orient by default). */
    public void toJson(String path) throws Exception {
        JsonWriter.write(this, path, JsonOptions.defaults());
    }

    public void toJson(String path, JsonOptions options) throws Exception {
        JsonWriter.write(this, path, options);
    }

    public void toJson(java.nio.file.Path path, JsonOptions options) throws Exception {
        JsonWriter.write(this, path, options);
    }

    public void toJson(java.io.Writer writer, JsonOptions options) throws Exception {
        JsonWriter.write(this, writer, options);
    }

    /** Serialize this DataFrame to a JSON string. */
    public String toJsonString() {
        return JsonWriter.toString(this, JsonOptions.defaults());
    }

    public String toJsonString(JsonOptions options) {
        return JsonWriter.toString(this, options);
    }

    /** Write this DataFrame as JSON Lines / NDJSON. */
    public void toJsonl(String path) throws Exception {
        JsonWriter.writeJsonl(this, path);
    }

    public void toJsonl(String path, JsonOptions options) throws Exception {
        JsonWriter.writeJsonl(this, java.nio.file.Path.of(path),
            options == null ? JsonOptions.lines() : options);
    }

    public void toJsonl(java.nio.file.Path path, JsonOptions options) throws Exception {
        JsonWriter.writeJsonl(this, path, options == null ? JsonOptions.lines() : options);
    }

    // ---- I/O: Arrow IPC (Feather v2) ----

    /** Read an Arrow IPC / Feather v2 file into a DataFrame. */
    public static DataFrame readArrow(String path) throws Exception {
        return org.bytedeco.pytorch.data.arrow.LocalArrowIpcReader.read(path);
    }

    /** Alias for {@link #readArrow(String)}. */
    public static DataFrame readIpc(String path) throws Exception {
        return readArrow(path);
    }

    /** Write this DataFrame as an Arrow IPC / Feather v2 file. */
    public void writeArrow(String path) throws Exception {
        org.bytedeco.pytorch.data.arrow.LocalArrowIpcWriter.write(this, path);
    }

    /** Alias for {@link #writeArrow(String)}. */
    public void writeIpc(String path) throws Exception {
        writeArrow(path);
    }

    /** Feather v2 alias for {@link #readArrow(String)} (Arrow IPC file format). */
    public static DataFrame readFeather(String path) throws Exception {
        return readArrow(path);
    }

    /** Feather v2 alias for {@link #writeArrow(String)}. */
    public void toFeather(String path) throws Exception {
        writeArrow(path);
    }

    /** Alias for {@link #toFeather(String)}. */
    public void writeFeather(String path) throws Exception {
        toFeather(path);
    }

    // ---- I/O: TSV ----

    /** Read a TSV file (tab-delimited, pandas/IMDb-style {@code \\N} nulls). */
    public static DataFrame readTsv(String path) throws Exception {
        return CsvReader.read(path, CsvOptions.tsv());
    }

    public static DataFrame readTsv(String path, CsvOptions options) throws Exception {
        CsvOptions opt = options == null ? CsvOptions.tsv()
            : CsvOptions.builder()
                .header(options.header())
                .delimiter(options.delimiter() == ',' ? '\t' : options.delimiter())
                .quote(options.quote())
                .escape(options.escape())
                .charset(options.charset())
                .nullValues(options.nullValues().toArray(new String[0]))
                .comment(options.comment())
                .skipRows(options.skipRows())
                .maxRows(options.maxRows())
                .inferSchema(options.inferSchema())
                .inferSampleSize(options.inferSampleSize())
                .strict(options.strict())
                .typeHeader(options.typeHeader())
                .quoteMode(options.quoteMode())
                .writeNullToken(options.writeNullToken())
                .stripBom(options.stripBom())
                .columnNames(options.columnNames())
                .schema(options.schema())
                .build();
        // Force tab if caller passed CSV defaults by mistake
        if (opt.delimiter() != '\t') {
            opt = CsvOptions.builder()
                .header(opt.header())
                .delimiter('\t')
                .quote(opt.quote())
                .escape(opt.escape())
                .charset(opt.charset())
                .nullValues(opt.nullValues().toArray(new String[0]))
                .comment(opt.comment())
                .skipRows(opt.skipRows())
                .maxRows(opt.maxRows())
                .inferSchema(opt.inferSchema())
                .inferSampleSize(opt.inferSampleSize())
                .strict(opt.strict())
                .typeHeader(opt.typeHeader())
                .quoteMode(opt.quoteMode())
                .writeNullToken(opt.writeNullToken())
                .stripBom(opt.stripBom())
                .columnNames(opt.columnNames())
                .schema(opt.schema())
                .build();
        }
        return CsvReader.read(path, opt);
    }

    public static DataFrame readTsv(java.nio.file.Path path, CsvOptions options) throws Exception {
        return readTsv(path.toString(), options);
    }

    /** Write this DataFrame as TSV (tab-delimited). */
    public void toTsv(String path) throws Exception {
        CsvWriter.write(this, path, CsvOptions.tsv());
    }

    public void toTsv(String path, CsvOptions options) throws Exception {
        CsvOptions opt = options == null ? CsvOptions.tsv() : options;
        if (opt.delimiter() != '\t') {
            opt = CsvOptions.builder()
                .header(opt.header())
                .delimiter('\t')
                .quote(opt.quote())
                .escape(opt.escape())
                .charset(opt.charset())
                .nullValues(opt.nullValues().toArray(new String[0]))
                .comment(opt.comment())
                .skipRows(opt.skipRows())
                .maxRows(opt.maxRows())
                .inferSchema(opt.inferSchema())
                .inferSampleSize(opt.inferSampleSize())
                .strict(opt.strict())
                .typeHeader(opt.typeHeader())
                .quoteMode(opt.quoteMode())
                .writeNullToken(opt.writeNullToken().isEmpty() ? "\\N" : opt.writeNullToken())
                .stripBom(opt.stripBom())
                .columnNames(opt.columnNames())
                .schema(opt.schema())
                .build();
        }
        CsvWriter.write(this, path, opt);
    }

    public void toTsv(java.nio.file.Path path, CsvOptions options) throws Exception {
        toTsv(path.toString(), options);
    }

    // ---- I/O: NPZ (NumPy zip archive) ----

    /**
     * Read a {@code .npz} archive into a DataFrame.
     * <ul>
     *   <li>Multiple 1D arrays of equal length → one column per array</li>
     *   <li>Single 2D array → columns {@code col_0..} (or use {@link #readNpz(String, String)})</li>
     *   <li>Single 1D array → column named after the array key</li>
     * </ul>
     */
    public static DataFrame readNpz(String path) throws Exception {
        Map<String, NDArray> arrays = NP.loadz(path);
        if (arrays.isEmpty()) return DataFrame.create();

        // Prefer multi 1D equal-length columns
        boolean all1d = true;
        long len = -1;
        for (NDArray a : arrays.values()) {
            if (a.shape.length != 1) { all1d = false; break; }
            if (len < 0) len = a.shape[0];
            else if (a.shape[0] != len) { all1d = false; break; }
        }
        if (all1d && arrays.size() >= 1) {
            DataFrame df = DataFrame.create();
            for (Map.Entry<String, NDArray> e : arrays.entrySet()) {
                df.addColumn(ndarrayToColumn(e.getKey(), e.getValue()));
            }
            return df;
        }

        // Single 2D → matrix columns
        if (arrays.size() == 1) {
            Map.Entry<String, NDArray> only = arrays.entrySet().iterator().next();
            return ndarrayToDataFrame(only.getKey(), only.getValue());
        }

        // Mixed: expose each array as a TENSOR column of length 1 (metadata style)
        DataFrame df = DataFrame.create();
        df.addColumn("key", Column.DType.STRING);
        df.addColumn("shape", Column.DType.STRING);
        df.addColumn("dtype", Column.DType.STRING);
        df.addColumn("size", Column.DType.INT64);
        for (Map.Entry<String, NDArray> e : arrays.entrySet()) {
            NDArray a = e.getValue();
            int ri = df.addEmptyRow();
            df.set(ri, "key", e.getKey());
            df.set(ri, "shape", Arrays.toString(a.shape));
            df.set(ri, "dtype", a.dtype.getDescriptor());
            df.set(ri, "size", a.size);
        }
        return df;
    }

    /** Read a single named array from a {@code .npz} file into a DataFrame. */
    public static DataFrame readNpz(String path, String key) throws Exception {
        Map<String, NDArray> arrays = NP.loadz(path);
        NDArray a = arrays.get(key);
        if (a == null) {
            // try with/without .npy suffix semantics
            a = arrays.get(key.endsWith(".npy") ? key.substring(0, key.length() - 4) : key + ".npy");
        }
        if (a == null) {
            throw new IllegalArgumentException("NPZ key not found: " + key + " in " + arrays.keySet());
        }
        return ndarrayToDataFrame(key, a);
    }

    /** Write numeric columns as 1D arrays into a {@code .npz} archive. */
    public void toNpz(String path) throws Exception {
        Map<String, NDArray> arrays = new LinkedHashMap<>();
        for (Column col : columns) {
            if (!isNumeric(col.dtype()) && col.dtype() != Column.DType.BOOLEAN) continue;
            arrays.put(col.name(), columnToNDArray(col));
        }
        if (arrays.isEmpty()) {
            throw new IllegalStateException("toNpz requires at least one numeric/boolean column");
        }
        NP.savez(path, arrays);
    }

    public void writeNpz(String path) throws Exception { toNpz(path); }

    // ---- I/O: auto-detect ----

    /**
     * Read a DataFrame by file extension (csv/tsv/json/parquet/arrow/feather/pkl/…),
     * with magic-byte fallback when the extension is missing or wrong.
     * @see FormatDetect#read(String)
     * @see org.bytedeco.pytorch.data.dataframe.io.SchemaInfer
     */
    public static DataFrame read(String path) throws Exception {
        return FormatDetect.read(path);
    }

    /**
     * Infer column schema for a dataset path without necessarily loading all rows
     * (parquet reads footer only; other formats may load once).
     */
    public static Schema inferSchema(String path) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.SchemaInfer.infer(path);
    }

    /** Print inferred schema for a path (Spark-style). */
    public static void printSchema(String path) throws Exception {
        org.bytedeco.pytorch.data.dataframe.io.SchemaInfer.print(inferSchema(path));
    }

    // ---- I/O: Excel / SQL / HDF5 / Avro / ORC (delegating packages) ----

    public static DataFrame readExcel(String path) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.excel.ExcelReader.read(path);
    }

    public static DataFrame readExcel(String path,
            org.bytedeco.pytorch.data.dataframe.excel.ExcelOptions options) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.excel.ExcelReader.read(path, options);
    }

    public static DataFrame readExcel(java.io.InputStream in,
            org.bytedeco.pytorch.data.dataframe.excel.ExcelOptions options) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.excel.ExcelReader.read(in, options);
    }

    public static Map<String, DataFrame> readExcelAll(String path) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.excel.ExcelReader.readAll(path,
            org.bytedeco.pytorch.data.dataframe.excel.ExcelOptions.defaults());
    }

    public static Map<String, DataFrame> readExcelAll(String path,
            org.bytedeco.pytorch.data.dataframe.excel.ExcelOptions options) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.excel.ExcelReader.readAll(path, options);
    }

    public void toExcel(String path) throws Exception {
        org.bytedeco.pytorch.data.dataframe.excel.ExcelWriter.write(this, path);
    }

    public void toExcel(String path,
            org.bytedeco.pytorch.data.dataframe.excel.ExcelOptions options) throws Exception {
        org.bytedeco.pytorch.data.dataframe.excel.ExcelWriter.write(this, path, options);
    }

    public static void writeExcelSheets(String path, Map<String, DataFrame> sheets) throws Exception {
        org.bytedeco.pytorch.data.dataframe.excel.ExcelWriter.writeSheets(path, sheets,
            org.bytedeco.pytorch.data.dataframe.excel.ExcelOptions.defaults());
    }

    public static void writeExcelSheets(String path, Map<String, DataFrame> sheets,
            org.bytedeco.pytorch.data.dataframe.excel.ExcelOptions options) throws Exception {
        org.bytedeco.pytorch.data.dataframe.excel.ExcelWriter.writeSheets(path, sheets, options);
    }

    public static DataFrame readSql(java.sql.Connection c, String sql) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.sql.SqlReader.read(c, sql);
    }

    public static DataFrame readSql(java.sql.Connection c, String sql,
            org.bytedeco.pytorch.data.dataframe.sql.SqlOptions options) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.sql.SqlReader.read(c, sql, options);
    }

    public static DataFrame readSqlTable(java.sql.Connection c, String table) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.sql.SqlReader.readTable(c, table);
    }

    public static DataFrame readSql(String sqlitePath, String sql) throws Exception {
        try (java.sql.Connection c = org.bytedeco.pytorch.data.dataframe.sql.Sqlite.open(sqlitePath)) {
            return readSql(c, sql);
        }
    }

    // ---- I/O: SQLite (dedicated, pandas read_sql / to_sql style) ----

    /**
     * Read a SQLite table into a DataFrame (opens {@code sqlitePath}, SELECT * FROM table).
     *
     * @see #readSqlite(String, String)
     * @see #toSqlite(String, String)
     */
    public static DataFrame readSqliteTable(String sqlitePath, String table) throws Exception {
        try (java.sql.Connection c = org.bytedeco.pytorch.data.dataframe.sql.Sqlite.open(sqlitePath)) {
            return readSqlTable(c, table);
        }
    }

    /**
     * Run SQL against a SQLite database file → DataFrame.
     * Alias of {@link #readSql(String, String)} with a clearer name.
     */
    public static DataFrame readSqlite(String sqlitePath, String sql) throws Exception {
        return readSql(sqlitePath, sql);
    }

    /**
     * Read SQLite with options (fetch size, dtype overrides, …).
     */
    public static DataFrame readSqlite(String sqlitePath, String sql,
                                       org.bytedeco.pytorch.data.dataframe.sql.SqlOptions options)
            throws Exception {
        try (java.sql.Connection c = org.bytedeco.pytorch.data.dataframe.sql.Sqlite.open(sqlitePath)) {
            return readSql(c, sql, options);
        }
    }

    /**
     * Write this DataFrame into a SQLite table (default {@code if_exists=REPLACE}).
     *
     * <pre>{@code
     * df.toSqlite("data.db", "people");
     * DataFrame back = DataFrame.readSqliteTable("data.db", "people");
     * }</pre>
     */
    public void toSqlite(String sqlitePath, String table) throws Exception {
        toSqlite(sqlitePath, table,
                org.bytedeco.pytorch.data.dataframe.sql.SqlOptions.builder()
                        .ifExists(org.bytedeco.pytorch.data.dataframe.sql.SqlOptions.IfExists.REPLACE)
                        .build());
    }

    /**
     * Write this DataFrame into a SQLite table with {@link org.bytedeco.pytorch.data.dataframe.sql.SqlOptions}.
     */
    public void toSqlite(String sqlitePath, String table,
                         org.bytedeco.pytorch.data.dataframe.sql.SqlOptions options) throws Exception {
        try (java.sql.Connection c = org.bytedeco.pytorch.data.dataframe.sql.Sqlite.open(sqlitePath)) {
            toSql(c, table, options == null
                    ? org.bytedeco.pytorch.data.dataframe.sql.SqlOptions.defaults()
                    : options);
        }
    }

    /** Alias of {@link #toSqlite(String, String)}. */
    public void writeSqlite(String sqlitePath, String table) throws Exception {
        toSqlite(sqlitePath, table);
    }

    public void writeSqlite(String sqlitePath, String table,
                            org.bytedeco.pytorch.data.dataframe.sql.SqlOptions options) throws Exception {
        toSqlite(sqlitePath, table, options);
    }

    /** Open a SQLite connection (caller owns / must close). */
    public static java.sql.Connection openSqlite(String path) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.sql.Sqlite.open(path);
    }

    public static java.sql.Connection openSqliteInMemory() throws Exception {
        return org.bytedeco.pytorch.data.dataframe.sql.Sqlite.openInMemory();
    }

    public void toSql(java.sql.Connection c, String table) throws Exception {
        org.bytedeco.pytorch.data.dataframe.sql.SqlWriter.write(this, c, table);
    }

    public void toSql(java.sql.Connection c, String table,
            org.bytedeco.pytorch.data.dataframe.sql.SqlOptions options) throws Exception {
        org.bytedeco.pytorch.data.dataframe.sql.SqlWriter.write(this, c, table, options);
    }

    public void toSql(String sqlitePath, String table,
            org.bytedeco.pytorch.data.dataframe.sql.SqlOptions options) throws Exception {
        try (java.sql.Connection c = org.bytedeco.pytorch.data.dataframe.sql.Sqlite.open(sqlitePath)) {
            toSql(c, table, options);
        }
    }

    /** Convenience: write to SQLite path with default REPLACE (pandas-style). */
    public void toSql(String sqlitePath, String table) throws Exception {
        toSqlite(sqlitePath, table);
    }

    // ---- I/O: Redis (pure RESP via dataframe.redis.Redis — no Jedis) ----

    /**
     * Open a Redis connection (caller owns / must {@code close()}).
     *
     * @see org.bytedeco.pytorch.data.dataframe.redis.Redis#connect(String, int)
     */
    public static org.bytedeco.pytorch.data.dataframe.redis.Redis openRedis() {
        return org.bytedeco.pytorch.data.dataframe.redis.Redis.connect();
    }

    public static org.bytedeco.pytorch.data.dataframe.redis.Redis openRedis(String host, int port) {
        return org.bytedeco.pytorch.data.dataframe.redis.Redis.connect(host, port);
    }

    public static org.bytedeco.pytorch.data.dataframe.redis.Redis openRedis(
            String host, int port, String password) {
        return org.bytedeco.pytorch.data.dataframe.redis.Redis.connect(host, port, password);
    }

    /** Parse {@code redis://[:password@]host:port[/db]}. */
    public static org.bytedeco.pytorch.data.dataframe.redis.Redis openRedis(String uri) {
        return org.bytedeco.pytorch.data.dataframe.redis.Redis.connectUri(uri);
    }

    /**
     * Write this DataFrame to Redis using {@link org.bytedeco.pytorch.data.dataframe.redis.RedisOptions}
     * (HASH / JSON / FRAME_JSON + optional TTL).
     *
     * <pre>{@code
     * try (Redis r = DataFrame.openRedis("127.0.0.1", 6379)) {
     *     df.toRedis(r, RedisOptions.hash("df:people:", Duration.ofHours(1)));
     * }
     * }</pre>
     *
     * @return number of keys written
     */
    public int toRedis(org.bytedeco.pytorch.data.dataframe.redis.Redis redis,
                       org.bytedeco.pytorch.data.dataframe.redis.RedisOptions options) {
        return redis.writeDataFrame(this, options);
    }

    /** HASH layout: one Redis hash per row under {@code prefix + id}. */
    public int toRedis(org.bytedeco.pytorch.data.dataframe.redis.Redis redis, String prefix) {
        return toRedis(redis, org.bytedeco.pytorch.data.dataframe.redis.RedisOptions.hash(prefix));
    }

    /**
     * HASH layout with TTL on every row key.
     *
     * @param ttl row-key expiry; {@code null} = no expiry
     */
    public int toRedis(org.bytedeco.pytorch.data.dataframe.redis.Redis redis,
                       String prefix, java.time.Duration ttl) {
        return toRedis(redis, org.bytedeco.pytorch.data.dataframe.redis.RedisOptions.hash(prefix, ttl));
    }

    /**
     * HASH layout with explicit id column + TTL.
     */
    public int toRedisHash(org.bytedeco.pytorch.data.dataframe.redis.Redis redis,
                           String prefix, String idColumn, java.time.Duration ttl) {
        return redis.writeHash(this, prefix, idColumn, ttl);
    }

    public int toRedisHash(org.bytedeco.pytorch.data.dataframe.redis.Redis redis, String prefix) {
        return toRedisHash(redis, prefix, null, null);
    }

    public int toRedisHash(org.bytedeco.pytorch.data.dataframe.redis.Redis redis,
                           String prefix, java.time.Duration ttl) {
        return toRedisHash(redis, prefix, null, ttl);
    }

    /** Per-row JSON strings ({@code SET prefix{id} json [EX ttl]}). */
    public int toRedisJson(org.bytedeco.pytorch.data.dataframe.redis.Redis redis,
                           String prefix, String idColumn, java.time.Duration ttl) {
        return redis.writeJson(this, prefix, idColumn, ttl);
    }

    public int toRedisJson(org.bytedeco.pytorch.data.dataframe.redis.Redis redis, String prefix) {
        return toRedisJson(redis, prefix, null, null);
    }

    /** Entire frame as one JSON array key. */
    public int toRedisFrame(org.bytedeco.pytorch.data.dataframe.redis.Redis redis,
                            String key, java.time.Duration ttl) {
        return redis.writeFrame(this, key, ttl);
    }

    public int toRedisFrame(org.bytedeco.pytorch.data.dataframe.redis.Redis redis, String key) {
        return toRedisFrame(redis, key, null);
    }

    /**
     * Connect, write, disconnect — one-shot convenience.
     * URI examples: {@code 127.0.0.1:6379}, {@code redis://:pass@host:6379/0}.
     */
    public int toRedis(String redisUri, org.bytedeco.pytorch.data.dataframe.redis.RedisOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.redis.Redis r =
                     org.bytedeco.pytorch.data.dataframe.redis.Redis.connectUri(redisUri)) {
            return toRedis(r, options);
        }
    }

    public int toRedis(String redisUri, String prefix, java.time.Duration ttl) {
        return toRedis(redisUri, org.bytedeco.pytorch.data.dataframe.redis.RedisOptions.hash(prefix, ttl));
    }

    /** Read DataFrame previously stored with {@link #toRedis}. */
    public static DataFrame readRedis(org.bytedeco.pytorch.data.dataframe.redis.Redis redis,
                                      org.bytedeco.pytorch.data.dataframe.redis.RedisOptions options) {
        return redis.readDataFrame(options);
    }

    public static DataFrame readRedis(org.bytedeco.pytorch.data.dataframe.redis.Redis redis, String prefix) {
        return redis.readHash(prefix);
    }

    public static DataFrame readRedisHash(org.bytedeco.pytorch.data.dataframe.redis.Redis redis, String prefix) {
        return redis.readHash(prefix);
    }

    public static DataFrame readRedisJson(org.bytedeco.pytorch.data.dataframe.redis.Redis redis, String prefix) {
        return redis.readJsonRows(prefix);
    }

    public static DataFrame readRedisFrame(org.bytedeco.pytorch.data.dataframe.redis.Redis redis, String key) {
        return redis.readFrame(key);
    }

    /** One-shot: connect → read → close. */
    public static DataFrame readRedis(String redisUri,
                                      org.bytedeco.pytorch.data.dataframe.redis.RedisOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.redis.Redis r =
                     org.bytedeco.pytorch.data.dataframe.redis.Redis.connectUri(redisUri)) {
            return readRedis(r, options);
        }
    }

    public static DataFrame readRedis(String redisUri, String prefix) {
        return readRedis(redisUri, org.bytedeco.pytorch.data.dataframe.redis.RedisOptions.hash(prefix));
    }

    /**
     * Read specific Redis hash keys into a DataFrame (order preserved).
     */
    public static DataFrame readRedisKeys(org.bytedeco.pytorch.data.dataframe.redis.Redis redis,
                                          java.util.Collection<String> keys) {
        return redis.readHashes(keys);
    }

    public static DataFrame readHdf(String path, String key) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.hdf5.Hdf5Reader.read(path, key);
    }

    public static DataFrame readHdf(String path, String key,
            org.bytedeco.pytorch.data.dataframe.hdf5.Hdf5Options options) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.hdf5.Hdf5Reader.read(path, key, options);
    }

    public void toHdf(String path, String key) throws Exception {
        org.bytedeco.pytorch.data.dataframe.hdf5.Hdf5Writer.write(this, path, key);
    }

    public void toHdf(String path, String key,
            org.bytedeco.pytorch.data.dataframe.hdf5.Hdf5Options options) throws Exception {
        org.bytedeco.pytorch.data.dataframe.hdf5.Hdf5Writer.write(this, path, key, options);
    }

    public static DataFrame readAvro(String path) throws Exception {
        return org.bytedeco.pytorch.data.avro.LocalAvroReader.read(path);
    }

    public static DataFrame readAvro(String path,
            org.bytedeco.pytorch.data.avro.AvroOptions options) throws Exception {
        return org.bytedeco.pytorch.data.avro.LocalAvroReader.read(path, options);
    }

    public void toAvro(String path) throws Exception {
        org.bytedeco.pytorch.data.avro.LocalAvroWriter.write(this, path);
    }

    public void toAvro(String path, org.bytedeco.pytorch.data.avro.AvroOptions options) throws Exception {
        org.bytedeco.pytorch.data.avro.LocalAvroWriter.write(this, path, options);
    }

    /**
     * Read ORC via legacy DuckDB {@code read_orc}.
     * @see #readOrcFormat(String) pure-Java orc-format path
     */
    public static DataFrame readOrc(String path) throws Exception {
        return org.bytedeco.pytorch.data.orc.LocalOrcReader.read(path);
    }

    /** @see #readOrc(String) */
    public static DataFrame readOrc(String path,
            org.bytedeco.pytorch.data.orc.OrcOptions options) throws Exception {
        return org.bytedeco.pytorch.data.orc.LocalOrcReader.read(path, options);
    }

    /**
     * Write ORC via legacy DuckDB {@code COPY ... FORMAT ORC} (may be unsupported).
     * @see #toOrcFormat(String) pure-Java orc-format path (reliable write)
     */
    public void toOrc(String path) throws Exception {
        org.bytedeco.pytorch.data.orc.LocalOrcWriter.write(this, path);
    }

    /** @see #toOrc(String) */
    public void toOrc(String path, org.bytedeco.pytorch.data.orc.OrcOptions options) throws Exception {
        org.bytedeco.pytorch.data.orc.LocalOrcWriter.write(this, path, options);
    }

    /**
     * Read ORC with the pure-Java {@code orc-format} backend (no Hadoop / no orc-core).
     * @see #readOrc(String) legacy DuckDB backend
     */
    public static DataFrame readOrcFormat(String path) throws Exception {
        return org.bytedeco.pytorch.data.orc.LocalOrcFormatReader.read(path);
    }

    /** @see #readOrcFormat(String) */
    public static DataFrame readOrcFormat(String path,
            org.bytedeco.pytorch.data.orc.OrcOptions options) throws Exception {
        return org.bytedeco.pytorch.data.orc.LocalOrcFormatReader.read(path, options);
    }

    /**
     * Write ORC with the pure-Java {@code orc-format} backend (no Hadoop / no orc-core).
     * Preferred when reliable local ORC write is required.
     * @see #toOrc(String) legacy DuckDB backend
     */
    public void toOrcFormat(String path) throws Exception {
        org.bytedeco.pytorch.data.orc.LocalOrcFormatWriter.write(this, path);
    }

    /** @see #toOrcFormat(String) */
    public void toOrcFormat(String path, org.bytedeco.pytorch.data.orc.OrcOptions options) throws Exception {
        org.bytedeco.pytorch.data.orc.LocalOrcFormatWriter.write(this, path, options);
    }

    // ---- I/O: Tensor (rank-aware 0–4+) ----

    /**
     * How a multi-dimensional {@link Tensor} is laid out as a DataFrame.
     * <ul>
     *   <li>{@link #COLUMNS} — rank≤2 as scalar columns (legacy default for 2-D)</li>
     *   <li>{@link #ROWS_AS_TENSOR} — leading axis = rows; each row is a {@code TENSOR}/{@code VECTOR} cell</li>
     *   <li>{@link #FLAT} — single numeric column of all elements</li>
     * </ul>
     */
    public enum TensorLayout {
        /** Rank 0/1 → one numeric col; rank 2 → {@code col_0..col_n-1}; rank≥3 → ROWS_AS_TENSOR. */
        COLUMNS,
        /** Leading dim = rows; remaining dims packed per-row as TENSOR (or VECTOR if rank-1 tail). */
        ROWS_AS_TENSOR,
        /** Flatten entire tensor into one numeric column. */
        FLAT
    }

    /** Rank-aware: 0/1 → one col; 2 → scalar columns; ≥3 → one TENSOR column of row slices. */
    public static DataFrame fromTensor(Tensor t, String... colNames) {
        return fromTensor(t, TensorLayout.COLUMNS, colNames);
    }

    public static DataFrame fromTensor(Tensor t, TensorLayout layout, String... colNames) {
        Objects.requireNonNull(t, "tensor");
        TensorLayout mode = layout == null ? TensorLayout.COLUMNS : layout;
        long[] shape = sizesAsArray(t.sizes());
        int rank = shape.length;

        if (mode == TensorLayout.FLAT || rank == 0) {
            return fromTensorFlat(t, colNames.length > 0 ? colNames[0] : "data");
        }
        if (mode == TensorLayout.ROWS_AS_TENSOR || rank >= 3) {
            String name = colNames.length > 0 ? colNames[0] : "tensor";
            return fromTensorRowsAsCells(t, name, rank == 2 /* prefer VECTOR for 2-D rows? no: rank tail */);
        }
        // COLUMNS: rank 1 or 2
        if (rank == 1) {
            return fromTensorFlat(t, colNames.length > 0 ? colNames[0] : "data");
        }
        // rank == 2 → columns
        DataFrame df = DataFrame.create();
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        String[] names = colNames;
        if (names.length < cols) {
            names = new String[cols];
            for (int i = 0; i < cols; i++) names[i] = "col_" + i;
        }
        Column.DType dtype = scalarTypeToDType(t.scalar_type());
        for (int c = 0; c < cols; c++) df.addColumn(names[c], dtype);
        Tensor flat = t.contiguous().cpu().reshape(new long[]{-1})
            .to(org.bytedeco.pytorch.global.torch.ScalarType.Double);
        org.bytedeco.javacpp.DoublePointer ptr = flat.data_ptr_double();
        for (int r = 0; r < rows; r++) {
            int ri = df.addRow();
            for (int c = 0; c < cols; c++) {
                long idx = (long) r * cols + c;
                df.set(ri, names[c], Double.valueOf(ptr.get(idx)));
            }
        }
        return df;
    }

    /** One TENSOR (or VECTOR for 1-D rows) column; leading axis = row count. */
    public static DataFrame fromTensorRows(Tensor t, String colName) {
        return fromTensor(t, TensorLayout.ROWS_AS_TENSOR, colName == null ? "tensor" : colName);
    }

    /** Build a frame from a map of named tensors (each becomes one TENSOR cell of a single-row frame,
     *  or a column of row-slices when rank≥1 with shared leading dim — here: one row, one cell each). */
    public static DataFrame fromTensors(Map<String, Tensor> tensors) {
        DataFrame df = DataFrame.create();
        if (tensors == null || tensors.isEmpty()) return df;
        for (String k : tensors.keySet()) {
            df.addColumn(k, Column.DType.TENSOR);
        }
        int ri = df.addEmptyRow();
        for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
            df.set(ri, e.getKey(),
                org.bytedeco.pytorch.data.dataframe.dtype.TensorData.fromTensor(e.getValue()));
        }
        return df;
    }

    public static DataFrame fromTensorData(String colName, org.bytedeco.pytorch.data.dataframe.dtype.TensorData... cells) {
        DataFrame df = DataFrame.create();
        String name = colName == null ? "tensor" : colName;
        df.addColumn(name, Column.DType.TENSOR);
        if (cells != null) {
            for (org.bytedeco.pytorch.data.dataframe.dtype.TensorData td : cells) {
                int ri = df.addEmptyRow();
                df.set(ri, name, td);
            }
        }
        return df;
    }

    public static DataFrame fromNDArray(org.bytedeco.pytorch.data.numpy.NDArray arr) {
        return fromNDArray(arr, TensorLayout.COLUMNS);
    }

    public static DataFrame fromNDArray(org.bytedeco.pytorch.data.numpy.NDArray arr, TensorLayout layout) {
        Objects.requireNonNull(arr, "ndarray");
        // Reuse tensor path for consistent rank policy
        Tensor t = org.bytedeco.pytorch.data.dataframe.tensor.TensorBridge.toTensor(arr);
        return fromTensor(t, layout);
    }

    /**
     * First numeric column → 1-D tensor (legacy), or if a VECTOR/EMBEDDING/TENSOR column
     * exists first, pack it via {@link #toTensorColumn(String)}.
     */
    public Tensor toTensor() {
        for (Column col : columns) {
            if (col.dtype() == Column.DType.VECTOR
                || col.dtype() == Column.DType.EMBEDDING
                || col.dtype() == Column.DType.TENSOR) {
                return toTensorColumn(col.name());
            }
        }
        Column col = findFirstNumeric();
        return columnToTensor(col);
    }

    /** Stack named numeric columns into a 2-D tensor {@code [n_rows, n_cols]}. */
    public Tensor toTensor(String... colNames) {
        if (colNames == null || colNames.length == 0) return toTensor();
        int n = rowCount();
        int c = colNames.length;
        double[] data = new double[n * c];
        for (int j = 0; j < c; j++) {
            Column col = column(colNames[j]);
            for (int i = 0; i < n; i++) {
                Object v = col.get(i);
                data[i * c + j] = v instanceof Number ? ((Number) v).doubleValue() : 0.0;
            }
        }
        Tensor t = torch.tensor(data);
        return t.reshape(new long[]{n, c});
    }

    /**
     * Pack a VECTOR / EMBEDDING / TENSOR / float[] column into a dense tensor.
     * <ul>
     *   <li>1-D cells → {@code [n, dim]}</li>
     *   <li>TENSOR cells with uniform shape {@code S} → {@code [n, ...S]}</li>
     * </ul>
     */
    public Tensor toTensorColumn(String colName) {
        Column col = column(colName);
        int n = col.size();
        if (n == 0) return torch.tensor(new float[0]);

        // Try TensorData path for multi-dim
        Object first = null;
        for (int i = 0; i < n; i++) {
            first = col.get(i);
            if (first != null) break;
        }
        if (first instanceof org.bytedeco.pytorch.data.dataframe.dtype.TensorData td0) {
            int[] cellShape = td0.getShape();
            int cellSize = td0.size();
            float[] matrix = new float[n * cellSize];
            for (int i = 0; i < n; i++) {
                Object cell = col.get(i);
                float[] row = cell instanceof org.bytedeco.pytorch.data.dataframe.dtype.TensorData td
                    ? td.getData()
                    : org.bytedeco.pytorch.data.dataframe.tensor.TensorBridge.asFloatVector(cell);
                if (row != null) {
                    System.arraycopy(row, 0, matrix, i * cellSize, Math.min(cellSize, row.length));
                }
            }
            long[] full = new long[1 + cellShape.length];
            full[0] = n;
            for (int d = 0; d < cellShape.length; d++) full[d + 1] = cellShape[d];
            Tensor t = torch.tensor(matrix);
            return t.reshape(full);
        }

        // VECTOR / EMBEDDING / float[] → [n, dim]
        float[] packed = org.bytedeco.pytorch.data.dataframe.ann.VectorColumn.pack(col);
        int dim = org.bytedeco.pytorch.data.dataframe.ann.VectorColumn.dimOf(col);
        if (dim <= 0) {
            // fall back to scalar numeric
            return columnToTensor(col);
        }
        Tensor t = torch.tensor(packed);
        return t.reshape(new long[]{n, dim});
    }

    /** Convert a numeric / vector column to an {@link org.bytedeco.pytorch.data.numpy.NDArray}. */
    public org.bytedeco.pytorch.data.numpy.NDArray toNDArray(String colName) {
        Column col = column(colName);
        if (col.dtype() == Column.DType.VECTOR
            || col.dtype() == Column.DType.EMBEDDING
            || col.dtype() == Column.DType.TENSOR) {
            Tensor t = toTensorColumn(colName);
            return org.bytedeco.pytorch.data.dataframe.tensor.TensorBridge.toNDArray(t);
        }
        return columnToNDArray(col);
    }

    public org.bytedeco.pytorch.data.numpy.NDArray toNDArray() {
        for (Column col : columns) {
            if (col.dtype() == Column.DType.VECTOR
                || col.dtype() == Column.DType.EMBEDDING
                || col.dtype() == Column.DType.TENSOR
                || isNumeric(col.dtype())) {
                return toNDArray(col.name());
            }
        }
        throw new IllegalStateException("No numeric/vector column found");
    }

    private static DataFrame fromTensorFlat(Tensor t, String colName) {
        DataFrame df = DataFrame.create();
        Column.DType dtype = scalarTypeToDType(t.scalar_type());
        df.addColumn(colName, dtype);
        Tensor flat = t.contiguous().cpu().reshape(new long[]{-1})
            .to(org.bytedeco.pytorch.global.torch.ScalarType.Double);
        int n = (int) flat.numel();
        org.bytedeco.javacpp.DoublePointer ptr = flat.data_ptr_double();
        for (int i = 0; i < n; i++) {
            int ri = df.addEmptyRow();
            df.set(ri, colName, Double.valueOf(ptr.get(i)));
        }
        return df;
    }

    private static DataFrame fromTensorRowsAsCells(Tensor t, String colName, boolean unused) {
        DataFrame df = DataFrame.create();
        long[] shape = sizesAsArray(t.sizes());
        if (shape.length == 0) {
            df.addColumn(colName, Column.DType.TENSOR);
            int ri = df.addEmptyRow();
            df.set(ri, colName, org.bytedeco.pytorch.data.dataframe.dtype.TensorData.fromTensor(t));
            return df;
        }
        int rows = (int) shape[0];
        int[] cellShape = new int[Math.max(0, shape.length - 1)];
        long cellSize = 1;
        for (int i = 1; i < shape.length; i++) {
            cellShape[i - 1] = (int) shape[i];
            cellSize *= shape[i];
        }
        boolean asVector = cellShape.length <= 1;
        df.addColumn(colName, asVector ? Column.DType.VECTOR : Column.DType.TENSOR);

        Tensor cpu = t.contiguous().cpu();
        Tensor flat = cpu.reshape(new long[]{rows, cellSize})
            .to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        org.bytedeco.javacpp.FloatPointer ptr = flat.data_ptr_float();
        for (int r = 0; r < rows; r++) {
            float[] row = new float[(int) cellSize];
            for (int j = 0; j < cellSize; j++) {
                row[j] = ptr.get((long) r * cellSize + j);
            }
            int ri = df.addEmptyRow();
            if (asVector) {
                df.set(ri, colName, row);
            } else {
                df.set(ri, colName,
                    new org.bytedeco.pytorch.data.dataframe.dtype.TensorData(row, cellShape));
            }
        }
        return df;
    }

    // ---- aggregations ----

    public Map<String, Double> sum() {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                double s = col.asDoubleList().stream()
                    .filter(d -> !Double.isNaN(d)).mapToDouble(Double::doubleValue).sum();
                result.put(col.name(), s);
            }
        }
        return result;
    }

    public Map<String, Double> mean() {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                double[] vals = col.asDoubleArray();
                double s = 0, n = 0;
                for (double v : vals) if (!Double.isNaN(v)) { s += v; n++; }
                result.put(col.name(), n > 0 ? s / n : Double.NaN);
            }
        }
        return result;
    }

    public Map<String, Double> median() {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                List<Double> vals = col.asDoubleList().stream()
                    .filter(d -> !Double.isNaN(d)).sorted()
                    .collect(Collectors.toList());
                int n = vals.size();
                result.put(col.name(), n > 0
                    ? (n % 2 == 0 ? (vals.get(n/2-1) + vals.get(n/2)) / 2.0 : vals.get(n/2))
                    : Double.NaN);
            }
        }
        return result;
    }

    public Map<String, Double> std() { return stdVar(true); }
    public Map<String, Double> var() { return stdVar(false); }

    private Map<String, Double> stdVar(boolean std) {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                double[] vals = col.asDoubleArray();
                double mean = 0, n = 0;
                for (double v : vals) if (!Double.isNaN(v)) { mean += v; n++; }
                if (n > 1) {
                    mean /= n;
                    double ss = 0;
                    for (double v : vals) if (!Double.isNaN(v)) ss += (v - mean) * (v - mean);
                    double v = ss / (n - 1);
                    result.put(col.name(), std ? Math.sqrt(v) : v);
                } else result.put(col.name(), Double.NaN);
            }
        }
        return result;
    }

    public Map<String, Double> min() {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                double m = Double.POSITIVE_INFINITY;
                for (double v : col.asDoubleArray()) if (!Double.isNaN(v) && v < m) m = v;
                result.put(col.name(), m == Double.POSITIVE_INFINITY ? Double.NaN : m);
            }
        }
        return result;
    }

    public Map<String, Double> max() {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                double m = Double.NEGATIVE_INFINITY;
                for (double v : col.asDoubleArray()) if (!Double.isNaN(v) && v > m) m = v;
                result.put(col.name(), m == Double.NEGATIVE_INFINITY ? Double.NaN : m);
            }
        }
        return result;
    }

    public Map<String, Integer> count() {
        Map<String, Integer> result = new LinkedHashMap<>();
        for (Column col : columns) {
            int n = (int) col.asDoubleList().stream().filter(d -> !Double.isNaN(d)).count();
            result.put(col.name(), n);
        }
        return result;
    }

    public Map<String, Integer> nunique() {
        Map<String, Integer> result = new LinkedHashMap<>();
        for (Column col : columns) {
            Set<Object> uniq = new HashSet<>();
            for (Object v : col.data()) if (v != null) uniq.add(v);
            result.put(col.name(), uniq.size());
        }
        return result;
    }

    public Map<String, Double> quantile(double q) {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                List<Double> vals = col.asDoubleList().stream()
                    .filter(d -> !Double.isNaN(d)).sorted().collect(Collectors.toList());
                int n = vals.size();
                if (n == 0) { result.put(col.name(), Double.NaN); continue; }
                double pos = q * (n - 1);
                int lo = (int) Math.floor(pos), hi = (int) Math.ceil(pos);
                result.put(col.name(), (vals.get(lo) + vals.get(hi)) / 2.0);
            }
        }
        return result;
    }

    public Map<String, Object> mode() {
        Map<String, Object> result = new LinkedHashMap<>();
        for (Column col : columns) {
            Map<Object, Long> freq = new HashMap<>();
            for (Object v : col.data()) if (v != null) freq.merge(v, 1L, Long::sum);
            Object mode = freq.entrySet().stream()
                .max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
            result.put(col.name(), mode);
        }
        return result;
    }

    public Map<String, Double> skew() {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                List<Double> vals = col.asDoubleList().stream()
                    .filter(d -> !Double.isNaN(d)).collect(Collectors.toList());
                int n = vals.size();
                if (n < 3) { result.put(col.name(), Double.NaN); continue; }
                double mean = vals.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double std = Math.sqrt(vals.stream().mapToDouble(d -> (d - mean) * (d - mean)).sum() / (n - 1));
                if (std == 0) { result.put(col.name(), 0.0); continue; }
                double skew = vals.stream().mapToDouble(d -> Math.pow((d - mean) / std, 3)).sum() / n;
                result.put(col.name(), skew);
            }
        }
        return result;
    }

    public Map<String, Double> kurt() {
        Map<String, Double> result = new LinkedHashMap<>();
        for (Column col : columns) {
            if (isNumeric(col.dtype())) {
                List<Double> vals = col.asDoubleList().stream()
                    .filter(d -> !Double.isNaN(d)).collect(Collectors.toList());
                int n = vals.size();
                if (n < 4) { result.put(col.name(), Double.NaN); continue; }
                double mean = vals.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                double std = Math.sqrt(vals.stream().mapToDouble(d -> (d - mean) * (d - mean)).sum() / (n - 1));
                if (std == 0) { result.put(col.name(), 0.0); continue; }
                double kurt = vals.stream().mapToDouble(d -> Math.pow((d - mean) / std, 4)).sum() / n - 3;
                result.put(col.name(), kurt);
            }
        }
        return result;
    }

    public Map<String, List<Double>> describe() {
        Map<String, List<Double>> result = new LinkedHashMap<>();
        for (String k : mean().keySet()) {
            List<Double> stats = new ArrayList<>();
            stats.add(mean().get(k));
            stats.add(std().get(k));
            stats.add(min().get(k));
            stats.add(quantile(0.25).get(k));
            stats.add(quantile(0.5).get(k));
            stats.add(quantile(0.75).get(k));
            stats.add(max().get(k));
            result.put(k, stats);
        }
        return result;
    }

    // ---- groupby ----

    public GroupedDataFrame groupby(String... cols) {
        Map<String, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < rowCount; i++) {
            String key = "";
            for (int ci = 0; ci < cols.length; ci++) {
                Object v = column(cols[ci]).get(i);
                if (ci > 0) key += "|";
                key += v != null ? v.toString() : "null";
            }
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
        }
        return new GroupedDataFrame(this, cols, groups);
    }

    // ---- merge / join ----

    public DataFrame merge(DataFrame right, String leftOn, String rightOn, String how) throws Exception {
        DataFrame result = DataFrame.create();
        List<String> leftCols = columns.stream().map(Column::name).filter(n -> !n.equals(leftOn)).collect(Collectors.toList());
        List<String> rightCols = right.columns.stream().map(Column::name).filter(n -> !n.equals(rightOn)).collect(Collectors.toList());
        for (String c : leftCols) result.addColumn(c, column(c).dtype());
        result.addColumn(rightOn, column(leftOn).dtype());
        for (String c : rightCols) result.addColumn(c, right.column(c).dtype());

        Map<Object, Integer> rightIndex = new HashMap<>();
        for (int i = 0; i < right.rowCount; i++) rightIndex.put(right.column(rightOn).get(i), i);

        for (int li = 0; li < rowCount; li++) {
            Object key = column(leftOn).get(li);
            Integer ri = rightIndex.get(key);
            if (ri == null && ("right".equals(how) || "outer".equals(how))) {
                int nri = result.addRow();
                for (String c : leftCols) result.set(nri, c, null);
                result.set(nri, rightOn, key);
                for (String c : rightCols) result.set(nri, c, right.column(c).get(ri));
            } else if (ri != null) {
                int nri = result.addRow();
                for (String c : leftCols) result.set(nri, c, column(c).get(li));
                result.set(nri, rightOn, key);
                for (String c : rightCols) result.set(nri, c, right.column(c).get(ri));
            } else if ("left".equals(how) || "outer".equals(how)) {
                int nri = result.addRow();
                for (String c : leftCols) result.set(nri, c, column(c).get(li));
                result.set(nri, rightOn, key);
                for (String c : rightCols) result.set(nri, c, null);
            }
        }
        return result;
    }

    public DataFrame join(DataFrame other, String on, String how) throws Exception {
        return merge(other, on, on, how);
    }

    /**
     * Join with how = {@code "semi"} / {@code "anti"} / {@code "cross"} / {@code "asof"}
     * or standard {@code inner}/{@code left}/{@code right}/{@code outer}.
     */
    public DataFrame join(DataFrame other, String on, String how, String direction) throws Exception {
        String h = how == null ? "inner" : how.toLowerCase(Locale.ROOT);
        return switch (h) {
            case "semi" -> AdvancedOps.joinSemi(this, other, on);
            case "anti" -> AdvancedOps.joinAnti(this, other, on);
            case "cross" -> AdvancedOps.joinCross(this, other);
            case "asof" -> AdvancedOps.joinAsof(this, other, on, on, direction, null);
            default -> merge(other, on, on, h);
        };
    }

    /** Polars {@code join_asof} / Pandas {@code merge_asof}. Frames must be sorted by key. */
    public DataFrame joinAsof(DataFrame other, String on) throws Exception {
        return AdvancedOps.joinAsof(this, other, on);
    }

    public DataFrame joinAsof(DataFrame other, String leftOn, String rightOn,
                              String strategy, Double tolerance) throws Exception {
        return AdvancedOps.joinAsof(this, other, leftOn, rightOn, strategy, tolerance);
    }

    public static DataFrame mergeAsof(DataFrame left, DataFrame right, String on) throws Exception {
        return AdvancedOps.joinAsof(left, right, on);
    }

    public static DataFrame mergeAsof(DataFrame left, DataFrame right,
                                      String leftOn, String rightOn,
                                      String direction, Double tolerance) throws Exception {
        return AdvancedOps.mergeAsof(left, right, leftOn, rightOn, direction, tolerance);
    }

    /**
     * Full Pandas {@code merge_asof} with {@code by} group keys and {@code allow_exact_matches}.
     */
    public static DataFrame mergeAsof(DataFrame left, DataFrame right,
                                      String leftOn, String rightOn,
                                      String direction, Double tolerance,
                                      String[] by, boolean allowExactMatches) throws Exception {
        return AdvancedOps.mergeAsof(left, right, leftOn, rightOn, direction, tolerance, by, allowExactMatches);
    }

    public DataFrame joinAsof(DataFrame other, String leftOn, String rightOn,
                              String strategy, Double tolerance,
                              String[] by, boolean allowExactMatches) throws Exception {
        return AdvancedOps.joinAsof(this, other, leftOn, rightOn, strategy, tolerance, by, allowExactMatches);
    }

    public DataFrame joinSemi(DataFrame other, String on) {
        return AdvancedOps.joinSemi(this, other, on);
    }

    public DataFrame joinAnti(DataFrame other, String on) {
        return AdvancedOps.joinAnti(this, other, on);
    }

    public DataFrame joinCross(DataFrame other) throws Exception {
        return AdvancedOps.joinCross(this, other);
    }

    public DataFrame setDifference(DataFrame other) {
        return AdvancedOps.setDifference(this, other);
    }

    public DataFrame setIntersection(DataFrame other) {
        return AdvancedOps.setIntersection(this, other);
    }

    public DataFrame setUnion(DataFrame other) throws Exception {
        return AdvancedOps.setUnion(this, other);
    }

    // ---- concat ----

    public static DataFrame concat(List<DataFrame> dfs, int axis) throws Exception {
        if (dfs == null || dfs.isEmpty()) return DataFrame.create();
        DataFrame result = DataFrame.create();
        if (axis == 0) {
            List<String> allColNames = new ArrayList<>();
            Set<String> seen = new LinkedHashSet<>();
            for (DataFrame df : dfs) {
                for (Column c : df.columns) {
                    if (!seen.contains(c.name())) { seen.add(c.name()); allColNames.add(c.name()); }
                }
            }
            for (String n : allColNames) {
                Column.DType dtype = Column.DType.STRING;
                for (DataFrame df : dfs) if (df.hasColumn(n)) { dtype = df.column(n).dtype(); break; }
                result.addColumn(n, dtype);
            }
            for (DataFrame df : dfs) {
                for (int i = 0; i < df.rowCount; i++) {
                    int ri = result.addRow();
                    for (String cn : allColNames) {
                        if (df.hasColumn(cn)) result.set(ri, cn, df.get(i, cn));
                    }
                }
            }
        } else {
            for (Column c : dfs.get(0).columns) result.addColumn(c.copy());
            for (int i = 1; i < dfs.size(); i++) {
                for (Column c : dfs.get(i).columns) result.addColumn(c.copy());
            }
        }
        return result;
    }

    // ---- transform ----

    public DataFrame sortValues(String column, boolean ascending) {
        return sortValues(new String[]{column}, new boolean[]{ascending});
    }

    public DataFrame sortValues(String[] cols, boolean[] ascending) {
        List<Integer> order = IntStream.range(0, rowCount).boxed()
            .sorted((a, b) -> {
                for (int i = 0; i < cols.length; i++) {
                    String c = cols[i];
                    boolean asc = i < ascending.length ? ascending[i] : true;
                    Object va = column(c).get(a), vb = column(c).get(b);
                    int cmp = compareVals(va, vb);
                    if (cmp != 0) return asc ? cmp : -cmp;
                }
                return 0;
            })
            .collect(Collectors.toList());
        int[] idx = order.stream().mapToInt(Integer::intValue).toArray();
        return loc(idx);
    }

    /**
     * Drop rows with any null (Pandas default {@code how='any'} / Polars {@code drop_nulls}).
     */
    public DataFrame dropna() throws Exception {
        return dropnaHow("any");
    }

    /**
     * Drop null rows.
     * @param how {@code "any"} drops a row if any cell is null; {@code "all"} only if every cell is null
     */
    public DataFrame dropnaHow(String how) throws Exception {
        boolean any = how == null || !"all".equalsIgnoreCase(how);
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < rowCount; i++) {
            int nulls = 0;
            for (Column c : columns) if (c.get(i) == null) nulls++;
            boolean drop = any ? nulls > 0 : nulls == columns.size();
            if (!drop) keep.add(i);
        }
        return loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /** Polars {@code drop_nulls} alias for {@link #dropna()}. */
    public DataFrame dropNulls() throws Exception { return dropna(); }

    public DataFrame fillna(Object value) throws Exception {
        DataFrame result = copy();
        for (Column c : result.columns) {
            for (int i = 0; i < result.rowCount; i++) {
                if (c.get(i) == null) c.set(i, value);
            }
        }
        return result;
    }

    /** Polars {@code fill_null} alias for {@link #fillna(Object)}. */
    public DataFrame fillNull(Object value) throws Exception { return fillna(value); }

    public DataFrame dropDuplicates() {
        Set<String> seen = new LinkedHashSet<>();
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < rowCount; i++) {
            StringBuilder key = new StringBuilder();
            for (Column c : columns) key.append(c.get(i)).append('|');
            if (seen.add(key.toString())) keep.add(i);
        }
        return loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    public DataFrame select(String... colNames) {
        DataFrame result = DataFrame.create();
        for (String n : colNames) {
            if (hasColumn(n)) result.addColumn(column(n).copy());
        }
        return result;
    }

    /**
     * Select by expressions (Polars-style). Each expression becomes a column named via
     * {@link Expression#suggestedName()} (use {@code .alias("name")} to rename).
     */
    public DataFrame select(Expression... exprs) {
        DataFrame result = DataFrame.create();
        for (Expression e : exprs) {
            Column c = e.evaluate(this);
            if (result.hasColumn(c.name())) result.removeColumn(c.name());
            result.addColumn(c);
        }
        return result;
    }

    /**
     * Filter rows where {@code condition} evaluates to true (Polars-style).
     * Null / non-true results drop the row.
     */
    public DataFrame filter(Expression condition) {
        List<Integer> keep = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) {
            if (Expression.isTrue(condition.eval(i, this))) keep.add(i);
        }
        return loc(keep.stream().mapToInt(Integer::intValue).toArray());
    }

    /**
     * Add or replace a column by evaluating {@code expr} (Polars-style {@code with_column}).
     */
    public DataFrame withColumn(String name, Expression expr) {
        Column computed = expr.evaluate(this);
        Column named = new Column(name, computed.dtype(), computed.data());
        // pad / trim to rowCount
        while (named.size() < rowCount) named.add(null);
        DataFrame result = DataFrame.create();
        boolean replaced = false;
        for (Column c : columns) {
            if (c.name().equals(name)) {
                result.addColumn(named);
                replaced = true;
            } else {
                result.addColumn(c.copy());
            }
        }
        if (!replaced) result.addColumn(named);
        result.syncRowCount();
        return result;
    }

    /**
     * Convenience: {@code withColumn(name, windowExpr.over(spec))}.
     * Example: {@code df.withWindow("rn", row_number(), window().partitionBy("dept").orderBy(asc("sal")));}
     */
    public DataFrame withWindow(String name, Expression windowExpr,
                                org.bytedeco.pytorch.data.dataframe.window.WindowSpec spec) {
        return withColumn(name, windowExpr.over(spec));
    }

    public DataFrame drop(String... colNames) {
        DataFrame result = DataFrame.create();
        for (Column c : columns) {
            if (Arrays.stream(colNames).noneMatch(n -> n.equals(c.name()))) {
                result.addColumn(c.copy());
            }
        }
        return result;
    }

    public List<Boolean> between(String column, double left, double right) {
        return between(column, left, right, true);
    }

    public List<Boolean> between(String column, double left, double right, boolean inclusive) {
        Column col = column(column);
        List<Boolean> mask = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) {
            Object v = col.get(i);
            if (!(v instanceof Number)) { mask.add(false); continue; }
            double d = ((Number) v).doubleValue();
            boolean ok = inclusive ? (d >= left && d <= right) : (d > left && d < right);
            mask.add(ok);
        }
        return mask;
    }

    public DataFrame cumsum() throws Exception {
        DataFrame result = copy();
        for (Column c : result.columns) {
            if (!isNumeric(c.dtype())) continue;
            double running = 0;
            for (int i = 0; i < result.rowCount; i++) {
                Object v = c.get(i);
                if (v instanceof Number) { running += ((Number) v).doubleValue(); c.set(i, running); }
            }
        }
        return result;
    }

    public DataFrame shift(String column, int periods) {
        DataFrame result = copy();
        Column src = result.column(column);
        for (int i = 0; i < result.rowCount; i++) {
            int srcIdx = i - periods;
            if (srcIdx >= 0 && srcIdx < rowCount) src.set(i, column(column).get(srcIdx));
            else src.set(i, null);
        }
        return result;
    }

    public DataFrame pct_change(int periods) throws Exception {
        DataFrame result = copy();
        for (Column c : result.columns) {
            if (!isNumeric(c.dtype())) continue;
            for (int i = 0; i < result.rowCount; i++) {
                int prev = i - periods;
                if (prev >= 0 && prev < rowCount) {
                    Object cv = c.get(i), pv = c.get(prev);
                    if (cv instanceof Number && pv instanceof Number) {
                        double denom = ((Number) pv).doubleValue();
                        c.set(i, denom != 0
                            ? (((Number) cv).doubleValue() - denom) / denom
                            : Double.NaN);
                    }
                }
            }
        }
        return result;
    }

    public DataFrame clip(Double lower, Double upper) throws Exception {
        DataFrame result = copy();
        for (Column c : result.columns) {
            if (!isNumeric(c.dtype())) continue;
            for (int i = 0; i < result.rowCount; i++) {
                Object v = c.get(i);
                if (!(v instanceof Number)) continue;
                double d = ((Number) v).doubleValue();
                if (lower != null && d < lower) c.set(i, lower);
                else if (upper != null && d > upper) c.set(i, upper);
            }
        }
        return result;
    }

    // ---- utility ----

    public void info() {
        System.out.println("DataFrame: " + rowCount + " rows x " + columnCount() + " columns");
        for (Column c : columns) {
            System.out.printf("  %-20s %-10s %d non-null%n",
                c.name(), c.dtype().name(), c.data().stream().filter(Objects::nonNull).count());
        }
    }

    /**
     * Print column names and dtypes (Spark / Polars style).
     * Nested LIST/VECTOR/MAP/STRUCT columns are labelled with their cell runtime shape when available.
     */
    public void printSchema() {
        System.out.println("root");
        System.out.println(" |-- " + rowCount + " rows x " + columnCount() + " columns");
        for (Column c : columns) {
            String extra = schemaHint(c);
            System.out.printf(" |-- %s: %s%s%n", c.name(), c.dtype().name(), extra);
        }
    }

    private static String schemaHint(Column c) {
        if (c.size() == 0) return "";
        Object sample = null;
        for (int i = 0; i < c.size(); i++) {
            sample = c.get(i);
            if (sample != null) break;
        }
        if (sample == null) return " (nullable)";
        if (sample instanceof long[]) return " (long[" + ((long[]) sample).length + "])";
        if (sample instanceof int[]) return " (int[" + ((int[]) sample).length + "])";
        if (sample instanceof float[]) return " (float[" + ((float[]) sample).length + "])";
        if (sample instanceof double[]) return " (double[" + ((double[]) sample).length + "])";
        if (sample instanceof List) return " (List, len=" + ((List<?>) sample).size() + ")";
        if (sample instanceof Map) return " (Map, size=" + ((Map<?, ?>) sample).size() + ")";
        return "";
    }

    /** Print the first 20 rows to stdout (Pandas / Spark {@code show()}). */
    public void show() { show(20); }

    /** Print the first {@code n} rows to stdout. */
    public void show(int n) {
        System.out.print(toString(n));
        if (n >= rowCount) {
            System.out.printf("[%d rows x %d columns]%n", rowCount, columnCount());
        }
    }

    /**
     * Print a Pandas-style numeric summary ({@link #describeFrame()}) to stdout.
     * Returns the summary DataFrame for further use.
     */
    public DataFrame describeAndShow() {
        DataFrame d = describeFrame();
        d.show(d.rowCount());
        return d;
    }

    public String toString() { return toString(20); }

    public String toString(int maxRows) {
        if (rowCount == 0) return "Empty DataFrame\n";
        StringBuilder sb = new StringBuilder();
        int[] widths = new int[columnCount()];
        for (int i = 0; i < columnCount(); i++) {
            final int colIdx = i;
            widths[i] = Math.min(50, Math.max(
                columns.get(i).name().length(),
                IntStream.range(0, Math.min(maxRows, rowCount))
                    .map(r -> formatCell(columns.get(colIdx).get(r)).length())
                    .max().orElse(0)));
        }
        for (int i = 0; i < columnCount(); i++) {
            if (i > 0) sb.append(" | ");
            String n = columns.get(i).name();
            sb.append(String.format("%-" + widths[i] + "s", n.substring(0, Math.min(n.length(), widths[i]))));
        }
        sb.append("\n");
        for (int i = 0; i < columnCount(); i++) {
            if (i > 0) sb.append("-+-");
            sb.append("-".repeat(widths[i]));
        }
        sb.append("\n");
        int show = Math.min(maxRows, rowCount);
        for (int r = 0; r < show; r++) {
            for (int c = 0; c < columnCount(); c++) {
                if (c > 0) sb.append(" | ");
                String s = formatCell(columns.get(c).get(r));
                sb.append(String.format("%-" + widths[c] + "s", s.substring(0, Math.min(s.length(), widths[c]))));
            }
            sb.append("\n");
        }
        if (rowCount > maxRows) sb.append(String.format("[%d rows total]%n", rowCount));
        return sb.toString();
    }

    /** Human-readable cell for tables (arrays / lists expand; no {@code [J@hash}). */
    static String formatCell(Object v) {
        if (v == null) return "null";
        if (v instanceof long[]) return Arrays.toString((long[]) v);
        if (v instanceof int[]) return Arrays.toString((int[]) v);
        if (v instanceof short[]) return Arrays.toString((short[]) v);
        if (v instanceof byte[]) {
            byte[] b = (byte[]) v;
            // short binary preview
            if (b.length > 16) return "bytes[" + b.length + "]";
            return Arrays.toString(b);
        }
        if (v instanceof float[]) return Arrays.toString((float[]) v);
        if (v instanceof double[]) return Arrays.toString((double[]) v);
        if (v instanceof boolean[]) return Arrays.toString((boolean[]) v);
        if (v instanceof Object[]) return Arrays.deepToString((Object[]) v);
        if (v instanceof List || v instanceof Map) return String.valueOf(v);
        return v.toString();
    }

    public DataFrame copy() {
        DataFrame result = DataFrame.create();
        for (Column c : columns) result.addColumn(c.copy());
        return result;
    }

    public Map<String, Object> toDict(int rowIndex) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (Column c : columns) m.put(c.name(), c.get(rowIndex));
        return m;
    }

    public List<Map<String, Object>> toRecords() {
        List<Map<String, Object>> records = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) records.add(toDict(i));
        return records;
    }

    @Override public void close() {
        for (Column c : columns) {
            try { c.close(); } catch (Exception ignored) {}
        }
        for (int i = resources.size() - 1; i >= 0; i--) {
            try { resources.get(i).close(); } catch (Exception ignored) {}
        }
        resources.clear();
    }

    /** Register an external resource (e.g. Arrow allocator/root) closed with this frame. */
    public void addResource(AutoCloseable resource) {
        if (resource != null) resources.add(resource);
    }

    /** Public sync for Arrow reader after attaching vector-backed columns. */
    public void syncRowCountPublic() { syncRowCount(); }

    public Schema schema() { return Schema.fromDataFrame(this); }

    /** Polars-style alias for {@link #groupby(String...)}. */
    public GroupedDataFrame groupBy(String... cols) { return groupby(cols); }

    // ---- reshape: pivot / melt / explode / dummies ----

    /**
     * Pivot with FIRST aggregation (pandas {@code pivot}).
     * Empty cells become {@code null} (not 0).
     */
    public DataFrame pivot(String index, String columns, String values) {
        return pivotTable(index, columns, values, AggFunction.FIRST);
    }

    /**
     * Pivot table with aggregation. Composite (index, column) keys avoid delimiter collisions.
     */
    public DataFrame pivotTable(String index, String columns, String values, AggFunction aggFunc) {
        Column indexCol = column(index);
        Column colsCol = column(columns);
        Column valsCol = column(values);

        LinkedHashSet<Object> indexValues = new LinkedHashSet<>();
        LinkedHashSet<Object> columnValues = new LinkedHashSet<>();
        Map<AbstractMap.SimpleImmutableEntry<Object, Object>, List<Double>> pivotData = new HashMap<>();

        for (int i = 0; i < rowCount; i++) {
            Object idxVal = DataValues.unwrap(indexCol.get(i));
            Object colVal = DataValues.unwrap(colsCol.get(i));
            indexValues.add(idxVal);
            columnValues.add(colVal);
            double d = DataValues.asDouble(valsCol.get(i));
            if (!Double.isNaN(d)) {
                AbstractMap.SimpleImmutableEntry<Object, Object> key =
                    new AbstractMap.SimpleImmutableEntry<>(idxVal, colVal);
                pivotData.computeIfAbsent(key, k -> new ArrayList<>()).add(d);
            }
        }

        DataFrame result = DataFrame.create();
        result.addColumn(index, indexCol.dtype());
        for (Object colVal : columnValues) {
            String name = colVal == null ? "null" : colVal.toString();
            // avoid collision with index col name
            if (name.equals(index)) name = name + "_1";
            result.addColumn(name, Column.DType.FLOAT64);
        }

        List<Object> colList = new ArrayList<>(columnValues);
        for (Object idxVal : indexValues) {
            int row = result.addEmptyRow();
            result.set(row, index, idxVal);
            int ci = 0;
            for (Object colVal : colList) {
                String name = colVal == null ? "null" : colVal.toString();
                if (name.equals(index)) name = name + "_1";
                AbstractMap.SimpleImmutableEntry<Object, Object> key =
                    new AbstractMap.SimpleImmutableEntry<>(idxVal, colVal);
                List<Double> vals = pivotData.get(key);
                if (vals == null || vals.isEmpty()) {
                    result.set(row, name, null);
                } else {
                    result.set(row, name, aggregateList(vals, aggFunc == null ? AggFunction.FIRST : aggFunc));
                }
                ci++;
            }
        }
        return result;
    }

    private static double aggregateList(List<Double> values, AggFunction func) {
        if (values == null || values.isEmpty()) return Double.NaN;
        return switch (func) {
            case SUM -> values.stream().mapToDouble(Double::doubleValue).sum();
            case MEAN -> values.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
            case MAX -> values.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
            case MIN -> values.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN);
            case COUNT, NUNIQUE -> values.size();
            case MEDIAN -> {
                List<Double> s = new ArrayList<>(values);
                Collections.sort(s);
                int n = s.size();
                yield (n % 2 == 1) ? s.get(n / 2) : (s.get(n / 2 - 1) + s.get(n / 2)) / 2.0;
            }
            case STD -> {
                if (values.size() < 2) yield Double.NaN;
                double m = values.stream().mapToDouble(d -> d).average().orElse(0);
                double ss = 0; for (double v : values) ss += (v - m) * (v - m);
                yield Math.sqrt(ss / (values.size() - 1));
            }
            case VAR -> {
                if (values.size() < 2) yield Double.NaN;
                double m = values.stream().mapToDouble(d -> d).average().orElse(0);
                double ss = 0; for (double v : values) ss += (v - m) * (v - m);
                yield ss / (values.size() - 1);
            }
            case LAST -> values.get(values.size() - 1);
            default -> values.get(0); // FIRST and others
        };
    }

    /**
     * Unpivot wide → long (pandas {@code melt}). Preserves value dtype when all value
     * columns share the same dtype; otherwise values are STRING.
     */
    public DataFrame melt(List<String> idVars, List<String> valueVars, String varName, String valueName) {
        if (varName == null || varName.isEmpty()) varName = "variable";
        if (valueName == null || valueName.isEmpty()) valueName = "value";
        if (valueVars == null || valueVars.isEmpty()) {
            valueVars = new ArrayList<>();
            Set<String> ids = idVars == null ? Set.of() : new HashSet<>(idVars);
            for (Column c : columns) if (!ids.contains(c.name())) valueVars.add(c.name());
        }

        Column.DType valueDtype = null;
        boolean homogeneous = true;
        for (String vv : valueVars) {
            Column.DType dt = column(vv).dtype();
            if (valueDtype == null) valueDtype = dt;
            else if (valueDtype != dt) { homogeneous = false; break; }
        }
        if (!homogeneous || valueDtype == null) valueDtype = Column.DType.STRING;

        DataFrame result = DataFrame.create();
        if (idVars != null) {
            for (String id : idVars) result.addColumn(id, column(id).dtype());
        }
        result.addColumn(varName, Column.DType.STRING);
        result.addColumn(valueName, valueDtype);

        for (int i = 0; i < rowCount; i++) {
            for (String vv : valueVars) {
                int row = result.addEmptyRow();
                int k = 0;
                if (idVars != null) {
                    for (String id : idVars) {
                        result.set(row, id, column(id).get(i));
                        k++;
                    }
                }
                result.set(row, varName, vv);
                Object v = column(vv).get(i);
                if (!homogeneous && v != null) v = v.toString();
                result.set(row, valueName, v);
            }
        }
        return result;
    }

    public DataFrame melt(List<String> idVars, List<String> valueVars) {
        return melt(idVars, valueVars, "variable", "value");
    }

    /** Polars {@code unpivot} alias for {@link #melt}. */
    public DataFrame unpivot(List<String> idVars, List<String> valueVars) {
        return melt(idVars, valueVars);
    }

    public DataFrame unpivot(List<String> idVars, List<String> valueVars, String varName, String valueName) {
        return melt(idVars, valueVars, varName, valueName);
    }

    /**
     * Transpose rows ↔ columns (Pandas {@code transpose} / Polars {@code transpose}).
     * First column becomes header names when {@code includeHeader=true}; otherwise
     * columns are named {@code column_0..n}.
     */
    public DataFrame transpose() {
        return transpose(false);
    }

    public DataFrame transpose(boolean includeHeader) {
        DataFrame result = DataFrame.create();
        if (rowCount == 0 && columns.isEmpty()) return result;

        if (includeHeader && !columns.isEmpty()) {
            // use first column values as new column names; remaining cols become rows
            Column header = columns.get(0);
            result.addColumn("index", Column.DType.STRING);
            for (int r = 0; r < rowCount; r++) {
                Object h = header.get(r);
                String name = h == null ? ("column_" + r) : h.toString();
                // de-dup
                String base = name;
                int k = 1;
                while (result.hasColumn(name)) name = base + "_" + (k++);
                result.addColumn(name, Column.DType.STRING);
            }
            for (int c = 1; c < columns.size(); c++) {
                int ri = result.addEmptyRow();
                result.set(ri, "index", columns.get(c).name());
                for (int r = 0; r < rowCount; r++) {
                    Object v = columns.get(c).get(r);
                    result.set(ri, result.column(r + 1).name(), v == null ? null : v.toString());
                }
            }
            return result;
        }

        // standard: new columns = old rows; first column holds old column names
        result.addColumn("index", Column.DType.STRING);
        for (int r = 0; r < rowCount; r++) {
            result.addColumn("column_" + r, Column.DType.STRING);
        }
        for (Column col : columns) {
            int ri = result.addEmptyRow();
            result.set(ri, "index", col.name());
            for (int r = 0; r < rowCount; r++) {
                Object v = col.get(r);
                result.set(ri, "column_" + r, v == null ? null : v.toString());
            }
        }
        return result;
    }

    /**
     * Select columns by dtype (Pandas {@code select_dtypes}).
     * Pass one or more {@link Column.DType} values to keep.
     */
    public DataFrame selectDtypes(Column.DType... types) {
        if (types == null || types.length == 0) return DataFrame.create();
        Set<Column.DType> want = EnumSet.noneOf(Column.DType.class);
        for (Column.DType t : types) if (t != null) want.add(t);
        List<String> keep = new ArrayList<>();
        for (Column c : columns) {
            if (want.contains(c.dtype())) keep.add(c.name());
        }
        return select(keep.toArray(new String[0]));
    }

    /** Alias of {@link #selectDtypes(Column.DType...)}. */
    public DataFrame selectDtypes(Collection<Column.DType> types) {
        if (types == null || types.isEmpty()) return DataFrame.create();
        return selectDtypes(types.toArray(new Column.DType[0]));
    }

    /**
     * Add a 0-based row index column (Polars {@code with_row_index}).
     * Default name is {@code "index"}.
     */
    public DataFrame withRowIndex() {
        return withRowIndex("index", 0);
    }

    public DataFrame withRowIndex(String name) {
        return withRowIndex(name, 0);
    }

    public DataFrame withRowIndex(String name, long offset) {
        String colName = name == null || name.isEmpty() ? "index" : name;
        if (hasColumn(colName)) {
            // de-dup name
            String base = colName;
            int k = 1;
            while (hasColumn(colName) || colName.equals(base + "_" + k)) {
                colName = base + "_" + (k++);
                if (!hasColumn(colName)) break;
            }
        }
        DataFrame result = DataFrame.create();
        // build index data first so column sizes stay aligned
        List<Object> idxData = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) idxData.add(offset + i);
        result.addColumn(new Column(colName, Column.DType.INT64, idxData));
        for (Column c : columns) result.addColumn(c.copy());
        result.syncRowCount();
        return result;
    }

    /**
     * Pandas-style set_index: move {@code col} to the front as a named index column
     * (DataFrame has no separate Index object — column is reordered to position 0).
     */
    public DataFrame setIndex(String col) {
        if (!hasColumn(col)) throw new IllegalArgumentException("No such column: " + col);
        List<String> order = new ArrayList<>();
        order.add(col);
        for (Column c : columns) if (!c.name().equals(col)) order.add(c.name());
        return reorderColumns(order.toArray(new String[0]));
    }

    /**
     * Drop a row-index column if present (default name {@code "index"}), otherwise no-op.
     * Polars {@code drop_index} / Pandas {@code reset_index(drop=True)} analogue.
     */
    public DataFrame resetIndex() {
        return resetIndex("index", true);
    }

    public DataFrame resetIndex(String indexName, boolean drop) {
        String name = indexName == null ? "index" : indexName;
        if (!hasColumn(name)) {
            // just add a fresh 0..n-1 index if not dropping
            return drop ? copy() : withRowIndex(name);
        }
        if (drop) {
            DataFrame result = copy();
            result.removeColumn(name);
            return result;
        }
        // keep existing index column, add a new positional index
        return withRowIndex("_pos");
    }

    /** Horizontal stack (Polars {@code hstack} / Pandas {@code concat(axis=1)}). */
    public static DataFrame hstack(DataFrame... frames) throws Exception {
        if (frames == null || frames.length == 0) return DataFrame.create();
        return concat(Arrays.asList(frames), 1);
    }

    public static DataFrame hstack(List<DataFrame> frames) throws Exception {
        return concat(frames, 1);
    }

    /** Vertical stack (Polars {@code vstack} / Pandas {@code concat(axis=0)}). */
    public static DataFrame vstack(DataFrame... frames) throws Exception {
        if (frames == null || frames.length == 0) return DataFrame.create();
        return concat(Arrays.asList(frames), 0);
    }

    public static DataFrame vstack(List<DataFrame> frames) throws Exception {
        return concat(frames, 0);
    }

    /**
     * Bin a numeric column into discrete labels (Pandas {@code cut}).
     * Adds/replaces column {@code outName} with bin labels.
     */
    public DataFrame cut(String columnName, double[] bins, String[] labels, String outName) {
        String out = outName == null || outName.isEmpty() ? columnName + "_bin" : outName;
        return withColumn(out, Expression.col(columnName).cut(bins, labels));
    }

    public DataFrame cut(String columnName, double[] bins, String[] labels) {
        return cut(columnName, bins, labels, columnName + "_bin");
    }

    public DataFrame cut(String columnName, double[] bins) {
        return cut(columnName, bins, null, columnName + "_bin");
    }

    /** Sort alias matching Polars {@code sort}. */
    public DataFrame sort(String by) {
        return sortValues(by, true);
    }

    public DataFrame sort(String by, boolean ascending) {
        return sortValues(by, ascending);
    }

    public DataFrame sort(String[] by, boolean[] ascending) {
        return sortValues(by, ascending);
    }

    /**
     * Element-wise membership test for a column → boolean list
     * (Pandas {@code Series.isin} / expression {@link Expression#isIn}).
     */
    public List<Boolean> isin(String columnName, Object... values) {
        Set<Object> set = new HashSet<>();
        if (values != null) Collections.addAll(set, values);
        Column c = column(columnName);
        List<Boolean> out = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) {
            Object v = c.get(i);
            out.add(v != null && set.contains(v));
        }
        return out;
    }

    /** Return a DataFrame with a boolean column {@code outName} for isin. */
    public DataFrame isin(String columnName, Collection<?> values, String outName) {
        Object[] arr = values == null ? new Object[0] : values.toArray();
        return withColumn(outName, Expression.col(columnName).isIn(arr));
    }

    /**
     * Explode a list-like column into one row per element. Non-list cells become a single-element list.
     */
    public DataFrame explode(String columnName) {
        Column explodeCol = column(columnName);
        DataFrame result = DataFrame.create();
        for (Column c : columns) result.addColumn(c.name(), c.dtype());

        for (int i = 0; i < rowCount; i++) {
            Object v = explodeCol.get(i);
            List<?> list;
            if (v instanceof List) {
                list = (List<?>) v;
            } else if (v instanceof Object[]) {
                list = Arrays.asList((Object[]) v);
            } else if (v instanceof org.bytedeco.pytorch.data.dataframe.dtype.DataValue) {
                Object raw = ((org.bytedeco.pytorch.data.dataframe.dtype.DataValue) v).toArrowCompatible();
                if (raw instanceof List) list = (List<?>) raw;
                else list = Collections.singletonList(v);
            } else {
                list = Collections.singletonList(v);
            }
            if (list.isEmpty()) {
                int row = result.addEmptyRow();
                for (Column c : columns) {
                    result.set(row, c.name(), c.name().equals(columnName) ? null : c.get(i));
                }
            } else {
                for (Object elem : list) {
                    int row = result.addEmptyRow();
                    for (Column c : columns) {
                        result.set(row, c.name(), c.name().equals(columnName) ? elem : c.get(i));
                    }
                }
            }
        }
        return result;
    }

    /**
     * One-hot encode a column. Dummy columns are INT32 0/1; source column is kept.
     */
    public DataFrame getDummies(String columnName, String prefix) {
        Column col = column(columnName);
        LinkedHashSet<String> uniques = new LinkedHashSet<>();
        for (int i = 0; i < rowCount; i++) {
            Object v = DataValues.unwrap(col.get(i));
            uniques.add(v == null ? "null" : v.toString());
        }

        DataFrame result = copy();
        String pfx = prefix == null ? columnName : prefix;
        List<String> dummyNames = new ArrayList<>();
        for (String u : uniques) {
            String name = pfx + "_" + u;
            // ensure unique name
            String base = name;
            int n = 1;
            while (result.hasColumn(name)) name = base + "_" + (n++);
            result.addColumn(name, Column.DType.INT32);
            // size dummy column to rowCount via set on empty rows
            Column dcol = result.column(name);
            while (dcol.size() < result.rowCount) dcol.add(0);
            for (int i = 0; i < result.rowCount; i++) dcol.set(i, 0);
            dummyNames.add(name);
        }

        for (int i = 0; i < result.rowCount; i++) {
            Object raw = DataValues.unwrap(col.get(i));
            String v = raw == null ? "null" : raw.toString();
            String target = pfx + "_" + v;
            for (String dn : dummyNames) {
                result.set(i, dn, dn.equals(target) ? 1 : 0);
            }
        }
        return result;
    }

    public DataFrame getDummies(String columnName) {
        return getDummies(columnName, columnName);
    }

    /** Factorize a column into integer codes + unique labels (order of appearance). */
    public FactorizeResult factorize(String columnName) {
        Column col = column(columnName);
        LinkedHashMap<String, Integer> map = new LinkedHashMap<>();
        int[] codes = new int[rowCount];
        for (int i = 0; i < rowCount; i++) {
            Object v = DataValues.unwrap(col.get(i));
            String key = v == null ? "null" : v.toString();
            Integer code = map.get(key);
            if (code == null) {
                code = map.size();
                map.put(key, code);
            }
            codes[i] = code;
        }
        return new FactorizeResult(codes, map.keySet().toArray(new String[0]));
    }

    /**
     * Cross-tabulation of two factors with optional values + aggregation.
     * {@code aggFunc}: "count" (default), "sum", "mean".
     */
    public static DataFrame crosstab(DataFrame df, String indexCol, String columnsCol,
                                     String valuesCol, String aggFunc) {
        Column idxC = df.column(indexCol);
        Column colC = df.column(columnsCol);
        LinkedHashSet<String> rowKeys = new LinkedHashSet<>();
        LinkedHashSet<String> colKeys = new LinkedHashSet<>();
        for (int i = 0; i < df.rowCount; i++) {
            Object r = DataValues.unwrap(idxC.get(i));
            Object c = DataValues.unwrap(colC.get(i));
            rowKeys.add(r == null ? "null" : r.toString());
            colKeys.add(c == null ? "null" : c.toString());
        }

        Map<String, Map<String, List<Double>>> acc = new LinkedHashMap<>();
        for (String rk : rowKeys) {
            Map<String, List<Double>> m = new LinkedHashMap<>();
            for (String ck : colKeys) m.put(ck, new ArrayList<>());
            acc.put(rk, m);
        }
        for (int i = 0; i < df.rowCount; i++) {
            Object ro = DataValues.unwrap(idxC.get(i));
            Object co = DataValues.unwrap(colC.get(i));
            String rk = ro == null ? "null" : ro.toString();
            String ck = co == null ? "null" : co.toString();
            double v = 1.0;
            if (valuesCol != null) v = DataValues.asDouble(df.column(valuesCol).get(i));
            if (Double.isNaN(v)) v = 0.0;
            acc.get(rk).get(ck).add(v);
        }

        DataFrame result = DataFrame.create();
        result.addColumn(indexCol, Column.DType.STRING);
        for (String ck : colKeys) result.addColumn(ck, Column.DType.FLOAT64);
        String agg = aggFunc == null ? "count" : aggFunc.toLowerCase(Locale.ROOT);
        for (String rk : rowKeys) {
            int row = result.addEmptyRow();
            result.set(row, indexCol, rk);
            for (String ck : colKeys) {
                List<Double> vals = acc.get(rk).get(ck);
                double out = switch (agg) {
                    case "sum" -> vals.stream().mapToDouble(d -> d).sum();
                    case "mean" -> vals.isEmpty() ? Double.NaN
                        : vals.stream().mapToDouble(d -> d).average().orElse(Double.NaN);
                    default -> (double) vals.size();
                };
                result.set(row, ck, out);
            }
        }
        return result;
    }

    public static DataFrame crosstab(DataFrame df, String indexCol, String columnsCol) {
        return crosstab(df, indexCol, columnsCol, null, "count");
    }

    /** Value counts sorted by frequency descending. */
    public Map<Object, Integer> valueCounts(String columnName) {
        Map<Object, Integer> result = new LinkedHashMap<>();
        Column col = column(columnName);
        for (int i = 0; i < rowCount; i++) {
            Object val = DataValues.unwrap(col.get(i));
            result.merge(val, 1, Integer::sum);
        }
        return result.entrySet().stream()
            .sorted(Map.Entry.<Object, Integer>comparingByValue().reversed())
            .collect(LinkedHashMap::new, (m, e) -> m.put(e.getKey(), e.getValue()), Map::putAll);
    }

    // ---- stats matrix: corr / cov ----

    /** Pearson correlation matrix over numeric columns. */
    public DataFrame corr() {
        List<Column> numericCols = new ArrayList<>();
        for (Column c : columns) if (isNumeric(c.dtype())) numericCols.add(c);

        DataFrame result = DataFrame.create();
        result.addColumn("index", Column.DType.STRING);
        for (Column c : numericCols) result.addColumn(c.name(), Column.DType.FLOAT64);

        for (Column c1 : numericCols) {
            int row = result.addEmptyRow();
            result.set(row, "index", c1.name());
            for (Column c2 : numericCols) {
                result.set(row, c2.name(), computeCorrelation(c1, c2));
            }
        }
        return result;
    }

    /** Sample covariance matrix over numeric columns. */
    public DataFrame cov() {
        List<Column> numericCols = new ArrayList<>();
        for (Column c : columns) if (isNumeric(c.dtype())) numericCols.add(c);

        DataFrame result = DataFrame.create();
        result.addColumn("index", Column.DType.STRING);
        for (Column c : numericCols) result.addColumn(c.name(), Column.DType.FLOAT64);

        for (Column c1 : numericCols) {
            int row = result.addEmptyRow();
            result.set(row, "index", c1.name());
            for (Column c2 : numericCols) {
                result.set(row, c2.name(), computeCovariance(c1, c2));
            }
        }
        return result;
    }

    private double computeCorrelation(Column c1, Column c2) {
        List<Double> x = new ArrayList<>(), y = new ArrayList<>();
        int n = Math.min(c1.size(), c2.size());
        for (int i = 0; i < n; i++) {
            double a = DataValues.asDouble(c1.get(i));
            double b = DataValues.asDouble(c2.get(i));
            if (!Double.isNaN(a) && !Double.isNaN(b)) { x.add(a); y.add(b); }
        }
        int nv = x.size();
        if (nv < 2) return Double.NaN;
        double mx = x.stream().mapToDouble(d -> d).average().orElse(0);
        double my = y.stream().mapToDouble(d -> d).average().orElse(0);
        double num = 0, dx = 0, dy = 0;
        for (int i = 0; i < nv; i++) {
            double a = x.get(i) - mx, b = y.get(i) - my;
            num += a * b; dx += a * a; dy += b * b;
        }
        double den = Math.sqrt(dx * dy);
        return den == 0 ? Double.NaN : num / den;
    }

    private double computeCovariance(Column c1, Column c2) {
        List<Double> x = new ArrayList<>(), y = new ArrayList<>();
        int n = Math.min(c1.size(), c2.size());
        for (int i = 0; i < n; i++) {
            double a = DataValues.asDouble(c1.get(i));
            double b = DataValues.asDouble(c2.get(i));
            if (!Double.isNaN(a) && !Double.isNaN(b)) { x.add(a); y.add(b); }
        }
        int nv = x.size();
        if (nv < 2) return Double.NaN;
        double mx = x.stream().mapToDouble(d -> d).average().orElse(0);
        double my = y.stream().mapToDouble(d -> d).average().orElse(0);
        double sum = 0;
        for (int i = 0; i < nv; i++) sum += (x.get(i) - mx) * (y.get(i) - my);
        return sum / (nv - 1);
    }

    // ---- rank / sample / unique / apply ----

    /**
     * Rank values in a column. {@code method}: average|min|max|first|dense.
     * Returns a copy with a new {@code <column>_rank} FLOAT64 column.
     */
    public DataFrame rank(String columnName, String method, boolean ascending) {
        Column col = column(columnName);
        double[] vals = new double[rowCount];
        for (int i = 0; i < rowCount; i++) vals[i] = DataValues.asDouble(col.get(i));

        Integer[] sortedIdx = IntStream.range(0, rowCount).boxed()
            .sorted((a, b) -> {
                double da = vals[a], db = vals[b];
                if (Double.isNaN(da) && Double.isNaN(db)) return 0;
                if (Double.isNaN(da)) return 1;
                if (Double.isNaN(db)) return -1;
                int cmp = Double.compare(da, db);
                return ascending ? cmp : -cmp;
            }).toArray(Integer[]::new);

        double[] ranks = new double[rowCount];
        Arrays.fill(ranks, Double.NaN);
        String m = method == null ? "average" : method.toLowerCase(Locale.ROOT);
        int i = 0;
        int dense = 0;
        while (i < sortedIdx.length) {
            if (Double.isNaN(vals[sortedIdx[i]])) break;
            int j = i;
            while (j + 1 < sortedIdx.length
                && !Double.isNaN(vals[sortedIdx[j + 1]])
                && vals[sortedIdx[j + 1]] == vals[sortedIdx[i]]) j++;
            dense++;
            for (int k = i; k <= j; k++) {
                ranks[sortedIdx[k]] = switch (m) {
                    case "min" -> i + 1;
                    case "max" -> j + 1;
                    case "first" -> k + 1;
                    case "dense" -> dense;
                    default -> (i + j + 2) / 2.0; // average
                };
            }
            i = j + 1;
        }

        DataFrame result = copy();
        String outName = columnName + "_rank";
        if (result.hasColumn(outName)) result.removeColumn(outName);
        result.addColumn(outName, Column.DType.FLOAT64);
        Column rc = result.column(outName);
        while (rc.size() < result.rowCount) rc.add(null);
        for (int r = 0; r < result.rowCount; r++) rc.set(r, ranks[r]);
        return result;
    }

    public DataFrame rank(String columnName) {
        return rank(columnName, "average", true);
    }

    /** Sample {@code n} rows without replacement. */
    public DataFrame sample(int n) {
        return sample(n, null);
    }

    public DataFrame sample(int n, Long seed) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        if (n > rowCount) n = rowCount;
        Random rng = seed == null ? new Random() : new Random(seed);
        List<Integer> idx = IntStream.range(0, rowCount).boxed().collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(idx, rng);
        int[] take = idx.subList(0, n).stream().mapToInt(Integer::intValue).toArray();
        return loc(take);
    }

    public DataFrame sampleFrac(double fraction) {
        return sampleFrac(fraction, null);
    }

    public DataFrame sampleFrac(double fraction, Long seed) {
        if (fraction < 0 || fraction > 1) throw new IllegalArgumentException("fraction in [0,1]");
        int n = (int) Math.round(rowCount * fraction);
        return sample(n, seed);
    }

    /**
     * Weighted sample (Pandas {@code sample(n, weights=...)}).
     * @param weights per-row weights (length = rowCount); null = uniform
     * @param replace with-replacement if true
     */
    public DataFrame sample(int n, double[] weights, boolean replace, Long seed) {
        return AdvancedOps.sampleWeighted(this, n, weights, replace, seed);
    }

    /** Pandas {@code DataFrame.pipe(func)} — chain external function. */
    public <R> R pipe(java.util.function.Function<DataFrame, R> func) {
        return AdvancedOps.pipe(this, func);
    }

    /** Pandas {@code DataFrame.take(indices)} — positional row selection. */
    public DataFrame take(int... indices) {
        return AdvancedOps.take(this, indices);
    }

    /** Pandas {@code mask(cond, other)} — replace where boolean column is true. */
    public DataFrame mask(String boolCol, Object other) {
        return AdvancedOps.mask(this, boolCol, other);
    }

    /** Pandas {@code where(cond, other)} — keep where boolean column is true, else other. */
    public DataFrame whereMask(String boolCol, Object other) {
        return AdvancedOps.where(this, boolCol, other);
    }

    /** Pandas {@code combine_first(other)}. */
    public DataFrame combineFirst(DataFrame other) throws Exception {
        return AdvancedOps.combineFirst(this, other);
    }

    /** Pandas {@code combine(other, func)}. */
    public DataFrame combine(DataFrame other, java.util.function.BiFunction<Object, Object, Object> func)
            throws Exception {
        return AdvancedOps.combine(this, other, func);
    }

    /**
     * Pandas {@code reindex(labels)} on a key column with optional fill method
     * ({@code null}/ {@code "ffill"}/ {@code "bfill"}).
     */
    public DataFrame reindex(String keyCol, List<?> labels, String method) throws Exception {
        return AdvancedOps.reindex(this, keyCol, labels, method);
    }

    public DataFrame reindex(String keyCol, List<?> labels) throws Exception {
        return reindex(keyCol, labels, null);
    }

    /** Pandas {@code Series.searchsorted}. Column must be sorted ascending. */
    public int[] searchsorted(String col, Object[] values, String side) {
        return AdvancedOps.searchsorted(this, col, values, side);
    }

    public int[] searchsorted(String col, Object... values) {
        return searchsorted(col, values, "left");
    }

    /**
     * Interpolate nulls in a numeric column.
     * Methods: {@code "linear"}, {@code "ffill"}, {@code "bfill"}, {@code "nearest"}.
     */
    public DataFrame interpolate(String col, String method) {
        return AdvancedOps.interpolate(this, col, method);
    }

    public DataFrame interpolate(String col) {
        return interpolate(col, "linear");
    }

    /** Pandas {@code qcut} — equal-frequency binning; adds {@code col_qcut} column. */
    public DataFrame qcut(String col, int q) {
        return AdvancedOps.qcut(this, col, q);
    }

    public DataFrame qcut(String col, int q, String[] labels, String duplicates) {
        return AdvancedOps.qcut(this, col, q, labels, duplicates);
    }

    public boolean isMonotonicIncreasing(String col) {
        return AdvancedOps.isMonotonicIncreasing(this, col);
    }

    public boolean isMonotonicDecreasing(String col) {
        return AdvancedOps.isMonotonicDecreasing(this, col);
    }

    public boolean isUnique(String col) {
        return AdvancedOps.isUnique(this, col);
    }

    /** Per-column memory usage in bytes (deep string accounting if {@code deep}). */
    public Map<String, Long> memoryUsage(boolean deep) {
        return AdvancedOps.memoryUsage(this, deep);
    }

    public Map<String, Long> memoryUsage() {
        return memoryUsage(true);
    }

    /** Polars {@code estimate_size()} — total estimated bytes. */
    public long estimateSize() {
        return AdvancedOps.estimateSize(this);
    }

    /**
     * Unnest a STRUCT/MAP column into top-level columns (Polars {@code unnest} /
     * {@code struct.unnest}). Field names become column names; conflicts get {@code _1} suffix.
     */
    public DataFrame unnest(String structCol) {
        Column src = column(structCol);
        // collect union of field names in order of first appearance
        LinkedHashSet<String> fields = new LinkedHashSet<>();
        List<Map<String, Object>> maps = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) {
            Object v = src.get(i);
            Map<String, Object> m = null;
            if (v instanceof Map<?, ?> raw) {
                m = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : raw.entrySet()) {
                    m.put(String.valueOf(e.getKey()), e.getValue());
                }
            } else if (v instanceof org.bytedeco.pytorch.data.dataframe.dtype.StructData sd) {
                m = new LinkedHashMap<>(sd.getFieldValues());
            } else if (v != null) {
                m = org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec.asStringMap(v);
            }
            if (m == null) m = Map.of();
            fields.addAll(m.keySet());
            maps.add(m);
        }
        DataFrame result = DataFrame.create();
        for (Column c : columns) {
            if (!c.name().equals(structCol)) result.addColumn(c.copy());
        }
        for (String f : fields) {
            String name = f;
            if (result.hasColumn(name) || name.equals(structCol)) name = f + "_1";
            result.addColumn(name, Column.DType.STRING);
            Column out = result.column(name);
            for (int i = 0; i < rowCount; i++) {
                Object val = maps.get(i).get(f);
                out.add(val);
            }
        }
        result.syncRowCountPublic();
        return result;
    }

    // ================================================================
    // MultiIndex-style level ops / resample / query-eval / advanced IO
    // ================================================================

    /** Pandas {@code swaplevel(i, j)} on named level columns. */
    public DataFrame swaplevel(int i, int j, String... levels) {
        return MultiIndexOps.swaplevel(this, i, j, levels);
    }

    public DataFrame swaplevel(String levelA, String levelB, String... levels) {
        return MultiIndexOps.swaplevel(this, levelA, levelB, levels);
    }

    /** Pandas {@code reorder_levels}. */
    public DataFrame reorderLevels(String[] order, String... levels) {
        return MultiIndexOps.reorderLevels(this, order, levels);
    }

    /** Pandas {@code droplevel}. */
    public DataFrame droplevel(Object level, String... levels) {
        return MultiIndexOps.droplevel(this, level, levels);
    }

    /**
     * Pandas {@code stack} analogue — melt value columns while keeping id levels.
     */
    public DataFrame stackLevels(List<String> idLevels, List<String> valueVars,
                                 String varName, String valueName) {
        return MultiIndexOps.stack(this, idLevels, valueVars, varName, valueName);
    }

    /** Pandas {@code unstack} analogue — pivot levelCol values into columns. */
    public DataFrame unstack(String levelCol, String valueCol) throws Exception {
        return MultiIndexOps.unstack(this, levelCol, valueCol);
    }

    public DataFrame unstack(List<String> indexCols, String levelCol, String valueCol,
                             Object fillValue) throws Exception {
        return MultiIndexOps.unstack(this, indexCols, levelCol, valueCol, fillValue);
    }

    /**
     * Pandas {@code resample(rule, on=...)}.
     * @param on time/key column (numeric epoch millis or temporal)
     * @param rule e.g. {@code "5s"}, {@code "1h"}, {@code "1d"}, or raw millis string
     */
    public Resampler resample(String on, String rule) {
        return Resampler.of(this, on, rule);
    }

    public Resampler resample(String on, String rule, String origin, long offsetMillis) {
        return Resampler.of(this, on, rule, origin, offsetMillis);
    }

    /** Pandas {@code DataFrame.query(expr)}. */
    public DataFrame query(String expr) {
        return QueryEval.query(this, expr);
    }

    public DataFrame query(String expr, Map<String, Object> localDict) {
        return QueryEval.query(this, expr, localDict);
    }

    /**
     * Pandas {@code DataFrame.eval}. Supports {@code "z = x + y"} assignment form
     * or bare expression → column {@code "result"}.
     */
    public DataFrame eval(String expr) {
        return QueryEval.eval(this, expr);
    }

    public DataFrame eval(String expr, Map<String, Object> localDict) {
        return QueryEval.eval(this, expr, localDict);
    }

    /**
     * Polars {@code map_batches} eager form — apply a batch UDF to this frame.
     * Prefer {@code lazy().mapBatches(fn).collect()} for pipeline composition.
     */
    public DataFrame mapBatches(java.util.function.Function<DataFrame, DataFrame> fn) {
        return fn.apply(this);
    }

    /** Advanced Parquet read: column projection + optional row filter + row-group parallel. */
    public static DataFrame readParquetAdvanced(String path,
            org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.ReadOptions opt) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.read(path, opt);
    }

    public static DataFrame readParquet(String path, String... columns) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.read(path,
            org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.ReadOptions.defaults()
                .columns(columns));
    }

    /**
     * Hive-style partitioned parquet write:
     * {@code root/key=val/.../part-N.parquet}.
     */
    public void writeParquetPartitioned(String root, String... partitionBy) throws Exception {
        org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.writePartitioned(this, root, partitionBy);
    }

    /** Scan Hive-partitioned parquet directory tree. */
    public static DataFrame readParquetHive(String root, String... partitionKeys) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.scanHive(root, partitionKeys);
    }

    public static DataFrame readParquetHive(String root, String[] partitionKeys,
            org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.ReadOptions opt) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.scanHive(root, partitionKeys, opt);
    }

    /** Stream advanced parquet batches (column prune + filter). */
    public static long streamParquetAdvanced(String path,
            org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.ReadOptions opt,
            java.util.function.Consumer<DataFrame> consumer) throws Exception {
        return org.bytedeco.pytorch.data.dataframe.io.ParquetAdvanced.stream(path, opt, consumer);
    }

    /** Distinct rows across all columns (or subset). */
    public DataFrame unique(String... subset) {
        return dropDuplicates(subset == null || subset.length == 0 ? null : subset, "first");
    }

    public DataFrame distinct() {
        return unique();
    }

    /**
     * Drop duplicate rows. {@code keep}: "first" (default), "last", "none"/false.
     */
    public DataFrame dropDuplicates(String[] subset, String keep) {
        String[] cols = (subset == null || subset.length == 0)
            ? columns.stream().map(Column::name).toArray(String[]::new)
            : subset;
        String k = keep == null ? "first" : keep.toLowerCase(Locale.ROOT);

        Map<String, Integer> first = new LinkedHashMap<>();
        Map<String, Integer> last = new LinkedHashMap<>();
        for (int i = 0; i < rowCount; i++) {
            String key = rowKey(i, cols);
            first.putIfAbsent(key, i);
            last.put(key, i);
        }

        List<Integer> keepIdx = new ArrayList<>();
        if ("none".equals(k) || "false".equals(k)) {
            Set<String> dups = new HashSet<>();
            Map<String, Integer> counts = new HashMap<>();
            for (int i = 0; i < rowCount; i++) {
                String key = rowKey(i, cols);
                counts.merge(key, 1, Integer::sum);
            }
            for (int i = 0; i < rowCount; i++) {
                String key = rowKey(i, cols);
                if (counts.get(key) == 1) keepIdx.add(i);
            }
        } else if ("last".equals(k)) {
            keepIdx.addAll(last.values());
            Collections.sort(keepIdx);
        } else {
            keepIdx.addAll(first.values());
            Collections.sort(keepIdx);
        }
        return loc(keepIdx.stream().mapToInt(Integer::intValue).toArray());
    }

    private String rowKey(int row, String[] cols) {
        StringBuilder sb = new StringBuilder();
        for (String c : cols) {
            Object v = DataValues.unwrap(column(c).get(row));
            sb.append(v == null ? "\0" : v.toString()).append('');
        }
        return sb.toString();
    }

    /** Boolean mask of duplicate rows (keep=first semantics: True if not the first occurrence). */
    public List<Boolean> duplicated(String... subset) {
        String[] cols = (subset == null || subset.length == 0)
            ? columns.stream().map(Column::name).toArray(String[]::new)
            : subset;
        Set<String> seen = new HashSet<>();
        List<Boolean> out = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) {
            String key = rowKey(i, cols);
            out.add(!seen.add(key));
        }
        return out;
    }

    /** Apply a function to each value of a column; returns a new frame with the column replaced. */
    public DataFrame apply(String columnName, Function<Object, Object> func) {
        DataFrame result = copy();
        Column col = result.column(columnName);
        for (int i = 0; i < result.rowCount; i++) {
            col.set(i, func.apply(col.get(i)));
        }
        return result;
    }

    /**
     * Apply a row-wise function; result stored in new column {@code apply_result}.
     * Type is inferred from the first non-null return value.
     */
    public DataFrame applyRows(Function<Map<String, Object>, Object> func) {
        Object[] outputs = new Object[rowCount];
        Class<?> detected = null;
        for (int i = 0; i < rowCount; i++) {
            Object out = func.apply(toDict(i));
            outputs[i] = out;
            if (out != null && detected == null) detected = out.getClass();
        }
        Column.DType outType = Column.DType.STRING;
        if (detected != null) {
            if (detected == Integer.class) outType = Column.DType.INT32;
            else if (detected == Long.class) outType = Column.DType.INT64;
            else if (detected == Float.class) outType = Column.DType.FLOAT32;
            else if (Number.class.isAssignableFrom(detected)) outType = Column.DType.FLOAT64;
            else if (detected == Boolean.class) outType = Column.DType.BOOLEAN;
        }
        DataFrame result = copy();
        String resultCol = "apply_result";
        if (result.hasColumn(resultCol)) result.removeColumn(resultCol);
        result.addColumn(resultCol, outType);
        Column outCol = result.column(resultCol);
        while (outCol.size() < result.rowCount) outCol.add(null);
        for (int i = 0; i < rowCount; i++) outCol.set(i, outputs[i]);
        return result;
    }

    public DataFrame applyRow(Function<Map<String, Object>, Object> func) {
        return applyRows(func);
    }

    // ---- column helpers: diff / abs / round / astype ----

    public DataFrame diff(String columnName, int periods) {
        DataFrame result = copy();
        Column col = result.column(columnName);
        Column src = column(columnName);
        for (int i = 0; i < result.rowCount; i++) {
            int prev = i - periods;
            if (prev < 0 || prev >= rowCount) {
                col.set(i, null);
            } else {
                double a = DataValues.asDouble(src.get(i));
                double b = DataValues.asDouble(src.get(prev));
                col.set(i, (Double.isNaN(a) || Double.isNaN(b)) ? null : a - b);
            }
        }
        return result;
    }

    public DataFrame diff(String columnName) { return diff(columnName, 1); }

    public DataFrame abs(String columnName) {
        DataFrame result = copy();
        Column col = result.column(columnName);
        for (int i = 0; i < result.rowCount; i++) {
            double v = DataValues.asDouble(col.get(i));
            if (!Double.isNaN(v)) col.set(i, Math.abs(v));
        }
        return result;
    }

    public DataFrame round(String columnName, int decimals) {
        DataFrame result = copy();
        Column col = result.column(columnName);
        double factor = Math.pow(10, decimals);
        for (int i = 0; i < result.rowCount; i++) {
            double v = DataValues.asDouble(col.get(i));
            if (!Double.isNaN(v)) col.set(i, Math.round(v * factor) / factor);
        }
        return result;
    }

    /** Cast a column to a new dtype (returns new frame). */
    public DataFrame astype(String columnName, Column.DType newType) {
        DataFrame result = copy();
        Column src = result.column(columnName);
        Column neu = new Column(columnName, newType);
        for (int i = 0; i < result.rowCount; i++) {
            Object v = src.get(i);
            if (v == null) { neu.add(null); continue; }
            switch (newType) {
                case INT32 -> neu.add((int) DataValues.asDouble(v));
                case INT64 -> neu.add((long) DataValues.asDouble(v));
                case FLOAT32 -> neu.add((float) DataValues.asDouble(v));
                case FLOAT64 -> neu.add(DataValues.asDouble(v));
                case BOOLEAN -> {
                    if (v instanceof Boolean) neu.add(v);
                    else neu.add(DataValues.asDouble(v) != 0);
                }
                case STRING -> neu.add(DataValues.asString(v));
                default -> neu.add(v);
            }
        }
        result.removeColumn(columnName);
        result.addColumn(neu);
        return result;
    }

    // ---- matrix bridge for ML ----

    /**
     * Convert selected numeric columns to a dense {@code double[rows][cols]} matrix.
     * Null/non-numeric → NaN.
     */
    public double[][] toMatrix(String... numericCols) {
        List<Column> cols;
        if (numericCols == null || numericCols.length == 0) {
            cols = new ArrayList<>();
            for (Column c : columns) if (isNumeric(c.dtype())) cols.add(c);
        } else {
            cols = new ArrayList<>(numericCols.length);
            for (String n : numericCols) cols.add(column(n));
        }
        int m = rowCount, n = cols.size();
        double[][] out = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                out[i][j] = DataValues.asDouble(cols.get(j).get(i));
            }
        }
        return out;
    }

    /** Alias for {@link #toMatrix(String...)} with all numeric columns. */
    public double[][] to_numpy() {
        return toMatrix();
    }

    /** Build a DataFrame from a list of row maps (column order = first row key order). */
    public static DataFrame fromRecords(List<Map<String, Object>> records) {
        DataFrame df = DataFrame.create();
        if (records == null || records.isEmpty()) return df;
        Map<String, Object> first = records.get(0);
        for (Map.Entry<String, Object> e : first.entrySet()) {
            df.addColumn(e.getKey(), inferDType(e.getValue()));
        }
        // ensure all keys from later rows exist
        for (Map<String, Object> rec : records) {
            for (String k : rec.keySet()) {
                if (!df.hasColumn(k)) df.addColumn(k, inferDType(rec.get(k)));
            }
        }
        for (Map<String, Object> rec : records) {
            int row = df.addEmptyRow();
            for (Column c : df.columns) {
                df.set(row, c.name(), rec.get(c.name()));
            }
        }
        return df;
    }

    // ---- rolling / expanding / ewm ----

    public org.bytedeco.pytorch.data.dataframe.window.Rolling rolling(int window) {
        return new org.bytedeco.pytorch.data.dataframe.window.Rolling(this, window);
    }

    public org.bytedeco.pytorch.data.dataframe.window.Rolling rolling(int window, int minPeriods) {
        return new org.bytedeco.pytorch.data.dataframe.window.Rolling(this, window, minPeriods);
    }

    public org.bytedeco.pytorch.data.dataframe.window.Expanding expanding() {
        return new org.bytedeco.pytorch.data.dataframe.window.Expanding(this);
    }

    public org.bytedeco.pytorch.data.dataframe.window.Expanding expanding(int minPeriods) {
        return new org.bytedeco.pytorch.data.dataframe.window.Expanding(this, minPeriods);
    }

    public org.bytedeco.pytorch.data.dataframe.window.Ewm ewm(double alpha) {
        return new org.bytedeco.pytorch.data.dataframe.window.Ewm(this, alpha);
    }

    public org.bytedeco.pytorch.data.dataframe.window.Ewm ewm(double alpha, boolean adjust) {
        return new org.bytedeco.pytorch.data.dataframe.window.Ewm(this, alpha, adjust);
    }

    /** Feature-engineering façade entry. */
    public org.bytedeco.pytorch.data.dataframe.feature.FeatureEngineering feature() {
        return new org.bytedeco.pytorch.data.dataframe.feature.FeatureEngineering(this);
    }

    /**
     * Spark-MLlib-style pipeline entry bound to this DataFrame.
     * <pre>
     *   DataFrame out = df.pipeline()
     *       .append("impute", new SimpleImputer("mean", "age"))
     *       .append("scale",  new StandardScaler("age","score"))
     *       .append("clf",    new LogisticRegression())
     *       .fitTransform();
     * </pre>
     */
    public org.bytedeco.pytorch.data.dataframe.feature.pipeline.DataFramePipeline pipeline() {
        return new org.bytedeco.pytorch.data.dataframe.feature.pipeline.DataFramePipeline(this);
    }

    // ---- structural ops: split / cat / stack / expand / compress / train_test_split ----

    /** Split into {@code n} roughly equal row chunks. */
    public DataFrame[] splitRows(int n) { return DataFrameOps.splitRows(this, n); }

    /** Split rows at exclusive cut indices. */
    public DataFrame[] splitRowsAt(int... indices) { return DataFrameOps.splitRowsAt(this, indices); }

    /** Split into {@code n} roughly equal column chunks. */
    public DataFrame[] splitCols(int n) { return DataFrameOps.splitCols(this, n); }

    /** Split columns by named groups. */
    public DataFrame[] splitColsByName(String[]... groups) {
        return DataFrameOps.splitColsByName(this, groups);
    }

    /** Vertical concat alias of {@link #vstack(DataFrame...)}. */
    public static DataFrame cat(DataFrame... frames) throws Exception {
        return DataFrameOps.cat(frames);
    }

    /** Horizontal stack with collision rename. */
    public static DataFrame stack(DataFrame... frames) throws Exception {
        return DataFrameOps.stack(frames);
    }

    /** Expand a LIST/VECTOR column into wide scalar columns {@code col_0..N}. */
    public DataFrame expand(String listCol) { return DataFrameOps.expand(this, listCol); }

    public DataFrame expand(String listCol, int maxWidth, boolean dropOriginal, String prefix) {
        return DataFrameOps.expand(this, listCol, maxWidth, dropOriginal, prefix);
    }

    /** Compress wide columns into one LIST column. */
    public DataFrame compress(String outCol, String... sourceCols) {
        return DataFrameOps.compress(this, outCol, sourceCols);
    }

    public DataFrame compressPrefix(String outCol, String prefix, int width) {
        return DataFrameOps.compressPrefix(this, outCol, prefix, width);
    }

    /** Random train/test split (testSize as fraction or absolute count if ≥1). */
    public DataFrameOps.TrainTestSplit trainTestSplit(double testSize) {
        return DataFrameOps.trainTestSplit(this, testSize);
    }

    public DataFrameOps.TrainTestSplit trainTestSplit(double testSize, long seed) {
        return DataFrameOps.trainTestSplit(this, testSize, seed);
    }

    public DataFrameOps.TrainTestSplit trainTestSplit(double testSize, boolean shuffle,
                                                      Long seed, String stratifyCol) {
        return DataFrameOps.trainTestSplit(this, testSize, shuffle, seed, stratifyCol);
    }

    /**
     * Select feature columns + label columns.
     * <pre>
     *   var fl = df.featureLabel(
     *       new String[]{"user_id","item_id","likes_level","views_level","item_seq"},
     *       "label");
     *   DataFrameDataset ds = fl.toDataset();
     * </pre>
     */
    public DataFrameOps.FeatureLabelSplit featureLabel(String[] featureCols, String... labelCols) {
        return DataFrameOps.featureLabel(this, featureCols, labelCols);
    }

    /** Features = all columns except labels. */
    public DataFrameOps.FeatureLabelSplit featureLabelExclude(String... labelCols) {
        return DataFrameOps.featureLabelExclude(this, labelCols);
    }

    /**
     * Build a {@link org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataset}
     * with optional feature-engineering pipeline.
     *
     * <pre>
     *   DataFrameDataset ds = df.toDataset()
     *       .features("user_id", "item_id", "likes_level", "views_level")
     *       .sequenceFeature("item_seq")
     *       .labels("label")
     *       .pipeline(pipe)   // optional
     *       .build();
     *   DataFrameDataLoader loader = ds.dataloader().batchSize(256).shuffle(true).build();
     * </pre>
     */
    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataset.Builder toDataset() {
        return org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataset.builder(this);
    }

    /**
     * One-shot dataset: all non-label columns as features (LIST/VECTOR → sequence),
     * with optional pipeline.
     */
    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataset toDataset(
            String[] featureCols, String[] sequenceCols, String[] labelCols,
            org.bytedeco.pytorch.data.dataframe.feature.pipeline.DataFramePipeline pipe)
            throws Exception {
        var b = toDataset();
        if (featureCols != null) b.features(featureCols);
        if (sequenceCols != null) b.sequenceFeatures(sequenceCols);
        if (labelCols != null) b.labels(labelCols);
        if (pipe != null) b.pipeline(pipe);
        return b.build();
    }

    // ---- preprocessing operators (chainable) ----

    /** Rename columns: {@code df.rename(Map.of("old","new"))}. Returns a copy. */
    public DataFrame rename(Map<String, String> mapping) {
        DataFrame result = copy();
        if (mapping == null) return result;
        for (Map.Entry<String, String> e : mapping.entrySet()) {
            if (result.hasColumn(e.getKey()) && e.getValue() != null
                && !e.getKey().equals(e.getValue())) {
                result.renameColumn(e.getKey(), e.getValue());
            }
        }
        return result;
    }

    /** Rename a single column (returns copy). */
    public DataFrame rename(String oldName, String newName) {
        DataFrame result = copy();
        result.renameColumn(oldName, newName);
        return result;
    }

    /** Replace values across the whole frame or selected columns. */
    public DataFrame replace(Object oldValue, Object newValue) {
        return replace(oldValue, newValue, (String[]) null);
    }

    public DataFrame replace(Object oldValue, Object newValue, String... cols) {
        DataFrame result = copy();
        List<Column> targets;
        if (cols == null || cols.length == 0) targets = result.columns;
        else {
            targets = new ArrayList<>(cols.length);
            for (String n : cols) targets.add(result.column(n));
        }
        for (Column c : targets) {
            for (int i = 0; i < c.size(); i++) {
                Object v = c.get(i);
                if (Objects.equals(v, oldValue)
                    || (oldValue != null && v != null && String.valueOf(v).equals(String.valueOf(oldValue)))) {
                    c.set(i, newValue);
                }
            }
        }
        return result;
    }

    /** Replace via map of old→new for one column. */
    public DataFrame replace(String column, Map<?, ?> mapping) {
        DataFrame result = copy();
        Column c = result.column(column);
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            if (mapping.containsKey(v)) c.set(i, mapping.get(v));
            else if (v != null && mapping.containsKey(String.valueOf(v))) {
                c.set(i, mapping.get(String.valueOf(v)));
            }
        }
        return result;
    }

    /** Fill NA in selected columns with a constant. */
    public DataFrame fillna(Object value, String... cols) throws Exception {
        if (cols == null || cols.length == 0) return fillna(value);
        DataFrame result = copy();
        for (String n : cols) {
            Column c = result.column(n);
            for (int i = 0; i < c.size(); i++) {
                if (c.get(i) == null) c.set(i, value);
            }
        }
        return result;
    }

    /** Drop rows where any of the given columns is null. */
    public DataFrame dropna(String... subset) throws Exception {
        if (subset == null || subset.length == 0) return dropna();
        DataFrame result = DataFrame.create();
        for (Column c : columns) result.addColumn(c.name(), c.dtype());
        for (int r = 0; r < rowCount; r++) {
            boolean keep = true;
            for (String n : subset) {
                if (column(n).get(r) == null) { keep = false; break; }
            }
            if (!keep) continue;
            Object[] row = new Object[columns.size()];
            for (int c = 0; c < columns.size(); c++) row[c] = columns.get(c).get(r);
            result.addRow(row);
        }
        return result;
    }

    /** Cast multiple columns: {@code df.astype(Map.of("a", INT64, "b", FLOAT64))}. */
    public DataFrame astype(Map<String, Column.DType> typeMap) {
        DataFrame result = this;
        if (typeMap == null) return copy();
        for (Map.Entry<String, Column.DType> e : typeMap.entrySet()) {
            result = result.astype(e.getKey(), e.getValue());
        }
        return result;
    }

    /**
     * Polars-style multi-column add/replace via expressions
     * ({@code with_columns} / Pandas {@code assign} with callables).
     */
    public DataFrame withColumns(Expression... exprs) {
        DataFrame result = this;
        if (exprs == null) return copy();
        for (Expression e : exprs) {
            result = result.withColumn(e.suggestedName(), e);
        }
        return result;
    }

    /** Alias of {@link #withColumns(Expression...)}. */
    public DataFrame withColumn(Expression... exprs) {
        return withColumns(exprs);
    }

    /**
     * Describe numeric columns as a small summary DataFrame
     * (rows = stats, columns = feature names) — Pandas {@code describe()}.
     */
    public DataFrame describeFrame() {
        Map<String, List<Double>> stats = describe();
        DataFrame out = DataFrame.create();
        out.addColumn("stat", Column.DType.STRING);
        List<String> features = new ArrayList<>(stats.keySet());
        for (String f : features) out.addColumn(f, Column.DType.FLOAT64);
        String[] labels = {"mean", "std", "min", "25%", "50%", "75%", "max"};
        for (int si = 0; si < labels.length; si++) {
            int row = out.addEmptyRow();
            out.set(row, "stat", labels[si]);
            for (String f : features) {
                List<Double> vals = stats.get(f);
                out.set(row, f, vals != null && si < vals.size() ? vals.get(si) : null);
            }
        }
        return out;
    }

    /**
     * Value counts as a 2-column DataFrame ({@code value}, {@code count}),
     * sorted by count descending.
     */
    public DataFrame valueCountsFrame(String columnName) {
        Map<Object, Integer> counts = valueCounts(columnName);
        DataFrame out = DataFrame.create();
        out.addColumn("value", Column.DType.STRING);
        out.addColumn("count", Column.DType.INT64);
        List<Map.Entry<Object, Integer>> entries = new ArrayList<>(counts.entrySet());
        entries.sort((a, b) -> Integer.compare(b.getValue(), a.getValue()));
        for (Map.Entry<Object, Integer> e : entries) {
            int row = out.addEmptyRow();
            out.set(row, "value", e.getKey() == null ? null : String.valueOf(e.getKey()));
            out.set(row, "count", e.getValue().longValue());
        }
        return out;
    }

    /** Alias for {@link #to_numpy()}. */
    public double[][] toNumpy() {
        return to_numpy();
    }

    /** Assign / overwrite columns from a map of name → list values. */
    public DataFrame assign(Map<String, ? extends List<?>> columnsData) {
        DataFrame result = copy();
        if (columnsData == null) return result;
        for (Map.Entry<String, ? extends List<?>> e : columnsData.entrySet()) {
            result = result.withColumn(e.getKey(), e.getValue());
        }
        return result;
    }

    /** Map a column through a function (alias of {@link #apply}). */
    public DataFrame map(String columnName, Function<Object, Object> func) {
        return apply(columnName, func);
    }

    /** Keep rows matching a predicate over the row map. */
    public DataFrame filterRows(Predicate<Map<String, Object>> predicate) {
        DataFrame result = DataFrame.create();
        for (Column c : columns) result.addColumn(c.name(), c.dtype());
        for (int r = 0; r < rowCount; r++) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (Column c : columns) row.put(c.name(), c.get(r));
            if (predicate != null && !predicate.test(row)) continue;
            Object[] vals = new Object[columns.size()];
            for (int i = 0; i < columns.size(); i++) vals[i] = columns.get(i).get(r);
            result.addRow(vals);
        }
        return result;
    }

    /** String uppercase for a STRING column. */
    public DataFrame strUpper(String columnName) {
        return apply(columnName, v -> v == null ? null : String.valueOf(v).toUpperCase(Locale.ROOT));
    }

    /** String lowercase for a STRING column. */
    public DataFrame strLower(String columnName) {
        return apply(columnName, v -> v == null ? null : String.valueOf(v).toLowerCase(Locale.ROOT));
    }

    /** String trim for a STRING column. */
    public DataFrame strStrip(String columnName) {
        return apply(columnName, v -> v == null ? null : String.valueOf(v).trim());
    }

    /** String contains → new boolean column. */
    public DataFrame strContains(String columnName, String needle, String outColumn) {
        DataFrame result = copy();
        List<Boolean> flags = new ArrayList<>(rowCount);
        Column c = column(columnName);
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            flags.add(v != null && String.valueOf(v).contains(needle));
        }
        return result.withColumnForBool(outColumn, flags);
    }

    /** Numeric clip on selected columns only. */
    public DataFrame clip(Double lower, Double upper, String... cols) throws Exception {
        if (cols == null || cols.length == 0) return clip(lower, upper);
        DataFrame result = copy();
        for (String n : cols) {
            Column c = result.column(n);
            for (int i = 0; i < c.size(); i++) {
                Object v = c.get(i);
                if (!(v instanceof Number)) continue;
                double d = ((Number) v).doubleValue();
                if (lower != null && d < lower) d = lower;
                if (upper != null && d > upper) d = upper;
                c.set(i, d);
            }
        }
        return result;
    }

    /** Reorder columns; missing names are ignored, unspecified columns dropped. */
    public DataFrame reorderColumns(String... names) {
        DataFrame result = DataFrame.create();
        for (String n : names) {
            if (hasColumn(n)) result.addColumn(column(n).copy());
        }
        result.syncRowCount();
        return result;
    }

    /** Add a constant column. */
    public DataFrame withConstant(String name, Object value) {
        List<Object> data = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) data.add(value);
        return withColumn(name, data);
    }

    /** Binary numeric column op: {@code out = left op right} where op in + - * /. */
    public DataFrame withArithmetic(String outName, String left, String right, char op) {
        Column l = column(left);
        Column r = column(right);
        List<Double> data = new ArrayList<>(rowCount);
        for (int i = 0; i < rowCount; i++) {
            Object a = l.get(i), b = r.get(i);
            if (!(a instanceof Number) || !(b instanceof Number)) {
                data.add(Double.NaN);
                continue;
            }
            double x = ((Number) a).doubleValue();
            double y = ((Number) b).doubleValue();
            switch (op) {
                case '+': data.add(x + y); break;
                case '-': data.add(x - y); break;
                case '*': data.add(x * y); break;
                case '/': data.add(y == 0.0 ? Double.NaN : x / y); break;
                default: throw new IllegalArgumentException("Unsupported op: " + op);
            }
        }
        return withColumnForDouble(outName, data);
    }

    // ---- API aliases used by migrated feature/ML packages ----

    /** Alias for {@link #rowCount()}. */
    public int len() { return rowCount; }

    /** Alias for {@link #rowCount()}. */
    public int getRowCount() { return rowCount; }

    /** Alias for {@link #columnCount()}. */
    public int getColumnCount() { return columns.size(); }

    /** Alias for {@link #column(String)}. */
    public Column getColumn(String name) { return column(name); }

    /** Alias for {@link #column(String)}. */
    public Column getColumnByName(String name) { return column(name); }

    /** Column names in order. */
    public List<String> getColumnNames() {
        List<String> names = new ArrayList<>(columns.size());
        for (Column c : columns) names.add(c.name());
        return names;
    }

    /**
     * Add or replace a FLOAT64 column from a list of doubles (feature/ML transform helper).
     */
    public DataFrame withColumnForDouble(String name, List<Double> data) {
        DataFrame result = copy();
        if (result.hasColumn(name)) result.removeColumn(name);
        result.addColumn(name, Column.DType.FLOAT64);
        Column col = result.column(name);
        int n = Math.max(rowCount, data == null ? 0 : data.size());
        while (col.size() < n) col.add(null);
        if (data != null) {
            for (int i = 0; i < data.size(); i++) col.set(i, data.get(i));
        }
        result.syncRowCount();
        return result;
    }

    public DataFrame withColumnForInt(String name, List<Integer> data) {
        DataFrame result = copy();
        if (result.hasColumn(name)) result.removeColumn(name);
        result.addColumn(name, Column.DType.INT32);
        Column col = result.column(name);
        int n = Math.max(rowCount, data == null ? 0 : data.size());
        while (col.size() < n) col.add(null);
        if (data != null) for (int i = 0; i < data.size(); i++) col.set(i, data.get(i));
        result.syncRowCount();
        return result;
    }

    public DataFrame withColumnForLong(String name, List<Long> data) {
        DataFrame result = copy();
        if (result.hasColumn(name)) result.removeColumn(name);
        result.addColumn(name, Column.DType.INT64);
        Column col = result.column(name);
        int n = Math.max(rowCount, data == null ? 0 : data.size());
        while (col.size() < n) col.add(null);
        if (data != null) for (int i = 0; i < data.size(); i++) col.set(i, data.get(i));
        result.syncRowCount();
        return result;
    }

    public DataFrame withColumnForString(String name, List<String> data) {
        DataFrame result = copy();
        if (result.hasColumn(name)) result.removeColumn(name);
        result.addColumn(name, Column.DType.STRING);
        Column col = result.column(name);
        int n = Math.max(rowCount, data == null ? 0 : data.size());
        while (col.size() < n) col.add(null);
        if (data != null) for (int i = 0; i < data.size(); i++) col.set(i, data.get(i));
        result.syncRowCount();
        return result;
    }

    public DataFrame withColumnForBool(String name, List<Boolean> data) {
        DataFrame result = copy();
        if (result.hasColumn(name)) result.removeColumn(name);
        result.addColumn(name, Column.DType.BOOLEAN);
        Column col = result.column(name);
        int n = Math.max(rowCount, data == null ? 0 : data.size());
        while (col.size() < n) col.add(null);
        if (data != null) for (int i = 0; i < data.size(); i++) col.set(i, data.get(i));
        result.syncRowCount();
        return result;
    }

    /**
     * Add or replace a column from a generic list (infers dtype from first non-null).
     */
    public DataFrame withColumn(String name, List<?> data) {
        Column.DType dt = Column.DType.STRING;
        if (data != null) {
            for (Object v : data) {
                if (v != null) { dt = inferDType(v); break; }
            }
        }
        DataFrame result = copy();
        if (result.hasColumn(name)) result.removeColumn(name);
        result.addColumn(name, dt);
        Column col = result.column(name);
        int n = Math.max(rowCount, data == null ? 0 : data.size());
        while (col.size() < n) col.add(null);
        if (data != null) for (int i = 0; i < data.size(); i++) col.set(i, data.get(i));
        result.syncRowCount();
        return result;
    }

    // ---- Plot ----

    /**
     * Matplotlib/seaborn-style plotting entry.
     * <pre>
     *   df.plot().line("x", "y").savefig("out.png");
     *   df.plot().scatter("x", "y").show();
     * </pre>
     */
    public org.bytedeco.pytorch.data.dataframe.plot.DataFramePlot plot() {
        return new org.bytedeco.pytorch.data.dataframe.plot.DataFramePlot(this);
    }

    // ---- ANN / HNSW ----

    /**
     * Create a one-column (plus optional ids) DataFrame of dense float vectors.
     */
    public static DataFrame fromVectors(String vectorCol, float[][] data) {
        return org.bytedeco.pytorch.data.dataframe.ann.VectorColumn.fromVectors(vectorCol, data);
    }

    public static DataFrame fromVectors(String vectorCol, float[][] data, String idCol, long[] ids) {
        return org.bytedeco.pytorch.data.dataframe.ann.VectorColumn.fromVectors(vectorCol, data, idCol, ids);
    }

    /**
     * Fluent HNSW builder bound to a VECTOR column of this frame.
     * <pre>
     *   HnswIndex idx = df.buildHnsw("emb").M(16).efConstruction(200).space(Distance.L2).build();
     * </pre>
     */
    public HnswBuild buildHnsw(String vectorCol) {
        return new HnswBuild(this, vectorCol);
    }

    /**
     * Build an ephemeral HNSW on {@code vectorCol}, search {@code query}, return
     * a DataFrame of the k nearest rows with {@code _distance} and {@code _rank} columns.
     */
    public DataFrame annSearch(String vectorCol, float[] query, int k) throws Exception {
        return annSearch(vectorCol, query, k, Math.max(64, k * 2),
            org.bytedeco.pytorch.data.dataframe.ann.Distance.L2);
    }

    public DataFrame annSearch(String vectorCol, float[] query, int k, int efSearch,
                               org.bytedeco.pytorch.data.dataframe.ann.Distance space) throws Exception {
        org.bytedeco.pytorch.data.dataframe.ann.HnswIndex idx = buildHnsw(vectorCol)
            .space(space)
            .build();
        org.bytedeco.pytorch.data.dataframe.ann.AnnSearchResult r = idx.search(query, k, efSearch);
        return annResultToFrame(r);
    }

    private DataFrame annResultToFrame(org.bytedeco.pytorch.data.dataframe.ann.AnnSearchResult r) {
        int[] indices = r.indices();
        float[] dists = r.distances();
        DataFrame out = DataFrame.create();
        for (Column c : columns) out.addColumn(c.name(), c.dtype());
        out.addColumn("_distance", Column.DType.FLOAT64);
        out.addColumn("_rank", Column.DType.INT64);
        for (int i = 0; i < indices.length; i++) {
            int ri = indices[i];
            int row = out.addEmptyRow();
            for (Column c : columns) {
                out.set(row, c.name(), c.get(ri));
            }
            out.set(row, "_distance", (double) dists[i]);
            out.set(row, "_rank", (long) (i + 1));
        }
        return out;
    }

    /** Fluent builder returned by {@link #buildHnsw(String)}. */
    public static final class HnswBuild {
        private final DataFrame df;
        private final String vectorCol;
        private int M = 16;
        private int efConstruction = 200;
        private org.bytedeco.pytorch.data.dataframe.ann.Distance space =
            org.bytedeco.pytorch.data.dataframe.ann.Distance.L2;
        private boolean normalize = false;

        HnswBuild(DataFrame df, String vectorCol) {
            this.df = df;
            this.vectorCol = vectorCol;
        }

        public HnswBuild M(int m) { this.M = m; return this; }
        public HnswBuild efConstruction(int ef) { this.efConstruction = ef; return this; }
        public HnswBuild space(org.bytedeco.pytorch.data.dataframe.ann.Distance d) {
            this.space = d; return this;
        }
        public HnswBuild normalize(boolean v) { this.normalize = v; return this; }

        public org.bytedeco.pytorch.data.dataframe.ann.HnswIndex build() {
            Column col = df.column(vectorCol);
            int dim = org.bytedeco.pytorch.data.dataframe.ann.VectorColumn.dimOf(col);
            if (dim <= 0) throw new IllegalStateException("VECTOR column empty or dim=0: " + vectorCol);
            float[] matrix = org.bytedeco.pytorch.data.dataframe.ann.VectorColumn.pack(col);
            return org.bytedeco.pytorch.data.dataframe.ann.HnswIndex.builder(dim)
                .M(M)
                .efConstruction(efConstruction)
                .space(space)
                .normalize(normalize)
                .vectors(matrix, col.size())
                .build();
        }
    }

    // ---- Vector store (remote ANN backends, zero vendor SDKs) ----

    /**
     * Push VECTOR/EMBEDDING column rows into a {@link org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore}.
     * <pre>
     *   try (VectorStore vs = VectorStores.qdrant(url, "clips", 768, VectorMetric.COSINE)) {
     *       vs.ensureCollection();
     *       df.toVectorStore(vs, "id", "emb", "title", "year");
     *   }
     * </pre>
     */
    public void toVectorStore(org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore store,
                              String idCol, String vectorCol, String... payloadCols) {
        store.upsertDataFrame(this, idCol, vectorCol, payloadCols);
    }

    /**
     * k-NN search against an external vector store; returns hit DataFrame
     * ({@code id}, {@code score}, {@code distance}, {@code rank}, + payload).
     */
    public static DataFrame vectorSearch(org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore store,
                                         float[] query, int topK) {
        return store.search(query, topK).toDataFrame();
    }

    /**
     * Scroll an entire vector store into a DataFrame
     * ({@code id}, {@code vector}, + payload columns). Cap with {@code limit}.
     */
    public static DataFrame fromVectorStore(
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore store) {
        return store.toDataFrame();
    }

    public static DataFrame fromVectorStore(
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore store, int limit) {
        return store.toDataFrame(limit);
    }

    public static DataFrame fromVectorStore(
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore store,
            String idCol, String vectorCol) {
        return store.toDataFrame(idCol, vectorCol);
    }

    /** Alias for {@link #vectorSearch(org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore, float[], int)}. */
    public static DataFrame fromVectorSearch(
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore store,
            float[] query, int topK) {
        return vectorSearch(store, query, topK);
    }

    // ---- Vector store: dedicated per-backend convenience (memory / qdrant / redis / …) ----

    /**
     * Open a vector store by scheme + free-form config (same as {@link
     * org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores#open(String, java.util.Map)}).
     */
    public static org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore openVectorStore(
            String scheme, java.util.Map<String, Object> config) {
        return org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(scheme, config);
    }

    /**
     * Open a vector store from a URI:
     * {@code qdrant://localhost:6333/clips?dim=768&metric=cosine}.
     */
    public static org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore openVectorStore(String uri) {
        return org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(uri);
    }

    /**
     * Create (or open) an in-process HNSW store, ensure collection, upsert this DataFrame,
     * and return the open store (caller must {@code close()}).
     *
     * <pre>{@code
     * try (VectorStore vs = df.toMemoryVectorStore("docs", "id", "emb", 384, VectorMetric.COSINE, "title")) {
     *     DataFrame hits = DataFrame.vectorSearch(vs, query, 5);
     * }
     * }</pre>
     */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toMemoryVectorStore(
            String name, String idCol, String vectorCol, int dim,
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric metric,
            String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.memory(
                        name, dim, metric == null
                                ? org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE
                                : metric);
        vs.ensureCollection();
        toVectorStore(vs, idCol, vectorCol, payloadCols);
        return vs;
    }

    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toMemoryVectorStore(
            String name, String idCol, String vectorCol, int dim, String... payloadCols) {
        return toMemoryVectorStore(name, idCol, vectorCol, dim,
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE, payloadCols);
    }

    /**
     * Upsert into Qdrant (HTTP REST, no SDK). Opens store, ensures collection, upserts, returns it.
     * Caller must {@code close()}.
     */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toQdrant(
            String url, String collection, String idCol, String vectorCol, int dim,
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric metric,
            String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.qdrant(
                        url, collection, dim, metric == null
                                ? org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE
                                : metric);
        vs.ensureCollection();
        toVectorStore(vs, idCol, vectorCol, payloadCols);
        return vs;
    }

    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toQdrant(
            String url, String collection, String idCol, String vectorCol, int dim,
            String... payloadCols) {
        return toQdrant(url, collection, idCol, vectorCol, dim,
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE, payloadCols);
    }

    /** Upsert into Redis / RediSearch (RESP, no Jedis). Caller must {@code close()}. */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toRedisVectorStore(
            String host, int port, String index, String idCol, String vectorCol, int dim,
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric metric,
            String... payloadCols) {
        return toRedisVectorStore(host, port, index, idCol, vectorCol, dim, metric, null, payloadCols);
    }

    /**
     * Upsert into Redis / RediSearch with per-doc TTL ({@code EXPIRE} on {@code doc:{id}}).
     * Caller must {@code close()}.
     *
     * @param ttl document key expiry; {@code null} = no expiry
     */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toRedisVectorStore(
            String host, int port, String index, String idCol, String vectorCol, int dim,
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric metric,
            java.time.Duration ttl, String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.redis.RedisVectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.redis.RedisVectorStore.builder()
                        .host(host).port(port)
                        .index(index).dim(dim)
                        .metric(metric == null
                                ? org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE
                                : metric)
                        .ttl(ttl)
                        .build();
        vs.ensureCollection();
        if (ttl != null && !ttl.isZero() && !ttl.isNegative()) {
            vs.upsertDataFrame(this, idCol, vectorCol, ttl, payloadCols);
        } else {
            vs.upsertDataFrame(this, idCol, vectorCol, payloadCols);
        }
        return vs;
    }

    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toRedisVectorStore(
            String host, int port, String index, String idCol, String vectorCol, int dim,
            String... payloadCols) {
        return toRedisVectorStore(host, port, index, idCol, vectorCol, dim,
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE, payloadCols);
    }

    /**
     * Upsert into Redis / RediSearch with TTL (default COSINE metric).
     */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toRedisVectorStore(
            String host, int port, String index, String idCol, String vectorCol, int dim,
            java.time.Duration ttl, String... payloadCols) {
        return toRedisVectorStore(host, port, index, idCol, vectorCol, dim,
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE, ttl, payloadCols);
    }

    /**
     * Upsert this DataFrame's vectors into an existing {@link
     * org.bytedeco.pytorch.data.dataframe.vectorstore.redis.RedisVectorStore} with TTL.
     */
    public void toRedisVectorStore(
            org.bytedeco.pytorch.data.dataframe.vectorstore.redis.RedisVectorStore store,
            String idCol, String vectorCol, java.time.Duration ttl, String... payloadCols) {
        store.upsertDataFrame(this, idCol, vectorCol, ttl, payloadCols);
    }

    /** Upsert into Milvus REST v2 (no milvus-sdk-java). Caller must {@code close()}. */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toMilvus(
            String url, String collection, String idCol, String vectorCol, int dim,
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric metric,
            String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.milvus(
                        url, collection, dim, metric == null
                                ? org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE
                                : metric);
        vs.ensureCollection();
        toVectorStore(vs, idCol, vectorCol, payloadCols);
        return vs;
    }

    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toMilvus(
            String url, String collection, String idCol, String vectorCol, int dim,
            String... payloadCols) {
        return toMilvus(url, collection, idCol, vectorCol, dim,
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE, payloadCols);
    }

    /** Upsert into OpenSearch k-NN (HTTP, no opensearch-java). Caller must {@code close()}. */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toOpenSearch(
            String url, String index, String idCol, String vectorCol, int dim,
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric metric,
            String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.openSearch(
                        url, index, dim, metric == null
                                ? org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE
                                : metric);
        vs.ensureCollection();
        toVectorStore(vs, idCol, vectorCol, payloadCols);
        return vs;
    }

    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toOpenSearch(
            String url, String index, String idCol, String vectorCol, int dim,
            String... payloadCols) {
        return toOpenSearch(url, index, idCol, vectorCol, dim,
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE, payloadCols);
    }

    /** Upsert into pgvector via JDBC (driver on app classpath). Caller must {@code close()}. */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toPgVector(
            String jdbcUrl, String user, String password, String table,
            String idCol, String vectorCol, int dim,
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric metric,
            String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.pgvector(
                        jdbcUrl, user, password, table, dim, metric == null
                                ? org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE
                                : metric);
        vs.ensureCollection();
        toVectorStore(vs, idCol, vectorCol, payloadCols);
        return vs;
    }

    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toPgVector(
            String jdbcUrl, String user, String password, String table,
            String idCol, String vectorCol, int dim, String... payloadCols) {
        return toPgVector(jdbcUrl, user, password, table, idCol, vectorCol, dim,
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE, payloadCols);
    }

    /** Upsert into Mongo Atlas Vector Search (Data API). Caller must {@code close()}. */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toMongoAtlas(
            String dataApiUrl, String apiKey, String dataSource, String database, String collection,
            String idCol, String vectorCol, int dim,
            org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric metric,
            String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.mongoAtlas(
                        dataApiUrl, apiKey, dataSource, database, collection, dim, metric == null
                                ? org.bytedeco.pytorch.data.dataframe.vectorstore.VectorMetric.COSINE
                                : metric);
        vs.ensureCollection();
        toVectorStore(vs, idCol, vectorCol, payloadCols);
        return vs;
    }

    /**
     * Generic write: open store by scheme/config, ensure collection, upsert this DataFrame.
     * Returns the open store — caller must {@code close()}.
     *
     * <pre>{@code
     * try (VectorStore vs = df.toVectorStore("memory",
     *         Map.of("name","docs","dim",384,"metric","cosine"),
     *         "id", "emb", "title")) {
     *     ...
     * }
     * }</pre>
     */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toVectorStore(
            String scheme, java.util.Map<String, Object> config,
            String idCol, String vectorCol, String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(scheme, config);
        vs.ensureCollection();
        toVectorStore(vs, idCol, vectorCol, payloadCols);
        return vs;
    }

    /**
     * Generic write via URI (e.g. {@code memory://x/docs?dim=384&metric=cosine}).
     * Returns the open store — caller must {@code close()}.
     */
    public org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore toVectorStore(
            String uri, String idCol, String vectorCol, String... payloadCols) {
        org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(uri);
        vs.ensureCollection();
        toVectorStore(vs, idCol, vectorCol, payloadCols);
        return vs;
    }

    /** Read all (or up to limit) points from a scheme/config store into a DataFrame. */
    public static DataFrame readVectorStore(String scheme, java.util.Map<String, Object> config)
            throws Exception {
        try (org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                     org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(scheme, config)) {
            return vs.toDataFrame();
        }
    }

    public static DataFrame readVectorStore(String scheme, java.util.Map<String, Object> config, int limit)
            throws Exception {
        try (org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                     org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(scheme, config)) {
            return vs.toDataFrame(limit);
        }
    }

    public static DataFrame readVectorStore(String uri) throws Exception {
        try (org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                     org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(uri)) {
            return vs.toDataFrame();
        }
    }

    public static DataFrame readVectorStore(String uri, int limit) throws Exception {
        try (org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                     org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(uri)) {
            return vs.toDataFrame(limit);
        }
    }

    /** k-NN search against a scheme/config store → hit DataFrame (store is opened and closed). */
    public static DataFrame searchVectorStore(
            String scheme, java.util.Map<String, Object> config, float[] query, int topK)
            throws Exception {
        try (org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                     org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(scheme, config)) {
            return vs.search(query, topK).toDataFrame();
        }
    }

    public static DataFrame searchVectorStore(String uri, float[] query, int topK) throws Exception {
        try (org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStore vs =
                     org.bytedeco.pytorch.data.dataframe.vectorstore.VectorStores.open(uri)) {
            return vs.search(query, topK).toDataFrame();
        }
    }

    // ---- Milvus / OpenSearch / Mongo / PgVector full-client DataFrame I/O ----

    /**
     * Write this DataFrame via the full {@link org.bytedeco.pytorch.data.dataframe.milvus.Milvus}
     * client (REST v2, no milvus-sdk-java). Mirrors Redis-style {@code toRedis}.
     */
    public int toMilvus(org.bytedeco.pytorch.data.dataframe.milvus.Milvus milvus,
                       org.bytedeco.pytorch.data.dataframe.milvus.MilvusOptions options) {
        return milvus.writeDataFrame(this, options);
    }

    public int toMilvus(org.bytedeco.pytorch.data.dataframe.milvus.Milvus milvus, String collection) {
        return toMilvus(milvus, org.bytedeco.pytorch.data.dataframe.milvus.MilvusOptions.collection(collection));
    }

    /** One-shot: connect URI → write → close. */
    public int toMilvus(String milvusUri,
                        org.bytedeco.pytorch.data.dataframe.milvus.MilvusOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.milvus.Milvus m =
                     org.bytedeco.pytorch.data.dataframe.milvus.Milvus.connectUri(milvusUri)) {
            return toMilvus(m, options);
        }
    }

    public static DataFrame readMilvus(org.bytedeco.pytorch.data.dataframe.milvus.Milvus milvus,
                                       org.bytedeco.pytorch.data.dataframe.milvus.MilvusOptions options) {
        return milvus.readDataFrame(options);
    }

    public static DataFrame readMilvus(org.bytedeco.pytorch.data.dataframe.milvus.Milvus milvus,
                                       String collection) {
        return readMilvus(milvus, org.bytedeco.pytorch.data.dataframe.milvus.MilvusOptions.collection(collection));
    }

    public static DataFrame readMilvus(String milvusUri,
                                       org.bytedeco.pytorch.data.dataframe.milvus.MilvusOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.milvus.Milvus m =
                     org.bytedeco.pytorch.data.dataframe.milvus.Milvus.connectUri(milvusUri)) {
            return readMilvus(m, options);
        }
    }

    public static DataFrame searchMilvus(org.bytedeco.pytorch.data.dataframe.milvus.Milvus milvus,
                                         float[] query, int topK,
                                         org.bytedeco.pytorch.data.dataframe.milvus.MilvusOptions options) {
        return milvus.searchDataFrame(query, topK, options);
    }

    /**
     * Write via full {@link org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch} client
     * (REST + _bulk, no opensearch-java).
     */
    public int toOpenSearch(org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch client,
                            org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearchOptions options) {
        return client.writeDataFrame(this, options);
    }

    public int toOpenSearch(org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch client, String index) {
        return toOpenSearch(client, org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearchOptions.index(index));
    }

    public int toOpenSearch(String openSearchUri,
                            org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearchOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch os =
                     org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch.connectUri(openSearchUri)) {
            return toOpenSearch(os, options);
        }
    }

    public static DataFrame readOpenSearch(org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch client,
                                           org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearchOptions options) {
        return client.readDataFrame(options);
    }

    public static DataFrame readOpenSearch(org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch client,
                                           String index) {
        return readOpenSearch(client, org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearchOptions.index(index));
    }

    public static DataFrame readOpenSearch(String openSearchUri,
                                           org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearchOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch os =
                     org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch.connectUri(openSearchUri)) {
            return readOpenSearch(os, options);
        }
    }

    public static DataFrame searchOpenSearch(org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearch client,
                                             float[] query, int topK,
                                             org.bytedeco.pytorch.data.dataframe.opensearch.OpenSearchOptions options) {
        return client.searchDataFrame(query, topK, options);
    }

    /**
     * Write via full {@link org.bytedeco.pytorch.data.dataframe.mongo.Mongo} client
     * (Atlas Data API; official driver via {@code MongoBackend} SPI).
     */
    public int toMongo(org.bytedeco.pytorch.data.dataframe.mongo.Mongo mongo,
                       org.bytedeco.pytorch.data.dataframe.mongo.MongoOptions options) {
        return mongo.writeDataFrame(this, options);
    }

    public int toMongo(org.bytedeco.pytorch.data.dataframe.mongo.Mongo mongo,
                       String database, String collection) {
        return toMongo(mongo, org.bytedeco.pytorch.data.dataframe.mongo.MongoOptions.collection(database, collection));
    }

    public int toMongo(String mongoUri,
                       org.bytedeco.pytorch.data.dataframe.mongo.MongoOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.mongo.Mongo m =
                     org.bytedeco.pytorch.data.dataframe.mongo.Mongo.connectUri(mongoUri)) {
            return toMongo(m, options);
        }
    }

    public static DataFrame readMongo(org.bytedeco.pytorch.data.dataframe.mongo.Mongo mongo,
                                      org.bytedeco.pytorch.data.dataframe.mongo.MongoOptions options) {
        return mongo.readDataFrame(options);
    }

    public static DataFrame readMongo(org.bytedeco.pytorch.data.dataframe.mongo.Mongo mongo,
                                      String database, String collection) {
        return readMongo(mongo, org.bytedeco.pytorch.data.dataframe.mongo.MongoOptions.collection(database, collection));
    }

    public static DataFrame readMongo(String mongoUri,
                                      org.bytedeco.pytorch.data.dataframe.mongo.MongoOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.mongo.Mongo m =
                     org.bytedeco.pytorch.data.dataframe.mongo.Mongo.connectUri(mongoUri)) {
            return readMongo(m, options);
        }
    }

    public static DataFrame searchMongo(org.bytedeco.pytorch.data.dataframe.mongo.Mongo mongo,
                                        float[] query, int topK,
                                        org.bytedeco.pytorch.data.dataframe.mongo.MongoOptions options) {
        return mongo.searchDataFrame(query, topK, options);
    }

    /**
     * Write via full {@link org.bytedeco.pytorch.data.dataframe.pgvector.PgVector} client
     * (JDBC + pgvector; driver on app classpath).
     */
    public int toPgVector(org.bytedeco.pytorch.data.dataframe.pgvector.PgVector pg,
                          org.bytedeco.pytorch.data.dataframe.pgvector.PgVectorOptions options) {
        return pg.writeDataFrame(this, options);
    }

    public int toPgVector(org.bytedeco.pytorch.data.dataframe.pgvector.PgVector pg, String table) {
        return toPgVector(pg, org.bytedeco.pytorch.data.dataframe.pgvector.PgVectorOptions.table(table));
    }

    public int toPgVector(String jdbcUri,
                          org.bytedeco.pytorch.data.dataframe.pgvector.PgVectorOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.pgvector.PgVector pg =
                     org.bytedeco.pytorch.data.dataframe.pgvector.PgVector.connectUri(jdbcUri)) {
            return toPgVector(pg, options);
        }
    }

    public static DataFrame readPgVector(org.bytedeco.pytorch.data.dataframe.pgvector.PgVector pg,
                                         org.bytedeco.pytorch.data.dataframe.pgvector.PgVectorOptions options) {
        return pg.readDataFrame(options);
    }

    public static DataFrame readPgVector(org.bytedeco.pytorch.data.dataframe.pgvector.PgVector pg,
                                         String table) {
        return readPgVector(pg, org.bytedeco.pytorch.data.dataframe.pgvector.PgVectorOptions.table(table));
    }

    public static DataFrame readPgVector(String jdbcUri,
                                         org.bytedeco.pytorch.data.dataframe.pgvector.PgVectorOptions options) {
        try (org.bytedeco.pytorch.data.dataframe.pgvector.PgVector pg =
                     org.bytedeco.pytorch.data.dataframe.pgvector.PgVector.connectUri(jdbcUri)) {
            return readPgVector(pg, options);
        }
    }

    public static DataFrame searchPgVector(org.bytedeco.pytorch.data.dataframe.pgvector.PgVector pg,
                                           float[] query, int topK,
                                           org.bytedeco.pytorch.data.dataframe.pgvector.PgVectorOptions options) {
        return pg.searchDataFrame(query, topK, options);
    }

    // ---- private helpers ----

    void syncRowCount() {
        rowCount = columns.isEmpty() ? 0
            : columns.stream().mapToInt(Column::size).max().orElse(0);
    }

    /** Append a row of nulls and return its index (package + public for groupby/arrow). */
    public int addEmptyRow() {
        for (Column c : columns) c.add(null);
        return rowCount++;
    }

    private int addRow() {
        return addEmptyRow();
    }

    private int compareVals(Object a, Object b) {
        if (a == null && b == null) return 0;
        if (a == null) return -1;
        if (b == null) return 1;
        if (a instanceof Number && b instanceof Number)
            return Double.compare(((Number) a).doubleValue(), ((Number) b).doubleValue());
        return a.toString().compareTo(b.toString());
    }

    private static boolean isNumeric(Column.DType dtype) {
        return dtype == Column.DType.INT32 || dtype == Column.DType.INT64
            || dtype == Column.DType.FLOAT32 || dtype == Column.DType.FLOAT64;
    }

    private static Column.DType inferDType(Object v) {
        if (v == null) return Column.DType.STRING;
        if (v instanceof Integer) return Column.DType.INT32;
        if (v instanceof Long) return Column.DType.INT64;
        if (v instanceof Float) return Column.DType.FLOAT32;
        if (v instanceof Double) return Column.DType.FLOAT64;
        if (v instanceof Boolean) return Column.DType.BOOLEAN;
        if (v instanceof LocalDate) return Column.DType.DATE;
        if (v instanceof LocalTime) return Column.DType.TIME;
        if (v instanceof Instant || v instanceof LocalDateTime || v instanceof ZonedDateTime)
            return Column.DType.DATETIME;
        if (v instanceof Duration) return Column.DType.DURATION;
        if (v instanceof float[] || v instanceof double[]) return Column.DType.VECTOR;
        return Column.DType.STRING;
    }

    private Column findFirstNumeric() {
        for (Column c : columns) if (isNumeric(c.dtype())) return c;
        throw new IllegalStateException("No numeric column found");
    }

    private static Column.DType numpyDType(DType dtype) {
        return switch (dtype) {
            case FLOAT64 -> Column.DType.FLOAT64;
            case FLOAT32 -> Column.DType.FLOAT32;
            case INT64 -> Column.DType.INT64;
            case INT32 -> Column.DType.INT32;
            default -> Column.DType.STRING;
        };
    }

    private static Column.DType scalarTypeToDType(org.bytedeco.pytorch.global.torch.ScalarType st) {
        // JavaCPP: Tensor.scalar_type() returns a non-canonical proxy — intern first
        // or switch falls through to Byte (ordinal 0).
        if (st == null) return Column.DType.FLOAT64;
        return switch (st.intern()) {
            case Double -> Column.DType.FLOAT64;
            case Float -> Column.DType.FLOAT32;
            case Long -> Column.DType.INT64;
            case Int -> Column.DType.INT32;
            case Bool -> Column.DType.BOOLEAN;
            default -> Column.DType.FLOAT64;
        };
    }

    private NDArray columnToNDArray(Column col) {
        long[] shape = new long[]{col.size()};
        DType dtype = switch (col.dtype()) {
            case FLOAT64 -> DType.FLOAT64;
            case FLOAT32 -> DType.FLOAT32;
            case INT64 -> DType.INT64;
            case INT32 -> DType.INT32;
            case BOOLEAN -> DType.INT8;
            default -> DType.FLOAT64;
        };
        NDArray arr = new NDArray(dtype, shape);
        for (int i = 0; i < col.size(); i++) {
            Object v = col.get(i);
            if (v instanceof Number) {
                arr.setDouble(i, ((Number) v).doubleValue());
            } else if (v instanceof Boolean) {
                arr.setDouble(i, ((Boolean) v) ? 1.0 : 0.0);
            }
        }
        return arr;
    }

    private static Column ndarrayToColumn(String name, NDArray arr) {
        Column.DType dtype = numpyDType(arr.dtype);
        // BOOLEAN-ish: INT8 0/1 from toNpz
        if (arr.dtype == DType.INT8 || arr.dtype == DType.UINT8) {
            boolean allBool = true;
            long n = 1;
            for (long d : arr.shape) n *= d;
            for (int i = 0; i < n; i++) {
                long v = arr.getLong(i);
                if (v != 0 && v != 1) { allBool = false; break; }
            }
            if (allBool && arr.shape.length == 1) {
                Column col = new Column(name, Column.DType.BOOLEAN);
                for (int i = 0; i < arr.shape[0]; i++) col.add(arr.getLong(i) != 0);
                return col;
            }
        }
        Column col = new Column(name, dtype);
        if (arr.shape.length == 1) {
            long n = arr.shape[0];
            if (NDArray.isFloatFamily(arr.dtype)) {
                for (int i = 0; i < n; i++) col.add(arr.getDouble(i));
            } else {
                for (int i = 0; i < n; i++) col.add(arr.getLong(i));
            }
        } else {
            // store each leading-axis slice as vector/row blob
            long rows = arr.shape[0];
            long inner = 1;
            for (int i = 1; i < arr.shape.length; i++) inner *= arr.shape[i];
            for (int r = 0; r < rows; r++) {
                if (inner == 1) {
                    col.add(NDArray.isFloatFamily(arr.dtype) ? arr.getDouble(r) : arr.getLong(r));
                } else {
                    double[] row = new double[(int) inner];
                    for (int j = 0; j < inner; j++) {
                        int idx = (int) (r * inner + j);
                        row[j] = NDArray.isFloatFamily(arr.dtype) ? arr.getDouble(idx) : arr.getLong(idx);
                    }
                    col.add(row);
                }
            }
        }
        return col;
    }

    private static DataFrame ndarrayToDataFrame(String key, NDArray arr) {
        DataFrame df = DataFrame.create();
        if (arr.shape.length == 1) {
            df.addColumn(ndarrayToColumn(key, arr));
            return df;
        }
        if (arr.shape.length == 2) {
            int rows = (int) arr.shape[0];
            int cols = (int) arr.shape[1];
            Column.DType dtype = numpyDType(arr.dtype);
            for (int c = 0; c < cols; c++) df.addColumn("col_" + c, dtype);
            for (int r = 0; r < rows; r++) {
                int ri = df.addEmptyRow();
                for (int c = 0; c < cols; c++) {
                    int idx = r * cols + c;
                    Object v = NDArray.isFloatFamily(arr.dtype) ? arr.getDouble(idx) : arr.getLong(idx);
                    df.set(ri, "col_" + c, v);
                }
            }
            return df;
        }
        // higher rank → flatten to 1D column
        Column.DType dtype = numpyDType(arr.dtype);
        Column col = new Column(key, dtype);
        long n = arr.size;
        if (NDArray.isFloatFamily(arr.dtype)) {
            for (int i = 0; i < n; i++) col.add(arr.getDouble(i));
        } else {
            for (int i = 0; i < n; i++) col.add(arr.getLong(i));
        }
        df.addColumn(col);
        return df;
    }

    private static org.bytedeco.pytorch.global.torch.ScalarType toTorchScalarType(Column.DType dtype) {
        return switch (dtype) {
            case FLOAT64 -> org.bytedeco.pytorch.global.torch.ScalarType.Double;
            case FLOAT32 -> org.bytedeco.pytorch.global.torch.ScalarType.Float;
            case INT64 -> org.bytedeco.pytorch.global.torch.ScalarType.Long;
            case INT32 -> org.bytedeco.pytorch.global.torch.ScalarType.Int;
            case BOOLEAN -> org.bytedeco.pytorch.global.torch.ScalarType.Bool;
            default -> org.bytedeco.pytorch.global.torch.ScalarType.Float;
        };
    }

    private Tensor columnToTensor(Column col) {
        org.bytedeco.pytorch.global.torch.ScalarType st = toTorchScalarType(col.dtype());
        TensorOptions opts = new TensorOptions(st);
        double[] data = new double[col.size()];
        for (int i = 0; i < col.size(); i++) {
            Object v = col.get(i);
            data[i] = v instanceof Number ? ((Number) v).doubleValue() : 0;
        }
        return torch.tensor(data, opts).reshape(new long[]{col.size()});
    }

    private static Column tensorToColumn(String name, Tensor t) {
        long[] shape = sizesAsArray(t.sizes());
        Column.DType dtype = scalarTypeToDType(t.scalar_type());
        Column col = new Column(name, dtype);
        Tensor flat = t.contiguous().cpu().reshape(new long[]{-1})
            .to(org.bytedeco.pytorch.global.torch.ScalarType.Double);
        int n = (int) flat.numel();
        org.bytedeco.javacpp.DoublePointer ptr = flat.data_ptr_double();
        int rows = shape.length >= 1 ? (int) shape[0] : n;
        int cols = shape.length >= 2 ? (int) shape[1] : 1;
        if (shape.length <= 1) {
            for (int i = 0; i < n; i++) col.add(ptr.get(i));
        } else {
            for (int r = 0; r < rows; r++) {
                if (cols == 1) col.add(ptr.get(r));
                else {
                    double[] row = new double[cols];
                    for (int c = 0; c < cols; c++) row[c] = ptr.get((long) r * cols + c);
                    col.add(row);
                }
            }
        }
        return col;
    }

    private static long[] sizesAsArray(org.bytedeco.pytorch.c10.LongHeaderOnlyArrayRef ref) {
        long len = ref.size();
        if (len == 0) return new long[0];
        return ref.vec().get();
    }

    private static Column.DType parquetTypeToDType(org.apache.parquet.schema.Type ft) {
        if (ft.isPrimitive()) {
            org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName ptn =
                ft.asPrimitiveType().getPrimitiveTypeName();
            return switch (ptn) {
                case INT32 -> Column.DType.INT32;
                case INT64 -> Column.DType.INT64;
                case FLOAT -> Column.DType.FLOAT32;
                case DOUBLE -> Column.DType.FLOAT64;
                case BOOLEAN -> Column.DType.BOOLEAN;
                case BINARY, FIXED_LEN_BYTE_ARRAY -> {
                    org.apache.parquet.schema.LogicalTypeAnnotation lta = ft.getLogicalTypeAnnotation();
                    if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.StringLogicalTypeAnnotation
                        || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.EnumLogicalTypeAnnotation
                        || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.JsonLogicalTypeAnnotation) {
                        yield Column.DType.STRING;
                    }
                    yield Column.DType.BINARY;
                }
                default -> Column.DType.STRING;
            };
        }
        // Nested group: LIST / MAP / STRUCT (e.g. item_seq: list<int64>)
        org.apache.parquet.schema.LogicalTypeAnnotation lta = ft.getLogicalTypeAnnotation();
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.ListLogicalTypeAnnotation
            || isLegacyRepeatedList(ft)) {
            // float/double lists → VECTOR (ANN/HNSW embeddings); int/string/nested → LIST
            Column.DType elem = parquetListElementDType(ft);
            if (elem == Column.DType.FLOAT32 || elem == Column.DType.FLOAT64) {
                return Column.DType.VECTOR;
            }
            return Column.DType.LIST;
        }
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapLogicalTypeAnnotation
            || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapKeyValueTypeAnnotation) {
            return Column.DType.MAP;
        }
        return Column.DType.STRUCT;
    }

    /** True for pre-parquet-format-2.0 "repeated primitive/group" list encoding. */
    private static boolean isLegacyRepeatedList(org.apache.parquet.schema.Type ft) {
        return !ft.isPrimitive()
            && ft.getRepetition() == org.apache.parquet.schema.Type.Repetition.REPEATED;
    }

    /** Infer element dtype of a LIST (or legacy repeated) field for VECTOR vs LIST choice. */
    private static Column.DType parquetListElementDType(org.apache.parquet.schema.Type ft) {
        try {
            org.apache.parquet.schema.GroupType gt = ft.asGroupType();
            if (gt.getFieldCount() == 0) return Column.DType.STRING;
            org.apache.parquet.schema.Type mid = gt.getType(0);
            // 3-level LIST: group list { repeated group list { optional T item } }
            if (!mid.isPrimitive()) {
                org.apache.parquet.schema.GroupType midG = mid.asGroupType();
                if (midG.getFieldCount() > 0) {
                    org.apache.parquet.schema.Type elem = midG.getType(0);
                    if (elem.isPrimitive()) return parquetTypeToDType(elem);
                    // nested list → LIST
                    return Column.DType.LIST;
                }
            } else {
                // 2-level / legacy: repeated T element
                return parquetTypeToDType(mid);
            }
        } catch (Exception ignored) { /* fall through */ }
        return Column.DType.STRING;
    }

    private static Object readGroupValue(org.apache.parquet.example.data.Group row,
                                         String field, org.apache.parquet.schema.Type ft) {
        int fieldIndex;
        try {
            fieldIndex = row.getType().getFieldIndex(field);
        } catch (Exception e) {
            return null;
        }
        int n = row.getFieldRepetitionCount(fieldIndex);
        if (n == 0) return null; // optional / missing

        if (ft.isPrimitive()) {
            // REPEATED primitive at this level → materialize as dense array / List (legacy list encoding)
            if (ft.getRepetition() == org.apache.parquet.schema.Type.Repetition.REPEATED) {
                List<Object> out = new ArrayList<>(n);
                for (int i = 0; i < n; i++) out.add(readPrimitiveAt(row, fieldIndex, i, ft));
                return densifyNumericList(out, parquetTypeToDType(ft));
            }
            return readPrimitiveAt(row, fieldIndex, 0, ft);
        }

        // Nested group field
        org.apache.parquet.schema.LogicalTypeAnnotation lta = ft.getLogicalTypeAnnotation();
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.ListLogicalTypeAnnotation
            || isLegacyRepeatedList(ft)) {
            return readParquetList(row, fieldIndex, ft);
        }
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapLogicalTypeAnnotation
            || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapKeyValueTypeAnnotation) {
            return readParquetMap(row, fieldIndex, ft);
        }
        // STRUCT: single group (or first if repeated)
        org.apache.parquet.example.data.Group g = row.getGroup(fieldIndex, 0);
        return readParquetStruct(g);
    }

    private static Object readPrimitiveAt(org.apache.parquet.example.data.Group row,
                                          int fieldIndex, int idx,
                                          org.apache.parquet.schema.Type ft) {
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName ptn =
            ft.asPrimitiveType().getPrimitiveTypeName();
        try {
            return switch (ptn) {
                case INT32 -> row.getInteger(fieldIndex, idx);
                case INT64 -> row.getLong(fieldIndex, idx);
                case FLOAT -> Float.valueOf(row.getFloat(fieldIndex, idx));
                case DOUBLE -> row.getDouble(fieldIndex, idx);
                case BOOLEAN -> Boolean.valueOf(row.getBoolean(fieldIndex, idx));
                case BINARY, FIXED_LEN_BYTE_ARRAY -> {
                    org.apache.parquet.schema.LogicalTypeAnnotation lta = ft.getLogicalTypeAnnotation();
                    org.apache.parquet.io.api.Binary bin = row.getBinary(fieldIndex, idx);
                    if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.StringLogicalTypeAnnotation
                        || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.EnumLogicalTypeAnnotation
                        || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.JsonLogicalTypeAnnotation
                        || lta == null) {
                        // UTF8 string annotation, or unannotated binary often used as string by writers
                        if (lta != null
                            || looksLikeUtf8(bin)) {
                            yield bin.toStringUsingUTF8();
                        }
                        yield bin.getBytes();
                    }
                    yield bin.getBytes();
                }
                case INT96 -> row.getInt96(fieldIndex, idx).toStringUsingUTF8();
                default -> row.getValueToString(fieldIndex, idx);
            };
        } catch (Exception e) {
            try { return row.getValueToString(fieldIndex, idx); }
            catch (Exception e2) { return null; }
        }
    }

    private static boolean looksLikeUtf8(org.apache.parquet.io.api.Binary bin) {
        if (bin == null || bin.length() == 0) return true;
        try {
            String s = bin.toStringUsingUTF8();
            return s != null;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Materialize a Parquet LIST field.
     * Standard 3-level encoding (Arrow / Spark / fixed_size_list on disk):
     *   optional group item_seq (LIST) {
     *     repeated group list { optional int64 item; }
     *   }
     * Also supports:
     *   - 2-level: optional group X (LIST) { repeated T element; }
     *   - legacy REPEATED primitive at the parent (handled in {@link #readGroupValue})
     *   - legacy REPEATED group { T item } at this field
     * Numeric lists densify to long[] / int[] / float[] / double[] when homogeneous;
     * otherwise {@code List<Object>}. Empty list → empty array of inferred element type
     * when known, else empty List.
     */
    private static Object readParquetList(org.apache.parquet.example.data.Group row,
                                          int fieldIndex,
                                          org.apache.parquet.schema.Type ft) {
        org.apache.parquet.schema.GroupType listType = ft.asGroupType();
        // Outer may be optional group with 0/1 occurrence; elements live one level down.
        if (row.getFieldRepetitionCount(fieldIndex) == 0) return null;

        // Legacy: this field itself is REPEATED group (not the standard LIST annotation outer).
        // Each repetition is one list element (struct, or single-child wrapper around a primitive).
        if (ft.getRepetition() == org.apache.parquet.schema.Type.Repetition.REPEATED) {
            int n = row.getFieldRepetitionCount(fieldIndex);
            List<Object> elems = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                org.apache.parquet.example.data.Group g = row.getGroup(fieldIndex, i);
                if (listType.getFieldCount() == 1) {
                    org.apache.parquet.schema.Type child = listType.getType(0);
                    if (child.isPrimitive()) {
                        if (g.getFieldRepetitionCount(0) == 0) elems.add(null);
                        else elems.add(readPrimitiveAt(g, 0, 0, child));
                    } else {
                        org.apache.parquet.schema.LogicalTypeAnnotation cl =
                            child.getLogicalTypeAnnotation();
                        if (cl instanceof org.apache.parquet.schema.LogicalTypeAnnotation.ListLogicalTypeAnnotation
                            || isLegacyRepeatedList(child)) {
                            elems.add(readParquetList(g, 0, child));
                        } else {
                            elems.add(readParquetStruct(g));
                        }
                    }
                } else {
                    elems.add(readParquetStruct(g));
                }
            }
            return densifyNumericList(elems, parquetListElementDType(ft));
        }

        org.apache.parquet.example.data.Group listGroup = row.getGroup(fieldIndex, 0);
        if (listType.getFieldCount() == 0) {
            return densifyNumericList(Collections.emptyList(), parquetListElementDType(ft));
        }
        org.apache.parquet.schema.Type mid = listType.getType(0);
        int midIndex = 0;
        int midCount = listGroup.getFieldRepetitionCount(midIndex);

        // 3-level: mid is repeated group "list" with one child element
        if (!mid.isPrimitive()) {
            org.apache.parquet.schema.GroupType midG = mid.asGroupType();
            org.apache.parquet.schema.Type elemType =
                midG.getFieldCount() > 0 ? midG.getType(0) : null;
            List<Object> elems = new ArrayList<>(midCount);
            for (int i = 0; i < midCount; i++) {
                org.apache.parquet.example.data.Group elemGroup = listGroup.getGroup(midIndex, i);
                if (elemType == null) {
                    elems.add(readParquetStruct(elemGroup));
                    continue;
                }
                // empty element group / missing optional item
                if (elemGroup.getType().getFieldCount() == 0
                    || elemGroup.getFieldRepetitionCount(0) == 0) {
                    elems.add(null);
                    continue;
                }
                if (elemType.isPrimitive()) {
                    elems.add(readPrimitiveAt(elemGroup, 0, 0, elemType));
                } else {
                    // nested list/struct element
                    org.apache.parquet.schema.LogicalTypeAnnotation elLta =
                        elemType.getLogicalTypeAnnotation();
                    if (elLta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.ListLogicalTypeAnnotation
                        || isLegacyRepeatedList(elemType)) {
                        elems.add(readParquetList(elemGroup, 0, elemType));
                    } else if (elemType.isPrimitive()) {
                        elems.add(readPrimitiveAt(elemGroup, 0, 0, elemType));
                    } else {
                        // struct element: the child group itself is the struct
                        elems.add(readParquetStruct(elemGroup.getGroup(0, 0)));
                    }
                }
            }
            return densifyNumericList(elems, parquetListElementDType(ft));
        }

        // 2-level: mid is repeated primitive element
        List<Object> elems = new ArrayList<>(midCount);
        for (int i = 0; i < midCount; i++) {
            elems.add(readPrimitiveAt(listGroup, midIndex, i, mid));
        }
        return densifyNumericList(elems, parquetListElementDType(ft));
    }

    /** Convert homogeneous numeric List → primitive array for LIST/VECTOR cells. */
    private static Object densifyNumericList(List<Object> elems) {
        return densifyNumericList(elems, null);
    }

    private static Object densifyNumericList(List<Object> elems, Column.DType preferred) {
        if (elems == null) return null;
        if (elems.isEmpty()) {
            // Prefer typed empty arrays so schema remains stable for fixed-size-list columns.
            if (preferred == Column.DType.INT64) return new long[0];
            if (preferred == Column.DType.INT32) return new int[0];
            if (preferred == Column.DType.FLOAT32) return new float[0];
            if (preferred == Column.DType.FLOAT64) return new double[0];
            return elems;
        }
        boolean allLong = true, allInt = true, allFloat = true, allDouble = true;
        for (Object o : elems) {
            if (o == null) {
                allLong = allInt = allFloat = allDouble = false;
                break;
            }
            if (!(o instanceof Number)) {
                allLong = allInt = allFloat = allDouble = false;
                break;
            }
            if (!(o instanceof Long) && !(o instanceof Integer) && !(o instanceof Short) && !(o instanceof Byte))
                allLong = false;
            if (!(o instanceof Integer) && !(o instanceof Short) && !(o instanceof Byte))
                allInt = false;
            if (!(o instanceof Float)) allFloat = false;
            if (!(o instanceof Double) && !(o instanceof Float)) allDouble = false;
        }
        // Prefer preferred dtype when it matches (e.g. INT64 lists stay long[] even if all values fit int)
        if (preferred == Column.DType.INT64 && allLong) {
            long[] a = new long[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).longValue();
            return a;
        }
        if (preferred == Column.DType.FLOAT32 && (allFloat || allDouble || allLong || allInt)) {
            float[] a = new float[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).floatValue();
            return a;
        }
        if (preferred == Column.DType.FLOAT64 && (allDouble || allFloat || allLong || allInt)) {
            double[] a = new double[elems.size()];
            for (int i = 0; i < elems.size(); i++) a[i] = ((Number) elems.get(i)).doubleValue();
            return a;
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

    private static Map<String, Object> readParquetMap(org.apache.parquet.example.data.Group row,
                                                      int fieldIndex,
                                                      org.apache.parquet.schema.Type ft) {
        Map<String, Object> map = new LinkedHashMap<>();
        if (row.getFieldRepetitionCount(fieldIndex) == 0) return map;
        org.apache.parquet.example.data.Group mapGroup = row.getGroup(fieldIndex, 0);
        // MAP: optional group X (MAP) { repeated group key_value { required K key; optional V value; } }
        if (mapGroup.getType().getFieldCount() == 0) return map;
        int kvIndex = 0;
        int n = mapGroup.getFieldRepetitionCount(kvIndex);
        org.apache.parquet.schema.GroupType kvType = mapGroup.getType().getType(kvIndex).asGroupType();
        org.apache.parquet.schema.Type keyType = kvType.getType(0);
        org.apache.parquet.schema.Type valType = kvType.getFieldCount() > 1 ? kvType.getType(1) : null;
        for (int i = 0; i < n; i++) {
            org.apache.parquet.example.data.Group kv = mapGroup.getGroup(kvIndex, i);
            Object key = keyType.isPrimitive()
                ? readPrimitiveAt(kv, 0, 0, keyType)
                : readParquetStruct(kv.getGroup(0, 0));
            Object val = null;
            if (valType != null && kv.getFieldRepetitionCount(1) > 0) {
                if (valType.isPrimitive()) val = readPrimitiveAt(kv, 1, 0, valType);
                else {
                    org.apache.parquet.schema.LogicalTypeAnnotation vl =
                        valType.getLogicalTypeAnnotation();
                    if (vl instanceof org.apache.parquet.schema.LogicalTypeAnnotation.ListLogicalTypeAnnotation)
                        val = readParquetList(kv, 1, valType);
                    else
                        val = readParquetStruct(kv.getGroup(1, 0));
                }
            }
            map.put(String.valueOf(key), val);
        }
        return map;
    }

    private static Map<String, Object> readParquetStruct(org.apache.parquet.example.data.Group g) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (g == null) return out;
        org.apache.parquet.schema.GroupType st = g.getType();
        for (int i = 0; i < st.getFieldCount(); i++) {
            org.apache.parquet.schema.Type child = st.getType(i);
            String name = child.getName();
            if (g.getFieldRepetitionCount(i) == 0) {
                out.put(name, null);
                continue;
            }
            out.put(name, readGroupValue(g, name, child));
        }
        return out;
    }

    private static void writeGroupField(org.apache.parquet.example.data.simple.SimpleGroup g,
                                        String name, Column.DType dtype, Object val) {
        if (val == null) return;
        switch (dtype) {
            case INT32:    g.add(name, ((Number) val).intValue()); break;
            case INT64:    g.add(name, ((Number) val).longValue()); break;
            case FLOAT32:  g.add(name, ((Number) val).floatValue()); break;
            case FLOAT64:  g.add(name, ((Number) val).doubleValue()); break;
            case BOOLEAN:  g.add(name, (Boolean) val); break;
            case LIST:
            case VECTOR:
            case EMBEDDING:
                writeListField(g, name, val);
                break;
            case MAP:
                writeMapField(g, name, val);
                break;
            case STRUCT:
                writeStructField(g, name, val);
                break;
            case JSON:
                // JSON logical annotation on BINARY/UTF8
                g.add(name, val instanceof String ? (String) val
                    : org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec.encodeText(val));
                break;
            default:
                if (val instanceof long[] || val instanceof int[] || val instanceof float[]
                    || val instanceof double[] || val instanceof List) {
                    writeListField(g, name, val);
                } else if (val instanceof java.util.Map) {
                    writeMapField(g, name, val);
                } else {
                    g.add(name, val.toString());
                }
                break;
        }
    }

    /**
     * Write a MAP cell into standard Parquet MAP encoding:
     *   optional group name (MAP) {
     *     repeated group key_value { required binary key (STRING); optional binary value (STRING); }
     *   }
     * Nested values are JSON-encoded as strings for portability.
     */
    private static void writeMapField(org.apache.parquet.example.data.simple.SimpleGroup g,
                                      String name, Object val) {
        org.apache.parquet.example.data.simple.SimpleGroup mapOuter =
            (org.apache.parquet.example.data.simple.SimpleGroup) g.addGroup(name);
        if (val == null) return;
        java.util.Map<String, Object> map =
            org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec.asStringMap(val);
        if (map == null) return;
        for (java.util.Map.Entry<String, Object> e : map.entrySet()) {
            org.apache.parquet.example.data.simple.SimpleGroup kv =
                (org.apache.parquet.example.data.simple.SimpleGroup) mapOuter.addGroup(0);
            kv.add(0, e.getKey() == null ? "" : e.getKey());
            Object v = e.getValue();
            if (v == null) continue;
            if (v instanceof Number || v instanceof Boolean || v instanceof String) {
                kv.add(1, String.valueOf(v));
            } else {
                kv.add(1, org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec.encodeText(v));
            }
        }
    }

    /**
     * Write a STRUCT cell as a group of string fields (values JSON-encoded when nested).
     * Schema uses optional binary (STRING) children named by map keys from first non-null sample
     * when building schema; at write time unknown keys are skipped if not in schema — we write
     * all keys present in the group type.
     */
    private static void writeStructField(org.apache.parquet.example.data.simple.SimpleGroup g,
                                         String name, Object val) {
        org.apache.parquet.example.data.simple.SimpleGroup st =
            (org.apache.parquet.example.data.simple.SimpleGroup) g.addGroup(name);
        if (val == null) return;
        java.util.Map<String, Object> map =
            org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec.asStringMap(val);
        if (map == null) return;
        org.apache.parquet.schema.GroupType gt = st.getType();
        for (int i = 0; i < gt.getFieldCount(); i++) {
            String fname = gt.getType(i).getName();
            if (!map.containsKey(fname)) continue;
            Object v = map.get(fname);
            if (v == null) continue;
            if (v instanceof Number || v instanceof Boolean || v instanceof String) {
                st.add(fname, String.valueOf(v));
            } else {
                st.add(fname, org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec.encodeText(v));
            }
        }
        // If schema is empty (flexible struct), add all keys as string fields dynamically —
        // SimpleGroup requires schema fields; keys not in schema are ignored above.
        // When schema was built with all union keys, they are present.
    }

    /**
     * Write a LIST/VECTOR cell into a 3-level Parquet LIST group:
     *   optional group name (LIST) { repeated group list { optional T item; } }
     */
    private static void writeListField(org.apache.parquet.example.data.simple.SimpleGroup g,
                                       String name, Object val) {
        org.apache.parquet.example.data.simple.SimpleGroup listOuter =
            (org.apache.parquet.example.data.simple.SimpleGroup) g.addGroup(name);
        if (val instanceof long[]) {
            long[] a = (long[]) val;
            for (long v : a) {
                org.apache.parquet.example.data.simple.SimpleGroup mid =
                    (org.apache.parquet.example.data.simple.SimpleGroup) listOuter.addGroup(0);
                mid.add(0, v);
            }
            return;
        }
        if (val instanceof int[]) {
            int[] a = (int[]) val;
            for (int v : a) {
                org.apache.parquet.example.data.simple.SimpleGroup mid =
                    (org.apache.parquet.example.data.simple.SimpleGroup) listOuter.addGroup(0);
                mid.add(0, v);
            }
            return;
        }
        if (val instanceof float[]) {
            float[] a = (float[]) val;
            for (float v : a) {
                org.apache.parquet.example.data.simple.SimpleGroup mid =
                    (org.apache.parquet.example.data.simple.SimpleGroup) listOuter.addGroup(0);
                mid.add(0, v);
            }
            return;
        }
        if (val instanceof double[]) {
            double[] a = (double[]) val;
            for (double v : a) {
                org.apache.parquet.example.data.simple.SimpleGroup mid =
                    (org.apache.parquet.example.data.simple.SimpleGroup) listOuter.addGroup(0);
                mid.add(0, v);
            }
            return;
        }
        if (val instanceof List) {
            for (Object o : (List<?>) val) {
                org.apache.parquet.example.data.simple.SimpleGroup mid =
                    (org.apache.parquet.example.data.simple.SimpleGroup) listOuter.addGroup(0);
                if (o == null) continue;
                if (o instanceof Number) {
                    Number n = (Number) o;
                    // Prefer widest write; schema decides actual type
                    if (o instanceof Double || o instanceof Float) mid.add(0, n.doubleValue());
                    else mid.add(0, n.longValue());
                } else {
                    mid.add(0, String.valueOf(o));
                }
            }
        }
    }

    private org.apache.parquet.schema.MessageType buildParquetSchema() {
        // Build MessageType with proper LIST logical types for LIST/VECTOR columns.
        List<org.apache.parquet.schema.Type> fields = new ArrayList<>();
        for (Column c : columns) {
            fields.add(parquetFieldForColumn(c));
        }
        return new org.apache.parquet.schema.MessageType("root", fields);
    }

    private static org.apache.parquet.schema.Type parquetFieldForColumn(Column c) {
        String name = c.name();
        switch (c.dtype()) {
            case INT32:   return org.apache.parquet.schema.Types.optional(
                org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT32).named(name);
            case INT64:   return org.apache.parquet.schema.Types.optional(
                org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT64).named(name);
            case FLOAT32: return org.apache.parquet.schema.Types.optional(
                org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.FLOAT).named(name);
            case FLOAT64: return org.apache.parquet.schema.Types.optional(
                org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.DOUBLE).named(name);
            case BOOLEAN: return org.apache.parquet.schema.Types.optional(
                org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BOOLEAN).named(name);
            case LIST:
            case VECTOR:
            case EMBEDDING:
                return parquetListField(name, inferListElementPrimitive(c));
            case MAP:
                return parquetMapField(name);
            case STRUCT:
                return parquetStructField(name, c);
            case JSON:
                return org.apache.parquet.schema.Types.optional(
                        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BINARY)
                    .as(org.apache.parquet.schema.LogicalTypeAnnotation.jsonType())
                    .named(name);
            case BINARY:
                return org.apache.parquet.schema.Types.optional(
                    org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BINARY).named(name);
            default:
                // STRING and fallbacks
                return org.apache.parquet.schema.Types.optional(
                        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BINARY)
                    .as(org.apache.parquet.schema.LogicalTypeAnnotation.stringType())
                    .named(name);
        }
    }

    /**
     * Standard MAP:
     *   optional group name (MAP) {
     *     repeated group key_value {
     *       required binary key (STRING);
     *       optional binary value (STRING);
     *     }
     *   }
     */
    private static org.apache.parquet.schema.Type parquetMapField(String name) {
        return org.apache.parquet.schema.Types.optionalMap()
            .key(org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BINARY)
            .value(org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BINARY,
                org.apache.parquet.schema.Type.Repetition.OPTIONAL)
            .named(name);
    }

    /**
     * STRUCT as group of optional string fields. Field names are the union of keys
     * observed in the column samples (LinkedHashMap order of first row that has each key).
     */
    private static org.apache.parquet.schema.Type parquetStructField(String name, Column c) {
        java.util.LinkedHashSet<String> keys = new java.util.LinkedHashSet<>();
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            if (v == null) continue;
            java.util.Map<String, Object> m =
                org.bytedeco.pytorch.data.dataframe.io.ComplexCellCodec.asStringMap(v);
            if (m != null) keys.addAll(m.keySet());
            if (keys.size() > 64) break; // cap width
        }
        if (keys.isEmpty()) {
            // empty struct placeholder with a dummy optional string field
            return org.apache.parquet.schema.Types.buildGroup(
                    org.apache.parquet.schema.Type.Repetition.OPTIONAL)
                .optional(org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BINARY)
                    .as(org.apache.parquet.schema.LogicalTypeAnnotation.stringType())
                    .named("_empty")
                .named(name);
        }
        org.apache.parquet.schema.Types.GroupBuilder<org.apache.parquet.schema.GroupType> gb =
            org.apache.parquet.schema.Types.buildGroup(
                org.apache.parquet.schema.Type.Repetition.OPTIONAL);
        for (String k : keys) {
            String fieldName = k == null || k.isEmpty() ? "_k" : k;
            gb = gb.optional(org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.BINARY)
                .as(org.apache.parquet.schema.LogicalTypeAnnotation.stringType())
                .named(fieldName);
        }
        return gb.named(name);
    }

    /** Infer primitive element type of a LIST/VECTOR column from the first non-null cell. */
    private static org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName
            inferListElementPrimitive(Column c) {
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            if (v == null) continue;
            if (v instanceof int[]) return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT32;
            if (v instanceof long[]) return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT64;
            if (v instanceof float[]) return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.FLOAT;
            if (v instanceof double[]) return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.DOUBLE;
            if (v instanceof List && !((List<?>) v).isEmpty()) {
                Object e = ((List<?>) v).get(0);
                if (e instanceof Integer) return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT32;
                if (e instanceof Long) return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT64;
                if (e instanceof Float) return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.FLOAT;
                if (e instanceof Double) return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.DOUBLE;
            }
            break;
        }
        // default: INT64 for LIST (item sequences), FLOAT for VECTOR/EMBEDDING
        if (c.dtype() == Column.DType.VECTOR || c.dtype() == Column.DType.EMBEDDING) {
            return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.FLOAT;
        }
        return org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT64;
    }

    /**
     * Standard 3-level LIST:
     *   optional group name (LIST) {
     *     repeated group list {
     *       optional T item;
     *     }
     *   }
     */
    private static org.apache.parquet.schema.Type parquetListField(
            String name, org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName elem) {
        return org.apache.parquet.schema.Types.optionalList()
            .element(elem, org.apache.parquet.schema.Type.Repetition.OPTIONAL)
            .named(name);
    }
}
