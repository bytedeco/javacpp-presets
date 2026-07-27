package org.bytedeco.pytorch.data.parquet;

import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.ColumnWriteStore;
import org.apache.parquet.column.Encoding;
import org.apache.parquet.column.ParquetProperties;
import org.apache.parquet.column.impl.ColumnWriteStoreV1;
import org.apache.parquet.column.page.DictionaryPage;
import org.apache.parquet.column.page.PageWriteStore;
import org.apache.parquet.column.page.PageWriter;
import org.apache.parquet.column.statistics.Statistics;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.GroupWriter;
import org.apache.parquet.example.data.simple.SimpleGroup;
import org.apache.parquet.format.ColumnChunk;
import org.apache.parquet.format.ColumnMetaData;
import org.apache.parquet.format.CompressionCodec;
import org.apache.parquet.format.FileMetaData;
import org.apache.parquet.format.PageHeader;
import org.apache.parquet.format.RowGroup;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.ColumnIOFactory;
import org.apache.parquet.io.MessageColumnIO;
import org.apache.parquet.io.api.RecordConsumer;
import org.apache.parquet.schema.MessageType;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure-Java Parquet file writer (our {@code ParquetOutputFormat}).
 *
 * <p>Uses {@code parquet-column} encodings + thrift footer from
 * {@code parquet-format-structures}. <b>No Hadoop, no parquet-hadoop.</b>
 *
 * <pre>
 *   try (ParquetOutputFormat out = ParquetOutputFormat.builder(path, schema)
 *           .withCompression(CompressionCodecName.ZSTD)
 *           .build()) {
 *       SimpleGroup g = out.makeGroup();
 *       g.add("id", 1L);
 *       out.write(g);
 *   }
 * </pre>
 */
public final class ParquetOutputFormat implements Closeable {
    public static final class Builder {
        private final String path;
        private final MessageType schema;
        private CompressionCodecName codec = CompressionCodecName.UNCOMPRESSED;
        private int rowGroupSize = 128 * 1024 * 1024;
        private int pageSize = ParquetProperties.DEFAULT_PAGE_SIZE;
        private boolean dictionary = true;

        public Builder(String path, MessageType schema) {
            this.path = path;
            this.schema = schema;
        }

        public Builder withCompression(CompressionCodecName codec) {
            this.codec = codec == null ? CompressionCodecName.UNCOMPRESSED : codec;
            return this;
        }

        public Builder withRowGroupSize(int bytes) {
            this.rowGroupSize = bytes;
            return this;
        }

        public Builder withPageSize(int bytes) {
            this.pageSize = bytes;
            return this;
        }

        public Builder withDictionary(boolean enabled) {
            this.dictionary = enabled;
            return this;
        }

        public ParquetOutputFormat build() throws IOException {
            return new ParquetOutputFormat(this);
        }
    }

    public static Builder builder(String path, MessageType schema) {
        return new Builder(path, schema);
    }

    private final Path path;
    private final MessageType schema;
    private final CompressionCodec thriftCodec;
    private final ParquetProperties props;
    private final MessageColumnIO columnIO;
    private final List<ColumnDescriptor> columns;

    private OutputStream out;
    private long fileOffset; // bytes written so far (after magic)
    private long rowCount;
    private long rowsInGroup;
    private long bufferedEstimate;

    private ColumnChunkPageWriteStore pageStore;
    private ColumnWriteStore writeStore;
    private RecordConsumer recordConsumer;
    private GroupWriter groupWriter;
    private final List<RowGroup> rowGroups = new ArrayList<>();
    private boolean closed;

    private ParquetOutputFormat(Builder b) throws IOException {
        this.path = Paths.get(b.path);
        this.schema = b.schema;
        this.thriftCodec = ParquetFormatCodec.toThrift(b.codec);
        // Nested LIST/VECTOR: intermediate page flushes can leave a column with
        // rowsWrittenSoFar < rowCount but valueCount==0 → "writing empty page"
        // on final flush. Disable mid-chunk page splitting by row/value limits;
        // page size still bounds memory via withPageSize.
        this.props = ParquetProperties.builder()
            .withPageSize(b.pageSize)
            .withDictionaryEncoding(b.dictionary)
            .withPageRowCountLimit(Integer.MAX_VALUE)
            .withPageValueCountThreshold(Integer.MAX_VALUE)
            .withMinRowCountForPageSizeCheck(Integer.MAX_VALUE)
            .withMaxRowCountForPageSizeCheck(Integer.MAX_VALUE)
            .build();
        this.columnIO = new ColumnIOFactory().getColumnIO(schema);
        this.columns = schema.getColumns();

        Path parent = path.getParent();
        if (parent != null) Files.createDirectories(parent);
        this.out = Files.newOutputStream(path);
        ParquetFormatCodec.writeMagic(out);
        this.fileOffset = 4; // after PAR1
        startRowGroup();
    }

    public SimpleGroup makeGroup() {
        return new SimpleGroup(schema);
    }

    public MessageType getSchema() { return schema; }

    public long getRowCount() { return rowCount; }

    public void write(Group row) throws IOException {
        ensureOpen();
        groupWriter.write(row);
        writeStore.endRecord();
        rowCount++;
        rowsInGroup++;
        bufferedEstimate += 64; // rough; flush on page store size too
        if (writeStore.isColumnFlushNeeded()
            || writeStore.getBufferedSize() >= props.getPageSizeThreshold() * Math.max(1, columns.size())
            || bufferedEstimate >= 128 * 1024 * 1024L) {
            // keep accumulating until row-group threshold via buffered size
        }
        if (writeStore.getBufferedSize() >= 128L * 1024 * 1024) {
            flushRowGroup();
            startRowGroup();
        }
    }

    private void startRowGroup() {
        pageStore = new ColumnChunkPageWriteStore(columns, thriftCodec);
        writeStore = new ColumnWriteStoreV1(schema, pageStore, props);
        recordConsumer = columnIO.getRecordWriter(writeStore);
        groupWriter = new GroupWriter(recordConsumer, schema);
        rowsInGroup = 0;
        bufferedEstimate = 0;
    }

    private void flushRowGroup() throws IOException {
        if (writeStore == null) return;
        if (rowsInGroup == 0) {
            writeStore.close();
            writeStore = null;
            pageStore = null;
            groupWriter = null;
            recordConsumer = null;
            return;
        }
        // close() flushes pages; do NOT call flush()+close() (triggers empty-page error).
        writeStore.close();
        RowGroup rg = pageStore.writeTo(out, fileOffset, rowsInGroup, columns);
        rowGroups.add(rg);
        long off = 4; // after PAR1 magic
        for (RowGroup g : rowGroups) {
            if (g.getColumns() == null) continue;
            for (ColumnChunk cc : g.getColumns()) {
                if (cc.isSetMeta_data()) off += cc.getMeta_data().getTotal_compressed_size();
            }
        }
        fileOffset = off;
        rowsInGroup = 0;
        writeStore = null;
        pageStore = null;
        groupWriter = null;
        recordConsumer = null;
    }

    @Override
    public void close() throws IOException {
        if (closed) return;
        closed = true;
        try {
            if (rowsInGroup > 0) {
                flushRowGroup();
            } else if (writeStore != null) {
                writeStore.close();
            }
            FileMetaData meta = new FileMetaData(
                1,
                ParquetFormatCodec.toThriftSchema(schema),
                rowCount,
                rowGroups);
            meta.setCreated_by(ParquetFormatCodec.CREATED_BY);
            ParquetFormatCodec.writeFileMetaDataTrailer(out, meta);
        } finally {
            if (out != null) {
                out.close();
                out = null;
            }
        }
    }

    private void ensureOpen() throws IOException {
        if (closed) throw new IOException("ParquetOutputFormat closed");
    }

    // ---- page write store: buffer pages per column, then dump to file ------

    static final class ColumnChunkPageWriteStore implements PageWriteStore {
        private final Map<ColumnDescriptor, BufferingPageWriter> writers = new LinkedHashMap<>();
        private final CompressionCodec codec;

        ColumnChunkPageWriteStore(List<ColumnDescriptor> columns, CompressionCodec codec) {
            this.codec = codec;
            for (ColumnDescriptor c : columns) {
                writers.put(c, new BufferingPageWriter(c, codec));
            }
        }

        @Override
        public PageWriter getPageWriter(ColumnDescriptor descriptor) {
            return writers.get(descriptor);
        }

        @Override
        public void close() {
            for (BufferingPageWriter w : writers.values()) w.close();
        }

        /**
         * Write all column chunks sequentially; return RowGroup with metadata.
         * {@code startOffset} is current file offset (where first column chunk begins).
         */
        RowGroup writeTo(OutputStream out, long startOffset, long numRows,
                         List<ColumnDescriptor> ordered) throws IOException {
            List<ColumnChunk> chunks = new ArrayList<>();
            long offset = startOffset;
            long totalCompressed = 0;
            long totalUncompressed = 0;
            for (ColumnDescriptor col : ordered) {
                BufferingPageWriter w = writers.get(col);
                ChunkBytes cb = w.finish();
                long chunkStart = offset;
                out.write(cb.bytes);
                offset += cb.bytes.length;
                totalCompressed += cb.bytes.length;
                totalUncompressed += cb.uncompressedTotal;

                ColumnMetaData md = new ColumnMetaData(
                    ParquetFormatCodec.toThriftTypePublic(col.getType()),
                    cb.encodings,
                    Arrays.asList(col.getPath()),
                    codec,
                    cb.numValues,
                    cb.uncompressedTotal,
                    cb.bytes.length,
                    chunkStart);
                // data_page_offset: first data page; dict offset if present
                if (cb.dictionaryOffset >= 0) {
                    md.setDictionary_page_offset(chunkStart + cb.dictionaryOffset);
                    md.setData_page_offset(chunkStart + cb.firstDataOffset);
                } else {
                    md.setData_page_offset(chunkStart + cb.firstDataOffset);
                }
                ColumnChunk cc = new ColumnChunk(chunkStart);
                cc.setMeta_data(md);
                cc.setFile_offset(chunkStart);
                chunks.add(cc);
            }
            RowGroup rg = new RowGroup(chunks, totalUncompressed, numRows);
            rg.setFile_offset(startOffset);
            rg.setTotal_compressed_size(totalCompressed);
            return rg;
        }
    }

    static final class ChunkBytes {
        final byte[] bytes;
        final long numValues;
        final long uncompressedTotal;
        final List<org.apache.parquet.format.Encoding> encodings;
        final long dictionaryOffset; // relative to chunk start, or -1
        final long firstDataOffset;  // relative to chunk start

        ChunkBytes(byte[] bytes, long numValues, long uncompressedTotal,
                   List<org.apache.parquet.format.Encoding> encodings,
                   long dictionaryOffset, long firstDataOffset) {
            this.bytes = bytes;
            this.numValues = numValues;
            this.uncompressedTotal = uncompressedTotal;
            this.encodings = encodings;
            this.dictionaryOffset = dictionaryOffset;
            this.firstDataOffset = firstDataOffset;
        }
    }

    static final class BufferingPageWriter implements PageWriter {
        private final ColumnDescriptor col;
        private final CompressionCodec codec;
        private final ByteArrayOutputStream buf = new ByteArrayOutputStream();
        private final List<org.apache.parquet.format.Encoding> encodings = new ArrayList<>();
        private long numValues;
        private long uncompressedTotal;
        private long dictionaryOffset = -1;
        private long firstDataOffset = -1;
        private DictionaryPage pendingDict;
        private boolean closed;

        BufferingPageWriter(ColumnDescriptor col, CompressionCodec codec) {
            this.col = col;
            this.codec = codec;
        }

        @Override
        public void writePage(BytesInput bytesInput, int valueCount, Statistics<?> statistics,
                              Encoding rlEncoding, Encoding dlEncoding, Encoding valuesEncoding)
                throws IOException {
            writePage(bytesInput, valueCount, -1, statistics, rlEncoding, dlEncoding, valuesEncoding);
        }

        @Override
        public void writePage(BytesInput bytesInput, int valueCount, int rowCount,
                              Statistics<?> statistics,
                              Encoding rlEncoding, Encoding dlEncoding, Encoding valuesEncoding)
                throws IOException {
            flushDictIfNeeded();
            byte[] raw = bytesInput.toByteArray();
            byte[] compressed = ParquetFormatCodec.compress(codec, raw);
            if (firstDataOffset < 0) firstDataOffset = buf.size();
            PageHeader header = ParquetFormatCodec.dataPageV1Header(
                valueCount, raw.length, compressed.length, valuesEncoding, rlEncoding, dlEncoding);
            ParquetFormatCodec.writePageHeader(header, buf);
            buf.write(compressed);
            numValues += valueCount;
            uncompressedTotal += raw.length;
            addEncoding(rlEncoding);
            addEncoding(dlEncoding);
            addEncoding(valuesEncoding);
        }

        // parquet-column 1.17 default methods throw UOE — delegate to V1 path.
        @Override
        public void writePage(BytesInput bytesInput, int valueCount, int rowCount,
                              Statistics<?> statistics,
                              org.apache.parquet.column.statistics.SizeStatistics sizeStatistics,
                              Encoding rlEncoding, Encoding dlEncoding, Encoding valuesEncoding)
                throws IOException {
            writePage(bytesInput, valueCount, rowCount, statistics, rlEncoding, dlEncoding, valuesEncoding);
        }

        @Override
        public void writePage(BytesInput bytesInput, int valueCount, int rowCount,
                              Statistics<?> statistics,
                              org.apache.parquet.column.statistics.SizeStatistics sizeStatistics,
                              org.apache.parquet.column.statistics.geospatial.GeospatialStatistics geoStatistics,
                              Encoding rlEncoding, Encoding dlEncoding, Encoding valuesEncoding)
                throws IOException {
            writePage(bytesInput, valueCount, rowCount, statistics, rlEncoding, dlEncoding, valuesEncoding);
        }

        @Override
        public void writePageV2(int rowCount, int nullCount, int valueCount,
                                BytesInput repetitionLevels, BytesInput definitionLevels,
                                Encoding dataEncoding, BytesInput data,
                                Statistics<?> statistics) throws IOException {
            // Flatten V2 into a synthetic V1-like concatenated page is non-trivial;
            // ColumnWriteStoreV1 uses V1 pages by default — this path is rare.
            throw new IOException("DATA_PAGE_V2 write not supported in pure-Java ParquetOutputFormat");
        }

        @Override
        public void writePageV2(int rowCount, int nullCount, int valueCount,
                                BytesInput repetitionLevels, BytesInput definitionLevels,
                                Encoding dataEncoding, BytesInput data,
                                Statistics<?> statistics,
                                org.apache.parquet.column.statistics.SizeStatistics sizeStatistics)
                throws IOException {
            writePageV2(rowCount, nullCount, valueCount, repetitionLevels, definitionLevels,
                dataEncoding, data, statistics);
        }

        @Override
        public void writePageV2(int rowCount, int nullCount, int valueCount,
                                BytesInput repetitionLevels, BytesInput definitionLevels,
                                Encoding dataEncoding, BytesInput data,
                                Statistics<?> statistics,
                                org.apache.parquet.column.statistics.SizeStatistics sizeStatistics,
                                org.apache.parquet.column.statistics.geospatial.GeospatialStatistics geoStatistics)
                throws IOException {
            writePageV2(rowCount, nullCount, valueCount, repetitionLevels, definitionLevels,
                dataEncoding, data, statistics);
        }

        @Override
        public long getMemSize() { return buf.size(); }

        @Override
        public long allocatedSize() { return buf.size(); }

        @Override
        public void writeDictionaryPage(DictionaryPage dictionaryPage) throws IOException {
            // defer until first data page so dict is first in chunk
            pendingDict = dictionaryPage;
        }

        private void flushDictIfNeeded() throws IOException {
            if (pendingDict == null) return;
            DictionaryPage dp = pendingDict;
            pendingDict = null;
            if (dictionaryOffset < 0) dictionaryOffset = buf.size();
            byte[] raw = dp.getBytes().toByteArray();
            byte[] compressed = ParquetFormatCodec.compress(codec, raw);
            PageHeader header = ParquetFormatCodec.dictionaryPageHeader(
                dp.getDictionarySize(), raw.length, compressed.length, dp.getEncoding());
            ParquetFormatCodec.writePageHeader(header, buf);
            buf.write(compressed);
            uncompressedTotal += raw.length;
            addEncoding(dp.getEncoding());
        }

        @Override
        public String memUsageString(String prefix) {
            return prefix + " col=" + Arrays.toString(col.getPath()) + " size=" + buf.size();
        }

        @Override
        public void close() { closed = true; }

        ChunkBytes finish() throws IOException {
            flushDictIfNeeded();
            if (firstDataOffset < 0) firstDataOffset = 0;
            // ensure PLAIN present for readers
            if (encodings.isEmpty()) {
                encodings.add(org.apache.parquet.format.Encoding.PLAIN);
            }
            return new ChunkBytes(buf.toByteArray(), numValues, uncompressedTotal,
                new ArrayList<>(encodings), dictionaryOffset, firstDataOffset);
        }

        private void addEncoding(Encoding e) {
            org.apache.parquet.format.Encoding te = ParquetFormatCodec.toThrift(e);
            if (!encodings.contains(te)) encodings.add(te);
        }
    }
}
