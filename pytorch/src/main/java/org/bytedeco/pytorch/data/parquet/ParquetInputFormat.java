package org.bytedeco.pytorch.data.parquet;

import org.apache.parquet.bytes.BytesInput;
import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.Encoding;
import org.apache.parquet.column.page.DataPage;
import org.apache.parquet.column.page.DataPageV1;
import org.apache.parquet.column.page.DictionaryPage;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.column.page.PageReader;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.convert.GroupRecordConverter;
import org.apache.parquet.format.ColumnChunk;
import org.apache.parquet.format.ColumnMetaData;
import org.apache.parquet.format.CompressionCodec;
import org.apache.parquet.format.FileMetaData;
import org.apache.parquet.format.PageHeader;
import org.apache.parquet.format.PageType;
import org.apache.parquet.format.RowGroup;
import org.apache.parquet.io.ColumnIOFactory;
import org.apache.parquet.io.MessageColumnIO;
import org.apache.parquet.io.RecordReader;
import org.apache.parquet.schema.MessageType;

import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure-Java Parquet file reader (our {@code ParquetInputFormat}).
 *
 * <p>Reads PAR1 footer via thrift ({@code parquet-format-structures}), decodes
 * column pages with {@code parquet-column}, materializes rows as
 * {@link Group} via {@link GroupRecordConverter}. <b>No Hadoop, no
 * parquet-hadoop.</b>
 *
 * <pre>
 *   try (ParquetInputFormat in = ParquetInputFormat.open("data.parquet")) {
 *       MessageType schema = in.getSchema();
 *       for (Group row = in.read(); row != null; row = in.read()) { ... }
 *   }
 * </pre>
 */
public final class ParquetInputFormat implements Closeable {
    private final Path path;
    private final FileMetaData footer;
    private final MessageType schema;
    private final MessageColumnIO columnIO;
    private final List<ColumnDescriptor> columns;

    private int rowGroupIndex;
    private long remainingInGroup;
    private RecordReader<Group> recordReader;
    private boolean closed;

    private ParquetInputFormat(Path path, FileMetaData footer, MessageType schema) {
        this.path = path;
        this.footer = footer;
        this.schema = schema;
        this.columnIO = new ColumnIOFactory().getColumnIO(schema);
        this.columns = schema.getColumns();
        this.rowGroupIndex = 0;
        this.remainingInGroup = 0;
    }

    public static ParquetInputFormat open(String path) throws IOException {
        return open(Paths.get(path));
    }

    public static ParquetInputFormat open(Path path) throws IOException {
        FileMetaData footer = ParquetFormatCodec.readFileMetaData(path);
        MessageType schema = ParquetFormatCodec.fromThriftSchema(footer.getSchema());
        return new ParquetInputFormat(path, footer, schema);
    }

    public MessageType getSchema() { return schema; }

    public FileMetaData getFooter() { return footer; }

    public long getRecordCount() { return footer.getNum_rows(); }

    public int getRowGroupCount() {
        return footer.getRow_groups() == null ? 0 : footer.getRow_groups().size();
    }

    public List<String> getFieldNames() {
        List<String> names = new ArrayList<>();
        for (org.apache.parquet.schema.Type t : schema.getFields()) {
            names.add(t.getName());
        }
        return names;
    }

    /**
     * Read next row as {@link Group}, or {@code null} at EOF.
     */
    public Group read() throws IOException {
        ensureOpen();
        while (true) {
            if (remainingInGroup <= 0) {
                if (!advanceRowGroup()) return null;
            }
            remainingInGroup--;
            Group row = recordReader.read();
            if (row != null) return row;
            // skipped / null record — continue
        }
    }

    private boolean advanceRowGroup() throws IOException {
        List<RowGroup> groups = footer.getRow_groups();
        if (groups == null || rowGroupIndex >= groups.size()) {
            recordReader = null;
            remainingInGroup = 0;
            return false;
        }
        RowGroup rg = groups.get(rowGroupIndex++);
        PageReadStore pages = loadRowGroup(rg);
        GroupRecordConverter converter = new GroupRecordConverter(schema);
        recordReader = columnIO.getRecordReader(pages, converter);
        remainingInGroup = rg.getNum_rows();
        return remainingInGroup > 0 || advanceRowGroup();
    }

    private PageReadStore loadRowGroup(RowGroup rg) throws IOException {
        Map<String, ColumnChunk> byPath = new HashMap<>();
        if (rg.getColumns() != null) {
            for (ColumnChunk cc : rg.getColumns()) {
                ColumnMetaData md = cc.getMeta_data();
                if (md == null || md.getPath_in_schema() == null) continue;
                byPath.put(ParquetFormatCodec.pathKey(md.getPath_in_schema()), cc);
            }
        }
        Map<ColumnDescriptor, PageReader> readers = new HashMap<>();
        for (ColumnDescriptor col : columns) {
            String key = ParquetFormatCodec.pathKey(col.getPath());
            ColumnChunk cc = byPath.get(key);
            if (cc == null) {
                readers.put(col, EmptyPageReader.INSTANCE);
            } else {
                readers.put(col, new ColumnChunkPageReader(path, cc, col));
            }
        }
        long rowCount = rg.getNum_rows();
        return new SimplePageReadStore(readers, rowCount);
    }

    private void ensureOpen() throws IOException {
        if (closed) throw new IOException("ParquetInputFormat closed");
    }

    @Override
    public void close() {
        closed = true;
        recordReader = null;
    }

    // ---- PageReadStore / PageReader ----------------------------------------

    private static final class SimplePageReadStore implements PageReadStore {
        private final Map<ColumnDescriptor, PageReader> readers;
        private final long rowCount;

        SimplePageReadStore(Map<ColumnDescriptor, PageReader> readers, long rowCount) {
            this.readers = readers;
            this.rowCount = rowCount;
        }

        @Override
        public PageReader getPageReader(ColumnDescriptor descriptor) {
            PageReader r = readers.get(descriptor);
            return r == null ? EmptyPageReader.INSTANCE : r;
        }

        @Override
        public long getRowCount() { return rowCount; }
    }

    private static final class EmptyPageReader implements PageReader {
        static final EmptyPageReader INSTANCE = new EmptyPageReader();
        @Override public DictionaryPage readDictionaryPage() { return null; }
        @Override public long getTotalValueCount() { return 0; }
        @Override public DataPage readPage() { return null; }
    }

    /**
     * Streams pages from a single column chunk on disk.
     */
    private static final class ColumnChunkPageReader implements PageReader {
        private final Path path;
        private final ColumnChunk chunk;
        private final long chunkStart;
        private final long chunkEnd;
        private long pos;
        private DictionaryPage dictionaryPage;
        private boolean dictionaryLoaded;
        private final long totalValues;
        private long valuesRead;
        private final CompressionCodec codec;

        ColumnChunkPageReader(Path path, ColumnChunk chunk, ColumnDescriptor col) throws IOException {
            this.path = path;
            this.chunk = chunk;
            ColumnMetaData md = chunk.getMeta_data();
            this.codec = md.getCodec() == null ? CompressionCodec.UNCOMPRESSED : md.getCodec();
            this.totalValues = md.getNum_values();
            // Prefer dictionary_page_offset when present, else data_page_offset.
            long start = md.isSetDictionary_page_offset() && md.getDictionary_page_offset() > 0
                ? md.getDictionary_page_offset()
                : md.getData_page_offset();
            // file_offset on ColumnChunk is sometimes the first page; prefer meta offsets
            if (start <= 0 && chunk.isSetFile_offset()) start = chunk.getFile_offset();
            this.chunkStart = start;
            long compressedSize = md.getTotal_compressed_size();
            this.chunkEnd = start + Math.max(0, compressedSize);
            this.pos = start;
            this.valuesRead = 0;
        }

        @Override
        public DictionaryPage readDictionaryPage() {
            try {
                ensureDictionary();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            return dictionaryPage;
        }

        private void ensureDictionary() throws IOException {
            if (dictionaryLoaded) return;
            dictionaryLoaded = true;
            ColumnMetaData md = chunk.getMeta_data();
            // Arrow / many modern writers omit dictionary_page_offset even when the
            // first page at data_page_offset is a DICTIONARY_PAGE and data pages use
            // RLE_DICTIONARY. Always peek the first page of the chunk.
            long dictOff = md.isSetDictionary_page_offset() ? md.getDictionary_page_offset() : 0;
            long dataOff = md.isSetData_page_offset() ? md.getData_page_offset() : 0;
            long peekAt = dictOff > 0 ? dictOff : (dataOff > 0 ? dataOff : chunkStart);
            if (peekAt > 0 && pos != peekAt) {
                pos = peekAt;
            }
            PageAndBytes pb = readNextPageBytes();
            if (pb == null) return;
            if (pb.header.getType() == PageType.DICTIONARY_PAGE) {
                dictionaryPage = ParquetFormatCodec.toDictionaryPage(pb.header, pb.uncompressed);
            } else {
                // first page is data — keep it for readPage()
                pending = pb;
            }
        }

        private PageAndBytes pending;

        @Override
        public long getTotalValueCount() { return totalValues; }

        @Override
        public DataPage readPage() {
            try {
                ensureDictionary();
                if (valuesRead >= totalValues && pending == null) return null;
                PageAndBytes pb = pending;
                pending = null;
                if (pb == null) pb = readNextPageBytes();
                if (pb == null) return null;
                if (pb.header.getType() == PageType.DICTIONARY_PAGE) {
                    // skip extra dict
                    dictionaryPage = ParquetFormatCodec.toDictionaryPage(pb.header, pb.uncompressed);
                    return readPage();
                }
                if (pb.header.getType() == PageType.DATA_PAGE) {
                    DataPageV1 page = ParquetFormatCodec.toDataPageV1(pb.header, pb.uncompressed);
                    valuesRead += pb.header.getData_page_header().getNum_values();
                    return page;
                }
                if (pb.header.getType() == PageType.DATA_PAGE_V2) {
                    // Minimal V2 support: treat data bytes as PLAIN-like page without levels split —
                    // not fully correct for nested; still allow flat files.
                    throw new IOException("DATA_PAGE_V2 not yet supported by pure-Java ParquetInputFormat");
                }
                // INDEX_PAGE etc. — skip
                return readPage();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        private PageAndBytes readNextPageBytes() throws IOException {
            if (pos >= chunkEnd) return null;
            // Read a generous header window then parse thrift PageHeader
            int maxHeader = (int) Math.min(512, chunkEnd - pos);
            if (maxHeader <= 0) return null;
            // headers can be larger; grow if needed
            int attempt = 64;
            IOException last = null;
            while (attempt <= 8192 && pos + attempt <= chunkEnd + 8192) {
                int toRead = (int) Math.min(attempt, Math.max(1, chunkEnd - pos + 256));
                // read header candidate from pos
                long remain = Files.size(path) - pos;
                if (remain <= 0) return null;
                int window = (int) Math.min(Math.max(toRead, 64), remain);
                byte[] headBuf = ParquetFormatCodec.readFully(path, pos, window);
                try {
                    ByteArrayInputStream bin = new ByteArrayInputStream(headBuf);
                    long before = bin.available();
                    PageHeader header = ParquetFormatCodec.readPageHeader(bin);
                    int headerSize = (int) (before - bin.available());
                    int compressedSize = header.getCompressed_page_size();
                    int uncompressedSize = header.getUncompressed_page_size();
                    long dataStart = pos + headerSize;
                    byte[] compressed = ParquetFormatCodec.readFully(path, dataStart, compressedSize);
                    byte[] uncompressed = ParquetFormatCodec.decompress(
                        codec, compressed, uncompressedSize);
                    pos = dataStart + compressedSize;
                    return new PageAndBytes(header, uncompressed);
                } catch (Exception e) {
                    last = e instanceof IOException ? (IOException) e : new IOException(e);
                    attempt *= 2;
                }
            }
            if (last != null) throw last;
            return null;
        }
    }

    private static final class PageAndBytes {
        final PageHeader header;
        final byte[] uncompressed;
        PageAndBytes(PageHeader header, byte[] uncompressed) {
            this.header = header;
            this.uncompressed = uncompressed;
        }
    }
}
