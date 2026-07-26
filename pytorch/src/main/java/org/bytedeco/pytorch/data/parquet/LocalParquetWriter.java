package org.bytedeco.pytorch.data.parquet;

import java.io.IOException;

import org.apache.parquet.column.ParquetProperties;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroup;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.io.OutputFile;
import org.apache.parquet.schema.MessageType;

/**
 * Local-only Parquet row-writer (no Hadoop Path / FileSystem / UGI).
 *
 * <p>Builds a parquet file row-by-row from {@link Group} objects:
 * <pre>
 *   MessageType schema = SchemaBuilder.builder()
 *       .requiredInt64("id")
 *       .optionalString("name")
 *       .requiredFloat("emb")
 *       .build();
 *
 *   try (LocalParquetWriter w = LocalParquetWriter.builder(path, schema)
 *           .withCompression(CompressionCodecName.ZSTD)
 *           .withBlockSize(128 * 1024 * 1024)
 *           .build()) {
 *       for (MyRow r : rows) {
 *           SimpleGroup g = w.makeGroup();
 *           g.add("id", r.id());
 *           g.add("name", r.name());
 *           g.add("emb", r.emb());
 *           w.write(g);
 *       }
 *   }
 * </pre>
 *
 * <p>Compression uses {@link ZstdCodecFactory} (zstd-jni / snappy-java).
 */
public final class LocalParquetWriter implements AutoCloseable {

    /** Fluent builder for {@link LocalParquetWriter}. */
    public static class Builder {
        private final String path;
        private final MessageType schema;
        private CompressionCodecName codec = CompressionCodecName.UNCOMPRESSED;
        private int blockSize = 128 * 1024 * 1024;
        private int pageSize = ParquetProperties.DEFAULT_PAGE_SIZE;
        private boolean enableDictionary = ParquetProperties.DEFAULT_IS_DICTIONARY_ENABLED;
        private boolean enableValidation = false;

        public Builder(String path, MessageType schema) {
            this.path = path;
            this.schema = schema;
        }

        public Builder withCompression(CompressionCodecName codec) {
            this.codec = codec;
            return this;
        }

        /** Set block/row-group size in bytes. Default: 128 MiB. */
        public Builder withBlockSize(int bytes) {
            this.blockSize = bytes;
            return this;
        }

        /** Set page size in bytes. Default: 1 MiB. */
        public Builder withPageSize(int bytes) {
            this.pageSize = bytes;
            return this;
        }

        /** Enable dictionary encoding. Default: true. */
        public Builder withDictionary(boolean enabled) {
            this.enableDictionary = enabled;
            return this;
        }

        /** Enable validation. Default: false. */
        public Builder withValidation(boolean enabled) {
            this.enableValidation = enabled;
            return this;
        }

        public LocalParquetWriter build() throws IOException {
            return new LocalParquetWriter(this);
        }
    }

    public static Builder builder(String path, MessageType schema) {
        return new Builder(path, schema);
    }

    // ---- internal state ----

    private final ParquetWriter<Group> writer;
    private final MessageType schema;

    private long rowCount;

    private LocalParquetWriter(Builder b) throws IOException {
        this.schema = b.schema;
        OutputFile outputFile = new LocalOutputFile(b.path);
        // ExampleParquetWriter.Builder is public; .build() returns ParquetWriter<Group>
        @SuppressWarnings("unchecked")
        ParquetWriter<Group> w = (ParquetWriter<Group>)
            ExampleParquetWriter.builder(outputFile)
                .withType(b.schema)
                .withCompressionCodec(b.codec)
                .withRowGroupSize(b.blockSize)
                .withPageSize(b.pageSize)
                .withDictionaryEncoding(b.enableDictionary)
                .withValidation(b.enableValidation)
                .build();
        this.writer = w;
    }

    /**
     * Returns a fresh {@link SimpleGroup} for this schema.
     * Populate it with {@code g.add(fieldName, value)} then pass to {@link #write(Group)}.
     */
    public SimpleGroup makeGroup() {
        return new SimpleGroup(schema);
    }

    /**
     * Write one row. Rows are buffered into row-groups internally; flushing happens
     * automatically when the current row-group reaches its size limit.
     */
    public void write(Group row) throws IOException {
        writer.write(row);
        rowCount++;
    }

    /** Number of rows written so far. */
    public long getRowCount() { return rowCount; }

    @Override
    public void close() throws IOException {
        writer.close();
    }
}
