package org.bytedeco.pytorch.data.parquet;

import java.io.IOException;

import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroup;
import org.apache.parquet.hadoop.metadata.CompressionCodecName;
import org.apache.parquet.schema.MessageType;

/**
 * Local-only Parquet row-writer facade over pure-Java {@link ParquetOutputFormat}.
 * No Hadoop / {@code parquet-hadoop}.
 *
 * <pre>
 *   MessageType schema = SchemaBuilder.builder()
 *       .requiredInt64("id")
 *       .optionalString("name")
 *       .build();
 *   try (LocalParquetWriter w = LocalParquetWriter.builder(path, schema)
 *           .withCompression(CompressionCodecName.ZSTD)
 *           .build()) {
 *       SimpleGroup g = w.makeGroup();
 *       g.add("id", 1L);
 *       g.add("name", "alice");
 *       w.write(g);
 *   }
 * </pre>
 *
 * <p>{@link CompressionCodecName} lives in {@code parquet-common} (package name only;
 * not the {@code parquet-hadoop} artifact).
 */
public final class LocalParquetWriter implements AutoCloseable {

    public static class Builder {
        private final String path;
        private final MessageType schema;
        private CompressionCodecName codec = CompressionCodecName.UNCOMPRESSED;
        private int blockSize = 128 * 1024 * 1024;
        private int pageSize = 1024 * 1024;
        private boolean enableDictionary = true;

        public Builder(String path, MessageType schema) {
            this.path = path;
            this.schema = schema;
        }

        public Builder withCompression(CompressionCodecName codec) {
            this.codec = codec == null ? CompressionCodecName.UNCOMPRESSED : codec;
            return this;
        }

        public Builder withBlockSize(int bytes) {
            this.blockSize = bytes;
            return this;
        }

        public Builder withPageSize(int bytes) {
            this.pageSize = bytes;
            return this;
        }

        public Builder withDictionary(boolean enabled) {
            this.enableDictionary = enabled;
            return this;
        }

        /** Retained for API compatibility; validation is always off in pure-Java path. */
        public Builder withValidation(boolean enabled) { return this; }

        public LocalParquetWriter build() throws IOException {
            return new LocalParquetWriter(this);
        }
    }

    public static Builder builder(String path, MessageType schema) {
        return new Builder(path, schema);
    }

    private final ParquetOutputFormat out;
    private final MessageType schema;
    private long rowCount;

    private LocalParquetWriter(Builder b) throws IOException {
        this.schema = b.schema;
        this.out = ParquetOutputFormat.builder(b.path, b.schema)
            .withCompression(b.codec)
            .withRowGroupSize(b.blockSize)
            .withPageSize(b.pageSize)
            .withDictionary(b.enableDictionary)
            .build();
    }

    public SimpleGroup makeGroup() {
        return out.makeGroup();
    }

    public void write(Group row) throws IOException {
        out.write(row);
        rowCount++;
    }

    public long getRowCount() { return rowCount; }

    public MessageType getSchema() { return schema; }

    @Override
    public void close() throws IOException {
        out.close();
    }
}
