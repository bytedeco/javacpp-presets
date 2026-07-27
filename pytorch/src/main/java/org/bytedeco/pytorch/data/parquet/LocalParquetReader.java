package org.bytedeco.pytorch.data.parquet;

import java.io.IOException;
import java.util.List;

import org.apache.parquet.example.data.Group;
import org.apache.parquet.schema.MessageType;

/**
 * Local-only Parquet row-reader facade over pure-Java {@link ParquetInputFormat}.
 * No Hadoop / {@code parquet-hadoop}.
 */
public final class LocalParquetReader implements AutoCloseable {

    private final ParquetInputFormat in;

    private LocalParquetReader(ParquetInputFormat in) {
        this.in = in;
    }

    public static LocalParquetReader open(String path) throws IOException {
        return new LocalParquetReader(ParquetInputFormat.open(path));
    }

    public MessageType getSchema() { return in.getSchema(); }

    public List<String> getFieldNames() { return in.getFieldNames(); }

    public Group read() throws IOException { return in.read(); }

    @Override
    public void close() throws IOException { in.close(); }
}
