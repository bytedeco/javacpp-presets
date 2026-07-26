package org.bytedeco.pytorch.data.parquet;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.conf.PlainParquetConfiguration;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.convert.GroupRecordConverter;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.FileMetaData;
import org.apache.parquet.io.ColumnIOFactory;
import org.apache.parquet.io.LocalInputFile;
import org.apache.parquet.io.MessageColumnIO;
import org.apache.parquet.io.RecordReader;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.Type;

/**
 * Local-only Parquet row-reader (no Hadoop Path / FileSystem / UGI).
 *
 * <p>Mirrors the record-iteration logic of {@code ParquetReader}:
 * <pre>
 *   try (LocalParquetReader r = LocalParquetReader.open(path)) {
 *       for (Group row = r.read(); row != null; row = r.read()) {
 *           int id    = row.getInteger("id", 0);
 *           String name = row.getString("name", 0);
 *       }
 *   }
 * </pre>
 *
 * <p>Compression is handled by {@link ZstdCodecFactory} (zstd-jni / snappy-java),
 * not Hadoop codecs.
 */
public final class LocalParquetReader implements AutoCloseable {

    private final ParquetFileReader fileReader;
    private final MessageType schema;
    private final MessageColumnIO columnIO;

    private PageReadStore pages;
    private RecordReader<Group> recordReader;
    private long remainingInGroup;

    private LocalParquetReader(ParquetFileReader fileReader, MessageType schema,
                                MessageColumnIO columnIO) {
        this.fileReader = fileReader;
        this.schema = schema;
        this.columnIO = columnIO;
    }

    /** Returns the Parquet file schema. */
    public MessageType getSchema() { return schema; }

    /** Returns all field names in schema order. */
    public List<String> getFieldNames() {
        List<String> names = new ArrayList<>();
        for (Type field : schema.getFields()) {
            names.add(field.getName());
        }
        return names;
    }

    /**
     * Read the next row as a {@link Group}, or {@code null} at EOF.
     * Automatically advances to the next row-group when the current one is exhausted.
     */
    public Group read() throws IOException {
        while (true) {
            if (remainingInGroup <= 0) {
                pages = fileReader.readNextRowGroup();
                if (pages == null) return null; // EOF
                remainingInGroup = pages.getRowCount();
                GroupRecordConverter converter = new GroupRecordConverter(schema);
                recordReader = columnIO.getRecordReader(pages, converter);
            }
            remainingInGroup--;
            Group row = recordReader.read();
            // RecordReader may skip records internally; always re-read if null
            if (row != null) return row;
            // null means current record was skipped; loop to advance
        }
    }

    @Override
    public void close() throws IOException {
        fileReader.close();
    }

    /**
     * Open a local parquet file for row iteration.
     *
     * @param path  absolute or relative path to the parquet file
     */
    public static LocalParquetReader open(String path) throws IOException {
        LocalInputFile input = new LocalInputFile(Paths.get(path));
        PlainParquetConfiguration conf = new PlainParquetConfiguration();
        ParquetReadOptions options = ParquetReadOptions
            .builder(conf)
            .withCodecFactory(ZstdCodecFactory.INSTANCE)
            .build();
        ParquetFileReader fileReader = ParquetFileReader.open(input, options);
        FileMetaData meta = fileReader.getFooter().getFileMetaData();
        MessageType schema = meta.getSchema();
        MessageColumnIO columnIO = new ColumnIOFactory().getColumnIO(schema);
        return new LocalParquetReader(fileReader, schema, columnIO);
    }
}
