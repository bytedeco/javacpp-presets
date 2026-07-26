package org.bytedeco.pytorch.data.orc;
import org.bytedeco.pytorch.nn.options.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.ql.exec.vector.BytesColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.DoubleColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.LongColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.TimestampColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.VectorizedRowBatch;
import org.apache.orc.CompressionKind;
import org.apache.orc.OrcFile;
import org.apache.orc.TypeDescription;
import org.apache.orc.Writer;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

/**
 * Write a {@link DataFrame} as an ORC file on the local filesystem.
 */
public final class LocalOrcWriter {
    private LocalOrcWriter() {}

    public static void write(DataFrame df, String path) throws Exception {
        write(df, path, OrcOptions.defaults());
    }

    public static void write(DataFrame df, String path, OrcOptions options) throws Exception {
        if (df == null) throw new IllegalArgumentException("dataframe required");
        OrcOptions opt = options == null ? OrcOptions.defaults() : options;

        java.nio.file.Path p = java.nio.file.Path.of(path);
        if (Files.exists(p)) {
            if (!opt.overwrite()) {
                throw new IllegalStateException("ORC file exists and overwrite=false: " + path);
            }
            Files.delete(p);
        } else {
            java.nio.file.Path parent = p.getParent();
            if (parent != null) Files.createDirectories(parent);
        }

        TypeDescription schema = buildSchema(df);
        Configuration conf = localConf();
        org.apache.hadoop.fs.FileSystem localFs = rawLocal(conf);
        Path hPath = new Path(p.toAbsolutePath().toUri());
        OrcFile.WriterOptions wopts = OrcFile.writerOptions(conf)
            .fileSystem(localFs)
            .setSchema(schema)
            .stripeSize(opt.stripeSize())
            .compress(toCompress(opt.compress()))
            .overwrite(true);

        try (Writer writer = OrcFile.createWriter(hPath, wopts)) {
            int batchSize = opt.batchSize() > 0 ? opt.batchSize() : 1024;
            VectorizedRowBatch batch = schema.createRowBatch(batchSize);
            int cols = df.columnCount();

            for (int r = 0; r < df.rowCount(); r++) {
                int row = batch.size++;
                for (int c = 0; c < cols; c++) {
                    Column col = df.column(c);
                    writeCell(batch, c, row, col.get(r), col.dtype());
                }
                if (batch.size >= batchSize) {
                    writer.addRowBatch(batch);
                    batch.reset();
                }
            }
            if (batch.size != 0) {
                writer.addRowBatch(batch);
                batch.reset();
            }
        }
    }

    static TypeDescription buildSchema(DataFrame df) {
        TypeDescription struct = TypeDescription.createStruct();
        for (int i = 0; i < df.columnCount(); i++) {
            Column col = df.column(i);
            struct.addField(col.name(), dtypeToOrc(col.dtype()));
        }
        return struct;
    }

    static TypeDescription dtypeToOrc(Column.DType dt) {
        switch (dt) {
            case BOOLEAN: return TypeDescription.createBoolean();
            case INT32: return TypeDescription.createInt();
            case INT64: return TypeDescription.createLong();
            case FLOAT32: return TypeDescription.createFloat();
            case FLOAT64: return TypeDescription.createDouble();
            case DATE: return TypeDescription.createDate();
            case DATETIME: return TypeDescription.createTimestamp();
            case BINARY: return TypeDescription.createBinary();
            default: return TypeDescription.createString();
        }
    }

    /** Minimal local-FS config that avoids Hadoop UGI Subject lookup on modern JDKs. */
    static Configuration localConf() {
        Configuration conf = new Configuration(false);
        conf.set("fs.defaultFS", "file:///");
        conf.set("fs.file.impl", org.apache.hadoop.fs.RawLocalFileSystem.class.getName());
        conf.set("fs.hdfs.impl.disable.cache", "true");
        conf.set("hadoop.security.authentication", "simple");
        conf.setBoolean("hadoop.security.authorization", false);
        // Prevent UserGroupInformation from calling Subject.getSubject (removed/blocked on JDK 21+)
        System.setProperty("HADOOP_USER_NAME",
            System.getProperty("user.name", "dataframe"));
        conf.setClass("fs.file.impl", org.apache.hadoop.fs.RawLocalFileSystem.class,
            org.apache.hadoop.fs.FileSystem.class);
        return conf;
    }

    /** Construct a RawLocalFileSystem without going through FileSystem cache/UGI. */
    static org.apache.hadoop.fs.FileSystem rawLocal(Configuration conf) throws java.io.IOException {
        org.apache.hadoop.fs.RawLocalFileSystem raw = new org.apache.hadoop.fs.RawLocalFileSystem();
        raw.initialize(java.net.URI.create("file:///"), conf);
        return raw;
    }

    private static CompressionKind toCompress(OrcOptions.Compress c) {
        if (c == null) return CompressionKind.NONE;
        switch (c) {
            case ZLIB: return CompressionKind.ZLIB;
            case SNAPPY: return CompressionKind.SNAPPY;
            case LZ4: return CompressionKind.LZ4;
            case ZSTD: return CompressionKind.ZSTD;
            case NONE:
            default: return CompressionKind.NONE;
        }
    }

    private static void writeCell(VectorizedRowBatch batch, int colIdx, int row,
                                  Object val, Column.DType dtype) {
        if (val == null) {
            batch.cols[colIdx].noNulls = false;
            batch.cols[colIdx].isNull[row] = true;
            return;
        }
        batch.cols[colIdx].isNull[row] = false;
        switch (dtype) {
            case BOOLEAN: {
                LongColumnVector lcv = (LongColumnVector) batch.cols[colIdx];
                boolean b = val instanceof Boolean ? (Boolean) val
                    : Boolean.parseBoolean(String.valueOf(val));
                lcv.vector[row] = b ? 1 : 0;
                break;
            }
            case INT32:
            case INT64: {
                LongColumnVector lcv = (LongColumnVector) batch.cols[colIdx];
                lcv.vector[row] = val instanceof Number ? ((Number) val).longValue()
                    : Long.parseLong(String.valueOf(val));
                break;
            }
            case FLOAT32:
            case FLOAT64: {
                DoubleColumnVector dcv = (DoubleColumnVector) batch.cols[colIdx];
                dcv.vector[row] = val instanceof Number ? ((Number) val).doubleValue()
                    : Double.parseDouble(String.valueOf(val));
                break;
            }
            case DATE: {
                LongColumnVector lcv = (LongColumnVector) batch.cols[colIdx];
                if (val instanceof LocalDate) {
                    lcv.vector[row] = ((LocalDate) val).toEpochDay();
                } else {
                    lcv.vector[row] = LocalDate.parse(String.valueOf(val)).toEpochDay();
                }
                break;
            }
            case DATETIME: {
                TimestampColumnVector tcv = (TimestampColumnVector) batch.cols[colIdx];
                long millis;
                if (val instanceof LocalDateTime) {
                    millis = ((LocalDateTime) val).toInstant(ZoneOffset.UTC).toEpochMilli();
                } else if (val instanceof Instant) {
                    millis = ((Instant) val).toEpochMilli();
                } else if (val instanceof Number) {
                    millis = ((Number) val).longValue();
                } else {
                    millis = Instant.parse(String.valueOf(val)).toEpochMilli();
                }
                tcv.time[row] = millis;
                tcv.nanos[row] = (int) ((millis % 1000) * 1_000_000);
                break;
            }
            case BINARY: {
                BytesColumnVector bcv = (BytesColumnVector) batch.cols[colIdx];
                byte[] bytes = val instanceof byte[] ? (byte[]) val
                    : String.valueOf(val).getBytes(StandardCharsets.UTF_8);
                bcv.setRef(row, bytes, 0, bytes.length);
                break;
            }
            default: {
                BytesColumnVector bcv = (BytesColumnVector) batch.cols[colIdx];
                byte[] bytes = String.valueOf(val).getBytes(StandardCharsets.UTF_8);
                bcv.setRef(row, bytes, 0, bytes.length);
                break;
            }
        }
    }
}
