package org.bytedeco.pytorch.data.orc;
import org.bytedeco.pytorch.nn.options.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.ql.exec.vector.BytesColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.ColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.DoubleColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.LongColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.TimestampColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.VectorizedRowBatch;
import org.apache.orc.OrcFile;
import org.apache.orc.Reader;
import org.apache.orc.RecordReader;
import org.apache.orc.TypeDescription;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.nio.charset.StandardCharsets;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;

/**
 * Local-filesystem ORC reader → {@link DataFrame}.
 *
 * <p>Uses Hadoop local FS ({@code file://}) without requiring a cluster.
 */
public final class LocalOrcReader {
    private LocalOrcReader() {}

    public static DataFrame read(String path) throws Exception {
        return read(path, OrcOptions.defaults());
    }

    public static DataFrame read(String path, OrcOptions options) throws Exception {
        OrcOptions opt = options == null ? OrcOptions.defaults() : options;
        Configuration conf = LocalOrcWriter.localConf();
        org.apache.hadoop.fs.FileSystem localFs = LocalOrcWriter.rawLocal(conf);
        Path hPath = new Path(new java.io.File(path).getAbsoluteFile().toURI());
        try (Reader reader = OrcFile.createReader(hPath,
                OrcFile.readerOptions(conf).filesystem(localFs))) {
            TypeDescription schema = reader.getSchema();
            List<TypeDescription> children = schema.getChildren();
            List<String> fieldNames = schema.getFieldNames();

            DataFrame df = DataFrame.create();
            Column.DType[] dtypes = new Column.DType[fieldNames.size()];
            for (int i = 0; i < fieldNames.size(); i++) {
                dtypes[i] = orcToDType(children.get(i));
                df.addColumn(fieldNames.get(i), dtypes[i]);
            }

            try (RecordReader rows = reader.rows()) {
                VectorizedRowBatch batch = schema.createRowBatch(opt.batchSize() > 0 ? opt.batchSize() : 1024);
                int max = opt.maxRows();
                while (rows.nextBatch(batch)) {
                    for (int r = 0; r < batch.size; r++) {
                        if (max >= 0 && df.rowCount() >= max) return df;
                        int ri = df.addEmptyRow();
                        for (int c = 0; c < fieldNames.size(); c++) {
                            Object v = readCell(batch.cols[c], children.get(c), r, dtypes[c]);
                            df.set(ri, fieldNames.get(c), v);
                        }
                    }
                }
            }
            return df;
        }
    }

    static Column.DType orcToDType(TypeDescription t) {
        switch (t.getCategory()) {
            case BOOLEAN: return Column.DType.BOOLEAN;
            case BYTE:
            case SHORT:
            case INT: return Column.DType.INT32;
            case LONG: return Column.DType.INT64;
            case FLOAT: return Column.DType.FLOAT32;
            case DOUBLE:
            case DECIMAL: return Column.DType.FLOAT64;
            case DATE: return Column.DType.DATE;
            case TIMESTAMP:
            case TIMESTAMP_INSTANT: return Column.DType.DATETIME;
            case BINARY: return Column.DType.BINARY;
            case STRING:
            case CHAR:
            case VARCHAR:
            default: return Column.DType.STRING;
        }
    }

    private static Object readCell(ColumnVector col, TypeDescription type, int row, Column.DType dtype) {
        if (col.isNull[row]) return null;
        int idx = col.isRepeating ? 0 : row;
        switch (type.getCategory()) {
            case BOOLEAN: {
                long v = ((LongColumnVector) col).vector[idx];
                return v != 0;
            }
            case BYTE:
            case SHORT:
            case INT: {
                long v = ((LongColumnVector) col).vector[idx];
                return (int) v;
            }
            case LONG: {
                return ((LongColumnVector) col).vector[idx];
            }
            case FLOAT: {
                double v = ((DoubleColumnVector) col).vector[idx];
                return (float) v;
            }
            case DOUBLE:
            case DECIMAL: {
                return ((DoubleColumnVector) col).vector[idx];
            }
            case DATE: {
                long days = ((LongColumnVector) col).vector[idx];
                return LocalDate.ofEpochDay(days);
            }
            case TIMESTAMP:
            case TIMESTAMP_INSTANT: {
                TimestampColumnVector tcv = (TimestampColumnVector) col;
                long millis = tcv.time[idx];
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(millis), ZoneOffset.UTC);
            }
            case BINARY: {
                BytesColumnVector bcv = (BytesColumnVector) col;
                int start = bcv.start[idx];
                int len = bcv.length[idx];
                byte[] out = new byte[len];
                System.arraycopy(bcv.vector[idx], start, out, 0, len);
                return out;
            }
            case STRING:
            case CHAR:
            case VARCHAR:
            default: {
                if (col instanceof BytesColumnVector) {
                    BytesColumnVector bcv = (BytesColumnVector) col;
                    return new String(bcv.vector[idx], bcv.start[idx], bcv.length[idx], StandardCharsets.UTF_8);
                }
                return null;
            }
        }
    }
}
