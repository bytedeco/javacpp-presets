package org.bytedeco.pytorch.data.avro;
import org.bytedeco.pytorch.nn.options.*;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileReader;
import org.apache.avro.file.FileReader;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.util.Utf8;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.io.IoTypeCoercion;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;

/**
 * Local Avro data-file reader → {@link DataFrame}.
 */
public final class LocalAvroReader {
    private LocalAvroReader() {}

    public static DataFrame read(String path) throws Exception {
        return read(path, AvroOptions.defaults());
    }

    public static DataFrame read(String path, AvroOptions options) throws Exception {
        AvroOptions opt = options == null ? AvroOptions.defaults() : options;
        File file = Path.of(path).toFile();
        try (FileReader<GenericRecord> reader =
                 DataFileReader.openReader(file, new GenericDatumReader<>())) {
            Schema schema = reader.getSchema();
            List<Schema.Field> fields = schema.getFields();
            String[] names = new String[fields.size()];
            Column.DType[] dtypes = new Column.DType[fields.size()];
            for (int i = 0; i < fields.size(); i++) {
                Schema.Field f = fields.get(i);
                names[i] = f.name();
                if (opt.schema() != null && opt.schema().containsKey(f.name())) {
                    dtypes[i] = opt.schema().get(f.name());
                } else {
                    dtypes[i] = avroToDType(unwrapNullable(f.schema()));
                }
            }

            DataFrame df = DataFrame.create();
            for (int i = 0; i < names.length; i++) df.addColumn(names[i], dtypes[i]);

            int max = opt.maxRows();
            while (reader.hasNext()) {
                if (max >= 0 && df.rowCount() >= max) break;
                GenericRecord rec = reader.next();
                int ri = df.addEmptyRow();
                for (int i = 0; i < names.length; i++) {
                    Object raw = rec.get(i);
                    Object v = convertValue(raw, dtypes[i], unwrapNullable(fields.get(i).schema()));
                    df.set(ri, names[i], v);
                }
            }
            return df;
        }
    }

    static Schema unwrapNullable(Schema s) {
        if (s.getType() == Schema.Type.UNION) {
            List<Schema> types = s.getTypes();
            Schema nonNull = null;
            for (Schema t : types) {
                if (t.getType() != Schema.Type.NULL) {
                    if (nonNull != null) return s; // complex union
                    nonNull = t;
                }
            }
            return nonNull != null ? nonNull : s;
        }
        return s;
    }

    static Column.DType avroToDType(Schema s) {
        switch (s.getType()) {
            case BOOLEAN: return Column.DType.BOOLEAN;
            case INT: {
                if (s.getLogicalType() != null && "date".equals(s.getLogicalType().getName())) {
                    return Column.DType.DATE;
                }
                return Column.DType.INT32;
            }
            case LONG: {
                // logical types
                if (s.getLogicalType() != null) {
                    String n = s.getLogicalType().getName();
                    if ("timestamp-millis".equals(n) || "timestamp-micros".equals(n)) {
                        return Column.DType.DATETIME;
                    }
                    if ("time-millis".equals(n) || "time-micros".equals(n)) {
                        return Column.DType.TIME;
                    }
                }
                return Column.DType.INT64;
            }
            case FLOAT: return Column.DType.FLOAT32;
            case DOUBLE: return Column.DType.FLOAT64;
            case BYTES:
            case FIXED:
                return Column.DType.BINARY;
            default:
                if (s.getLogicalType() != null && "date".equals(s.getLogicalType().getName())) {
                    return Column.DType.DATE;
                }
                // Avro INT with logicalType date handled above via getLogicalType on INT
                if (s.getType() == Schema.Type.INT && s.getLogicalType() != null
                    && "date".equals(s.getLogicalType().getName())) {
                    return Column.DType.DATE;
                }
                return Column.DType.STRING;
        }
    }

    static Object convertValue(Object raw, Column.DType dtype, Schema schema) {
        if (raw == null) return null;
        if (raw instanceof Utf8) raw = raw.toString();
        if (raw instanceof ByteBuffer) {
            ByteBuffer bb = ((ByteBuffer) raw).duplicate();
            byte[] bytes = new byte[bb.remaining()];
            bb.get(bytes);
            raw = bytes;
        }
        if (schema != null && schema.getLogicalType() != null) {
            String ln = schema.getLogicalType().getName();
            if ("date".equals(ln) && raw instanceof Number) {
                return LocalDate.ofEpochDay(((Number) raw).intValue());
            }
            if ("timestamp-millis".equals(ln) && raw instanceof Number) {
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(((Number) raw).longValue()), ZoneOffset.UTC);
            }
            if ("timestamp-micros".equals(ln) && raw instanceof Number) {
                long us = ((Number) raw).longValue();
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(us / 1000), ZoneOffset.UTC);
            }
        }
        try {
            return IoTypeCoercion.coerce(raw, dtype);
        } catch (Exception e) {
            return raw instanceof byte[] ? raw : String.valueOf(raw);
        }
    }
}
