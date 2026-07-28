package org.bytedeco.pytorch.data.avro;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileReader;
import org.apache.avro.file.FileReader;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.util.Utf8;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.io.ComplexCellCodec;
import org.bytedeco.pytorch.dataframe.io.IoTypeCoercion;

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
                // also honor prop-style logicalType used by our writer
                if ("date".equals(s.getProp("logicalType"))) return Column.DType.DATE;
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
                String prop = s.getProp("logicalType");
                if ("timestamp-millis".equals(prop) || "timestamp-micros".equals(prop)) {
                    return Column.DType.DATETIME;
                }
                return Column.DType.INT64;
            }
            case FLOAT: return Column.DType.FLOAT32;
            case DOUBLE: return Column.DType.FLOAT64;
            case BYTES:
            case FIXED:
                return Column.DType.BINARY;
            case ARRAY: {
                Schema elem = unwrapNullable(s.getElementType());
                // array<float|double> → VECTOR; otherwise LIST
                if (elem.getType() == Schema.Type.FLOAT || elem.getType() == Schema.Type.DOUBLE) {
                    return Column.DType.VECTOR;
                }
                return Column.DType.LIST;
            }
            case MAP:
                return Column.DType.MAP;
            case RECORD:
                return Column.DType.STRUCT;
            default:
                if (s.getLogicalType() != null && "date".equals(s.getLogicalType().getName())) {
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
        // GenericData.Array / Collection / Map from Avro nested types
        if (raw instanceof org.apache.avro.generic.GenericData.Array) {
            List<Object> list = new ArrayList<>();
            for (Object o : (org.apache.avro.generic.GenericData.Array<?>) raw) {
                list.add(unwrapAvroScalar(o));
            }
            raw = list;
        } else if (raw instanceof Collection && !(raw instanceof List)) {
            raw = new ArrayList<>((Collection<?>) raw);
        }
        if (raw instanceof org.apache.avro.generic.GenericRecord) {
            org.apache.avro.generic.GenericRecord rec = (org.apache.avro.generic.GenericRecord) raw;
            Map<String, Object> m = new LinkedHashMap<>();
            for (Schema.Field f : rec.getSchema().getFields()) {
                m.put(f.name(), unwrapAvroScalar(rec.get(f.name())));
            }
            raw = m;
        } else if (raw instanceof Map) {
            Map<?, ?> src = (Map<?, ?>) raw;
            Map<String, Object> m = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : src.entrySet()) {
                Object k = e.getKey();
                if (k instanceof Utf8) k = k.toString();
                m.put(String.valueOf(k), unwrapAvroScalar(e.getValue()));
            }
            raw = m;
        }

        if (schema != null) {
            String ln = schema.getLogicalType() != null ? schema.getLogicalType().getName()
                : schema.getProp("logicalType");
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
            if (ComplexCellCodec.isComplex(dtype) || ComplexCellCodec.isListLike(dtype)
                || ComplexCellCodec.isMapLike(dtype)) {
                return ComplexCellCodec.coerceComplex(raw, dtype);
            }
            return IoTypeCoercion.coerce(raw, dtype);
        } catch (Exception e) {
            return raw instanceof byte[] ? raw : String.valueOf(raw);
        }
    }

    private static Object unwrapAvroScalar(Object o) {
        if (o == null) return null;
        if (o instanceof Utf8) return o.toString();
        if (o instanceof ByteBuffer) {
            ByteBuffer bb = ((ByteBuffer) o).duplicate();
            byte[] bytes = new byte[bb.remaining()];
            bb.get(bytes);
            return bytes;
        }
        if (o instanceof org.apache.avro.generic.GenericData.Array) {
            List<Object> list = new ArrayList<>();
            for (Object x : (org.apache.avro.generic.GenericData.Array<?>) o) {
                list.add(unwrapAvroScalar(x));
            }
            return list;
        }
        if (o instanceof org.apache.avro.generic.GenericRecord) {
            org.apache.avro.generic.GenericRecord rec = (org.apache.avro.generic.GenericRecord) o;
            Map<String, Object> m = new LinkedHashMap<>();
            for (Schema.Field f : rec.getSchema().getFields()) {
                m.put(f.name(), unwrapAvroScalar(rec.get(f.name())));
            }
            return m;
        }
        if (o instanceof Map) {
            Map<?, ?> src = (Map<?, ?>) o;
            Map<String, Object> m = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : src.entrySet()) {
                Object k = e.getKey();
                if (k instanceof Utf8) k = k.toString();
                m.put(String.valueOf(k), unwrapAvroScalar(e.getValue()));
            }
            return m;
        }
        return o;
    }
}

