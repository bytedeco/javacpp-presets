package org.bytedeco.pytorch.data.avro;

import org.bytedeco.pytorch.dataframe.Column;

import java.util.*;

/**
 * Options for Avro data-file read/write.
 */
public final class AvroOptions {
    public enum Codec { NULL, DEFLATE, SNAPPY }

    private final Codec codec;
    private final int deflateLevel;
    private final boolean nullableFields;
    private final Map<String, Column.DType> schema;
    private final boolean inferSchema;
    private final int maxRows;

    private AvroOptions(Builder b) {
        this.codec = b.codec;
        this.deflateLevel = b.deflateLevel;
        this.nullableFields = b.nullableFields;
        this.schema = b.schema == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.schema));
        this.inferSchema = b.inferSchema;
        this.maxRows = b.maxRows;
    }

    public static Builder builder() { return new Builder(); }
    public static AvroOptions defaults() { return builder().build(); }

    public Codec codec() { return codec; }
    public int deflateLevel() { return deflateLevel; }
    public boolean nullableFields() { return nullableFields; }
    public Map<String, Column.DType> schema() { return schema; }
    public boolean inferSchema() { return inferSchema; }
    public int maxRows() { return maxRows; }

    public static final class Builder {
        private Codec codec = Codec.NULL;
        private int deflateLevel = 6;
        private boolean nullableFields = true;
        private Map<String, Column.DType> schema = null;
        private boolean inferSchema = true;
        private int maxRows = -1;

        public Builder codec(Codec v) { this.codec = v == null ? Codec.NULL : v; return this; }
        public Builder deflateLevel(int v) { this.deflateLevel = v; return this; }
        public Builder nullableFields(boolean v) { this.nullableFields = v; return this; }
        public Builder schema(Map<String, Column.DType> v) { this.schema = v; return this; }
        public Builder inferSchema(boolean v) { this.inferSchema = v; return this; }
        public Builder maxRows(int v) { this.maxRows = v; return this; }

        public AvroOptions build() { return new AvroOptions(this); }
    }
}
