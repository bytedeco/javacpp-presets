package org.bytedeco.pytorch.dataframe.sql;

import org.bytedeco.pytorch.dataframe.Column;

import java.util.*;

/**
 * Options for JDBC / SQLite DataFrame I/O (pandas {@code read_sql}/{@code to_sql} style).
 */
public final class SqlOptions {

    public enum IfExists {
        /** Raise if table exists. */
        FAIL,
        /** DROP then recreate. */
        REPLACE,
        /** INSERT into existing table. */
        APPEND
    }

    private final IfExists ifExists;
    private final int chunksize;
    private final boolean index;
    private final String indexLabel;
    private final boolean quoteIdentifiers;
    private final Map<String, Column.DType> dtype;
    private final String schema; // optional DB schema / catalog prefix
    private final int fetchSize;
    private final boolean autoCommitAroundWrite;

    private SqlOptions(Builder b) {
        this.ifExists = b.ifExists;
        this.chunksize = b.chunksize;
        this.index = b.index;
        this.indexLabel = b.indexLabel;
        this.quoteIdentifiers = b.quoteIdentifiers;
        this.dtype = b.dtype == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.dtype));
        this.schema = b.schema;
        this.fetchSize = b.fetchSize;
        this.autoCommitAroundWrite = b.autoCommitAroundWrite;
    }

    public static Builder builder() { return new Builder(); }
    public static SqlOptions defaults() { return builder().build(); }

    public IfExists ifExists() { return ifExists; }
    public int chunksize() { return chunksize; }
    public boolean index() { return index; }
    public String indexLabel() { return indexLabel; }
    public boolean quoteIdentifiers() { return quoteIdentifiers; }
    public Map<String, Column.DType> dtype() { return dtype; }
    public String schema() { return schema; }
    public int fetchSize() { return fetchSize; }
    public boolean autoCommitAroundWrite() { return autoCommitAroundWrite; }

    public static final class Builder {
        private IfExists ifExists = IfExists.FAIL;
        private int chunksize = 1000;
        private boolean index = false;
        private String indexLabel = "index";
        private boolean quoteIdentifiers = true;
        private Map<String, Column.DType> dtype = null;
        private String schema = null;
        private int fetchSize = 1000;
        private boolean autoCommitAroundWrite = true;

        public Builder ifExists(IfExists v) { this.ifExists = v == null ? IfExists.FAIL : v; return this; }
        public Builder chunksize(int v) { this.chunksize = Math.max(1, v); return this; }
        public Builder index(boolean v) { this.index = v; return this; }
        public Builder indexLabel(String v) { this.indexLabel = v == null ? "index" : v; return this; }
        public Builder quoteIdentifiers(boolean v) { this.quoteIdentifiers = v; return this; }
        public Builder dtype(Map<String, Column.DType> v) { this.dtype = v; return this; }
        public Builder schema(String v) { this.schema = v; return this; }
        public Builder fetchSize(int v) { this.fetchSize = v; return this; }
        public Builder autoCommitAroundWrite(boolean v) { this.autoCommitAroundWrite = v; return this; }

        public SqlOptions build() { return new SqlOptions(this); }
    }
}
