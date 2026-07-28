package org.bytedeco.pytorch.dataframe.hdf5;

import org.bytedeco.pytorch.dataframe.Column;

import java.util.*;

/**
 * Options for HDF5 DataFrame I/O.
 *
 * <p>Default write layout is <em>columnar</em>: a group at {@code key} with one 1-D
 * dataset per column plus attributes {@code format=columnar}, {@code column_names},
 * {@code dtypes}. Readable by h5py and by {@link Hdf5Reader}.
 */
public final class Hdf5Options {

    public enum Format {
        /** One 1-D dataset per column under the key group (default, portable). */
        COLUMNAR,
        /** Single 2-D numeric dataset (rows × cols); string columns not supported. */
        MATRIX
    }

    private final Format format;
    private final List<String> columns;
    private final Map<String, Column.DType> schema;
    private final boolean overwrite;
    private final int maxRows;

    private Hdf5Options(Builder b) {
        this.format = b.format;
        this.columns = b.columns == null ? null : Collections.unmodifiableList(new ArrayList<>(b.columns));
        this.schema = b.schema == null ? null : Collections.unmodifiableMap(new LinkedHashMap<>(b.schema));
        this.overwrite = b.overwrite;
        this.maxRows = b.maxRows;
    }

    public static Builder builder() { return new Builder(); }
    public static Hdf5Options defaults() { return builder().build(); }

    public Format format() { return format; }
    public List<String> columns() { return columns; }
    public Map<String, Column.DType> schema() { return schema; }
    public boolean overwrite() { return overwrite; }
    public int maxRows() { return maxRows; }

    public static final class Builder {
        private Format format = Format.COLUMNAR;
        private List<String> columns = null;
        private Map<String, Column.DType> schema = null;
        private boolean overwrite = true;
        private int maxRows = -1;

        public Builder format(Format v) { this.format = v == null ? Format.COLUMNAR : v; return this; }
        public Builder columns(List<String> v) { this.columns = v; return this; }
        public Builder schema(Map<String, Column.DType> v) { this.schema = v; return this; }
        public Builder overwrite(boolean v) { this.overwrite = v; return this; }
        public Builder maxRows(int v) { this.maxRows = v; return this; }

        public Hdf5Options build() { return new Hdf5Options(this); }
    }
}
