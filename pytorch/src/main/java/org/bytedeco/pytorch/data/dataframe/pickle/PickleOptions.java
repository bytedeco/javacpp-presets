package org.bytedeco.pytorch.data.dataframe.pickle;

/**
 * Options for DataFrame pickle I/O.
 *
 * <p>Default write layout is {@link Layout#SELF_DESC} — a protocol-4 dict that
 * Python can {@code pickle.load} and turn into a DataFrame with
 * {@code pd.DataFrame(obj["data"], columns=obj["columns"])}.
 * {@link Layout#RECORDS} keeps backward compatibility with the previous
 * list-of-dicts encoding.
 */
public final class PickleOptions {

    public enum Layout {
        /** {@code List<Map<String,Object>>} — legacy Java / simple Python. */
        RECORDS,
        /** Column-oriented maps of equal-length lists. */
        COLUMNS,
        /**
         * Self-describing dict:
         * {@code {__pandas_dataframe__: true, columns, dtypes, data: [{…},…] | {col:[…]}}}.
         */
        SELF_DESC
    }

    private final Layout layout;
    private final int protocol;
    private final boolean pandasCompat;

    private PickleOptions(Builder b) {
        this.layout = b.layout;
        this.protocol = b.protocol;
        this.pandasCompat = b.pandasCompat;
    }

    public static Builder builder() { return new Builder(); }
    public static PickleOptions defaults() { return builder().build(); }
    public static PickleOptions records() {
        return builder().layout(Layout.RECORDS).build();
    }

    public Layout layout() { return layout; }
    public int protocol() { return protocol; }
    public boolean pandasCompat() { return pandasCompat; }

    public static final class Builder {
        private Layout layout = Layout.SELF_DESC;
        private int protocol = 4;
        private boolean pandasCompat = true;

        public Builder layout(Layout v) { this.layout = v == null ? Layout.SELF_DESC : v; return this; }
        public Builder protocol(int v) { this.protocol = v; return this; }
        public Builder pandasCompat(boolean v) { this.pandasCompat = v; return this; }

        public PickleOptions build() { return new PickleOptions(this); }
    }
}
