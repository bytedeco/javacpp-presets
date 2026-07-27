package org.bytedeco.pytorch.utils.lance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Options for scanning / reading an official Lance dataset into a DataFrame.
 *
 * <pre>{@code
 * DataFrame df = DataFrame.readLance("clips.lance",
 *     LanceReadOptions.builder()
 *         .columns("id", "emb")
 *         .filter("id > 10")
 *         .limit(100)
 *         .build());
 * }</pre>
 */
public final class LanceReadOptions {

    private List<String> columns;
    private String filter;
    private long limit = -1;
    private long offset = -1;
    private Long version;
    private String tag;
    private boolean withRowId;
    private boolean withRowAddress;
    private long batchSize = 64_000;
    private boolean useScalarIndex = true;

    public static LanceReadOptions defaults() {
        return new LanceReadOptions();
    }

    public static Builder builder() {
        return new Builder();
    }

    public List<String> columns() {
        return columns == null ? null : Collections.unmodifiableList(columns);
    }

    public String filter() { return filter; }
    public long limit() { return limit; }
    public long offset() { return offset; }
    public Long version() { return version; }
    public String tag() { return tag; }
    public boolean withRowId() { return withRowId; }
    public boolean withRowAddress() { return withRowAddress; }
    public long batchSize() { return batchSize; }
    public boolean useScalarIndex() { return useScalarIndex; }

    public static final class Builder {
        private final LanceReadOptions o = new LanceReadOptions();

        public Builder columns(String... cols) {
            if (cols == null || cols.length == 0) {
                o.columns = null;
            } else {
                o.columns = new ArrayList<>(Arrays.asList(cols));
            }
            return this;
        }

        public Builder columns(List<String> cols) {
            o.columns = cols == null ? null : new ArrayList<>(cols);
            return this;
        }

        public Builder filter(String expr) {
            o.filter = expr;
            return this;
        }

        public Builder limit(long n) {
            o.limit = n;
            return this;
        }

        public Builder offset(long n) {
            o.offset = n;
            return this;
        }

        /** Open a specific dataset version (time travel). */
        public Builder version(long v) {
            o.version = v;
            return this;
        }

        /** Open via tag name (time travel). */
        public Builder tag(String t) {
            o.tag = t;
            return this;
        }

        public Builder withRowId(boolean v) {
            o.withRowId = v;
            return this;
        }

        public Builder withRowAddress(boolean v) {
            o.withRowAddress = v;
            return this;
        }

        public Builder batchSize(long n) {
            o.batchSize = n;
            return this;
        }

        public Builder useScalarIndex(boolean v) {
            o.useScalarIndex = v;
            return this;
        }

        public LanceReadOptions build() {
            return o;
        }
    }
}
