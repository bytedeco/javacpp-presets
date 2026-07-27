package org.bytedeco.pytorch.data.orc;

/**
 * Options for ORC read/write.
 *
 * <p>After Hadoop/ORC-core removal: {@link #maxRows()} is honored on
 * {@link LocalOrcReader}; {@link #compress()}, {@link #stripeSize()} and
 * {@link #batchSize()} are retained for API stability but ignored on the
 * DuckDB {@code read_orc} path. {@link LocalOrcWriter} may fail-fast when
 * DuckDB cannot {@code COPY ... FORMAT ORC} — use Parquet instead.
 */
public final class OrcOptions {
    public enum Compress { NONE, ZLIB, SNAPPY, LZ4, ZSTD }

    private final Compress compress;
    private final long stripeSize;
    private final int batchSize;
    private final int maxRows;
    private final boolean overwrite;

    private OrcOptions(Builder b) {
        this.compress = b.compress;
        this.stripeSize = b.stripeSize;
        this.batchSize = b.batchSize;
        this.maxRows = b.maxRows;
        this.overwrite = b.overwrite;
    }

    public static Builder builder() { return new Builder(); }
    public static OrcOptions defaults() { return builder().build(); }

    public Compress compress() { return compress; }
    public long stripeSize() { return stripeSize; }
    public int batchSize() { return batchSize; }
    public int maxRows() { return maxRows; }
    public boolean overwrite() { return overwrite; }

    public static final class Builder {
        private Compress compress = Compress.ZLIB;
        private long stripeSize = 64L * 1024 * 1024;
        private int batchSize = 1024;
        private int maxRows = -1;
        private boolean overwrite = true;

        public Builder compress(Compress v) { this.compress = v == null ? Compress.NONE : v; return this; }
        public Builder stripeSize(long v) { this.stripeSize = v; return this; }
        public Builder batchSize(int v) { this.batchSize = v; return this; }
        public Builder maxRows(int v) { this.maxRows = v; return this; }
        public Builder overwrite(boolean v) { this.overwrite = v; return this; }

        public OrcOptions build() { return new OrcOptions(this); }
    }
}
