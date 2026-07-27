package org.bytedeco.pytorch.data.orc;

/**
 * Options for ORC read/write (shared by legacy DuckDB and pure-Java format paths).
 *
 * <p><b>Legacy DuckDB</b> ({@link LocalOrcReader}/{@link LocalOrcWriter}):
 * {@link #maxRows()} is honored on read; {@link #compress()}, {@link #stripeSize()}
 * and {@link #batchSize()} are retained for API stability but ignored on
 * {@code read_orc}. Write may fail-fast when DuckDB cannot {@code COPY FORMAT ORC}.
 *
 * <p><b>Pure-Java orc-format</b> ({@link LocalOrcFormatReader}/{@link LocalOrcFormatWriter}):
 * honors {@link #maxRows()}, {@link #overwrite()}, {@link #compress()} (NONE/ZLIB;
 * SNAPPY/LZ4/ZSTD soft-fallback to ZLIB), {@link #stripeSize()} and {@link #batchSize()}
 * as soft stripe targets. No Hadoop / no orc-core.
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
