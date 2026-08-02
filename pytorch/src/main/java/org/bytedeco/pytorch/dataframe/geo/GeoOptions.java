package org.bytedeco.pytorch.dataframe.geo;

import java.util.Objects;

/**
 * Options for geo parse / join / index operations.
 */
public final class GeoOptions {

    private final CRS crs;
    private final double tolerance;
    private final int h3Resolution;
    private final int s2Level;
    private final boolean validate;

    private GeoOptions(Builder b) {
        this.crs = b.crs;
        this.tolerance = b.tolerance;
        this.h3Resolution = b.h3Resolution;
        this.s2Level = b.s2Level;
        this.validate = b.validate;
    }

    public static GeoOptions defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public CRS crs() { return crs; }
    public double tolerance() { return tolerance; }
    public int h3Resolution() { return h3Resolution; }
    public int s2Level() { return s2Level; }
    public boolean validate() { return validate; }

    public static final class Builder {
        private CRS crs = CRS.WGS84;
        private double tolerance = 0.0;
        private int h3Resolution = 7;
        private int s2Level = 12;
        private boolean validate = true;

        public Builder crs(CRS crs) {
            this.crs = Objects.requireNonNull(crs, "crs");
            return this;
        }

        public Builder tolerance(double tolerance) {
            if (tolerance < 0) throw new IllegalArgumentException("tolerance must be >= 0");
            this.tolerance = tolerance;
            return this;
        }

        /** H3 resolution 0–15. */
        public Builder h3Resolution(int res) {
            if (res < 0 || res > 15) throw new IllegalArgumentException("h3 resolution 0..15");
            this.h3Resolution = res;
            return this;
        }

        /** S2 cell level 0–30. */
        public Builder s2Level(int level) {
            if (level < 0 || level > 30) throw new IllegalArgumentException("s2 level 0..30");
            this.s2Level = level;
            return this;
        }

        public Builder validate(boolean v) {
            this.validate = v;
            return this;
        }

        public GeoOptions build() {
            return new GeoOptions(this);
        }
    }
}
