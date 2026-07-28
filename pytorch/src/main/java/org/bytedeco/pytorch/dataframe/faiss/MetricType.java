package org.bytedeco.pytorch.dataframe.faiss;

/**
 * FAISS metric types — codes match C++ {@code faiss::MetricType} exactly
 * so native {@code write_index}/{@code read_index} files interoperate with
 * Python {@code faiss}.
 *
 * <pre>
 *   METRIC_INNER_PRODUCT = 0
 *   METRIC_L2            = 1
 * </pre>
 */
public enum MetricType {
    /** Inner product — higher is better (FAISS returns raw IP as "distance"). */
    METRIC_INNER_PRODUCT(0),
    /** Squared L2 (Euclidean) — lower is better. */
    METRIC_L2(1);

    private final int code;

    MetricType(int code) {
        this.code = code;
    }

    /** On-disk / C++ enum ordinal used by FAISS binary format. */
    public int code() {
        return code;
    }

    public boolean lowerIsBetter() {
        return this == METRIC_L2;
    }

    public static MetricType fromCode(int code) {
        return switch (code) {
            case 0 -> METRIC_INNER_PRODUCT;
            case 1 -> METRIC_L2;
            default -> METRIC_L2; // unknown → L2
        };
    }
}
