package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.Tensor;

/**
 * Thin facade over {@link AggrUtils} preserving the historical Scatter static API
 * used by GATConv and other layers.
 *
 * <p>All heavy lifting (sum/mean/max/min/mul, isolated-node fill, long index coercion)
 * lives in {@link AggrUtils}. Prefer calling AggrUtils directly in new code.
 */
public final class Scatter {

    private Scatter() {}

    /**
     * Scatter {@code src} [E, F...] by {@code index} [E] into [dimSize, F...] with reduce.
     * Supported reduce: add/sum/mean/max/min/mul/prod.
     */
    public static Tensor scatter(Tensor src, Tensor index, long dimSize, String reduce) {
        return AggrUtils.scatter(src, index, dimSize, reduce);
    }

    /** scatter_add along dim 0 (most common GNN path). */
    public static Tensor scatter_add(Tensor src, Tensor index, long dim, long dimSize) {
        if (dim != 0) {
            throw new UnsupportedOperationException(
                    "Scatter.scatter_add only supports dim=0 (got " + dim + "); use AggrUtils for general cases");
        }
        return AggrUtils.scatter(src, index, dimSize, "sum");
    }

    public static Tensor scatter_mean(Tensor src, Tensor index, long dim, long dimSize) {
        if (dim != 0) {
            throw new UnsupportedOperationException(
                    "Scatter.scatter_mean only supports dim=0 (got " + dim + ")");
        }
        return AggrUtils.scatter(src, index, dimSize, "mean");
    }

    public static Tensor scatter_max(Tensor src, Tensor index, long dim, long dimSize) {
        if (dim != 0) {
            throw new UnsupportedOperationException(
                    "Scatter.scatter_max only supports dim=0 (got " + dim + ")");
        }
        return AggrUtils.scatter(src, index, dimSize, "max");
    }

    public static Tensor scatter_min(Tensor src, Tensor index, long dim, long dimSize) {
        if (dim != 0) {
            throw new UnsupportedOperationException(
                    "Scatter.scatter_min only supports dim=0 (got " + dim + ")");
        }
        return AggrUtils.scatter(src, index, dimSize, "min");
    }

    /** @deprecated Use {@link #scatter(Tensor, Tensor, long, String)} */
    @Deprecated
    public static Tensor scatter1(Tensor src, Tensor index, long dimSize, String reduce) {
        return scatter(src, index, dimSize, reduce);
    }

    /** @deprecated Use {@link #scatter(Tensor, Tensor, long, String)} */
    @Deprecated
    public static Tensor scatter2(Tensor src, Tensor index, long dimSize, String reduce) {
        return scatter(src, index, dimSize, reduce);
    }
}
