package org.bytedeco.pytorch.dataframe.faiss;

/** {@code faiss.IndexFlatIP(d)} — brute-force inner product. */
public final class IndexFlatIP extends IndexFlat {
    private static final long serialVersionUID = 1L;

    public IndexFlatIP(int d) {
        super(d, MetricType.METRIC_INNER_PRODUCT);
    }

    @Override
    public String indexType() {
        return "FlatIP";
    }
}
