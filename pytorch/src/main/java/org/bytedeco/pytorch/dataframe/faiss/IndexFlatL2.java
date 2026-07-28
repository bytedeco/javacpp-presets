package org.bytedeco.pytorch.dataframe.faiss;

/** {@code faiss.IndexFlatL2(d)} — brute-force squared L2. */
public final class IndexFlatL2 extends IndexFlat {
    private static final long serialVersionUID = 1L;

    public IndexFlatL2(int d) {
        super(d, MetricType.METRIC_L2);
    }

    @Override
    public String indexType() {
        return "FlatL2";
    }
}
