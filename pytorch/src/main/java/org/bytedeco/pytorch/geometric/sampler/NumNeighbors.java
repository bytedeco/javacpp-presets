package org.bytedeco.pytorch.geometric.sampler;

public class NumNeighbors {
    // 每一层采样的数量。例如 [10, 5] 表示第一层10个，第二层5个
    private long[] values;

    public NumNeighbors(long... values) {
        this.values = values;
    }

    public long get(int hop) {
        return hop < values.length ? values[hop] : -1;
    }

    public int numHops() {
        return values.length;
    }
}
