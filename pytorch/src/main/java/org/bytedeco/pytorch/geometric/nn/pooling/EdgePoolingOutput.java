package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.Tensor;

public class EdgePoolingOutput {
    public Tensor x;
    public Tensor cluster;

    public EdgePoolingOutput(Tensor x, Tensor cluster) {
        this.x = x;
        this.cluster = cluster;
    }
}