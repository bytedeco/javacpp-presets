/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.NormalizeFeatures
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

/**
 * Row-normalize node features so each row sums to 1 (L1 / sum-normalize).
 * Matches PyG {@code NormalizeFeatures} (sum, not Euclidean).
 */
public class NormalizeFeatures implements BaseTransform {

    @Override
    public GraphData apply(GraphData data) {
        Tensor x = TransformUtils.requireX(data);
        // sum over feature dim, keepdim → [N,1]
        Tensor rowSum = x.abs().sum(new long[]{1}, /*keepdim=*/true, new org.bytedeco.pytorch.ScalarTypeOptional());
        data.x = x.div(rowSum.add(new Scalar(1e-12)));
        return data;
    }
}
