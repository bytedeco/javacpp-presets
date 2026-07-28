/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.Distance
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

/**
 * Append Euclidean distance of each edge's endpoints to {@code edge_attr}.
 */
public class Distance implements BaseTransform {
    private final boolean norm;

    public Distance() { this(false); }
    public Distance(boolean norm) { this.norm = norm; }

    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        Tensor pos = TransformUtils.requirePos(data);
        Tensor row = ei.select(0, 0);
        Tensor col = ei.select(0, 1);
        Tensor dist = pos.index_select(0, col).sub(pos.index_select(0, row))
                .pow(new Scalar(2)).sum(1).sqrt().view(new long[]{-1, 1});
        if (norm && dist.numel() > 0) {
            dist = dist.div(dist.max().clamp_min(new Scalar(1e-12)));
        }
        data.edge_attr = TransformUtils.catEdgeAttr(data.edge_attr, dist);
        return data;
    }
}
