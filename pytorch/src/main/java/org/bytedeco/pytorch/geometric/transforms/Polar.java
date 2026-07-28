/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.Polar
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.atan2;
import static org.bytedeco.pytorch.global.torch.stack;

/** 2-D polar edge attributes {@code [r, theta]} with theta in [0,1]. */
public class Polar implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        Tensor pos = TransformUtils.requirePos(data);
        if (pos.size(1) < 2) {
            throw new IllegalArgumentException("Polar requires pos with >= 2 dims");
        }
        Tensor v = pos.index_select(0, ei.select(0, 1))
                .sub(pos.index_select(0, ei.select(0, 0)));
        Tensor r = v.pow(new Scalar(2)).sum(1).sqrt();
        Tensor theta = atan2(v.select(1, 1), v.select(1, 0))
                .div(new Scalar(2 * Math.PI)).add(new Scalar(0.5));
        Tensor polar = stack(new TensorVector(r, theta), 1);
        data.edge_attr = TransformUtils.catEdgeAttr(data.edge_attr, polar);
        return data;
    }
}
