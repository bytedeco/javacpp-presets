/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.LocalCartesian
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

/** Local Cartesian offsets scaled by global max |pos|. */
public class LocalCartesian implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        Tensor pos = TransformUtils.requirePos(data);
        Tensor local = pos.index_select(0, ei.select(0, 1))
                .sub(pos.index_select(0, ei.select(0, 0)));
        Tensor scale = pos.abs().max().clamp_min(new Scalar(1e-12));
        local = local.div(scale);
        data.edge_attr = TransformUtils.catEdgeAttr(data.edge_attr, local);
        return data;
    }
}
