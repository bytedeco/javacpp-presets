/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.Cartesian
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

/** Append normalized Cartesian offsets {@code pos_j - pos_i} to edge_attr. */
public class Cartesian implements BaseTransform {
    private final boolean norm;
    public Cartesian() { this(true); }
    public Cartesian(boolean norm) { this.norm = norm; }

    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        Tensor pos = TransformUtils.requirePos(data);
        Tensor cart = pos.index_select(0, ei.select(0, 1))
                .sub(pos.index_select(0, ei.select(0, 0)));
        if (norm) {
            // Map roughly into [0,1] using global max extent (PyG-style)
            Tensor scale = pos.max().mul(new Scalar(2)).clamp_min(new Scalar(1e-12));
            cart = cart.div(scale).add(new Scalar(0.5));
        }
        data.edge_attr = TransformUtils.catEdgeAttr(data.edge_attr, cart);
        return data;
    }
}
