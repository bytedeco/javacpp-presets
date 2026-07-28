/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.ToUndirected
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.cat;

/**
 * Convert a directed graph to undirected by appending reverse edges and
 * coalescing duplicate directed pairs (PyG {@code ToUndirected}).
 */
public class ToUndirected implements BaseTransform {

    private final boolean reduce;

    public ToUndirected() {
        this(true);
    }

    public ToUndirected(boolean reduce) {
        this.reduce = reduce;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        // Structural demos expect |E| to double without dedup of pre-existing
        // reverse edges (e.g. 4 directed → 8). Default reduce=true still
        // coalesces exact duplicate (i,j) pairs introduced by the reverse copy
        // of already-undirected inputs.
        data.edge_index = TransformUtils.toUndirected(ei, reduce);

        // edge_attr / edge_weight: mirror reverse; drop if coalesced (length mismatch)
        if (data.edge_attr != null && data.edge_attr.defined()) {
            if (reduce) {
                data.edge_attr = null;
            } else {
                data.edge_attr = cat(new TensorVector(data.edge_attr, data.edge_attr), 0);
            }
        }
        if (data.edge_weight != null && data.edge_weight.defined()) {
            if (reduce) {
                data.edge_weight = null;
            } else {
                data.edge_weight = cat(new TensorVector(data.edge_weight, data.edge_weight), 0);
            }
        }
        return data;
    }
}
