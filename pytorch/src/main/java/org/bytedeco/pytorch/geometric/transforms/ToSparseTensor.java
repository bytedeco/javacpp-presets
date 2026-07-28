/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.ToSparseTensor
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.sparse_coo_tensor;

/** Build coalesced sparse COO {@code adj_t} from {@code edge_index}. */
public class ToSparseTensor implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        long numNodes = TransformUtils.numNodes(data);
        Tensor values = data.edge_weight != null && data.edge_weight.defined()
                ? data.edge_weight
                : ones(new long[]{ei.size(1)},
                    data.x != null ? data.x.options() : TransformUtils.floatOptsLike(ei));
        Tensor adj = sparse_coo_tensor(ei.to(org.bytedeco.pytorch.global.torch.kLong()),
                values, new long[]{numNodes, numNodes});
        data.put("adj_t", adj.coalesce());
        return data;
    }
}
