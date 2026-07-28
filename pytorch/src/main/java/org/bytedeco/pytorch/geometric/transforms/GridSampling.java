/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.GridSampling
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.kLong;
import static org.bytedeco.pytorch.global.torch.unique_dim;

/**
 * Voxel-grid downsampling: keep one representative point per occupied voxel
 * of edge length {@code size}.
 */
public class GridSampling implements BaseTransform {
    private final float size;
    public GridSampling(float size) {
        if (size <= 0) throw new IllegalArgumentException("size must be > 0");
        this.size = size;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        // voxel id per axis
        Tensor cluster = pos.div(new Scalar(size)).floor().to(kLong()); // [N, D]
        // unique rows → inverse maps each point → cluster id
        T_TensorTensorTensor_T out = unique_dim(cluster, 0, true, true, false);
        Tensor inverse = out.get1(); // [N]
        long numClusters = out.get0().size(0);

        // first occurrence of each cluster id in original order:
        // sort by inverse, take group starts
        Tensor rep = TransformUtils.firstOccurrenceIndex(inverse, numClusters);

        data.pos = pos.index_select(0, rep);
        if (data.x != null && data.x.defined()) {
            data.x = data.x.index_select(0, rep);
        }
        if (data.get("norm") != null && data.get("norm").defined()) {
            data.put("norm", data.get("norm").index_select(0, rep));
        }
        data.edge_index = null; // topology invalidated
        return data;
    }
}
