/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.Center
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.kFloat;

/** Subtract the centroid from {@code pos}. */
public class Center implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        Tensor mean = pos.mean(new long[]{0}, /*keepdim=*/true,
                new ScalarTypeOptional(kFloat()));
        data.pos = pos.sub(mean);
        return data;
    }
}
