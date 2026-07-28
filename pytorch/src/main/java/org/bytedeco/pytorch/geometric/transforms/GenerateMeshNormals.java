/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.GenerateMeshNormals
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.cross;
import static org.bytedeco.pytorch.global.torch.zeros_like;

/** Accumulate per-face normals onto vertices → {@code data['norm']}. */
public class GenerateMeshNormals implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        Tensor face = data.get("face");
        if (face == null || !face.defined()) {
            throw new IllegalArgumentException("GenerateMeshNormals requires data['face']");
        }
        Tensor v1 = pos.index_select(0, face.select(0, 0));
        Tensor v2 = pos.index_select(0, face.select(0, 1));
        Tensor v3 = pos.index_select(0, face.select(0, 2));
        Tensor faceNormals = cross(v2.sub(v1), v3.sub(v1), new LongOptional(1));
        Tensor nodeNormals = zeros_like(pos);
        for (int i = 0; i < 3; i++) {
            nodeNormals.scatter_add_(0,
                    face.select(0, i).unsqueeze(-1).expand_as(faceNormals),
                    faceNormals);
        }
        Tensor denom = nodeNormals.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true)
                .add(new Scalar(1e-7));
        data.put("norm", nodeNormals.div(denom));
        return data;
    }
}
