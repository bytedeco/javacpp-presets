/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.Spherical
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.acos;
import static org.bytedeco.pytorch.global.torch.atan2;
import static org.bytedeco.pytorch.global.torch.stack;

/** 3-D spherical edge attributes {@code [r, phi, theta]}. */
public class Spherical implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        Tensor pos = TransformUtils.requirePos(data);
        if (pos.size(1) < 3) {
            throw new IllegalArgumentException("Spherical requires pos [N,3]");
        }
        Tensor v = pos.index_select(0, ei.select(0, 1))
                .sub(pos.index_select(0, ei.select(0, 0)));
        Tensor x = v.select(1, 0);
        Tensor y = v.select(1, 1);
        Tensor z = v.select(1, 2);
        Tensor r = v.pow(new Scalar(2)).sum(1).sqrt();
        Tensor phi = atan2(y, x).div(new Scalar(2 * Math.PI)).add(new Scalar(0.5));
        // clamp z/r into [-1,1] for numerical stability (avoid acos domain errors
        // and the bias introduced by adding eps to r when r is already exact)
        Tensor cosTheta = z.div(r.clamp_min(new Scalar(1e-12)))
                .clamp(new ScalarOptional(new Scalar(-1.0)), new ScalarOptional(new Scalar(1.0)));
        Tensor theta = acos(cosTheta).div(new Scalar(Math.PI));
        Tensor sph = stack(new TensorVector(r, phi, theta), 1);
        data.edge_attr = TransformUtils.catEdgeAttr(data.edge_attr, sph);
        return data;
    }
}
