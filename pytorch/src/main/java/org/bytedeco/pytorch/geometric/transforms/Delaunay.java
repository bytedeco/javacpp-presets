/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * Approximate Delaunay via Gabriel graph (pure Tensor, no OpenCV).
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.kLong;

/**
 * Build an undirected Gabriel graph from {@code pos} (a subgraph of the
 * Delaunay triangulation). Sufficient for topology demos.
 */
public class Delaunay implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        long n = pos.size(0);
        if (n < 2) {
            data.edge_index = org.bytedeco.pytorch.global.torch.zeros(
                    new long[]{2, 0}, TransformUtils.longOptsLike(pos));
            return data;
        }
        // dist^2 [N,N]
        Tensor distMat = pos.unsqueeze(1).sub(pos.unsqueeze(0)).pow(new Scalar(2)).sum(2);
        Tensor mid = pos.unsqueeze(1).add(pos.unsqueeze(0)).div(new Scalar(2.0));
        Tensor distToMid = mid.unsqueeze(2).sub(pos.unsqueeze(0).unsqueeze(0))
                .pow(new Scalar(2)).sum(3); // [N,N,N]
        Tensor radiusSq = distMat.div(new Scalar(4.0));
        Tensor isWithin = distToMid.lt(radiusSq.unsqueeze(2).sub(new Scalar(1e-5)));
        Tensor countWithin = isWithin.sum(2);
        Tensor adjMask = countWithin.eq(new Scalar(0));
        adjMask.fill_diagonal_(new Scalar(0));
        data.edge_index = adjMask.nonzero().t().to(kLong());
        return data;
    }
}
