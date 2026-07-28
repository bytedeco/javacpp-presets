/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.PointPairFeatures
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.cat;

/** PPF edge features {@code [d, n_i·d̂, n_j·d̂, n_i·n_j]} (requires {@code norm}). */
public class PointPairFeatures implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        Tensor pos = TransformUtils.requirePos(data);
        Tensor norm = data.get("norm");
        if (norm == null || !norm.defined()) {
            throw new IllegalArgumentException("PointPairFeatures requires data['norm']");
        }
        Tensor row = ei.select(0, 0);
        Tensor col = ei.select(0, 1);
        Tensor d = pos.index_select(0, col).sub(pos.index_select(0, row));
        Tensor dist = d.pow(new Scalar(2)).sum(1).sqrt().view(new long[]{-1, 1});
        d = d.div(dist.add(new Scalar(1e-7)));
        Tensor ni = norm.index_select(0, row);
        Tensor nj = norm.index_select(0, col);
        Tensor a1 = ni.mul(d).sum(1).view(new long[]{-1, 1});
        Tensor a2 = nj.mul(d).sum(1).view(new long[]{-1, 1});
        Tensor a3 = ni.mul(nj).sum(1).view(new long[]{-1, 1});
        Tensor ppf = cat(new TensorVector(dist, a1, a2, a3), 1);
        data.edge_attr = TransformUtils.catEdgeAttr(data.edge_attr, ppf);
        return data;
    }
}
