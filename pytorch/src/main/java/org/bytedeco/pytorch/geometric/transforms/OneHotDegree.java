/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.OneHotDegree
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Append a one-hot encoding of in-degree (clamped to {@code [0, maxDegree]})
 * to node features (PyG {@code OneHotDegree}).
 */
public class OneHotDegree implements BaseTransform {

    private final int maxDegree;

    public OneHotDegree(int maxDegree) {
        if (maxDegree < 0) {
            throw new IllegalArgumentException("maxDegree must be >= 0");
        }
        this.maxDegree = maxDegree;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor x = TransformUtils.requireX(data);
        Tensor ei = TransformUtils.requireEdgeIndex(data);
        long numNodes = x.size(0);

        Tensor degree = zeros(new long[]{numNodes},
                x.options().dtype(new ScalarTypeOptional(kLong())));
        Tensor col = ei.select(0, 1).to(kLong());
        Tensor values = ones(new long[]{col.size(0)},
                x.options().dtype(new ScalarTypeOptional(kLong())));
        degree.scatter_add_(0, col, values);

        degree = degree.clamp(new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(maxDegree)));
        Tensor oneHot = one_hot(degree, maxDegree + 1).to(x.dtype());
        data.x = cat(new TensorVector(x, oneHot), 1);
        return data;
    }

    public int getMaxDegree() {
        return maxDegree;
    }
}
