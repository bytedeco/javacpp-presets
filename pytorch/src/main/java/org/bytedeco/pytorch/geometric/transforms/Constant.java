/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.Constant
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.full;

/** Append a constant feature column to {@code x}. */
public class Constant implements BaseTransform {

    private final double value;
    private final String attrName;

    public Constant(double value) {
        this(value, "x");
    }

    public Constant(double value, String attrName) {
        this.value = value;
        this.attrName = attrName == null ? "x" : attrName;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor x = TransformUtils.requireX(data);
        long numNodes = x.size(0);
        Tensor c = full(new long[]{numNodes, 1}, new Scalar(value), x.options());
        if ("x".equals(attrName)) {
            data.x = cat(new TensorVector(x, c), 1);
        } else {
            data.put(attrName, c);
        }
        return data;
    }
}
