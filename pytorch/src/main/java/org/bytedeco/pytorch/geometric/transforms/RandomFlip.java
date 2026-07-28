/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.RandomFlip
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

/** Randomly flip {@code pos} along {@code axis} with probability {@code p}. */
public class RandomFlip implements BaseTransform {
    private final int axis;
    private final double p;

    public RandomFlip(int axis, double p) {
        if (axis < 0) throw new IllegalArgumentException("axis must be >= 0");
        if (p < 0 || p > 1) throw new IllegalArgumentException("p must be in [0,1]");
        this.axis = axis;
        this.p = p;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        if (axis >= pos.size(1)) {
            throw new IllegalArgumentException(
                    "axis=" + axis + " out of range for pos dim=" + pos.size(1));
        }
        if (Math.random() < p) {
            pos.select(1, axis).mul_(new Scalar(-1));
        }
        return data;
    }
}
