/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.RandomScale
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.geometric.data.GraphData;

/** Isotropically scale {@code pos} by a uniform factor in {@code [min, max]}. */
public class RandomScale implements BaseTransform {
    private final float min, max;

    public RandomScale(float min, float max) {
        if (max < min) throw new IllegalArgumentException("max < min");
        this.min = min;
        this.max = max;
    }

    @Override
    public GraphData apply(GraphData data) {
        TransformUtils.requirePos(data);
        float scale = min + (float) Math.random() * (max - min);
        data.pos.mul_(new Scalar(scale));
        return data;
    }
}
