/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * PyG peer: torch_geometric.transforms.RandomRotate
 */
package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Rotate {@code pos} by a random angle in {@code [-degrees, +degrees]} around
 * the given axis (0=X, 1=Y, 2=Z). Requires 3-D coordinates.
 */
public class RandomRotate implements BaseTransform {
    private final float degrees;
    private final int axis;

    public RandomRotate(float degrees, int axis) {
        this.degrees = degrees;
        this.axis = axis;
    }

    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        if (pos.size(1) != 3) {
            throw new IllegalArgumentException(
                    "RandomRotate requires pos [N,3], got size(1)=" + pos.size(1));
        }
        double angle = (Math.random() * 2 - 1) * Math.toRadians(degrees);
        float s = (float) Math.sin(angle);
        float c = (float) Math.cos(angle);
        float[] flat;
        if (axis == 2) {
            flat = new float[]{ c, s, 0,  -s, c, 0,  0, 0, 1};
        } else if (axis == 1) {
            flat = new float[]{ c, 0,-s,   0, 1, 0,  s, 0, c};
        } else {
            flat = new float[]{ 1, 0, 0,   0, c, s,  0,-s, c};
        }
        Tensor rot = tensor(flat, pos.options()).view(3, 3);
        data.pos = pos.mm(rot);
        return data;
    }
}
