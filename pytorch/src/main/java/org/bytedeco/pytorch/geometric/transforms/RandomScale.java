package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

public  class RandomScale implements BaseTransform {
    private final float min, max;

    public RandomScale(float min, float max) {
        this.min = min;
        this.max = max;
    }

    @Override
    public GraphData apply(GraphData data) {
        float scale = min + (float) Math.random() * (max - min);
        data.pos.mul_(new Scalar(scale));
        return data;
    }
}