package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class RandomJitter implements BaseTransform {
    private final float sigma;

    public RandomJitter(float sigma) { this.sigma = sigma; }

    @Override
    public GraphData apply(GraphData data) {
        // 生成与 pos 形状一致的噪声
        Tensor noise = randn_like(data.pos).mul(new Scalar(sigma));
        data.pos = data.pos.add(noise);
        return data;
    }
}