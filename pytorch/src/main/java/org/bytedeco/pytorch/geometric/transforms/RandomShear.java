package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class RandomShear implements BaseTransform {
    private final float sigma;

    public RandomShear(float sigma) { this.sigma = sigma; }

    @Override
    public GraphData apply(GraphData data) {
        // 构造一个单位矩阵加上随机噪声的剪切矩阵
        // Shear matrix S = I + E, 其中 E 是采样自 N(0, sigma) 的噪声
        Tensor matrix = eye(data.pos.size(1), data.pos.options());
        matrix.add_(randn_like(matrix).mul_(new Scalar(sigma)));

        data.pos = data.pos.mm(matrix);
        return data;
    }
}