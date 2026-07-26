package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

public  class RandomFlip implements BaseTransform {
    private final int axis;
    private final double p;

    public RandomFlip(int axis, double p) {
        this.axis = axis;
        this.p = p;
    }

    @Override
    public GraphData apply(GraphData data) {
        if (Math.random() < p) {
            // 将指定轴的坐标取反
            // select(1, axis) 选中 [N, D] 中的那一列坐标
            data.pos.select(1, axis).mul_(new Scalar(-1));
        }
        return data;
    }
}
