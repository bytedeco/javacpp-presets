package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class SamplePoints implements BaseTransform {
    private final int numSamples;

    public SamplePoints(int numSamples) { this.numSamples = numSamples; }

    @Override
    public GraphData apply(GraphData data) {
        Tensor face = data.get("face");
        Tensor pos = data.pos;

        // 1. 计算面面积 (三角形面积公式)
        Tensor v1 = pos.index_select(0, face.select(0, 0));
        Tensor v2 = pos.index_select(0, face.select(0, 1));
        Tensor v3 = pos.index_select(0, face.select(0, 2));
        Tensor areas = cross(v2.sub(v1), v3.sub(v1), new LongOptional(1)).norm(new ScalarOptional(new Scalar(2)), 1).mul(new Scalar(0.5f));

        // 2. 根据面积权重随机抽取面的索引
        Tensor faceIdx = multinomial(areas, numSamples, true,new GeneratorOptional());

        // 3. 在选中的面内生成随机重心坐标 (Barycentric coordinates)
        Tensor r1 = rand(new long[]{numSamples, 1}, pos.options()).sqrt();
        Tensor r2 = rand(new long[]{numSamples, 1}, pos.options());
        // P = (1-sqrt(r1))v1 + (sqrt(r1)(1-r2))v2 + (sqrt(r1)r2)v3
        Tensor p1 = v1.index_select(0, faceIdx).mul(ones_like(r1).sub(r1));
        Tensor p2 = v2.index_select(0, faceIdx).mul(r1.mul(ones_like(r2).sub(r2)));
        Tensor p3 = v3.index_select(0, faceIdx).mul(r1.mul(r2));

        data.pos = p1.add(p2).add(p3);
        return data;
    }
}