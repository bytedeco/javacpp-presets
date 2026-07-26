package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

/**
 * NormalizeFeatures: 行归一化 (Sum to 1)
 */
public class NormalizeFeatures implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        // L1 归一化，使得每行之和为 1
        Tensor norm = data.x.norm(new ScalarOptional(new Scalar(1.0)), new long[] {1}, true);
        data.x = data.x.div(norm.add(new Scalar(1e-6)));
        return data;
    }
}