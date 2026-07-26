package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.full;

/**
 * Constant: 为每个节点特征追加常数值
 */
public class Constant implements BaseTransform {
    private double value;
    public Constant(double value) { this.value = value; }

    @Override
    public GraphData apply(GraphData data) {
        long numNodes = data.x.size(0);
        Tensor c = full(new long[]{numNodes, 1}, new Scalar(value) , data.x.options());
        data.x = cat(new TensorVector(data.x, c), 1);
        return data;
    }
}
