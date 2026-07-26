package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * OneHotDegree: 将节点度数作为 One-hot 编码追加到特征中
 */
public  class OneHotDegree implements BaseTransform {
    private int maxDegree;
    public OneHotDegree(int maxDegree) { this.maxDegree = maxDegree; }

//    @Override
    public GraphData call2(GraphData data) {
        long numNodes = data.x.size(0);
        // 计算入度 (对 edge_index 的第二行计数)
        Tensor degree = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kLong())));
        Tensor ones = ones(new long[]{data.edge_index.size(1)}, data.x.options().dtype(new ScalarTypeOptional(kLong())));
        degree.scatter_add_(0, data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1)))), ones);

        // 限制最大度数并转为 One-hot
        degree = degree.clamp(new ScalarOptional(new Scalar(0)),new ScalarOptional(new Scalar( maxDegree)) );
        Tensor oneHot = one_hot(degree, maxDegree + 1).to(data.x.dtype());

        data.x = cat(new TensorVector(data.x, oneHot), 1);
        return data;
    }

    @Override
    public GraphData apply(GraphData data) {
        long numNodes = data.numNodes(); // 使用我们重写的 numNodes 方法更安全

        // 1. 初始化 degree 为 1D: [numNodes]
        Tensor degree = zeros(new long[]{numNodes}, data.x.options().dtype(new ScalarTypeOptional(kLong())));

        // 2. 获取入度索引 (edge_index 的第二行)
        // 使用 select(0, 1) 明确获取第 0 维的第 1 个切片，结果是 1D: [num_edges]
        Tensor col = data.edge_index.select(0, 1);

        // 3. 创建 ones，确保也是 1D 且长度与 col 一致
        Tensor values = ones(col.sizes(), data.x.options().dtype(new ScalarTypeOptional(kLong())));

        // 4. 执行 scatter_add_ (此时 self, index, src 全是 1D)
        degree.scatter_add_(0, col, values);

        // 5. 限制最大度数并转为 One-hot
        // 注意：degree 需要是 Long 类型才能传给 one_hot
        degree = degree.clamp(new ScalarOptional(new Scalar(0)), new ScalarOptional(new Scalar(maxDegree)));
        Tensor oneHot = one_hot(degree, maxDegree + 1).to(data.x.dtype());

        // 6. 拼接特征
        data.x = cat(new TensorVector(data.x, oneHot), 1);

        return data;
    }
}
