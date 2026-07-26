package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * GraphSizeNorm
 * 根据图的大小归一化特征，常用于 Transformer GNN。
 */
public class GraphSizeNorm extends Module {

    public GraphSizeNorm() {
        super();
    }

    public Tensor forward(Tensor x, Tensor batch) {
        // 如果没有 batch，视作单个大图
        if (batch == null) {
            // size = N
            double size = x.size(0);
            double scale = 1.0 / Math.sqrt(size);
            return x.mul(new Scalar(scale));
        }

        // 1. 计算每个图的节点数 (Degree of batch index)
        long batchSize = batch.max().item().toLong() + 1;
        Tensor graphSizes = AggrUtils.compute_degree(batch, batchSize); // [BatchSize]

        // 2. 映射回每个节点 (Broadcasting)
        // [BatchSize] -> [N]
        Tensor nodeScale = graphSizes.index_select(0, batch);

        // 3. 计算系数 1 / sqrt(size)
        // inv_sqrt = size^(-0.5)
        Tensor scale = nodeScale.rsqrt();

        // 4. 乘法 (广播到特征维度)
        // [N] -> [N, 1] * [N, C]
        return x.mul(scale.unsqueeze(1));
    }
}