package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Global Pooling (Readout) Functions
 * 将节点特征聚合为图特征 [N, C] -> [BatchSize, C]
 */
public class GlobalPooling {

    /**
     * @param x     节点特征 [N, hidden_channels]
     * @param batch 节点所属图的索引 [N] (如果单图训练，可传入全 0)
     * @param type  聚合类型: "sum", "mean", "max"
     */
    public static Tensor pool(Tensor x, Tensor batch, String type) {
        // 获取 BatchSize (图中最大索引 + 1)
        long batchSize = batch.max().item().toLong() + 1;

        switch (type.toLowerCase()) {
            case "sum":
                // 利用 index_add 实现全局求和
                Tensor sumOut = zeros(new long[]{batchSize, x.size(1)}, x.options());
                return sumOut.index_add_(0, batch, x);

            case "mean":
                // 先求和，再除以每个图的节点数
                Tensor meanSum = zeros(new long[]{batchSize, x.size(1)}, x.options());
                meanSum.index_add_(0, batch, x);

                // 计算每个 batch 的节点计数
                Tensor ones = ones(batch.sizes(), batch.options().dtype(new ScalarTypeOptional(kFloat())));
//                Tensor ones = ones_like(batch);//, x.options());
                Tensor count = zeros(new long[]{batchSize}, x.options());
                count.index_add_(0, batch, ones).clamp_min_(new Scalar(1.0)); // 防止除以 0

                return meanSum.divide(count.unsqueeze(1));

            case "max":
                // 注意：JavaCPP 中 scatter_max 的直接封装可能较复杂
                // 这里提供一个稳健的常用替代写法
                Tensor maxValues = full(new long[]{batchSize, x.size(1)}, new Scalar(-1e9), x.options());
                // 利用 scatter 实现
                return maxValues.index_reduce_(0, batch, x, "amax", true);
//                return scatter_max(x, batch, 0, maxOut);

            default:
                throw new IllegalArgumentException("Unsupported pooling type: " + type);
        }

    }
    /**
     * Global Add Pooling (Sum)
     * batch: [N] 标识每个节点属于哪个图
     */
    public static Tensor global_add_pool(Tensor x, Tensor batch) {
        long batchSize = (batch == null) ? 1 : batch.max().item().toLong() + 1;
        if (batch == null) {
            // Sum dim=0 -> [1, C]
            return x.sum(new long[]{0}, true, new ScalarTypeOptional());
        }
        return AggrUtils.scatter(x, batch, batchSize, "sum");
    }

    /**
     * Global Mean Pooling (Average)
     */
    public static Tensor global_mean_pool(Tensor x, Tensor batch) {
        long batchSize = (batch == null) ? 1 : batch.max().item().toLong() + 1;
        if (batch == null) {
            return x.mean(new long[]{0}, true, new ScalarTypeOptional());
        }
        return AggrUtils.scatter(x, batch, batchSize, "mean");
    }

    /**
     * Global Max Pooling
     */
    public static Tensor global_max_pool(Tensor x, Tensor batch) {
        long batchSize = (batch == null) ? 1 : batch.max().item().toLong() + 1;
        if (batch == null) {
            return x.max(0, true).get0(); // max returns (values, indices)
        }
        // org.bytedeco.pytorch.geometric.utils.AggrUtils.scatter 实现了 scatter_max
        return AggrUtils.scatter(x, batch, batchSize, "max");
    }
}